"""Stream a BF16 checkout into an fp4 record, or a mixed feed-forward record.

Language projections and the feed-forward are fp4 E2M1 with an E8M0 scale every
32 along K, except feed-forward layers selected by ``--ffn-fp8``. The embedding,
``lm_head``, MTP matrices, those promoted layers, and every vision-tower matrix
are fp8 E4M3. One-dimensional parameters and the depthwise convolution stay
Metal ``half``.
``repack_mtp_half`` rewrites MTP matrices that are still fp4 or fp8 into
``mtp.bin`` as Metal ``half``. A new quantize writes ``layout`` 2. ``resident.bin``
is name-sorted text, then layers as gate, up, and down. MTP tensors are
``mtp.bin`` and vision tensors are ``vision.bin``, each dense from offset 0.
An older manifest stays in shard order.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import BinaryIO

import numpy as np

from qwen38_27b.metal41.budget import (
    LAYOUT_PREFIX,
    LAYOUT_SCATTER,
    is_mtp,
    is_vision,
    language_ffn_layer,
    layout_version,
    resident_names,
    tensor_span,
)
from qwen38_27b.metal41.formats import (
    ffn_promote_count,
    pack_fp4_matrix,
    pack_fp8_matrix,
    promoted_layer_ids,
    relative_error,
    stored_k,
    stored_rows,
)

RECORD_NAME = "Qwen3.8-27B-fp4"
_FFN = {
    "mlp.gate_proj.weight": "gate",
    "mlp.up_proj.weight": "up",
    "mlp.down_proj.weight": "down",
}


def _align(value: int) -> int:
    return (value + 127) & ~127


def _bf16(raw: np.ndarray, shape: list[int]) -> np.ndarray:
    bits = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
    return (bits << 16).view(np.float32).reshape(shape)


def _header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as handle:
        size = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(size))
    return header, 8 + size


def _load(path: Path, header: dict, base: int, name: str) -> np.ndarray:
    info = header[name]
    start, stop = info["data_offsets"]
    with path.open("rb") as handle:
        handle.seek(base + start)
        raw = handle.read(stop - start)
    dtype = info["dtype"]
    shape = [int(value) for value in info["shape"]]
    if dtype == "BF16":
        return _bf16(np.frombuffer(raw, dtype=np.uint16).copy(), shape)
    if dtype == "F16":
        return np.frombuffer(raw, dtype=np.float16).copy().reshape(shape).astype(np.float32)
    if dtype == "F32":
        return np.frombuffer(raw, dtype=np.float32).copy().reshape(shape)
    message = f"{name} dtype {dtype} is not packed"
    raise ValueError(message)


def _durable(handle: BinaryIO) -> None:
    """Flush one file and wait until the bytes are on disk."""
    handle.flush()
    os.fsync(handle.fileno())


def _write_dense(path: Path, blob: bytes) -> int:
    """Append ``blob`` with no gap. Vision tensors use this."""
    with path.open("ab") as handle:
        offset = handle.seek(0, 2)
        handle.write(blob)
        _durable(handle)
    return offset


def _append(path: Path, blob: bytes) -> int:
    with path.open("ab") as handle:
        end = handle.seek(0, 2)
        pad = _align(end) - end
        if pad:
            handle.write(b"\x00" * pad)
            end += pad
        handle.write(blob)
        _durable(handle)
    return end


def _half_entry(array: np.ndarray) -> tuple[bytes, dict]:
    """Pack as IEEE binary16, the same width Metal ``half`` uses."""
    flat = np.ascontiguousarray(array, dtype=np.float16)
    body = {"dtype": "half", "shape": list(array.shape), "stored_shape": list(flat.shape)}
    return flat.tobytes(), body


def _half_matrix_entry(array: np.ndarray) -> tuple[bytes, dict]:
    """Pack a matrix as Metal ``half`` with rows and K padded to 64 for ``matmul2d``."""
    matrix = np.ascontiguousarray(array, dtype=np.float32)
    logical = list(matrix.shape)
    if matrix.ndim == 1:
        return _half_entry(matrix)
    if matrix.ndim > 2:
        matrix = matrix.reshape(matrix.shape[0], -1)
    rows = stored_rows(matrix.shape[0])
    cols = ((matrix.shape[1] + 63) // 64) * 64
    padded = np.zeros((rows, cols), dtype=np.float16)
    padded[: matrix.shape[0], : matrix.shape[1]] = matrix.astype(np.float16)
    body = {"dtype": "half", "shape": logical, "stored_shape": [rows, cols]}
    return padded.tobytes(), body


def record_name(percent: float | None, layers: int) -> str:
    """``<model>-fp4``, ``<model>-fp8``, or ``<model>-fp8-e<P>`` for a partial promotion.

    ``<P>`` is the ``--ffn-fp8`` percentage. ``20`` is ``Qwen3.8-27B-fp8-e20``.
    """
    if percent is None or float(percent) <= 0:
        return RECORD_NAME
    if ffn_promote_count(float(percent), layers) >= layers:
        return "Qwen3.8-27B-fp8"
    token = str(int(percent)) if float(percent) == int(percent) else f"{float(percent):g}"
    return f"Qwen3.8-27B-fp8-e{token}"


def _language_ffn(name: str) -> tuple[int, str] | None:
    parts = name.split(".")
    if "mtp" in parts or "layers" not in parts:
        return None
    index = parts.index("layers")
    tail = ".".join(parts[index + 2 :])
    kind = _FFN.get(tail)
    if kind is None:
        return None
    return int(parts[index + 1]), kind


def _fp8_entry(array: np.ndarray) -> tuple[bytes, dict]:
    matrix = np.ascontiguousarray(array, dtype=np.float32)
    logical = list(matrix.shape)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    elif matrix.ndim > 2:
        matrix = matrix.reshape(matrix.shape[0], -1)
    rows = stored_rows(matrix.shape[0])
    data, scales, stored_r, stored_c = pack_fp8_matrix(matrix, rows=rows)
    pad = _align(len(data)) - len(data)
    blob = data + (b"\x00" * pad) + scales
    body = {
        "dtype": "fp8",
        "shape": logical,
        "stored_shape": [stored_r, stored_c],
        "data_nbytes": len(data),
        "scale_nbytes": len(scales),
        "scale_pad": pad,
    }
    return blob, body


def _fp4_entry(array: np.ndarray) -> tuple[bytes, dict]:
    matrix = np.ascontiguousarray(array, dtype=np.float32)
    logical = list(matrix.shape)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    elif matrix.ndim > 2:
        matrix = matrix.reshape(matrix.shape[0], -1)
    rows = stored_rows(matrix.shape[0])
    data, scales, stored_r, stored_c = pack_fp4_matrix(matrix, rows=rows)
    pad = _align(len(data)) - len(data)
    blob = data + (b"\x00" * pad) + scales
    body = {
        "dtype": "fp4",
        "shape": logical,
        "stored_shape": [stored_r, stored_c],
        "data_nbytes": len(data),
        "scale_nbytes": len(scales),
        "scale_pad": pad,
    }
    return blob, body


def _mtp_needs_half(tensors: dict[str, dict]) -> bool:
    """True when an MTP matrix is still packed as fp4 or fp8."""
    return any(
        str(name).startswith("mtp.") and str(entry.get("dtype")) in {"fp4", "fp8"}
        for name, entry in tensors.items()
        if isinstance(entry, dict)
    )


def _keep_half(name: str, array: np.ndarray) -> bool:
    if name.endswith("conv1d.weight") or name.endswith("A_log") or name.endswith("dt_bias"):
        return True
    if name.endswith(".bias") or array.ndim == 1:
        return True
    return False


def _weight_map(src: Path) -> dict[str, str]:
    index = src / "model.safetensors.index.json"
    if index.is_file():
        body = json.loads(index.read_text())
        mapping = body.get("weight_map")
        if isinstance(mapping, dict):
            return {str(name): str(shard) for name, shard in mapping.items()}
    singles = sorted(src.glob("*.safetensors"))
    if len(singles) == 1:
        header, _base = _header(singles[0])
        return {name: singles[0].name for name in header if name != "__metadata__"}
    message = f"{src.name} has no safetensors index"
    raise FileNotFoundError(message)


def _groups(weight_map: dict[str, str]) -> dict[str, list[str]]:
    shards: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        shards.setdefault(shard, []).append(name)
    return shards


def _budget(body: dict, percent: float | None, count: int) -> None:
    stored_percent = body.get("ffn_fp8_percent")
    stored = body.get("ffn_fp8_layers")
    if count == 0:
        if stored_percent not in (None, 0, 0.0) or stored not in (None, []):
            message = "this record promotes feed-forward layers"
            raise RuntimeError(message)
        return
    if stored_percent is None or float(stored_percent) != float(percent):
        message = "record percentage does not match --ffn-fp8"
        raise RuntimeError(message)
    if not isinstance(stored, list) or len(stored) != count:
        message = "record layer list does not match --ffn-fp8"
        raise RuntimeError(message)


def _score_layers(
    src: Path, weight_map: dict[str, str], layers: int
) -> dict[int, dict[str, float]]:
    scores: dict[int, dict[str, float]] = {layer: {} for layer in range(layers)}
    for shard, names in sorted(_groups(weight_map).items()):
        path = src / shard
        header, base = _header(path)
        for name in names:
            found = _language_ffn(name)
            if found is None:
                continue
            layer, kind = found
            if layer < 0 or layer >= layers:
                message = f"{name} is outside {layers} language layers"
                raise RuntimeError(message)
            array = _load(path, header, base, name)
            scores[layer][kind] = relative_error(array)
            del array
            print(f"  score {name}", flush=True)
    return scores


def _promoted(
    src: Path,
    weight_map: dict[str, str],
    output: Path,
    percent: float | None,
    layers: int,
) -> list[int]:
    count = 0 if percent is None else ffn_promote_count(percent, layers)
    if count == 0:
        return []
    if count == layers:
        return list(range(layers))
    path = output / "ffn_error.json"
    progress = output / "quantize_progress.json"
    if path.is_file():
        body = json.loads(path.read_text())
        if float(body.get("percent", -1)) != float(percent) or int(body.get("count", -1)) != count:
            message = "ffn_error.json does not match --ffn-fp8"
            raise RuntimeError(message)
        chosen = sorted(int(item) for item in body["promoted"])
        if len(chosen) != count:
            message = "ffn_error.json promote count does not match"
            raise RuntimeError(message)
        return chosen
    if progress.is_file():
        message = "ffn_error.json is missing from an unfinished record"
        raise RuntimeError(message)
    scores = _score_layers(src, weight_map, layers)
    layer_scores: dict[int, float] = {}
    for layer in range(layers):
        parts = scores[layer]
        if set(parts) != {"gate", "up", "down"}:
            message = f"layer {layer} is missing a feed-forward score"
            raise RuntimeError(message)
        layer_scores[layer] = max(parts.values())
    chosen = promoted_layer_ids(layer_scores, count)
    body = {
        "percent": percent,
        "count": count,
        "layers": layers,
        "scores": {
            str(layer): {**scores[layer], "layer": layer_scores[layer]} for layer in range(layers)
        },
        "promoted": chosen,
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(body))
    os.replace(temporary, path)
    return chosen


def _numel(shape: list[int]) -> int:
    count = 1
    for dim in shape:
        count *= int(dim)
    return count


def _matrix_dims(shape: list[int]) -> tuple[int, int]:
    if len(shape) == 1:
        return 1, int(shape[0])
    cols = 1
    for dim in shape[1:]:
        cols *= int(dim)
    return int(shape[0]), cols


def _keep_half_shape(name: str, shape: list[int]) -> bool:
    if name.endswith("conv1d.weight") or name.endswith("A_log") or name.endswith("dt_bias"):
        return True
    if name.endswith(".bias") or len(shape) == 1:
        return True
    return False


def _vision_matrix(name: str, shape: list[int]) -> bool:
    """A vision-tower matrix. Norms and biases stay half."""
    return is_vision(name) and not _keep_half_shape(name, shape)


def _token_fp8(name: str) -> bool:
    """The embedding and ``lm_head`` are fp8. A bias or a norm stays half."""
    return name.endswith("embed_tokens.weight") or name.endswith("lm_head.weight")


def _mtp_matrix(name: str, shape: list[int]) -> bool:
    """An MTP matrix. Norms and biases stay half."""
    return is_mtp(name) and not _keep_half_shape(name, shape)


def _wide_fp8(name: str, shape: list[int], *, fp8: bool) -> bool:
    """True when the packed matrix is fp8 rather than fp4."""
    return fp8 or _vision_matrix(name, shape) or _mtp_matrix(name, shape) or _token_fp8(name)


def _planned_span(name: str, shape: list[int], *, fp8: bool) -> int:
    """Bytes ``_pack_array`` will write, including the intra-tensor scale pad."""
    if _keep_half_shape(name, shape):
        return _numel(shape) * 2
    rows, cols = _matrix_dims(shape)
    stored_r = stored_rows(rows)
    stored_c = stored_k(cols)
    wide = _wide_fp8(name, shape, fp8=fp8)
    data = stored_r * stored_c if wide else stored_r * (stored_c // 2)
    scales = stored_r * (stored_c // 32)
    return data + (_align(data) - data) + scales


def _pack_array(name: str, array: np.ndarray, chosen: set[int]) -> tuple[bytes, dict]:
    found = _language_ffn(name)
    if _keep_half(name, array):
        flat = array
        if name.endswith("conv1d.weight") and array.ndim >= 2:
            flat = array.reshape(array.shape[0], -1)
        return _half_entry(flat)
    shape = [int(item) for item in array.shape]
    promoted = found is not None and found[0] in chosen
    if _wide_fp8(name, shape, fp8=promoted):
        return _fp8_entry(array)
    return _fp4_entry(array)


def _promoted_fp8(name: str, chosen: set[int]) -> bool:
    found = _language_ffn(name)
    return found is not None and found[0] in chosen


def _resume_matches(name: str, entry: dict, chosen: set[int]) -> None:
    shape = entry.get("shape")
    dims = [int(item) for item in shape] if isinstance(shape, list) else []
    if _vision_matrix(name, dims) and entry.get("dtype") != "fp8":
        message = f"{name} is stored as {entry.get('dtype')}"
        raise RuntimeError(message)
    if _language_ffn(name) is not None:
        expected = "fp8" if _promoted_fp8(name, chosen) else "fp4"
        if entry.get("dtype") != expected:
            message = f"{name} is stored as {entry.get('dtype')}"
            raise RuntimeError(message)
    if _token_fp8(name) and len(dims) >= 2 and entry.get("dtype") != "fp8":
        message = f"{name} is stored as {entry.get('dtype')}"
        raise RuntimeError(message)
    if _mtp_matrix(name, dims) and entry.get("dtype") != "fp8":
        message = f"{name} is stored as {entry.get('dtype')}"
        raise RuntimeError(message)
    if is_mtp(name) and str(entry.get("file") or "") != "mtp.bin":
        message = f"{name} is not in mtp.bin"
        raise RuntimeError(message)


def _sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _digest_matches(path: Path, entry: dict) -> bool:
    """True when ``entry`` names a digest and those bytes are at its offset."""
    digest = entry.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        return False
    try:
        offset = int(entry["offset"])
        span = tensor_span(entry)
    except (KeyError, TypeError, ValueError):
        return False
    if span < 1 or offset < 0 or not path.is_file() or path.stat().st_size < offset + span:
        return False
    with path.open("rb") as handle:
        handle.seek(offset)
        found = handle.read(span)
    return len(found) == span and hashlib.sha256(found).hexdigest() == digest


def _save_progress(path: Path, done: dict[str, dict], layout: int) -> None:
    body: dict[str, object] = {"tensors": done}
    if layout == LAYOUT_PREFIX:
        body["layout"] = LAYOUT_PREFIX
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(body))
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _tensor_shapes(src: Path, weight_map: dict[str, str]) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    for shard, names in _groups(weight_map).items():
        header, _base = _header(src / shard)
        for name in names:
            info = header[name]
            shapes[name] = [int(value) for value in info["shape"]]
    return shapes


def _record_layers(manifest: dict, names: Iterable[str]) -> int:
    stored = manifest.get("layers")
    if type(stored) is int and stored > 0:
        return stored
    highest = 0
    for name in names:
        layer = language_ffn_layer(str(name))
        if layer is not None and layer + 1 > highest:
            highest = layer + 1
    return highest or 1


def _quantize_scatter(
    src: Path,
    resident: Path,
    weight_map: dict[str, str],
    done: dict[str, dict],
    chosen: set[int],
    progress_path: Path,
) -> dict[str, dict]:
    """Append in safetensors shard order. Each tensor starts on a 128-byte boundary."""
    for shard, names in sorted(_groups(weight_map).items()):
        path = src / shard
        header, base = _header(path)
        print(f"shard {shard}", flush=True)
        for name in names:
            if name in done:
                _resume_matches(name, done[name], chosen)
                recorded = done[name]
                target = resident.with_name(str(recorded.get("file") or "resident.bin"))
                if not isinstance(recorded.get("sha256"), str):
                    continue
                if _digest_matches(target, recorded):
                    continue
                _rewind(done, recorded, resident, progress_path, LAYOUT_SCATTER)
            array = _load(path, header, base, name)
            print(f"  {name}", flush=True)
            blob, body = _pack_array(name, array, chosen)
            if is_vision(name):
                offset = _write_dense(resident.with_name("vision.bin"), blob)
                filename = "vision.bin"
            elif is_mtp(name):
                offset = _write_dense(resident.with_name("mtp.bin"), blob)
                filename = "mtp.bin"
            else:
                offset = _append(resident, blob)
                filename = "resident.bin"
            done[name] = {**body, "file": filename, "offset": offset, "sha256": _sha256(blob)}
            del array
            _save_progress(progress_path, done, LAYOUT_SCATTER)
    return done


def _quantize_prefix(
    src: Path,
    resident: Path,
    weight_map: dict[str, str],
    done: dict[str, dict],
    chosen: set[int],
    layers: int,
    progress_path: Path,
) -> dict[str, dict]:
    """Write language weights to ``resident.bin``, MTP to ``mtp.bin``, and vision to ``vision.bin``."""
    shapes = _tensor_shapes(src, weight_map)
    order = resident_names(shapes, layers)
    vision = sorted(name for name in shapes if is_vision(name))
    mtp = sorted(name for name in shapes if is_mtp(name))
    covered = set(order) | set(vision) | set(mtp)
    if covered != set(shapes):
        message = "layout 2 dropped a tensor"
        raise RuntimeError(message)
    resident_plan = _plan(order, shapes, chosen)
    vision_plan = _plan(vision, shapes, chosen)
    mtp_plan = _plan(mtp, shapes, chosen)
    vision_path = resident.with_name("vision.bin")
    mtp_path = resident.with_name("mtp.bin")
    language_done = any(not is_vision(name) and not is_mtp(name) for name in done)
    vision_done = any(is_vision(name) for name in done)
    mtp_done = any(is_mtp(name) for name in done)
    if not language_done or not resident.is_file():
        resident.write_bytes(b"")
    if vision and (not vision_done or not vision_path.is_file()):
        vision_path.write_bytes(b"")
    if mtp and (not mtp_done or not mtp_path.is_file()):
        mtp_path.write_bytes(b"")
    handles: dict[Path, Any] = {}
    try:
        for shard, names in sorted(_groups(weight_map).items()):
            path = src / shard
            header, base = _header(path)
            print(f"shard {shard}", flush=True)
            for name in names:
                target, filename, offset, span = _place(
                    name, resident, vision_path, mtp_path, resident_plan, vision_plan, mtp_plan
                )
                if name in done:
                    _resume_matches(name, done[name], chosen)
                    if done[name].get("file") != filename or int(done[name]["offset"]) != offset:
                        message = f"{name} offset does not match layout 2"
                        raise RuntimeError(message)
                    if _digest_matches(target, done[name]):
                        continue
                    print(f"  rewrite {name}", flush=True)
                array = _load(path, header, base, name)
                print(f"  {name}", flush=True)
                blob, body = _pack_array(name, array, chosen)
                if len(blob) != span:
                    message = f"{name} packed {len(blob)} bytes and the layout planned {span}"
                    raise RuntimeError(message)
                handle = handles.get(target)
                if handle is None:
                    if not target.is_file():
                        target.write_bytes(b"")
                    handle = target.open("r+b")
                    handles[target] = handle
                handle.seek(offset)
                handle.write(blob)
                _durable(handle)
                done[name] = {**body, "file": filename, "offset": offset, "sha256": _sha256(blob)}
                del array
                _save_progress(progress_path, done, LAYOUT_PREFIX)
        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        for handle in handles.values():
            handle.close()
    _covers(resident, order, resident_plan)
    if vision:
        _covers(vision_path, vision, vision_plan)
    if mtp:
        _covers(mtp_path, mtp, mtp_plan)
    return done


def _rewind(
    done: dict[str, dict],
    entry: dict,
    resident: Path,
    progress_path: Path,
    layout: int,
) -> None:
    """Drop ``entry`` and every later tensor in its file, then truncate to its offset."""
    filename = str(entry.get("file") or "resident.bin")
    cut = int(entry["offset"])
    stale = [
        name
        for name, item in done.items()
        if str(item.get("file") or "resident.bin") == filename and int(item["offset"]) >= cut
    ]
    for name in stale:
        del done[name]
    path = resident if filename == "resident.bin" else resident.with_name(filename)
    if path.is_file() and path.stat().st_size > cut:
        with path.open("r+b") as handle:
            handle.truncate(cut)
            _durable(handle)
    _save_progress(progress_path, done, layout)


def _plan(
    order: list[str], shapes: dict[str, list[int]], chosen: set[int]
) -> tuple[dict[str, int], dict[str, int], int]:
    offsets: dict[str, int] = {}
    spans: dict[str, int] = {}
    cursor = 0
    for name in order:
        span = _planned_span(name, shapes[name], fp8=_promoted_fp8(name, chosen))
        offsets[name] = cursor
        spans[name] = span
        cursor += span
    return offsets, spans, cursor


def _place(
    name: str,
    resident: Path,
    vision_path: Path,
    mtp_path: Path,
    resident_plan: tuple[dict[str, int], dict[str, int], int],
    vision_plan: tuple[dict[str, int], dict[str, int], int],
    mtp_plan: tuple[dict[str, int], dict[str, int], int],
) -> tuple[Path, str, int, int]:
    if is_vision(name):
        offsets, spans, _end = vision_plan
        return vision_path, "vision.bin", offsets[name], spans[name]
    if is_mtp(name):
        offsets, spans, _end = mtp_plan
        return mtp_path, "mtp.bin", offsets[name], spans[name]
    offsets, spans, _end = resident_plan
    return resident, "resident.bin", offsets[name], spans[name]


def _covers(
    path: Path,
    order: list[str],
    plan: tuple[dict[str, int], dict[str, int], int],
) -> None:
    offsets, spans, cursor = plan
    if path.stat().st_size != cursor:
        message = f"{path.name} length does not match layout 2"
        raise RuntimeError(message)
    placed = 0
    for name in order:
        if int(offsets[name]) != placed:
            message = f"{name} is not at byte {placed}"
            raise RuntimeError(message)
        placed += int(spans[name])
    if placed != cursor:
        message = f"{path.name} spans do not cover layout 2"
        raise RuntimeError(message)


def quantize(src: Path, output: Path | None = None, *, ffn_fp8: float | None = None) -> Path:
    """Pack ``src``. A finished manifest with the same percentage is left in place."""
    src = src.expanduser().resolve()
    config = json.loads((src / "config.json").read_text())
    text = config.get("text_config") or config
    layers = int(text["num_hidden_layers"])
    count = 0 if ffn_fp8 is None else ffn_promote_count(ffn_fp8, layers)
    if output is None:
        output = Path("models") / record_name(ffn_fp8, layers)
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        body = json.loads(manifest_path.read_text())
        if body.get("complete"):
            _budget(body, ffn_fp8, count)
            return output
    weight_map = _weight_map(src)
    promoted = _promoted(src, weight_map, output, ffn_fp8, layers)
    if ffn_fp8 is not None:
        print(f"ffn fp8 layers {count}", flush=True)
    chosen = set(promoted)
    progress_path = output / "quantize_progress.json"
    done: dict[str, dict] = {}
    layout = LAYOUT_PREFIX
    if progress_path.is_file():
        loaded = json.loads(progress_path.read_text())
        done = loaded.get("tensors", {})
        layout = layout_version(loaded)
    resident = output / "resident.bin"
    if layout == LAYOUT_PREFIX:
        done = _quantize_prefix(src, resident, weight_map, done, chosen, layers, progress_path)
    else:
        done = _quantize_scatter(src, resident, weight_map, done, chosen, progress_path)
    if count == 0:
        feed = "fp4"
    elif count == layers:
        feed = "fp8"
    else:
        feed = "mixed"
    manifest = {
        "complete": True,
        "pipeline": "qwen38_27b",
        "source": src.name,
        "tensors": done,
        "kv_cache": "half",
        "weight": "fp4",
        "feed_forward": feed,
        "layers": layers,
        "hidden": int(text["hidden_size"]),
        "vocab": int(text["vocab_size"]),
        "layer_types": list(text["layer_types"]),
    }
    if count:
        manifest["ffn_fp8_percent"] = ffn_fp8
        manifest["ffn_fp8_layers"] = promoted
    if any(is_mtp(str(name)) and str(body.get("dtype")) == "fp8" for name, body in done.items()):
        manifest["mtp_weight"] = "fp8"
    if layout == LAYOUT_PREFIX:
        manifest["layout"] = LAYOUT_PREFIX
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest))
    os.replace(temporary, manifest_path)
    for extra in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
    ):
        src_file = src / extra
        if src_file.is_file():
            target = output / extra
            if not target.exists():
                shutil.copyfile(src_file, target)
    return output


def _read_span(record: Path, entry: dict) -> bytes:
    path = record / str(entry.get("file") or "resident.bin")
    span = tensor_span(entry)
    with path.open("rb") as handle:
        handle.seek(int(entry["offset"]))
        blob = handle.read(span)
    if len(blob) != span:
        message = f"{path.name} ended early"
        raise RuntimeError(message)
    return blob


def _write_span(record: Path, entry: dict, dst: BinaryIO) -> int:
    path = record / str(entry.get("file") or "resident.bin")
    span = tensor_span(entry)
    with path.open("rb") as src:
        src.seek(int(entry["offset"]))
        remaining = span
        while remaining:
            chunk = src.read(min(8 << 20, remaining))
            if len(chunk) != min(8 << 20, remaining):
                message = f"{path.name} ended early"
                raise RuntimeError(message)
            dst.write(chunk)
            remaining -= len(chunk)
    return span


def repack_mtp_half(record: Path, source: Path) -> Path:
    """Replace fp4 or fp8 MTP matrices in ``record`` with Metal ``half`` from the BF16 checkout.

    Only those matrices are read from ``source``. The rewritten ``mtp.bin`` holds
    every MTP tensor, dense from offset 0. ``resident.bin`` keeps the language
    tensors and drops any MTP bytes it still held. A record whose MTP matrices
    are already half is left in place.
    """
    record = record.expanduser().resolve()
    source = source.expanduser().resolve()
    manifest_path = record / "manifest.json"
    resident = record / "resident.bin"
    if not manifest_path.is_file() or not resident.is_file():
        message = f"{record.name} has no packed record"
        raise FileNotFoundError(message)
    manifest = json.loads(manifest_path.read_text())
    tensors: dict[str, dict] = manifest["tensors"]
    pending = [
        name
        for name, entry in tensors.items()
        if name.startswith("mtp.") and str(entry.get("dtype")) in {"fp4", "fp8"}
    ]
    if not pending:
        print("mtp half", flush=True)
        return record
    weight_map = _weight_map(source)
    missing = [name for name in pending if name not in weight_map]
    if missing:
        message = f"{missing[0]} is not in the BF16 checkout"
        raise RuntimeError(message)
    by_shard: dict[str, list[str]] = {}
    for name in pending:
        by_shard.setdefault(weight_map[name], []).append(name)
    packed: dict[str, tuple[bytes, dict]] = {}
    for shard, names in sorted(by_shard.items()):
        path = source / shard
        header, base = _header(path)
        print(f"shard {shard}", flush=True)
        for name in names:
            array = _load(path, header, base, name)
            print(f"  {name}", flush=True)
            packed[name] = _half_matrix_entry(array)
            del array
    prefix = layout_version(manifest) == LAYOUT_PREFIX
    language = [name for name in tensors if not is_vision(name) and not is_mtp(name)]
    if prefix:
        ordered = resident_names(language, _record_layers(manifest, tensors))
        if set(ordered) != set(language):
            message = "layout 2 rewrite dropped a tensor"
            raise RuntimeError(message)
    else:
        ordered = sorted(language, key=lambda name: int(tensors[name]["offset"]))
    mtp_names = sorted(name for name in tensors if is_mtp(name))
    held: dict[str, tuple[bytes, dict]] = {}
    for name in mtp_names:
        if name in packed:
            blob, body = packed[name]
            held[name] = (blob, body)
            continue
        held[name] = (_read_span(record, tensors[name]), dict(tensors[name]))
    language_bytes = sum(tensor_span(tensors[name]) for name in ordered)
    need = language_bytes + sum(len(blob) for blob, _body in held.values()) + 4096
    fresh = record / "resident.bin.new"
    if shutil.disk_usage(record).free < need:
        cursor_path = source
        while True:
            if shutil.disk_usage(cursor_path).free >= need:
                fresh = cursor_path / f".{record.name}.resident.bin.new"
                break
            if cursor_path.parent == cursor_path:
                message = f"need {need} free bytes to rewrite the resident file"
                raise RuntimeError(message)
            cursor_path = cursor_path.parent
    if fresh.parent != record:
        fresh.parent.mkdir(parents=True, exist_ok=True)
    updated: dict[str, dict] = {
        name: dict(entry) for name, entry in tensors.items() if is_vision(name)
    }
    cursor = 0
    with fresh.open("wb") as dst:
        for name in ordered:
            if not prefix:
                pad = _align(cursor) - cursor
                if pad:
                    dst.write(b"\x00" * pad)
                    cursor += pad
            entry = tensors[name]
            span = _write_span(record, entry, dst)
            updated[name] = {**entry, "file": "resident.bin", "offset": cursor}
            cursor += span
        dst.flush()
        os.fsync(dst.fileno())
    if fresh.stat().st_size != cursor:
        message = "resident rewrite length does not match"
        raise RuntimeError(message)
    mtp_fresh = record / "mtp.bin.new"
    mtp_cursor = 0
    with mtp_fresh.open("wb") as dst:
        for name in mtp_names:
            blob, body = held[name]
            updated[name] = {**body, "file": "mtp.bin", "offset": mtp_cursor}
            dst.write(blob)
            mtp_cursor += len(blob)
        dst.flush()
        os.fsync(dst.fileno())
    if mtp_fresh.stat().st_size != mtp_cursor:
        message = "mtp rewrite length does not match"
        raise RuntimeError(message)
    # Keep the backup on the volume that held the temporary file when the
    # record disk cannot hold old and new residents at once.
    if fresh.parent == record:
        backup = record / "resident.bin.bak"
    else:
        backup = fresh.parent / f".{record.name}.resident.bin.bak"
    if backup.exists():
        backup.unlink()
    if backup.parent == record:
        os.replace(resident, backup)
    else:
        shutil.move(str(resident), str(backup))
    try:
        if fresh.parent == record:
            os.replace(fresh, resident)
        else:
            shutil.copyfile(fresh, resident)
            fresh.unlink(missing_ok=True)
        os.replace(mtp_fresh, record / "mtp.bin")
        manifest["tensors"] = updated
        manifest["mtp_weight"] = "half"
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest))
        os.replace(temporary, manifest_path)
    except Exception:
        if backup.exists() and not resident.exists():
            if backup.parent == record:
                os.replace(backup, resident)
            else:
                shutil.move(str(backup), str(resident))
        if fresh.exists() and fresh.parent != record:
            fresh.unlink(missing_ok=True)
        mtp_fresh.unlink(missing_ok=True)
        raise
    backup.unlink(missing_ok=True)
    print(f"mtp half {len(pending)}", flush=True)
    return record


# Legacy name kept for callers that still import the fp8 repack entry point.
repack_mtp_fp8 = repack_mtp_half
