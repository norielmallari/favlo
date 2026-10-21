"""Stream a BF16 checkout into an fp4 record, or a mixed feed-forward record.

Every matrix is fp4 E2M1 with an E8M0 scale every 32 along K, except language
feed-forward layers selected by ``--ffn-fp8``. Those three matrices are fp8
E4M3. One-dimensional parameters and the depthwise convolution stay float16.
The MTP feed-forward stays fp4.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np

from qwen38_27b.formats import (
    ffn_promote_count,
    pack_fp4_matrix,
    pack_fp8_matrix,
    promoted_layer_ids,
    relative_error,
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


def _append(path: Path, blob: bytes) -> int:
    with path.open("ab") as handle:
        end = handle.seek(0, 2)
        pad = _align(end) - end
        if pad:
            handle.write(b"\x00" * pad)
            end += pad
        handle.write(blob)
    return end


def _f16_entry(array: np.ndarray) -> tuple[bytes, dict]:
    flat = np.ascontiguousarray(array, dtype=np.float16)
    body = {"dtype": "f16", "shape": list(array.shape), "stored_shape": list(flat.shape)}
    return flat.tobytes(), body


def record_name(count: int, layers: int) -> str:
    """Uniform fp4, uniform fp8, or a mixed record whose name ends in ``fp8-fp4``."""
    if count <= 0:
        return RECORD_NAME
    if count >= layers:
        return "Qwen3.8-27B-fp8"
    return "Qwen3.8-27B-fp8-fp4"


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


def _keep_f16(name: str, array: np.ndarray) -> bool:
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


def quantize(src: Path, output: Path | None = None, *, ffn_fp8: float | None = None) -> Path:
    """Pack ``src``. A finished manifest with the same percentage is left in place."""
    src = src.expanduser().resolve()
    config = json.loads((src / "config.json").read_text())
    text = config.get("text_config") or config
    layers = int(text["num_hidden_layers"])
    count = 0 if ffn_fp8 is None else ffn_promote_count(ffn_fp8, layers)
    if output is None:
        output = Path("models") / record_name(count, layers)
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
    if progress_path.is_file():
        done = json.loads(progress_path.read_text()).get("tensors", {})
    resident = output / "resident.bin"
    for shard, names in sorted(_groups(weight_map).items()):
        path = src / shard
        header, base = _header(path)
        print(f"shard {shard}", flush=True)
        for name in names:
            found = _language_ffn(name)
            if name in done:
                if found is not None:
                    expected = "fp8" if found[0] in chosen else "fp4"
                    if done[name].get("dtype") != expected:
                        message = f"{name} is stored as {done[name].get('dtype')}"
                        raise RuntimeError(message)
                continue
            array = _load(path, header, base, name)
            print(f"  {name}", flush=True)
            if _keep_f16(name, array):
                flat = array
                if name.endswith("conv1d.weight") and array.ndim >= 2:
                    flat = array.reshape(array.shape[0], -1)
                blob, body = _f16_entry(flat)
            elif found is not None and found[0] in chosen:
                blob, body = _fp8_entry(array)
            else:
                blob, body = _fp4_entry(array)
            offset = _append(resident, blob)
            done[name] = {**body, "file": "resident.bin", "offset": offset}
            del array
            temporary = progress_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps({"tensors": done}))
            os.replace(temporary, progress_path)
    if count == 0:
        feed = "fp4"
    elif count == layers:
        feed = "fp8"
    else:
        feed = "mixed"
    manifest = {
        "complete": True,
        "pipeline": "qwen38-27b",
        "source": src.name,
        "tensors": done,
        "kv_cache": "e4m3",
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
