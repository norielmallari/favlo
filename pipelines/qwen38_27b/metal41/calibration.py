"""Rewrite a full fp4 record with GPTQ and FOEM.

The sequences are the shared corpus in ``artifacts/calibration/``: software-agent
traces, coding instructions, function-calling dialogs, then general chat.
Each pass prefills those sequences on the packed record. The input Gram is a
half-precision ``XᵀX`` through ``matmul2d``, accumulated in float32 on device.
The inverse factor is a float32 Cholesky on device, and the column update runs
on device. BF16 rows are read from the checkout in sequential 8 MiB pieces. The
next pass sees the replaced codes. The embedding and ``lm_head`` stay
round-to-nearest fp8. Vision-tower matrices stay the round-to-nearest fp8 pack
the tower kernels read. MTP matrices stay the round-to-nearest fp8 pack in
``mtp.bin``.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from favlo.calibration_corpus import QUOTAS as _QUOTAS
from qwen38_27b.metal41.convert import _bf16, quantize
from qwen38_27b.metal41.formats import quantize_fp4
from qwen38_27b.metal41.gptq import (
    DeviceFactor,
    device_relative_error,
    encode_gram,
    factor_gram,
    foem_fp4_rows,
    gram_bytes,
    pad_fp4,
)
from qwen38_27b.metal41.layout import ATTN_OUT, HIDDEN, INTERMEDIATE, Z_WIDTH

BETA = 0.1
SEQUENCE_LENGTH = 2048
_PIECE = 8 << 20
_WIDTH = {
    "attn": HIDDEN,
    "ffn": HIDDEN,
    "head": HIDDEN,
    "down": INTERMEDIATE,
    "gout": Z_WIDTH,
    "aout": ATTN_OUT,
}
_PASS = 8
_CAPTURE_CONTEXT = 4096
_PIECES = {"swe": 4, "code": 2, "tool": 2, "chat": 2}
_LAYER = re.compile(r"layers\.(\d+)\.")
_ATTN = (
    "in_proj_qkv.weight",
    "in_proj_z.weight",
    "in_proj_a.weight",
    "in_proj_b.weight",
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
)


def language_stream(name: str) -> tuple[int, str] | None:
    """``(layer, stream)`` for a language matrix FOEM rewrites. The head is layer -1."""
    if name == "lm_head.weight":
        return -1, "head"
    if "visual" in name or name.startswith("mtp.") or name.endswith("embed_tokens.weight"):
        return None
    found = _LAYER.search(name)
    if found is None:
        return None
    layer = int(found.group(1))
    if name.endswith("mlp.down_proj.weight"):
        return layer, "down"
    if name.endswith("mlp.gate_proj.weight") or name.endswith("mlp.up_proj.weight"):
        return layer, "ffn"
    if "linear_attn" in name and name.endswith("out_proj.weight"):
        return layer, "gout"
    if name.endswith("o_proj.weight"):
        return layer, "aout"
    if name.endswith(_ATTN):
        return layer, "attn"
    return None


class Grams:
    """Device float32 input Grams for the layers this pass is rewriting."""

    def __init__(self, wanted: set[tuple[int, str]]) -> None:
        self.wanted = wanted
        self.sums: dict[tuple[int, str], np.ndarray] = {}
        self.device: dict[tuple[int, str], tuple[Any, int]] = {}
        self.transposed: Any = None
        self.scratch: Any = None
        self._transpose_bytes = 0
        self._scratch_bytes = 0

    def wants(self, layer: int, stream: str) -> bool:
        return (layer, stream) in self.wanted

    def allocate(self, session: Any) -> None:
        for layer, stream in sorted(self.wanted):
            width = _WIDTH.get(stream)
            if width is None:
                message = f"layer {layer} stream {stream} has no Gram width"
                raise RuntimeError(message)
            nbytes = width * width * 4
            try:
                buffer = session.zeros(nbytes)
            except RuntimeError as exc:
                message = f"capture ran out of memory for a {nbytes} byte Gram"
                raise RuntimeError(message) from exc
            self.device[(layer, stream)] = (buffer, width)
        if not self.device:
            return
        widest = max(width for _buffer, width in self.device.values())
        self._transpose_bytes, self._scratch_bytes = gram_bytes(SEQUENCE_LENGTH, widest)
        try:
            self.transposed = session.zeros(self._transpose_bytes)
            self.scratch = session.zeros(self._scratch_bytes)
        except RuntimeError as exc:
            message = "capture ran out of memory for the Gram product"
            raise RuntimeError(message) from exc

    def observe(
        self,
        session: Any,
        buffer: Any,
        count: int,
        span: int,
        width: int,
        layer: int,
        stream: str,
        *,
        half: bool,
    ) -> None:
        if count < 1 or not self.wants(layer, stream):
            return
        found = self.device.get((layer, stream))
        if found is None:
            message = f"layer {layer} stream {stream} has no device Gram"
            raise RuntimeError(message)
        acc, held = found
        if held != width:
            message = f"layer {layer} stream {stream} width {width} is not {held}"
            raise RuntimeError(message)
        transpose_bytes, scratch_bytes = gram_bytes(count, width)
        if transpose_bytes > self._transpose_bytes or scratch_bytes > self._scratch_bytes:
            message = f"layer {layer} stream {stream} Gram is larger than the pass scratch"
            raise RuntimeError(message)
        encode_gram(
            session,
            buffer,
            acc,
            count,
            span,
            width,
            self.transposed,
            self.scratch,
            half=half,
        )

    def harvest(self, session: Any) -> None:
        from qwen38_27b.metal41.session import release_buffer

        try:
            for key, (acc, width) in self.device.items():
                self.sums[key] = session.download_f32(acc, width * width).reshape(width, width).copy()
                release_buffer(acc)
            self.device.clear()
        finally:
            for buffer in (self.transposed, self.scratch):
                if buffer is not None:
                    release_buffer(buffer)
            self.transposed = None
            self.scratch = None

    def gram(self, layer: int, stream: str) -> np.ndarray:
        found = self.sums.get((layer, stream))
        if found is None:
            message = f"no calibration inputs for layer {layer} {stream}"
            raise KeyError(message)
        return found


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _rows(kind: str) -> list[Any]:
    from favlo.calibration_corpus import load

    return load(kind)


def _call_start(root: Path, ids: list[int]) -> int | None:
    """Token index 128 before the first tool call, or None when the row has none."""
    from qwen38_27b.portable.chat import decode_text

    if "<tool_call>" not in decode_text(root, ids):
        return None
    lo = 0
    hi = len(ids)
    while lo < hi:
        mid = (lo + hi) // 2
        if "<tool_call>" in decode_text(root, ids[:mid]):
            hi = mid
        else:
            lo = mid + 1
    return max(0, lo - 128)


def _window(root: Path, kind: str, row: Any, cap: int) -> list[int]:
    from qwen38_27b.portable.chat import encode_chat

    messages = row["messages"] if isinstance(row, dict) else row
    tools = row.get("tools") if isinstance(row, dict) else None
    ids = encode_chat(root, messages, tools, False)
    start = 0
    if kind == "swe":
        found = _call_start(root, ids)
        if found is None:
            return []
        start = found
    return ids[start : start + _PIECES[kind] * cap]


def _sequence_label() -> str:
    return ",".join(f"{kind}:{quota}" for kind, quota in _QUOTAS)


def _sequences(root: Path) -> list[list[int]]:
    """128 sequences of 2048 tokens from the shared calibration corpus."""
    cached = _root() / "artifacts" / "qwen38_27b" / "sequences.npy"
    label_path = cached.with_suffix(".label")
    label = _sequence_label()
    count = sum(quota for _kind, quota in _QUOTAS)
    if cached.is_file() and label_path.is_file() and label_path.read_text() == label:
        held = np.load(cached)
        if held.ndim == 2 and held.shape == (count, SEQUENCE_LENGTH):
            return [row.astype(np.int64).tolist() for row in held]
    cap = SEQUENCE_LENGTH
    sequences: list[list[int]] = []
    for kind, quota in _QUOTAS:
        pending: list[int] = []
        got = 0
        for row in _rows(kind):
            if got >= quota:
                break
            try:
                ids = _window(root, kind, row, cap)
            except (RuntimeError, json.JSONDecodeError, ValueError, KeyError, TypeError):
                continue
            pending.extend(ids)
            while len(pending) >= cap and got < quota:
                sequences.append(pending[:cap])
                pending = pending[cap:]
                got += 1
                print(f"sequence {len(sequences)} {kind}", flush=True)
        if got < quota:
            message = f"{kind} packed {got} sequences, need {quota}"
            raise RuntimeError(message)
    array = np.asarray(sequences, dtype=np.int32)
    cached.parent.mkdir(parents=True, exist_ok=True)
    np.save(cached, array)
    label_path.write_text(label)
    return sequences


def _targets(tensors: dict[str, dict]) -> dict[int, list[tuple[str, str]]]:
    grouped: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for name, entry in tensors.items():
        if str(entry.get("dtype")) != "fp4":
            continue
        found = language_stream(name)
        if found is None:
            continue
        layer, stream = found
        grouped[layer].append((name, stream))
    return grouped


def _capture(
    record: Path, sequences: list[list[int]], wanted: set[tuple[int, str]]
) -> tuple[Grams, Any]:
    from qwen38_27b.metal41.forward import Qwen38Runtime
    from qwen38_27b.metal41.prefill import prefill_ids

    grams = Grams(wanted)
    runtime = Qwen38Runtime(
        record,
        kv_cache="half",
        owner="qwen38_27b:calibrate",
        mtp=False,
        context_limit=_CAPTURE_CONTEXT,
    )
    try:
        if runtime.max_pos < SEQUENCE_LENGTH:
            message = f"calibration context {runtime.max_pos} is below {SEQUENCE_LENGTH}"
            raise RuntimeError(message)
        grams.allocate(runtime.session)
        runtime.grams = grams
        for index, ids in enumerate(sequences):
            runtime.reset()
            prefill_ids(runtime, ids, 0, mtp=False, progress=False)
            print(f"prefill {index + 1}/{len(sequences)}", flush=True)
        runtime.grams = None
    except Exception:
        runtime.close()
        raise
    return grams, runtime


def _read_exact(handle: Any, nbytes: int, piece: int = _PIECE) -> bytearray:
    """Read ``nbytes`` at the handle's position, in sequential pieces of at least 8 MiB."""
    if nbytes < 0 or piece < 1:
        message = f"weight read length {nbytes} piece {piece} is not valid"
        raise ValueError(message)
    got = bytearray()
    left = nbytes
    while left:
        step = min(piece, left)
        chunk = handle.read(step)
        if len(chunk) != step:
            message = f"weight read stopped after {nbytes - left + len(chunk)} of {nbytes}"
            raise RuntimeError(message)
        got += chunk
        left -= step
    return got


class _Shards:
    """Open checkout shards kept for a forward scan. Each read is sequential."""

    def __init__(self, src: Path) -> None:
        self.src = src
        self.handles: dict[Path, Any] = {}

    def read(self, path: Path, offset: int, nbytes: int) -> bytearray:
        handle = self.handles.get(path)
        if handle is None:
            handle = path.open("rb")
            flag = getattr(fcntl, "F_RDAHEAD", None)
            if flag is not None:
                fcntl.fcntl(handle.fileno(), flag, 1)
            self.handles[path] = handle
        if handle.tell() != offset:
            handle.seek(offset)
        return _read_exact(handle, nbytes)

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()


class _Rows:
    """One BF16 matrix, read by contiguous row blocks."""

    def __init__(self, shards: _Shards, path: Path, header: dict, base: int, name: str) -> None:
        info = header[name]
        start, _stop = info["data_offsets"]
        shape = [int(value) for value in info["shape"]]
        if len(shape) != 2:
            message = f"{name} rank {len(shape)} is not a matrix"
            raise RuntimeError(message)
        self.shards = shards
        self.path = path
        self.rows, self.inner = shape
        self.dtype = str(info["dtype"])
        width = {"BF16": 2, "F16": 2, "F32": 4}.get(self.dtype)
        if width is None:
            message = f"{name} dtype {self.dtype} is not packed"
            raise RuntimeError(message)
        self._width = width * self.inner
        self._origin = base + int(start)

    def load(self, begin: int, end: int) -> np.ndarray:
        raw = self.shards.read(self.path, self._origin + begin * self._width, (end - begin) * self._width)
        if self.dtype == "BF16":
            return _bf16(raw, [end - begin, self.inner])
        if self.dtype == "F16":
            return np.frombuffer(raw, dtype=np.float16).reshape(end - begin, self.inner).astype(np.float32)
        return np.frombuffer(raw, dtype=np.float32).reshape(end - begin, self.inner).copy()


def _disk_order(
    names: list[str],
    weight_map: dict[str, str],
    headers: dict[str, tuple[dict, int]],
    src: Path,
) -> list[str]:
    from qwen38_27b.metal41.convert import _header

    keyed: list[tuple[str, int, str]] = []
    for name in names:
        shard = weight_map[name]
        if shard not in headers:
            headers[shard] = _header(src / shard)
        header, base = headers[shard]
        start, _stop = header[name]["data_offsets"]
        keyed.append((shard, base + int(start), name))
    keyed.sort()
    return [name for _shard, _offset, name in keyed]


def _sample_rows(rows: int, sample: int = 256) -> tuple[int, list[int]]:
    step = max(1, rows // sample)
    return step, list(range(0, rows, step))


def _nearest_rows(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    data, plane = quantize_fp4(matrix)
    count, inner = matrix.shape
    return (
        np.frombuffer(data, dtype=np.uint8).reshape(count, inner // 2).copy(),
        np.frombuffer(plane, dtype=np.uint8).reshape(count, inner // 32).copy(),
    )


def _write_pack(record: Path, name: str, entry: dict, packed: np.ndarray, scales: np.ndarray) -> None:
    stored_rows, stored_inner = (int(value) for value in entry["stored_shape"])
    data, plane = pad_fp4(packed, scales, stored_rows, stored_inner)
    if len(data) != int(entry["data_nbytes"]) or len(plane) != int(entry["scale_nbytes"]):
        message = f"{name} packed length does not match the record"
        raise RuntimeError(message)
    offset = int(entry["offset"])
    path = record / "resident.bin"
    with path.open("r+b") as handle:
        handle.seek(offset)
        handle.write(data)
        handle.seek(offset + int(entry["data_nbytes"]) + int(entry.get("scale_pad") or 0))
        handle.write(plane)
        handle.flush()
        os.fsync(handle.fileno())


def _nearest_matrix(rows: _Rows) -> tuple[np.ndarray, np.ndarray]:
    data_parts: list[np.ndarray] = []
    scale_parts: list[np.ndarray] = []
    for begin in range(0, rows.rows, 4096):
        end = min(begin + 4096, rows.rows)
        data, plane = _nearest_rows(rows.load(begin, end))
        data_parts.append(data)
        scale_parts.append(plane)
    return np.concatenate(data_parts), np.concatenate(scale_parts)


def _paired_error(
    session: Any,
    rows: _Rows,
    packed: np.ndarray,
    scales: np.ndarray,
    gram: Any,
    picked: list[int],
) -> tuple[float, float]:
    weight = np.stack([rows.load(index, index + 1)[0] for index in picked])
    near_p, near_s = _nearest_rows(weight)
    base = device_relative_error(session, gram, rows.inner, weight, near_p, near_s)
    error = device_relative_error(
        session,
        gram,
        rows.inner,
        weight,
        np.stack([packed[index] for index in picked]),
        np.stack([scales[index] for index in picked]),
    )
    return error, base


def _splice_device(
    session: Any,
    record: Path,
    name: str,
    entry: dict,
    rows: _Rows,
    gram: Any,
    factor: DeviceFactor,
) -> None:
    _step, picked = _sample_rows(rows.rows)
    sample_w: list[np.ndarray] = []
    chosen_rows = set(picked)

    def load(begin: int, end: int) -> np.ndarray:
        tile = rows.load(begin, end)
        for local, index in enumerate(range(begin, end)):
            if index in chosen_rows:
                sample_w.append(np.array(tile[local], dtype=np.float32, copy=True))
        return tile

    try:
        packed, scales = foem_fp4_rows(session, rows.rows, rows.inner, load, factor, beta=BETA)
        weight = np.stack(sample_w)
        near_p, near_s = _nearest_rows(weight)
        base = device_relative_error(session, gram, rows.inner, weight, near_p, near_s)
        error = device_relative_error(
            session,
            gram,
            rows.inner,
            weight,
            np.stack([packed[index] for index in picked]),
            np.stack([scales[index] for index in picked]),
        )
        chosen = "foem"
        if not np.isfinite(error) or error > base:
            packed, scales = _nearest_matrix(rows)
            chosen = "nearest"
            error = base
    except (RuntimeError, np.linalg.LinAlgError, ValueError):
        packed, scales = _nearest_matrix(rows)
        error, base = _paired_error(session, rows, packed, scales, gram, picked)
        chosen = "nearest"
        error = base
    _write_pack(record, name, entry, packed, scales)
    print(
        f"{name} {chosen} output error {error**0.5:.4f} nearest {base**0.5:.4f}",
        flush=True,
    )


def _progress(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    body = json.loads(path.read_text())
    done = body.get("done")
    if not isinstance(done, list):
        return set()
    return {str(name) for name in done}


def _store_progress(path: Path, done: set[str]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"beta": BETA, "done": sorted(done)}))
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _memory_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "memory" in text or "allocated" in text


def _pass(
    record: Path,
    src: Path,
    sequences: list[list[int]],
    weight_map: dict[str, str],
    headers: dict[str, tuple[dict, int]],
    tensors: dict[str, dict],
    needed: list[tuple[int, str, str]],
    done: set[str],
) -> None:
    wanted = {(layer, stream) for layer, stream, _name in needed}
    print(f"capture {len(wanted)} grams", flush=True)
    try:
        grams, runtime = _capture(record, sequences, wanted)
    except RuntimeError as exc:
        by_layer: dict[int, list[tuple[int, str, str]]] = defaultdict(list)
        for item in needed:
            by_layer[item[0]].append(item)
        if len(by_layer) > 1 and _memory_error(exc):
            print("capture ran out of memory, retrying one layer", flush=True)
            for layer in sorted(by_layer):
                _pass(record, src, sequences, weight_map, headers, tensors, by_layer[layer], done)
            return
        raise
    from qwen38_27b.metal41.convert import _header

    progress = record / "foem_progress.json"
    shards = _Shards(src)
    retry: list[int] | None = None
    try:
        by_stream: dict[tuple[int, str], list[str]] = defaultdict(list)
        for layer, stream, name in needed:
            if name not in done:
                by_stream[(layer, stream)].append(name)
        for (layer, stream), names in by_stream.items():
            gram, width = grams.device[(layer, stream)]
            factor = None
            try:
                factor = factor_gram(runtime.session, gram, width, workspace=grams.scratch)
                ordered = _disk_order(names, weight_map, headers, src)
                for name in ordered:
                    shard = weight_map[name]
                    if shard not in headers:
                        headers[shard] = _header(src / shard)
                    header, base = headers[shard]
                    rows = _Rows(shards, src / shard, header, base, name)
                    _splice_device(
                        runtime.session,
                        record,
                        name,
                        tensors[name],
                        rows,
                        gram,
                        factor,
                    )
                    done.add(name)
                    _store_progress(progress, done)
            finally:
                if factor is not None:
                    factor.release()
    except RuntimeError as exc:
        layers = {layer for layer, _stream, _name in needed}
        if len(layers) > 1 and _memory_error(exc):
            retry = sorted(layers)
        else:
            raise
    finally:
        shards.close()
        runtime.close()
    if retry is not None:
        print("capture ran out of memory, retrying one layer", flush=True)
        by_layer: dict[int, list[tuple[int, str, str]]] = defaultdict(list)
        for item in needed:
            by_layer[item[0]].append(item)
        for layer in retry:
            _pass(record, src, sequences, weight_map, headers, tensors, by_layer[layer], done)


def calibrate(src: Path, record: Path | None = None) -> Path:
    """Pack ``src`` as full fp4 when needed, then rewrite the language codes."""
    src = src.expanduser().resolve()
    if record is None:
        record = Path("models") / "Qwen3.8-27B-fp4"
    record = record.expanduser().resolve()
    manifest_path = record / "manifest.json"
    if not manifest_path.is_file() or not json.loads(manifest_path.read_text()).get("complete"):
        print("round-to-nearest fp4", flush=True)
        quantize(src, record, ffn_fp8=None)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("feed_forward") not in (None, "fp4") or manifest.get("ffn_fp8_layers"):
        message = "calibrate rewrites a full fp4 record"
        raise RuntimeError(message)
    if manifest.get("quantization") == "gptq-foem":
        print("gptq-foem", flush=True)
        return record
    tensors: dict[str, dict] = manifest["tensors"]
    grouped = _targets(tensors)
    pending = [name for pairs in grouped.values() for name, _stream in pairs]
    done = _progress(record / "foem_progress.json")
    if pending and set(pending) <= done:
        _mark(manifest_path, manifest)
        return record
    from qwen38_27b.metal41.convert import _weight_map

    weight_map = _weight_map(src)
    sequences = _sequences(record if (record / "tokenizer.json").is_file() else src)
    headers: dict[str, tuple[dict, int]] = {}
    layers = int(manifest.get("layers") or 64)
    order = list(range(layers))
    for start in range(0, layers, _PASS):
        group = order[start : start + _PASS]
        needed: list[tuple[int, str, str]] = []
        for layer in group:
            for name, stream in grouped.get(layer, []):
                if name not in done:
                    needed.append((layer, stream, name))
        if start + _PASS >= layers:
            for name, stream in grouped.get(-1, []):
                if name not in done:
                    needed.append((-1, stream, name))
        if not needed:
            continue
        print(f"layers {group[0]}-{group[-1]}", flush=True)
        _pass(record, src, sequences, weight_map, headers, tensors, needed, done)
    manifest = json.loads(manifest_path.read_text())
    _mark(manifest_path, manifest)
    return record


def _mark(path: Path, manifest: dict) -> None:
    manifest["quantization"] = "gptq-foem"
    manifest["quantization_beta"] = BETA
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest))
    os.replace(temporary, path)
    print("gptq-foem", flush=True)
