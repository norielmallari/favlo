"""Pack the BF16 checkpoint into the int4 record and register it."""

from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path

import numpy as np

from klein_4b.cuda.constants import FORMAT, RECORD, REPO, SOURCE_BYTES, record_dir, source_dir
from klein_4b.cuda.formats import pack_int4

_SKIP_INT = {"I64", "I32", "I16", "I8", "U8", "U32", "BOOL"}


def quantize() -> Path:
    source = source_dir()
    _require_source(source)
    target = record_dir()
    manifest_path = target / "manifest.json"
    if manifest_path.is_file():
        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        if stored.get("format") == FORMAT and stored.get("complete") is True:
            _register(source, target)
            return target
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    blob = target / "weights.bin"
    index: dict[str, dict[str, object]] = {}
    stages = {"text": 0, "dit": 0, "vae": 0}
    with blob.open("wb") as handle:
        cursor = 0
        for relative in SOURCE_BYTES:
            path = source / relative
            prefix = _prefix(relative)
            for name, dtype, shape, raw in _tensors(path):
                if prefix == "text" and _skip_text(name):
                    continue
                if dtype in _SKIP_INT or name.endswith("num_batches_tracked"):
                    continue
                key = f"{prefix}/{name}"
                entry, payload = _pack(name, dtype, shape, raw)
                handle.write(payload)
                entry["offset"] = cursor
                index[key] = entry
                stages[prefix] += len(payload)
                cursor += len(payload)
                print(f"packed {key}", flush=True)
    tokenizer = _find_tokenizer(source)
    destination = target / "tokenizer" / "tokenizer.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(tokenizer, destination)
    (target / "index.json").write_text(json.dumps(index), encoding="utf-8")
    manifest = {
        "format": FORMAT,
        "complete": True,
        "nbytes": cursor,
        "repo": REPO,
        "stage_bytes": stages,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _register(source, target)
    return target


def _require_source(source: Path) -> None:
    for relative, size in SOURCE_BYTES.items():
        path = source / relative
        if not path.is_file():
            message = f"missing {relative}"
            raise FileNotFoundError(message)
        found = path.stat().st_size
        if found != size:
            message = f"{relative} is {found} bytes, expected {size}"
            raise ValueError(message)


def _prefix(relative: str) -> str:
    if relative.startswith("text_encoder/"):
        return "text"
    if relative.startswith("transformer/"):
        return "dit"
    return "vae"


def _skip_text(name: str) -> bool:
    if name.startswith("lm_head") or name == "model.norm.weight" or name.startswith("model.norm."):
        return True
    for index in range(27, 36):
        if name.startswith(f"model.layers.{index}."):
            return True
    return False


def _pack(name: str, dtype: str, shape: list[int], raw: bytes) -> tuple[dict[str, object], bytes]:
    array = _decode(dtype, shape, raw)
    if name.endswith("running_mean") or name.endswith("running_var"):
        values = np.ascontiguousarray(array, dtype=np.float32)
        return {"kind": "f32", "shape": list(values.shape), "nbytes": values.nbytes}, values.tobytes()
    if array.ndim == 1 or name.endswith(".bias") or name.endswith("norm.weight") or "norm" in name.split(".")[-2:]:
        if array.ndim > 1 and not name.endswith(".bias"):
            return _matrix(name, array, shape)
        values = np.ascontiguousarray(array.reshape(-1), dtype=np.float16)
        return {"kind": "f16", "shape": list(array.shape), "nbytes": values.nbytes}, values.tobytes()
    return _matrix(name, array, shape)


def _matrix(name: str, array: np.ndarray, shape: list[int]) -> tuple[dict[str, object], bytes]:
    del name
    matrix = np.ascontiguousarray(array, dtype=np.float32)
    if matrix.ndim == 4:
        out, inner = matrix.shape[0], int(np.prod(matrix.shape[1:]))
        matrix = matrix.reshape(out, inner)
    elif matrix.ndim != 2:
        message = f"cannot pack shape {shape}"
        raise ValueError(message)
    codes, scales, rows, inner = pack_int4(matrix)
    payload = codes.tobytes() + np.ascontiguousarray(scales, dtype=np.float32).tobytes()
    entry = {
        "kind": "int4",
        "shape": list(shape),
        "rows": rows,
        "inner": inner,
        "stored_n": int(codes.shape[0]),
        "stored_k": int(codes.shape[1] * 2),
        "codes_bytes": codes.nbytes,
        "scales_bytes": scales.nbytes,
        "nbytes": len(payload),
    }
    return entry, payload


def _decode(dtype: str, shape: list[int], raw: bytes) -> np.ndarray:
    if dtype == "BF16":
        units = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
        values = units.view(np.float32)
    elif dtype == "F16":
        values = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    elif dtype == "F32":
        values = np.frombuffer(raw, dtype=np.float32).copy()
    else:
        message = f"unsupported dtype {dtype}"
        raise ValueError(message)
    return values.reshape(shape)


def _tensors(path: Path):
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
        data_at = 8 + header_size
        for name, meta in header.items():
            if name == "__metadata__" or not isinstance(meta, dict):
                continue
            start, end = meta["data_offsets"]
            handle.seek(data_at + start)
            yield name, meta["dtype"], list(meta["shape"]), handle.read(end - start)


def _find_tokenizer(source: Path) -> Path:
    candidates = [
        source / "tokenizer" / "tokenizer.json",
        source / "text_encoder" / "tokenizer.json",
        source / "tokenizer.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    found = list(source.rglob("tokenizer.json"))
    if len(found) == 1:
        return found[0]
    message = "tokenizer.json is missing"
    raise FileNotFoundError(message)


def _register(source: Path, target: Path) -> None:
    from favlo.registry import register_quantized

    register_quantized(
        RECORD,
        source=source,
        alias="klein_4b",
        directory=target,
        pipeline="klein_4b",
        runtime="cuda",
    )
