"""Read one bfloat16 safetensors checkout."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np


def tensor_names(root: Path) -> dict[str, str]:
    """Map a tensor name to the shard file name. A single file uses ``""``."""
    index = root / "model.safetensors.index.json"
    if index.is_file():
        body = json.loads(index.read_text(encoding="utf-8"))
        weight_map = body["weight_map"]
        return {str(name): str(shard) for name, shard in weight_map.items()}
    singles = list(root.glob("*.safetensors"))
    if len(singles) != 1:
        message = f"{root} has no safetensors index"
        raise FileNotFoundError(message)
    header, _data = _header(singles[0])
    return {name: singles[0].name for name in header if name != "__metadata__"}


def tensor_shape(root: Path, shard: str, name: str) -> tuple[int, ...]:
    header, _data = _cached(root / shard)
    meta = header[name]
    if not isinstance(meta, dict):
        message = f"{name} safetensors entry is not an object"
        raise TypeError(message)
    return tuple(int(item) for item in meta["shape"])


def load_tensor(
    root: Path,
    shard: str,
    name: str,
    index: int | None = None,
    rows: tuple[int, int] | None = None,
) -> np.ndarray:
    """Load one tensor. ``index`` reads one expert. ``rows`` reads a row span."""
    path = root / shard
    header, data = _cached(path)
    meta = header[name]
    if not isinstance(meta, dict):
        message = f"{name} safetensors entry is not an object"
        raise TypeError(message)
    start, end = (int(item) for item in meta["data_offsets"])
    shape = tuple(int(item) for item in meta["shape"])
    dtype = str(meta["dtype"])
    width = _width(dtype)
    if index is not None and rows is not None:
        message = f"{name} cannot take an expert index and a row span"
        raise ValueError(message)
    if rows is not None:
        if len(shape) != 2:
            message = f"{name} is not a matrix"
            raise ValueError(message)
        start_row, stop_row = rows
        if start_row < 0 or stop_row > shape[0] or start_row >= stop_row:
            message = f"{name} rows {rows} are outside {shape[0]}"
            raise IndexError(message)
        row = int(shape[1]) * width
        origin = start + start_row * row
        take = (stop_row - start_row) * row
        out_shape = (stop_row - start_row, shape[1])
    elif index is None:
        take = end - start
        origin = start
        out_shape = shape
    else:
        if len(shape) != 3:
            message = f"{name} is not a rank-3 tensor"
            raise ValueError(message)
        if index < 0 or index >= shape[0]:
            message = f"{name} index {index} is outside {shape[0]}"
            raise IndexError(message)
        row = int(np.prod(shape[1:])) * width
        origin = start + index * row
        take = row
        out_shape = shape[1:]
    with path.open("rb") as handle:
        handle.seek(data + origin)
        raw = handle.read(take)
    if len(raw) != take:
        message = f"{name} ended before its declared bytes"
        raise ValueError(message)
    return _decode(raw, dtype, name).reshape(out_shape)


_CACHE: dict[Path, tuple[dict[str, object], int]] = {}


def _cached(path: Path) -> tuple[dict[str, object], int]:
    found = _CACHE.get(path)
    if found is None:
        found = _header(path)
        _CACHE[path] = found
    return found


def _width(dtype: str) -> int:
    if dtype in {"BF16", "F16"}:
        return 2
    if dtype == "F32":
        return 4
    message = f"dtype {dtype} is not a float checkpoint tensor"
    raise TypeError(message)


def _decode(raw: bytes, dtype: str, name: str) -> np.ndarray:
    if dtype == "BF16":
        units = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
        return np.array((units << 16).view(np.float32), dtype=np.float32)
    if dtype == "F16":
        return np.array(np.frombuffer(raw, dtype="<f2"), dtype=np.float32)
    if dtype == "F32":
        return np.array(np.frombuffer(raw, dtype="<f4"), dtype=np.float32)
    message = f"{name} dtype {dtype} is not a float checkpoint tensor"
    raise TypeError(message)


def _header(path: Path) -> tuple[dict[str, object], int]:
    with path.open("rb") as handle:
        count = struct.unpack("<Q", handle.read(8))[0]
        payload = handle.read(count)
    header = json.loads(payload)
    if not isinstance(header, dict):
        message = f"{path.name} safetensors header is not an object"
        raise TypeError(message)
    return header, 8 + count
