"""Kernel-order weight files. One shared span, bound at a plane offset.

``planes`` is a list of ``{name, dtype, offset, nbytes, shape}``. A start is
128-byte aligned. ``nbytes`` is the array size. The bytes are the padded
layout the upload already builds: int8 N 128 and K 256, float16 GEMM N 32,
wide conv N and K 64. A wide conv is that flattened matrix, not the 5D
checkpoint order.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ALIGN = 128
PIECE = 8 * 1024 * 1024
F16_N = 32
CONV_OUT = 32
CONV_TAP = 64
CONV_ALIGN = 64
I8_N = 128
I8_K = 256


class BoundTensor:
    """One plane inside a shared span. The source shape is what the forward reads."""

    def __init__(self, handle: object, plane: dict) -> None:
        self.handle = handle
        self.name = str(plane["name"])
        self.offset = int(plane["offset"])
        self.nbytes = int(plane["nbytes"])
        self.dtype = np.dtype(plane["dtype"])
        self.stored_shape = tuple(int(value) for value in plane["shape"])
        source = plane.get("source_shape", plane["shape"])
        self.source_shape = tuple(int(value) for value in source)
        self.layout = str(plane.get("layout", ""))
        self.rank = int(plane.get("rank", len(self.source_shape)))
        self.logical_n = int(plane.get("logical_n", self.source_shape[0] if self.source_shape else 0))
        self.logical_k = int(
            plane.get("logical_k", self.source_shape[-1] if len(self.source_shape) > 1 else 0)
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self.source_shape

    @property
    def ndim(self) -> int:
        return self.rank


class BlobTable:
    """Name lookup over one packed file. Kernel planes stay in the span."""

    def __init__(self, handle: object, planes: dict[str, dict]) -> None:
        self.handle = handle
        self.planes = planes

    def keys(self) -> set[str]:
        return set(self.planes)

    def tensor(self, name: str) -> BoundTensor:
        plane = _require(self.planes, name)
        layout = str(plane.get("layout", ""))
        if layout == "f16":
            expect_f16(plane)
        elif layout in {"conv64", "conv32"}:
            expect_conv(plane)
        elif layout == "f32":
            expect_f32(plane)
        return BoundTensor(self.handle, plane)

    def host(self, name: str) -> np.ndarray:
        """A small host copy. Norms and biases that no kernel binds use this."""
        return copy_plane(self.handle, _require(self.planes, name))


class Writer:
    """Append arrays at 128-byte starts. ``nbytes`` stays the array size."""

    def __init__(self) -> None:
        self._parts: list[tuple[int, bytes, dict]] = []
        self._cursor = 0

    def add(self, name: str, array: np.ndarray, **extra: object) -> dict:
        raw = np.ascontiguousarray(array)
        start = align(self._cursor)
        blob = raw.tobytes()
        row = {
            "name": name,
            "dtype": np.dtype(raw.dtype).name,
            "offset": start,
            "nbytes": len(blob),
            "shape": [int(value) for value in raw.shape],
        }
        for key, value in extra.items():
            row[key] = _jsonable(value)
        self._parts.append((start, blob, row))
        self._cursor = start + len(blob)
        return row

    def finish(self, bin_path: Path, json_path: Path) -> Path:
        bin_path.parent.mkdir(parents=True, exist_ok=True)
        with bin_path.open("wb") as dst:
            for start, blob, _row in self._parts:
                if dst.tell() != start:
                    dst.seek(start)
                dst.write(blob)
        rows = [row for _start, _blob, row in self._parts]
        json_path.write_text(json.dumps(rows), encoding="utf-8")
        return bin_path


def align(value: int) -> int:
    return (int(value) + ALIGN - 1) & ~(ALIGN - 1)


def pad_multiple(count: int, multiple: int) -> int:
    """The upload pad: at least one tile, then a whole tile."""
    tiled = ((int(count) + multiple - 1) // multiple) * multiple
    return max(multiple, tiled)


def resolve_named(directory: Path, bin_name: str, json_name: str) -> Path | None:
    """Both files, or neither. One of the pair is a broken pack."""
    blob = directory / bin_name
    table = directory / json_name
    if blob.is_file() and table.is_file():
        return blob
    if blob.is_file() or table.is_file():
        missing = table if blob.is_file() else blob
        message = f"weight pack is missing {missing.name}"
        raise FileNotFoundError(message)
    return None


def load_planes(path: Path) -> dict[str, dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        message = f"{path.name} is not a plane list"
        raise RuntimeError(message)
    index: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            message = f"{path.name} has a plane that is not an object"
            raise RuntimeError(message)
        name = str(row["name"])
        if name in index:
            message = f"{path.name} repeats {name}"
            raise RuntimeError(message)
        offset = int(row["offset"])
        nbytes = int(row["nbytes"])
        if offset % ALIGN != 0:
            message = f"{name} starts at {offset}, not a {ALIGN}-byte boundary"
            raise RuntimeError(message)
        shape = tuple(int(value) for value in row["shape"])
        dtype = np.dtype(row["dtype"])
        expect = int(np.prod(shape, dtype=np.int64)) * int(dtype.itemsize)
        if nbytes != expect:
            message = f"{name} nbytes {nbytes} is not the array size {expect}"
            raise RuntimeError(message)
        stored = dict(row)
        stored["dtype"] = dtype.name
        stored["shape"] = [int(value) for value in shape]
        index[name] = stored
    return index


def fill(view: memoryview, path: Path) -> None:
    """Read ``path`` into ``view`` in 8 MiB steps. A short read continues."""
    size = len(view)
    with path.open("rb") as src:
        offset = 0
        while offset < size:
            piece = min(PIECE, size - offset)
            read = src.readinto(view[offset : offset + piece])
            if not read:
                message = f"{path.name} stopped at {offset} of {size}"
                raise OSError(message)
            offset += int(read)


def f16_matrix(weight: np.ndarray) -> np.ndarray:
    """Float16 GEMM weight with N padded to 32. K stays."""
    data = np.ascontiguousarray(weight, dtype=np.float16)
    if data.ndim != 2:
        message = "a float16 GEMM weight is a matrix"
        raise RuntimeError(message)
    rows, cols = int(data.shape[0]), int(data.shape[1])
    padded = pad_multiple(rows, F16_N)
    if padded == rows:
        return data
    stored = np.zeros((padded, cols), dtype=np.float16)
    stored[:rows] = data
    return stored


def conv_matrix(weight: np.ndarray) -> tuple[np.ndarray, dict]:
    """The bytes ``conv3d`` uploads. A 4D weight gains a time tap of 1 first."""
    rank = int(weight.ndim)
    shaped = weight
    if rank == 4:
        shaped = np.reshape(
            weight,
            (int(weight.shape[0]), int(weight.shape[1]), 1, int(weight.shape[2]), int(weight.shape[3])),
        )
    if shaped.ndim != 5:
        message = "a convolution weight is 4D or 5D"
        raise RuntimeError(message)
    out_c = int(shaped.shape[0])
    taps = int(np.prod(shaped.shape[1:], dtype=np.int64))
    source_shape = [int(value) for value in shaped.shape]
    meta: dict = {"source_shape": source_shape, "logical_n": out_c, "rank": rank}
    if out_c >= CONV_OUT and taps >= CONV_TAP:
        padded_n = pad_multiple(out_c, CONV_ALIGN)
        k_pad = pad_multiple(taps, CONV_ALIGN)
        flat = np.ascontiguousarray(shaped, dtype=np.float16).reshape(out_c, taps)
        stored = np.zeros((padded_n, k_pad), dtype=np.float16)
        stored[:out_c, :taps] = flat
        meta["layout"] = "conv64"
        return stored, meta
    meta["layout"] = "conv32"
    return np.ascontiguousarray(shaped, dtype=np.float32), meta


def i8_bias(codes: np.ndarray, bias: np.ndarray, logical_n: int) -> np.ndarray:
    """The float16 bias row ``scale_i8`` binds, padded to the stored N."""
    row = np.zeros(int(codes.shape[0]), dtype=np.float16)
    flat = np.ascontiguousarray(bias, dtype=np.float16).reshape(-1)
    count = min(int(logical_n), int(row.shape[0]), int(flat.shape[0]))
    row[:count] = flat[:count]
    return row


def expect_f16(plane: dict) -> None:
    logical_n = int(plane["logical_n"])
    logical_k = int(plane["logical_k"])
    want = (pad_multiple(logical_n, F16_N), logical_k)
    _expect_shape(plane, want, np.float16)


def expect_i8(plane: dict) -> None:
    logical_n = int(plane.get("logical_n", plane["shape"][0]))
    logical_k = int(plane.get("logical_k", plane["shape"][1]))
    want = (pad_multiple(logical_n, I8_N), pad_multiple(logical_k, I8_K))
    _expect_shape(plane, want, np.int8)


def expect_conv(plane: dict) -> None:
    source = tuple(int(value) for value in plane["source_shape"])
    if len(source) != 5:
        message = f"{plane['name']} convolution is not 5D"
        raise RuntimeError(message)
    out_c = int(source[0])
    taps = int(np.prod(source[1:], dtype=np.int64))
    layout = str(plane.get("layout"))
    if layout == "conv64":
        want = (pad_multiple(out_c, CONV_ALIGN), pad_multiple(taps, CONV_ALIGN))
        _expect_shape(plane, want, np.float16)
        return
    if layout == "conv32":
        _expect_shape(plane, source, np.float32)
        return
    message = f"{plane['name']} layout {layout} is not a convolution"
    raise RuntimeError(message)


def expect_f32(plane: dict) -> None:
    shape = tuple(int(value) for value in plane["shape"])
    _expect_shape(plane, shape, np.float32)


def write_vae_arrays(arrays: dict[str, np.ndarray], dest: Path) -> Path:
    """Pack VAE tensors in name order. Wide convs are the padded float16 matrix."""
    writer = Writer()
    for name in sorted(arrays):
        value = np.asarray(arrays[name])
        if name.endswith(".weight") and value.ndim in {2, 4, 5}:
            _add_weight(writer, name, value)
            continue
        if name.endswith(".weight"):
            message = f"{name} rank {value.ndim} is not a VAE plane"
            raise RuntimeError(message)
        stored = np.ascontiguousarray(value, dtype=np.float32).reshape(-1)
        writer.add(
            name,
            stored,
            layout="f32",
            source_shape=[int(stored.shape[0])],
            rank=1,
        )
    return writer.finish(dest, dest.with_suffix(".json"))


def copy_plane(handle: object, plane: dict) -> np.ndarray:
    from favlo.memory import memory

    raw = memory.load(handle, int(plane["offset"]), int(plane["nbytes"]))
    shape = tuple(int(value) for value in plane["shape"])
    return np.frombuffer(raw, dtype=np.dtype(plane["dtype"])).reshape(shape).copy()


def map_plane(handle: object, plane: dict) -> tuple[np.ndarray, memoryview]:
    """A view of one plane. The caller keeps the memoryview alive."""
    from favlo.memory import memory

    view = memory.map(handle, int(plane["offset"]), int(plane["nbytes"]))
    shape = tuple(int(value) for value in plane["shape"])
    array = np.ndarray(shape, dtype=np.dtype(plane["dtype"]), buffer=view)
    return array, view


def _add_weight(writer: Writer, name: str, value: np.ndarray) -> None:
    if value.ndim == 2:
        logical_n, logical_k = int(value.shape[0]), int(value.shape[1])
        writer.add(
            name,
            f16_matrix(value),
            layout="f16",
            logical_n=logical_n,
            logical_k=logical_k,
            source_shape=[logical_n, logical_k],
            rank=2,
        )
        return
    stored, meta = conv_matrix(value)
    writer.add(name, stored, **meta)


def _expect_shape(plane: dict, want: tuple[int, ...], dtype: np.dtype) -> None:
    got = tuple(int(value) for value in plane["shape"])
    if got != tuple(want) or np.dtype(plane["dtype"]) != np.dtype(dtype):
        message = (
            f"{plane['name']} is {got} {plane['dtype']}, kernel wants {tuple(want)} {np.dtype(dtype).name}"
        )
        raise RuntimeError(message)


def _require(planes: dict[str, dict], name: str) -> dict:
    found = planes.get(name)
    if found is None:
        message = f"weight pack is missing {name}"
        raise FileNotFoundError(message)
    return found


def _jsonable(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value
