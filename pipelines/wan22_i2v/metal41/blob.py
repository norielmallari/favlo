"""Raw weight planes. One file, 128-byte starts, 8 MiB reads.

``weights.bin`` holds the plane bytes with zero pad between them. ``planes.json``
is a list of ``{name, dtype, offset, nbytes, shape}``. ``nbytes`` is the plane.
The pad is not part of it. A missing pair means the caller keeps its old files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable
from typing import Any, Mapping

import numpy as np

ALIGN = 128
CHUNK = 8 * 1024 * 1024
BLOB_NAME = "weights.bin"
INDEX_NAME = "planes.json"


@dataclass(frozen=True)
class Plane:
    """One binding inside a raw blob."""

    name: str
    dtype: str
    offset: int
    nbytes: int
    shape: tuple[int, ...]
    kind: str | None = None
    logical_n: int | None = None
    logical_k: int | None = None
    padded_n: int | None = None
    stored_k: int | None = None
    source_shape: tuple[int, ...] | None = None


def align_up(offset: int, align: int = ALIGN) -> int:
    if offset < 0:
        message = f"offset must be non-negative, got {offset}"
        raise ValueError(message)
    return (int(offset) + align - 1) & ~(align - 1)


def blob_ready(
    directory: Path, *, blob_name: str = BLOB_NAME, index_name: str = INDEX_NAME
) -> bool:
    folder = Path(directory)
    return (folder / blob_name).is_file() and (folder / index_name).is_file()


def write_blob(
    directory: Path,
    planes: Iterable[tuple[str, np.ndarray]],
    *,
    blob_name: str = BLOB_NAME,
    index_name: str = INDEX_NAME,
    meta: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    """Write planes in order. Each start is 128-byte aligned."""
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    partial = folder / f"{blob_name}.partial"
    index_partial = folder / f"{index_name}.partial"
    try:
        rows: list[dict[str, Any]] = []
        offset = 0
        with partial.open("wb") as out:
            for name, array in planes:
                raw = np.ascontiguousarray(array)
                start = align_up(offset)
                if start > offset:
                    out.write(b"\x00" * (start - offset))
                payload = memoryview(raw).cast("B")
                out.write(payload)
                entry: dict[str, Any] = {
                    "name": name,
                    "dtype": raw.dtype.name,
                    "offset": start,
                    "nbytes": int(raw.nbytes),
                    "shape": [int(value) for value in raw.shape],
                }
                extra = None if meta is None else meta.get(name)
                if extra:
                    for key, value in extra.items():
                        if key not in entry:
                            entry[key] = value
                rows.append(entry)
                offset = start + int(raw.nbytes)
        index_partial.write_text(json.dumps(rows), encoding="utf-8")
        partial.replace(folder / blob_name)
        index_partial.replace(folder / index_name)
    except Exception:
        partial.unlink(missing_ok=True)
        index_partial.unlink(missing_ok=True)
        raise


def load_index(
    directory: Path, *, blob_name: str = BLOB_NAME, index_name: str = INDEX_NAME
) -> dict[str, Plane] | None:
    """Plane table, or ``None`` when either file is absent."""
    folder = Path(directory)
    blob = folder / blob_name
    index = folder / index_name
    if not blob.is_file() or not index.is_file():
        return None
    rows = json.loads(index.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        message = f"{index.name} is not a plane list"
        raise ValueError(message)
    size = blob.stat().st_size
    planes: dict[str, Plane] = {}
    for row in rows:
        plane = _plane(row)
        if plane.name in planes:
            message = f"duplicate plane {plane.name}"
            raise ValueError(message)
        if plane.offset % ALIGN:
            message = f"{plane.name} starts at {plane.offset}, not a {ALIGN}-byte boundary"
            raise ValueError(message)
        if plane.offset < 0 or plane.nbytes < 0 or plane.offset + plane.nbytes > size:
            message = f"{plane.name} spans {plane.offset}+{plane.nbytes} past {size}"
            raise ValueError(message)
        expect = int(np.dtype(plane.dtype).itemsize)
        count = 1
        for dim in plane.shape:
            count *= dim
        if count * expect != plane.nbytes:
            message = f"{plane.name} nbytes {plane.nbytes} does not match its shape"
            raise ValueError(message)
        planes[plane.name] = plane
    return planes


def read_into(path: Path, dest: memoryview, *, chunk: int = CHUNK) -> None:
    """Sequential read of ``path`` into ``dest`` in ``chunk`` steps."""
    size = Path(path).stat().st_size
    if size > len(dest):
        message = f"{Path(path).name} is {size} bytes and the span is {len(dest)}"
        raise OSError(message)
    step = int(chunk)
    if step < 1:
        message = f"chunk must be positive, got {step}"
        raise ValueError(message)
    with Path(path).open("rb") as src:
        offset = 0
        while offset < size:
            piece = min(step, size - offset)
            got = src.readinto(dest[offset : offset + piece])
            if not got:
                message = f"{Path(path).name} stopped at {offset} of {size}"
                raise OSError(message)
            offset += got


def fill_span(handle: Any, path: Path, *, chunk: int = CHUNK) -> None:
    """``readinto`` the file at the span's mapped address."""
    from favlo.memory import memory

    size = Path(path).stat().st_size
    read_into(path, memory.map(handle, 0, size), chunk=chunk)


def plane_array(directory: Path, plane: Plane, *, blob_name: str = BLOB_NAME) -> np.ndarray:
    """One plane as its own array. The pad before the next plane stays in the file."""
    out = np.empty(plane.shape, dtype=np.dtype(plane.dtype))
    view = memoryview(out).cast("B")
    blob = Path(directory) / blob_name
    with blob.open("rb") as src:
        src.seek(plane.offset)
        offset = 0
        while offset < plane.nbytes:
            piece = min(CHUNK, plane.nbytes - offset)
            got = src.readinto(view[offset : offset + piece])
            if not got:
                message = f"{plane.name} stopped at {offset} of {plane.nbytes}"
                raise OSError(message)
            offset += got
    return out


def _plane(row: Any) -> Plane:
    if not isinstance(row, dict):
        message = "a plane row is not an object"
        raise ValueError(message)
    name = str(row["name"])
    dtype = str(row["dtype"])
    shape = tuple(int(value) for value in row["shape"])
    source = row.get("source_shape")
    source_shape = None if source is None else tuple(int(value) for value in source)
    kind = row.get("kind")
    return Plane(
        name=name,
        dtype=dtype,
        offset=int(row["offset"]),
        nbytes=int(row["nbytes"]),
        shape=shape,
        kind=None if kind is None else str(kind),
        logical_n=_optional_int(row.get("logical_n")),
        logical_k=_optional_int(row.get("logical_k")),
        padded_n=_optional_int(row.get("padded_n")),
        stored_k=_optional_int(row.get("stored_k")),
        source_shape=source_shape,
    )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
