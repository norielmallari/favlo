"""Read one GGUF file. Tensor bytes stay on disk until a name is loaded."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_F32 = 0
_F16 = 1
_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


@dataclass(frozen=True)
class TensorInfo:
    name: str
    type_id: int
    dims: tuple[int, ...]
    offset: int


class Gguf:
    """Version 3 GGUF. ``dims[0]`` is the contiguous axis."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.metadata: dict[str, object] = {}
        self.tensors: dict[str, TensorInfo] = {}
        self._data = 0
        self._parse()

    def array(self, name: str) -> np.ndarray:
        """Float32. A matrix is ``[dims[1], dims[0]]``."""
        info = self.tensors[name]
        count = 1
        for dim in info.dims:
            count *= dim
        with self.path.open("rb") as handle:
            handle.seek(self._data + info.offset)
            if info.type_id == _F16:
                raw = np.frombuffer(handle.read(count * 2), dtype=np.float16).astype(np.float32)
            elif info.type_id == _F32:
                raw = np.frombuffer(handle.read(count * 4), dtype=np.float32).copy()
            else:
                message = f"{name} type {info.type_id} is not F16 or F32"
                raise ValueError(message)
        if len(info.dims) == 1:
            return raw.reshape(info.dims[0])
        if len(info.dims) == 2:
            return raw.reshape(info.dims[1], info.dims[0])
        message = f"{name} rank {len(info.dims)} is not packed"
        raise ValueError(message)

    def _parse(self) -> None:
        data = self.path.read_bytes() if self.path.stat().st_size < 64 * 1024 * 1024 else self._head()
        if data[:4] != b"GGUF":
            message = f"{self.path.name} is not GGUF"
            raise ValueError(message)
        version, n_tensors, n_kv = struct.unpack_from("<IQQ", data, 4)
        if version != 3:
            message = f"{self.path.name} GGUF version {version} is not 3"
            raise ValueError(message)
        offset = 24
        for _ in range(n_kv):
            key, offset = _string(data, offset)
            kind = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            value, offset = _value(data, offset, kind)
            self.metadata[key] = value
        infos: list[TensorInfo] = []
        for _ in range(n_tensors):
            name, offset = _string(data, offset)
            rank = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            dims = struct.unpack_from("<" + "Q" * rank, data, offset)
            offset += 8 * rank
            type_id = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            tensor_offset = struct.unpack_from("<Q", data, offset)[0]
            offset += 8
            infos.append(TensorInfo(name, type_id, tuple(int(dim) for dim in dims), int(tensor_offset)))
        alignment = int(self.metadata.get("general.alignment", 32))
        self._data = (offset + alignment - 1) & ~(alignment - 1)
        self.tensors = {info.name: info for info in infos}

    def _head(self) -> bytes:
        """The header is the metadata plus the tensor table. A few dozen megabytes cover it."""
        with self.path.open("rb") as handle:
            return handle.read(48 * 1024 * 1024)


def _string(data: bytes, offset: int) -> tuple[str, int]:
    size = struct.unpack_from("<Q", data, offset)[0]
    offset += 8
    text = data[offset : offset + size].decode()
    return text, offset + size


def _value(data: bytes, offset: int, kind: int) -> tuple[object, int]:
    if kind == 8:
        return _string(data, offset)
    if kind in _SIZES:
        view = memoryview(data)[offset : offset + _SIZES[kind]]
        value = struct.unpack("<" + _CODE[kind], view)[0]
        return value, offset + _SIZES[kind]
    if kind == 9:
        element = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        count = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        items: list[object] = []
        for _ in range(count):
            item, offset = _value(data, offset, element)
            items.append(item)
        return items, offset
    message = f"GGUF value type {kind} is not read"
    raise ValueError(message)


_CODE = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "B", 10: "Q", 11: "q", 12: "d"}
