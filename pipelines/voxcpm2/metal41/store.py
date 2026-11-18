"""Read bfloat16 tensors from the checkout, or spans in a packed blob."""

from __future__ import annotations

import json
import mmap
import struct
from pathlib import Path

import numpy as np

from favlo.memory import memory
from voxcpm2.metal41.metal import Resident, Session, Span, buffer_view

PACK_VERSION = 1
PACK_NAME = "pack.json"
WEIGHTS_NAME = "weights.bin"
ALIGN = 128
READ_BYTES = 8 << 20
EMBED_NAME = "base_lm.embed_tokens.weight"
_DTYPE = {"F16": np.float16, "F32": np.float32}


class Store:
    """Weights for one opening. A pack is spans. A safetensors file is read on use."""

    def __init__(self, path: Path) -> None:
        self.path: Path | None = path
        self._file = path.open("rb")
        self._map: mmap.mmap | None = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        count = struct.unpack_from("<Q", self._map, 0)[0]
        header = json.loads(self._map[8 : 8 + count])
        header.pop("__metadata__", None)
        self.header = header
        self._origin = 8 + count
        self._buffer: memoryview | None = None
        self._records: dict[str, dict[str, object]] | None = None
        self._handle: object = None
        self._embed: np.ndarray | None = None
        self._resident: dict[str, Resident] = {}
        self._vector: dict[str, np.ndarray] = {}
        self._bias: dict[str, object] = {}
        self._gain: dict[str, object] = {}

    @classmethod
    def bind(
        cls,
        buffer: object,
        records: dict[str, dict[str, object]],
        handle: object,
    ) -> Store:
        """Spans in ``buffer``. ``handle`` is the shared allocation those spans bind."""
        self = cls.__new__(cls)
        self.path = None
        self._file = None
        self._map = None
        self._origin = 0
        self._buffer = memoryview(buffer)
        self._records = records
        self._handle = handle
        self._embed = None
        self.header = {
            name: {"dtype": row.get("dtype"), "shape": row.get("shape")}
            for name, row in records.items()
            if row.get("kind") != "vae"
        }
        self._resident = {}
        self._vector = {}
        self._bias = {}
        self._gain = {}
        return self

    def close(self) -> None:
        self._buffer = None
        self._embed = None
        self._records = None
        self._handle = None
        if self._map is not None:
            self._map.close()
            self._map = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def vector(self, name: str) -> np.ndarray:
        """A one-dimensional float32 view of a tensor."""
        cached = self._vector.get(name)
        if cached is None:
            if self._records is not None:
                row = self._records[name]
                if row.get("kind") not in {"bias", "vector"}:
                    message = f"{name} is not a host vector"
                    raise ValueError(message)
                cached = np.ascontiguousarray(tensor_view(self._buffer, row), dtype=np.float32)
                cached = cached.reshape(-1)
            else:
                cached = np.ascontiguousarray(self._f32(name), dtype=np.float32).reshape(-1)
            self._vector[name] = cached
        return cached

    def linear(
        self,
        session: Session,
        name: str,
        hidden: np.ndarray,
        bias: str | None = None,
    ) -> np.ndarray:
        """Project ``hidden`` with a resident half weight."""
        shift = None if bias is None else self.vector(bias)
        return session.linear(hidden, self.weight(session, name), shift)

    def weight(self, session: Session, name: str) -> Resident:
        """The resident half matrix for a device linear."""
        if self._records is None:
            return self._matrix(session, name)
        cached = self._resident.get(name)
        if cached is None:
            row = self._kind(name, "linear")
            shape = _shape(row)
            pad = _shape(row, "pad")
            cached = Resident(
                self._handle,
                shape[0],
                shape[1],
                pad[0],
                pad[1],
                int(row["offset"]),
            )
            self._resident[name] = cached
        return cached

    def bias_handle(self, session: Session, name: str) -> object:
        """A resident float32 bias. The device linear adds it after the product."""
        if self._records is None:
            return self._upload_bias(session, name)
        cached = self._bias.get(name)
        if cached is None:
            row = self._kind(name, "bias")
            cached = Span(self._handle, int(row["offset"]), tuple(_shape(row)))
            self._bias[name] = cached
        return cached

    def gain(self, session: Session, name: str) -> object:
        """A resident half RMSNorm weight."""
        if self._records is None:
            return self._upload_gain(session, name)
        cached = self._gain.get(name)
        if cached is None:
            row = self._kind(name, "gain")
            cached = Span(self._handle, int(row["offset"]), tuple(_shape(row)))
            self._gain[name] = cached
        return cached

    def embed(self, tokens: np.ndarray) -> np.ndarray:
        """Gather token rows. The embedding scale is 1 when ``use_mup`` is false."""
        if self._records is not None:
            return _gather(self._embedding(), tokens)
        meta = self.header[EMBED_NAME]
        rows, dim = (int(item) for item in meta["shape"])
        start = self._origin + int(meta["data_offsets"][0])
        stride = dim * 2
        out = np.empty((len(tokens), dim), dtype=np.float32)
        mapped = self._map
        if mapped is None:
            message = "the embedding map is closed"
            raise RuntimeError(message)
        for index, token in enumerate(np.asarray(tokens, dtype=np.int64)):
            row = int(token)
            if row < 0 or row >= rows:
                message = f"token {row} is outside the embedding"
                raise ValueError(message)
            offset = start + row * stride
            out[index] = _bf16_f32(mapped[offset : offset + stride])
        return out

    def _embedding(self) -> np.ndarray:
        if self._embed is None:
            row = self._kind(EMBED_NAME, "embed")
            self._embed = tensor_view(self._buffer, row)
        return self._embed

    def _upload_bias(self, session: Session, name: str) -> object:
        cached = self._bias.get(name)
        if cached is None:
            raw = np.ascontiguousarray(self.vector(name), dtype=np.float32).tobytes()
            cached = session.upload(raw)
            self._bias[name] = cached
        return cached

    def _upload_gain(self, session: Session, name: str) -> object:
        cached = self._gain.get(name)
        if cached is None:
            raw = np.ascontiguousarray(self.vector(name), dtype=np.float16).tobytes()
            cached = session.upload(raw)
            self._gain[name] = cached
        return cached

    def _matrix(self, session: Session, name: str) -> Resident:
        cached = self._resident.get(name)
        if cached is None:
            cached = session.resident(self._f32(name))
            self._resident[name] = cached
        return cached

    def _kind(self, name: str, kind: str) -> dict[str, object]:
        if self._records is None:
            message = f"{name} is not in a packed checkout"
            raise RuntimeError(message)
        row = self._records.get(name)
        if not isinstance(row, dict) or row.get("kind") != kind:
            message = f"{name} is not a {kind}"
            raise ValueError(message)
        return row

    def _f32(self, name: str) -> np.ndarray:
        meta = self.header[name]
        if meta.get("dtype") != "BF16":
            message = f"{name} is {meta.get('dtype')}"
            raise ValueError(message)
        start, end = (int(item) for item in meta["data_offsets"])
        mapped = self._map
        if mapped is None:
            message = f"{name} has no safetensors map"
            raise RuntimeError(message)
        raw = mapped[self._origin + start : self._origin + end]
        return _bf16_f32(raw).reshape(meta["shape"])


def tensor_view(buffer: object, row: dict[str, object]) -> np.ndarray:
    """The packed array named by ``row``. The view aliases ``buffer``."""
    label = str(row.get("dtype"))
    if label not in _DTYPE:
        message = f"packed dtype {label} is not float16 or float32"
        raise ValueError(message)
    shape = _shape(row)
    expect = int(row["nbytes"])
    width = int(np.dtype(_DTYPE[label]).itemsize)
    count = 1
    for item in shape:
        count *= item
    if expect != count * width:
        message = "packed tensor length does not match its shape"
        raise ValueError(message)
    return buffer_view(buffer, int(row["offset"]), shape, _DTYPE[label])


def pack_ready(path: Path) -> bool:
    """True when ``weights.bin`` and a version-1 ``pack.json`` are both present."""
    if not (path / PACK_NAME).is_file() or not (path / WEIGHTS_NAME).is_file():
        return False
    try:
        load_pack(path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError):
        return False
    return True


def load_pack(path: Path) -> dict[str, object]:
    """The pack index. A wrong version raises."""
    body = json.loads((path / PACK_NAME).read_text(encoding="utf-8"))
    if body.get("version") != PACK_VERSION or body.get("file") != WEIGHTS_NAME:
        message = f"{path.name} pack is not version {PACK_VERSION}"
        raise ValueError(message)
    if not isinstance(body.get("tensors"), dict):
        message = f"{path.name} pack has no tensors"
        raise ValueError(message)
    return body


def weight_bytes(path: Path) -> int:
    """The weight file open reads. A pack replaces the safetensors file and the VAE zip."""
    if pack_ready(path):
        return (path / WEIGHTS_NAME).stat().st_size
    weights = path / "model.safetensors"
    vae = path / "audiovae.pth"
    return weights.stat().st_size + vae.stat().st_size


def read_into(path: Path, view: memoryview, chunk: int = READ_BYTES) -> None:
    """Fill ``view`` from ``path`` in sequential pieces of ``chunk`` bytes."""
    size = len(view)
    with path.open("rb", buffering=0) as handle:
        offset = 0
        while offset < size:
            piece = min(int(chunk), size - offset)
            got = handle.readinto(view[offset : offset + piece])
            if not got:
                message = f"{path.name} stopped at {offset} of {size}"
                raise OSError(message)
            offset += int(got)


def read_blob(path: Path, owner: str) -> tuple[object, memoryview]:
    """Allocate one shared buffer and read ``path`` into it."""
    size = path.stat().st_size
    if size < 1:
        message = f"{path.name} is empty"
        raise ValueError(message)
    handle = memory.allocate(size, owner=owner)
    view = memory.map(handle, 0, size)
    read_into(path, view)
    return handle, view


def _bf16_f32(raw: bytes) -> np.ndarray:
    units = np.frombuffer(raw, dtype="<u2")
    return (units.astype(np.uint32) << 16).view(np.float32)


def _shape(row: dict[str, object], key: str = "shape") -> tuple[int, ...]:
    raw = row.get(key)
    if not isinstance(raw, list) or not raw:
        message = f"packed tensor {key} is missing"
        raise ValueError(message)
    return tuple(int(item) for item in raw)


def _gather(table: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    rows, dim = int(table.shape[0]), int(table.shape[1])
    out = np.empty((len(tokens), dim), dtype=np.float32)
    for index, token in enumerate(np.asarray(tokens, dtype=np.int64)):
        row = int(token)
        if row < 0 or row >= rows:
            message = f"token {row} is outside the embedding"
            raise ValueError(message)
        out[index] = np.asarray(table[row], dtype=np.float32)
    return out
