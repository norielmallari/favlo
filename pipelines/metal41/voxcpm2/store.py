"""Read bfloat16 tensors from the checkout and keep Linear weights resident."""

from __future__ import annotations

import json
import mmap
import struct
from pathlib import Path

import numpy as np

from voxcpm2.metal import Resident, Session


class Store:
    """A memory map of ``model.safetensors``. Rows are converted when they are used."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file = path.open("rb")
        self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        count = struct.unpack_from("<Q", self._map, 0)[0]
        header = json.loads(self._map[8 : 8 + count])
        header.pop("__metadata__", None)
        self.header = header
        self._origin = 8 + count
        self._resident: dict[str, Resident] = {}
        self._vector: dict[str, np.ndarray] = {}

    def close(self) -> None:
        self._map.close()
        self._file.close()

    def vector(self, name: str) -> np.ndarray:
        """A one-dimensional float32 view of a tensor."""
        cached = self._vector.get(name)
        if cached is None:
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
        return session.linear(hidden, self._matrix(session, name), shift)

    def embed(self, tokens: np.ndarray) -> np.ndarray:
        """Gather token rows. The embedding scale is 1 when ``use_mup`` is false."""
        meta = self.header["base_lm.embed_tokens.weight"]
        rows, dim = (int(item) for item in meta["shape"])
        start = self._origin + int(meta["data_offsets"][0])
        stride = dim * 2
        out = np.empty((len(tokens), dim), dtype=np.float32)
        for index, token in enumerate(np.asarray(tokens, dtype=np.int64)):
            row = int(token)
            if row < 0 or row >= rows:
                message = f"token {row} is outside the embedding"
                raise ValueError(message)
            offset = start + row * stride
            out[index] = _bf16_f32(self._map[offset : offset + stride])
        return out

    def _matrix(self, session: Session, name: str) -> Resident:
        cached = self._resident.get(name)
        if cached is None:
            cached = session.resident(self._f32(name))
            self._resident[name] = cached
        return cached

    def _f32(self, name: str) -> np.ndarray:
        meta = self.header[name]
        if meta.get("dtype") != "BF16":
            message = f"{name} is {meta.get('dtype')}"
            raise ValueError(message)
        start, end = (int(item) for item in meta["data_offsets"])
        raw = self._map[self._origin + start : self._origin + end]
        return _bf16_f32(raw).reshape(meta["shape"])


def _bf16_f32(raw: bytes) -> np.ndarray:
    units = np.frombuffer(raw, dtype="<u2")
    return (units.astype(np.uint32) << 16).view(np.float32)
