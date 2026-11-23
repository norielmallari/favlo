"""Memory-mapped int4 record. Tensors upload when a stage asks for them."""

from __future__ import annotations

import json
import mmap
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from klein_4b.cuda import engine
from klein_4b.cuda.constants import FORMAT


@dataclass
class Tensor:
    kind: str
    ptr: int
    rows: int = 0
    inner: int = 0
    stored_k: int = 0
    codes: int = 0
    scales: int = 0
    bias: int | None = None
    shape: tuple[int, ...] = ()
    extra: tuple[int, ...] = ()


class Store:
    def __init__(self, folder: Path) -> None:
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format") != FORMAT or manifest.get("complete") is not True:
            message = "klein record is not a finished int4 pack"
            raise ValueError(message)
        self.folder = folder
        self.manifest = manifest
        self.index = json.loads((folder / "index.json").read_text(encoding="utf-8"))
        self._handle = (folder / "weights.bin").open("rb")
        self._map = mmap.mmap(self._handle.fileno(), 0, access=mmap.ACCESS_READ)
        self.device: dict[str, Tensor] = {}

    def close(self) -> None:
        self.free_all()
        self._map.close()
        self._handle.close()

    def stages(self) -> dict[str, int]:
        raw = self.manifest.get("stage_bytes")
        if isinstance(raw, dict):
            return {str(key): int(value) for key, value in raw.items()}
        return {}

    def has(self, name: str) -> bool:
        return name in self.index

    def get(self, name: str) -> Tensor:
        cached = self.device.get(name)
        if cached is not None:
            return cached
        meta = self.index.get(name)
        if not isinstance(meta, dict):
            tail = name.rsplit("/", 1)[-1]
            near = [key for key in self.index if key.endswith(tail)][:8]
            message = f"missing tensor {name}; near {near}"
            raise KeyError(message)
        tensor = self._upload(meta)
        bias = self.index.get(name.replace(".weight", ".bias") if name.endswith(".weight") else "")
        if name.endswith(".weight") and isinstance(bias, dict) and bias.get("kind") == "f16":
            loaded = self._upload(bias)
            tensor.bias = loaded.ptr
            tensor.extra = tensor.extra + (loaded.ptr,)
        self.device[name] = tensor
        return tensor

    def free_prefix(self, prefix: str) -> None:
        for name in [key for key in self.device if key.startswith(prefix)]:
            self._free_one(name)

    def free_all(self) -> None:
        for name in list(self.device):
            self._free_one(name)

    def _free_one(self, name: str) -> None:
        tensor = self.device.pop(name)
        seen: set[int] = set()
        for value in (tensor.ptr, tensor.codes, tensor.scales, tensor.bias, *tensor.extra):
            if isinstance(value, int) and value and value not in seen:
                seen.add(value)
                engine.free(value)

    def _upload(self, meta: dict[str, object]) -> Tensor:
        kind = str(meta["kind"])
        offset = int(meta["offset"])
        shape = tuple(int(item) for item in meta["shape"])  # type: ignore[union-attr]
        if kind == "f16":
            count = int(np.prod(shape)) if shape else int(meta["nbytes"]) // 2
            raw = self._map[offset : offset + count * 2]
            ptr = _upload(raw)
            return Tensor(kind="f16", ptr=ptr, rows=count, shape=shape)
        if kind == "f32":
            count = int(np.prod(shape))
            raw = self._map[offset : offset + count * 4]
            ptr = _upload(raw)
            return Tensor(kind="f32", ptr=ptr, rows=count, shape=shape)
        codes_bytes = int(meta["codes_bytes"])
        scales_bytes = int(meta["scales_bytes"])
        codes = _upload(self._map[offset : offset + codes_bytes])
        scales = _upload(self._map[offset + codes_bytes : offset + codes_bytes + scales_bytes])
        return Tensor(
            kind="int4",
            ptr=codes,
            rows=int(meta["rows"]),
            inner=int(meta["inner"]),
            stored_k=int(meta["stored_k"]),
            codes=codes,
            scales=scales,
            shape=shape,
        )


def _upload(raw: bytes | memoryview) -> int:
    payload = bytes(raw)
    ptr = engine.alloc(len(payload))
    host = (ctypes_buffer(payload))
    engine.call("klein_copy_h2d", engine.pointer(ptr), host, ctypes_u64(len(payload)))
    return ptr


def ctypes_buffer(payload: bytes):
    import ctypes

    buffer = ctypes.create_string_buffer(payload)
    return ctypes.cast(buffer, ctypes.c_void_p)


def ctypes_u64(value: int):
    import ctypes

    return ctypes.c_uint64(value)
