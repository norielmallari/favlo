"""Read one tensor from a PyTorch zip checkpoint. No torch import."""

from __future__ import annotations

import io
import pickle
import sys
import types
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

_DTYPE = {
    "FloatStorage": np.float32,
    "HalfStorage": np.float16,
    "BFloat16Storage": np.dtype("bfloat16") if hasattr(np, "bfloat16") else np.float32,
    "DoubleStorage": np.float64,
    "LongStorage": np.int64,
    "IntStorage": np.int32,
    "ShortStorage": np.int16,
    "ByteStorage": np.uint8,
    "BoolStorage": np.bool_,
}


def _module(name: str) -> types.ModuleType:
    found = sys.modules.get(name)
    if isinstance(found, types.ModuleType):
        return found
    module = types.ModuleType(name)
    sys.modules[name] = module
    if "." in name:
        parent, child = name.rsplit(".", 1)
        setattr(_module(parent), child, module)
    return module


class Archive:
    """Lazy state dict. Tensors are views into the zip storages."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._zip = zipfile.ZipFile(path)
        prefix = _prefix(self._zip)
        self._prefix = prefix
        order = self._zip.read(f"{prefix}/byteorder").decode().strip()
        self._little = order != "big"
        self._meta, self._ids = _unpickle(self._zip.read(f"{prefix}/data.pkl"))

    def close(self) -> None:
        self._zip.close()

    def choose(self, name: str | None) -> str:
        """Use the single state dict. ``name`` selects one when several are stored."""
        if not isinstance(self._meta, dict):
            message = "checkpoint is not a dict"
            raise RuntimeError(message)
        if any(str(key).startswith("blocks.0.") for key in self._meta):
            return "checkpoint"
        flat = any(
            "._fsdp_wrapped_module" in str(key) or str(key).startswith("model.blocks.")
            for key in self._meta
        )
        if flat:
            self._normalize()
            return "checkpoint"
        nested = [
            key
            for key, value in self._meta.items()
            if isinstance(value, dict) and "storage" not in value
        ]
        if name is None and len(nested) == 1:
            name = str(nested[0])
        if name is None or name not in self._meta:
            found = ", ".join(str(key) for key in self._meta)
            message = f"pass --copy to name one state dict: {found}"
            raise RuntimeError(message)
        chosen = self._meta[name]
        if not isinstance(chosen, dict):
            message = f"{name} is not a state dict"
            raise RuntimeError(message)
        self._meta = chosen
        self._normalize()
        return str(name)

    def _normalize(self) -> None:
        renamed: dict[str, Any] = {}
        for key, value in self._meta.items():
            name = str(key)
            if name.startswith("model."):
                name = name[len("model.") :]
            name = name.replace("._fsdp_wrapped_module", "")
            renamed[name] = value
        self._meta = renamed

    def keys(self) -> list[str]:
        if isinstance(self._meta, dict):
            return [str(key) for key in self._meta]
        message = "checkpoint is not a state dict"
        raise RuntimeError(message)

    def tensor(self, name: str) -> np.ndarray:
        item = self._meta[name]
        storage_index = item["storage"][1]
        pid = self._ids[storage_index]
        storage_type = pid[1]
        key = str(pid[2])
        numel = int(pid[4])
        type_name = getattr(storage_type, "__name__", "")
        dtype = _DTYPE.get(type_name)
        if dtype is None:
            message = f"unsupported storage {type_name} for {name}"
            raise RuntimeError(message)
        raw = self._zip.read(f"{self._prefix}/data/{key}")
        if not self._little:
            raw = _swap(raw, np.dtype(dtype).itemsize)
        array = np.frombuffer(raw, dtype=dtype, count=numel)
        offset = int(item["offset"])
        shape = tuple(int(v) for v in item["size"])
        stride = tuple(int(v) for v in item["stride"])
        view = np.lib.stride_tricks.as_strided(
            array[offset:],
            shape=shape,
            strides=tuple(step * array.dtype.itemsize for step in stride),
        )
        if dtype == np.float32:
            return np.array(view, dtype=np.float32, copy=True)
        return np.array(view, dtype=np.float32, copy=True)


def load_state(path: Path) -> Archive:
    return Archive(path)


def _prefix(archive: zipfile.ZipFile) -> str:
    for name in archive.namelist():
        if name.endswith("/data.pkl"):
            return name[: -len("/data.pkl")]
    message = "checkpoint has no data.pkl"
    raise RuntimeError(message)


def _swap(raw: bytes, width: int) -> bytes:
    if width <= 1:
        return raw
    out = bytearray(len(raw))
    for index in range(0, len(raw), width):
        out[index : index + width] = raw[index : index + width][::-1]
    return bytes(out)


def _unpickle(raw: bytes) -> tuple[Any, list[tuple]]:
    ids: list[tuple] = []

    class _Loader(pickle.Unpickler):
        def persistent_load(self, pid: tuple) -> tuple:
            ids.append(pid)
            return ("storage", len(ids) - 1)

        def find_class(self, module: str, name: str) -> Any:
            if module == "torch._utils" and name == "_rebuild_tensor_v2":
                return _rebuild
            if name.endswith("Storage"):
                kind = type(name, (), {})
                setattr(_module(module), name, kind)
                return kind
            return super().find_class(module, name)

    def _rebuild(
        storage: tuple,
        offset: int,
        size: tuple,
        stride: tuple,
        requires_grad: bool = False,
        backward_hooks: Any = None,
        metadata: Any = None,
    ) -> dict[str, Any]:
        del requires_grad, backward_hooks, metadata
        return {"storage": storage, "offset": offset, "size": size, "stride": stride}

    meta = _Loader(io.BytesIO(raw)).load()
    return meta, ids
