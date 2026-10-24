"""Read a PyTorch zip checkpoint without importing torch."""

from __future__ import annotations

import io
import pickle
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_DTYPE_BYTES = {
    "FloatStorage": (4, "<f4"),
    "HalfStorage": (2, "<f2"),
    "BFloat16Storage": (2, "bf16"),
    "DoubleStorage": (8, "<f8"),
    "LongStorage": (8, "<i8"),
    "IntStorage": (4, "<i4"),
    "ShortStorage": (2, "<i2"),
    "ByteStorage": (1, "u1"),
    "BoolStorage": (1, "b1"),
}


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple[int, ...]
    dtype: str
    key: str
    offset: int
    numel: int


class _Storage:
    def __init__(self, dtype: str, key: str, numel: int) -> None:
        self.dtype = dtype
        self.key = key
        self.numel = numel


class _View:
    def __init__(self, storage: _Storage, offset: int, shape: tuple[int, ...]) -> None:
        self.storage = storage
        self.offset = offset
        self.shape = shape


def _dtype_name(storage_type: object) -> str:
    name = getattr(storage_type, "__name__", None)
    if isinstance(name, str) and name:
        return name
    text = str(storage_type)
    for known in _DTYPE_BYTES:
        if known in text:
            return known
    return text


def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata=None):
    del stride, requires_grad, backward_hooks, metadata
    if not isinstance(storage, _Storage):
        message = "tensor storage was not a persistent id"
        raise TypeError(message)
    return _View(storage, int(storage_offset), tuple(int(item) for item in size))


def _rebuild_tensor(storage, storage_offset, size, stride):
    return _rebuild_tensor_v2(storage, storage_offset, size, stride, False, None)


class _Stub:
    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs

    def __setstate__(self, state) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.state = state


class _Unpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2
        if name == "_rebuild_tensor":
            return _rebuild_tensor
        if name in {"Size", "dtype"} or module.startswith("torch"):
            return type(name, (_Stub,), {})
        if module.startswith(("collections", "numpy", "builtins")):
            return super().find_class(module, name)
        return type(name, (_Stub,), {})

    def persistent_load(self, saved_id):
        if not isinstance(saved_id, tuple) or not saved_id:
            message = "checkpoint persistent id is not a storage tuple"
            raise pickle.UnpicklingError(message)
        kind = saved_id[0]
        kind_name = kind.decode("ascii") if isinstance(kind, bytes) else kind
        if kind_name != "storage":
            message = f"checkpoint persistent id is {kind_name}"
            raise pickle.UnpicklingError(message)
        if len(saved_id) < 5:
            message = "checkpoint storage id is short"
            raise pickle.UnpicklingError(message)
        _storage_type, key, _location, numel = saved_id[1], saved_id[2], saved_id[3], saved_id[4]
        key_text = key.decode("ascii") if isinstance(key, bytes) else str(key)
        return _Storage(_dtype_name(_storage_type), key_text, int(numel))


def _walk(value, prefix: str, found: list[Tensor]) -> None:
    if isinstance(value, _View):
        found.append(
            Tensor(
                name=prefix,
                shape=value.shape,
                dtype=value.storage.dtype,
                key=value.storage.key,
                offset=value.offset,
                numel=value.storage.numel,
            )
        )
        return
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            _walk(item, child, found)
        return
    state = getattr(value, "__dict__", None)
    if isinstance(state, dict):
        for key, item in state.items():
            if key.startswith("_"):
                continue
            child = f"{prefix}.{key}" if prefix else str(key)
            _walk(item, child, found)


def _member(archive: zipfile.ZipFile, key: str) -> str:
    suffix = f"data/{key}"
    for name in archive.namelist():
        if name == suffix or name.endswith("/" + suffix):
            return name
    message = f"checkpoint is missing storage {key}"
    raise KeyError(message)


def list_tensors(path: Path) -> list[Tensor]:
    """Return every tensor view stored in a ``.pth`` zip. Weights stay on disk."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        pickle_name = next(
            (name for name in names if name == "data.pkl" or name.endswith("/data.pkl")),
            None,
        )
        if pickle_name is None:
            message = "checkpoint zip has no data.pkl"
            raise ValueError(message)
        payload = archive.read(pickle_name)
    root = _Unpickler(io.BytesIO(payload)).load()
    found: list[Tensor] = []
    _walk(root, "", found)
    return found


def read_tensor(path: Path, tensor: Tensor) -> np.ndarray:
    """Load one tensor as float32 or int64. bfloat16 is widened on the host."""
    width, code = _DTYPE_BYTES[tensor.dtype]
    with zipfile.ZipFile(path) as archive:
        member = _member(archive, tensor.key)
        with archive.open(member) as handle:
            handle.seek(tensor.offset * width)
            raw = handle.read(tensor.numel * width)
    if code == "bf16":
        bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        values = bits.view(np.float32)
    else:
        values = np.frombuffer(raw, dtype=code)
    return np.array(values, copy=True).reshape(tensor.shape)


_ID_MARKERS = ("token", "vocab", "special", "eoa", "text_id")


def id_tables(tensors: list[Tensor]) -> list[Tensor]:
    """Named integer tables. A float embedding is not the unpublished text-id list."""
    found = []
    for tensor in tensors:
        if tensor.dtype not in {"LongStorage", "IntStorage", "ShortStorage", "ByteStorage"}:
            continue
        lowered = tensor.name.lower()
        if any(marker in lowered for marker in _ID_MARKERS):
            found.append(tensor)
    return found
