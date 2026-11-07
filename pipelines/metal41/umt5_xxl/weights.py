"""BF16 umT5 readers for the Wan zip and a Hugging Face safetensors file."""

from __future__ import annotations

import io
import json
import pickle
import struct
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Protocol

import numpy as np

FORMATS = ("wan", "hf")

_PREFIX = "umt5-xxl-enc-bf16/data"
_PICKLE = "umt5-xxl-enc-bf16/data.pkl"
class Weights(Protocol):
    def tensor(self, name: str) -> np.ndarray:
        """Float32 array for one Wan-layout tensor name."""

    def embed_shape(self) -> tuple[int, int]:
        """Rows and width of ``token_embedding.weight``."""

    def read_embed_rows(self, start: int, count: int) -> np.ndarray:
        """Float32 rows of the token embedding without loading the table."""

    def close(self) -> None:
        """Release file handles."""


def open_weights(kind: str, path: Path) -> Weights:
    """Open one BF16 encoder. ``kind`` is ``wan`` or ``hf``."""
    if kind == "wan":
        return WanZip(path)
    if kind == "hf":
        return HfSafetensors(path)
    message = f"encoder format {kind} is not wan or hf"
    raise ValueError(message)


def hf_key(name: str) -> str:
    """Map a Wan tensor name onto the Hugging Face umT5 key."""
    if name == "token_embedding.weight":
        return "shared.weight"
    if name == "norm.weight":
        return "encoder.final_layer_norm.weight"
    if name.startswith("blocks.") and name.count(".") >= 2:
        layer_text, rest = name.split(".", 2)[1:]
        try:
            layer = int(layer_text)
        except ValueError as exc:
            message = f"unknown umT5 tensor {name}"
            raise KeyError(message) from exc
        if rest == "norm1.weight":
            return f"encoder.block.{layer}.layer.0.layer_norm.weight"
        if rest == "norm2.weight":
            return f"encoder.block.{layer}.layer.1.layer_norm.weight"
        if rest == "pos_embedding.embedding.weight":
            return (
                f"encoder.block.{layer}.layer.0.SelfAttention.relative_attention_bias.weight"
            )
        if rest == "attn.q.weight":
            return f"encoder.block.{layer}.layer.0.SelfAttention.q.weight"
        if rest == "attn.k.weight":
            return f"encoder.block.{layer}.layer.0.SelfAttention.k.weight"
        if rest == "attn.v.weight":
            return f"encoder.block.{layer}.layer.0.SelfAttention.v.weight"
        if rest == "attn.o.weight":
            return f"encoder.block.{layer}.layer.0.SelfAttention.o.weight"
        if rest == "ffn.gate.0.weight":
            return f"encoder.block.{layer}.layer.1.DenseReluDense.wi_0.weight"
        if rest == "ffn.fc1.weight":
            return f"encoder.block.{layer}.layer.1.DenseReluDense.wi_1.weight"
        if rest == "ffn.fc2.weight":
            return f"encoder.block.{layer}.layer.1.DenseReluDense.wo.weight"
    message = f"unknown umT5 tensor {name}"
    raise KeyError(message)


class WanZip:
    """Wan ``models_t5_umt5-xxl-enc-bf16.pth`` zip with a pickled tensor map."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._archive = zipfile.ZipFile(self.path)
        self._meta = _wan_metadata(self._archive)

    def tensor(self, name: str) -> np.ndarray:
        key, offset, shape, stride = self._meta[name]
        raw = self._archive.read(f"{_PREFIX}/{key}")
        storage = np.frombuffer(raw, dtype="<u2")
        return np.ascontiguousarray(_bf16_view(storage, offset, shape, stride), dtype=np.float32)

    def read_embed_rows(self, start: int, count: int) -> np.ndarray:
        """Float32 rows of ``token_embedding.weight`` without loading the table."""
        key, offset, shape, stride = self._meta["token_embedding.weight"]
        if tuple(stride) != (shape[1], 1):
            message = "token embedding is not row major"
            raise RuntimeError(message)
        rows, width = int(shape[0]), int(shape[1])
        if start < 0 or count < 0 or start + count > rows:
            message = f"token embedding rows {start}:{start + count} outside {rows}"
            raise ValueError(message)
        member = self._archive.open(f"{_PREFIX}/{key}")
        try:
            member.seek((int(offset) + start * width) * 2)
            raw = np.frombuffer(member.read(count * width * 2), dtype="<u2").copy()
        finally:
            member.close()
        bits = raw.astype(np.uint32) << np.uint32(16)
        return bits.view(np.float32).reshape(count, width)

    def embed_shape(self) -> tuple[int, int]:
        _key, _offset, shape, stride = self._meta["token_embedding.weight"]
        if tuple(stride) != (shape[1], 1):
            message = "token embedding is not row major"
            raise RuntimeError(message)
        return int(shape[0]), int(shape[1])

    def close(self) -> None:
        self._archive.close()


class HfSafetensors:
    """Hugging Face umT5 encoder saved as one ``.safetensors`` file."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        header, data = _safetensors_header(self.path)
        self._header = header
        self._data = data

    def tensor(self, name: str) -> np.ndarray:
        key = hf_key(name)
        return self._load(key)

    def read_embed_rows(self, start: int, count: int) -> np.ndarray:
        key = hf_key("token_embedding.weight")
        dtype, shape, begin, end = self._entry(key)
        rows, width = int(shape[0]), int(shape[1])
        if start < 0 or count < 0 or start + count > rows:
            message = f"token embedding rows {start}:{start + count} outside {rows}"
            raise ValueError(message)
        width_bytes = _dtype_width(dtype)
        origin = begin + start * width * width_bytes
        take = count * width * width_bytes
        with self.path.open("rb") as handle:
            handle.seek(self._data + origin)
            raw = handle.read(take)
        if len(raw) != take:
            message = f"{key} ended before its declared bytes"
            raise ValueError(message)
        return _decode(raw, dtype, key).reshape(count, width)

    def embed_shape(self) -> tuple[int, int]:
        _dtype, shape, _begin, _end = self._entry(hf_key("token_embedding.weight"))
        if len(shape) != 2:
            message = "token embedding is not a matrix"
            raise RuntimeError(message)
        return int(shape[0]), int(shape[1])

    def close(self) -> None:
        return

    def _load(self, key: str) -> np.ndarray:
        dtype, shape, begin, end = self._entry(key)
        take = end - begin
        with self.path.open("rb") as handle:
            handle.seek(self._data + begin)
            raw = handle.read(take)
        if len(raw) != take:
            message = f"{key} ended before its declared bytes"
            raise ValueError(message)
        return _decode(raw, dtype, key).reshape(shape)

    def _entry(self, key: str) -> tuple[str, tuple[int, ...], int, int]:
        item = self._header.get(key)
        if not isinstance(item, dict):
            message = f"umT5 tensor {key} is missing"
            raise KeyError(message)
        dtype = item.get("dtype")
        shape = item.get("shape")
        offsets = item.get("data_offsets")
        if not isinstance(dtype, str):
            message = f"{key} dtype is missing"
            raise TypeError(message)
        if not isinstance(shape, list) or not all(isinstance(v, int) for v in shape):
            message = f"{key} shape is missing"
            raise TypeError(message)
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(v, int) for v in offsets)
        ):
            message = f"{key} data_offsets are missing"
            raise TypeError(message)
        return dtype, tuple(int(v) for v in shape), int(offsets[0]), int(offsets[1])


def _wan_metadata(archive: zipfile.ZipFile) -> dict[str, tuple]:
    class _Unpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> object:
            if name == "OrderedDict":
                return OrderedDict
            if name == "_rebuild_tensor_v2":
                return _rebuild
            if name.endswith("Storage"):
                return type(name, (), {})
            message = f"unexpected pickle type {module}.{name}"
            raise RuntimeError(message)

        def persistent_load(self, pid: tuple) -> tuple:
            return (str(pid[2]), int(pid[4]))

    def _rebuild(
        storage: tuple,
        offset: int,
        size: tuple[int, ...],
        stride: tuple[int, ...],
        requires_grad: bool,
        hooks: object,
        metadata: object = None,
    ) -> tuple:
        del requires_grad, hooks, metadata
        return (storage[0], int(offset), tuple(int(v) for v in size), tuple(int(v) for v in stride))

    loaded = _Unpickler(io.BytesIO(archive.read(_PICKLE))).load()
    if not isinstance(loaded, dict):
        message = "umT5 checkpoint is not a tensor map"
        raise RuntimeError(message)
    return {str(name): value for name, value in loaded.items()}


def _bf16_view(
    storage: np.ndarray, offset: int, shape: tuple[int, ...], stride: tuple[int, ...]
) -> np.ndarray:
    view = np.lib.stride_tricks.as_strided(
        storage[offset:],
        shape=shape,
        strides=tuple(step * 2 for step in stride),
    )
    copied = np.ascontiguousarray(view)
    bits = copied.astype(np.uint32) << np.uint32(16)
    return bits.view(np.float32)


def _safetensors_header(path: Path) -> tuple[dict[str, object], int]:
    with path.open("rb") as handle:
        count = struct.unpack("<Q", handle.read(8))[0]
        payload = handle.read(count)
    header = json.loads(payload)
    if not isinstance(header, dict):
        message = f"{path.name} safetensors header is not an object"
        raise TypeError(message)
    return header, 8 + count


def _dtype_width(dtype: str) -> int:
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


__all__ = [
    "FORMATS",
    "HfSafetensors",
    "WanZip",
    "Weights",
    "hf_key",
    "open_weights",
]
