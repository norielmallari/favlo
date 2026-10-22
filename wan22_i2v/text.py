"""umT5 encoder for the DiT text context.

The encoder is bidirectional. Each block is RMSNorm, attention with a relative
bucket bias, a residual, RMSNorm, then the gated GELU feed-forward. There is
no dropout and no linear bias. The final RMSNorm is the context the DiT reads.
Padding is not inside this pass: only the real tokens, including the end
token, are returned.
"""

from __future__ import annotations

import io
import math
import pickle
import time
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from wan22_i2v.source import source_dir

_LAYERS = 24
_HEADS = 64
_BUCKETS = 32
_MAX_DISTANCE = 128
_EPS = np.float32(1e-6)
_PREFIX = "umt5-xxl-enc-bf16/data"
_PICKLE = "umt5-xxl-enc-bf16/data.pkl"


def tokenizer_path(root: Path | None = None) -> Path:
    return source_dir(root) / "tokenizer.json"


def encoder_path(root: Path | None = None) -> Path:
    return source_dir(root) / "models_t5_umt5-xxl-enc-bf16.pth"


def relative_buckets(query: int, key: int) -> np.ndarray:
    """T5 bidirectional buckets. Rows are queries and columns are keys."""
    positions = np.arange(key, dtype=np.int64)[None, :] - np.arange(query, dtype=np.int64)[:, None]
    half = _BUCKETS // 2
    buckets = (positions > 0).astype(np.int64) * half
    distance = np.abs(positions)
    exact = half // 2
    small = distance < exact
    span = math.log(_MAX_DISTANCE / exact)
    safe = np.maximum(distance, 1)
    scaled = np.log(safe.astype(np.float64) / exact) / span * (half - exact)
    large = exact + scaled.astype(np.int64)
    large = np.minimum(large, half - 1)
    buckets += np.where(small, distance, large)
    return buckets.astype(np.intp)


def encode_prompt(
    prompt: str, weights: Path | None = None, tokenizer: Path | None = None
) -> np.ndarray:
    """Token vectors ``[tokens, 4096]`` for one prompt."""
    vocab = Tokenizer.from_file(str(tokenizer or tokenizer_path()))
    ids = vocab.encode(prompt).ids
    if not ids:
        return np.zeros((0, 4096), dtype=np.float32)
    started = time.perf_counter()
    vectors = _Encoder(weights or encoder_path()).run(np.asarray(ids, dtype=np.int64))
    elapsed = time.perf_counter() - started
    print(f"umt5 {len(ids)} tokens {elapsed:.2f}s", flush=True)
    return vectors


class _Encoder:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._zip: zipfile.ZipFile | None = None
        self._meta: dict[str, tuple] = {}

    def run(self, ids: np.ndarray) -> np.ndarray:
        archive = zipfile.ZipFile(self.path)
        self._zip = archive
        try:
            self._meta = _metadata(archive)
            hidden = self._rows(ids)
            buckets = relative_buckets(len(ids), len(ids))
            for layer in range(_LAYERS):
                hidden = self._block(hidden, layer, buckets)
            return _rms(hidden, self._tensor("norm.weight"))
        finally:
            archive.close()
            self._zip = None

    def _block(self, hidden: np.ndarray, layer: int, buckets: np.ndarray) -> np.ndarray:
        prefix = f"blocks.{layer}"
        normed = _rms(hidden, self._tensor(f"{prefix}.norm1.weight"))
        query = normed @ self._tensor(f"{prefix}.attn.q.weight").T
        key = normed @ self._tensor(f"{prefix}.attn.k.weight").T
        value = normed @ self._tensor(f"{prefix}.attn.v.weight").T
        table = self._tensor(f"{prefix}.pos_embedding.embedding.weight")
        hidden = hidden + _attend(query, key, value, table[buckets])
        normed = _rms(hidden, self._tensor(f"{prefix}.norm2.weight"))
        gate = _gelu(normed @ self._tensor(f"{prefix}.ffn.gate.0.weight").T)
        inner = normed @ self._tensor(f"{prefix}.ffn.fc1.weight").T
        return hidden + (gate * inner) @ self._tensor(f"{prefix}.ffn.fc2.weight").T

    def _tensor(self, name: str) -> np.ndarray:
        archive = self._zip
        if archive is None:
            message = "umT5 archive is closed"
            raise RuntimeError(message)
        key, offset, shape, stride = self._meta[name]
        raw = archive.read(f"{_PREFIX}/{key}")
        storage = np.frombuffer(raw, dtype="<u2")
        return _float(storage, offset, shape, stride)

    def _rows(self, ids: np.ndarray) -> np.ndarray:
        archive = self._zip
        if archive is None:
            message = "umT5 archive is closed"
            raise RuntimeError(message)
        _key, offset, shape, stride = self._meta["token_embedding.weight"]
        if tuple(stride) != (shape[1], 1):
            message = "token embedding is not row major"
            raise RuntimeError(message)
        width = int(shape[1])
        out = np.empty((len(ids), width), dtype=np.float32)
        member = archive.open(f"{_PREFIX}/{_key}")
        try:
            for index, token in enumerate(ids):
                member.seek((int(offset) + int(token) * width) * 2)
                raw = member.read(width * 2)
                bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << np.uint32(16)
                out[index] = bits.view(np.float32)
        finally:
            member.close()
        return out


def _metadata(archive: zipfile.ZipFile) -> dict[str, tuple]:
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


def _float(
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


def _rms(hidden: np.ndarray, weight: np.ndarray) -> np.ndarray:
    variance = np.mean(hidden * hidden, axis=-1, keepdims=True)
    return hidden * np.reciprocal(np.sqrt(variance + _EPS)) * weight


def _gelu(hidden: np.ndarray) -> np.ndarray:
    cube = hidden * hidden * hidden
    inner = np.float32(0.7978845608) * (hidden + np.float32(0.044715) * cube)
    return np.float32(0.5) * hidden * (np.float32(1.0) + np.tanh(inner))


def _attend(
    query: np.ndarray, key: np.ndarray, value: np.ndarray, bias: np.ndarray
) -> np.ndarray:
    length, width = query.shape
    dim = width // _HEADS
    query_h = query.reshape(length, _HEADS, dim).transpose(1, 0, 2)
    key_h = key.reshape(length, _HEADS, dim).transpose(1, 0, 2)
    value_h = value.reshape(length, _HEADS, dim).transpose(1, 0, 2)
    scores = np.matmul(query_h, key_h.transpose(0, 2, 1))
    scores = scores + bias.transpose(2, 0, 1)
    scores -= np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= np.sum(weights, axis=-1, keepdims=True)
    mixed = np.matmul(weights, value_h)
    return np.ascontiguousarray(mixed.transpose(1, 0, 2).reshape(length, width))
