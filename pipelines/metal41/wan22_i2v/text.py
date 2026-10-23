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
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer

from wan22_i2v.formats import TILE_M, pad_to
from wan22_i2v.metal import Gpu
from wan22_i2v.source import ENCODER_FILE, TOKENIZER_FILE, umt5_dir

_LAYERS = 24
_HEADS = 64
_WIDTH = 4096
_FFN = 10240
_BUCKETS = 32
_MAX_DISTANCE = 128
_EPS = np.float32(1e-6)
_PREFIX = "umt5-xxl-enc-bf16/data"
_PICKLE = "umt5-xxl-enc-bf16/data.pkl"


def tokenizer_path(root: Path | None = None, repo: str | None = None) -> Path:
    return umt5_dir(root, repo) / TOKENIZER_FILE


def encoder_path(root: Path | None = None, repo: str | None = None) -> Path:
    return umt5_dir(root, repo) / ENCODER_FILE


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
    prompt: str,
    weights: Path | None = None,
    tokenizer: Path | None = None,
    on_layer: Callable[[], None] | None = None,
    gpu: Gpu | None = None,
) -> np.ndarray:
    """Token vectors ``[tokens, 4096]`` for one prompt."""
    vocab = Tokenizer.from_file(str(tokenizer or tokenizer_path()))
    ids = vocab.encode(prompt).ids
    if not ids:
        for _ in range(_LAYERS):
            if on_layer is not None:
                on_layer()
        return np.zeros((0, 4096), dtype=np.float32)
    started = time.perf_counter()
    own = gpu is None
    device = Gpu() if own else gpu
    try:
        vectors = _Encoder(weights or encoder_path()).run(
            np.asarray(ids, dtype=np.int64), on_layer, device
        )
    finally:
        if own:
            device.close()
    elapsed = time.perf_counter() - started
    print(f"umt5 {len(ids)} tokens {elapsed:.2f}s", flush=True)
    return vectors


class _Encoder:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._zip: zipfile.ZipFile | None = None
        self._meta: dict[str, tuple] = {}

    def run(
        self, ids: np.ndarray, on_layer: Callable[[], None] | None, gpu: Gpu
    ) -> np.ndarray:
        archive = zipfile.ZipFile(self.path)
        self._zip = archive
        rows = int(ids.shape[0])
        hidden: Any = None
        buckets: Any = None
        try:
            self._meta = _metadata(archive)
            hidden = gpu.upload(self._rows(ids))
            buckets = gpu.upload(relative_buckets(rows, rows).astype(np.int32))
            for layer in range(_LAYERS):
                hidden = self._block(gpu, hidden, rows, layer, buckets)
                if on_layer is not None:
                    on_layer()
            normed = self._norm(gpu, hidden, rows, self._tensor("norm.weight"))
            gpu.drop(hidden)
            hidden = None
            vectors = gpu.read_f32(normed, rows * _WIDTH).reshape(rows, _WIDTH)
            gpu.drop(normed)
            return vectors
        finally:
            if hidden is not None:
                gpu.drop(hidden)
            if buckets is not None:
                gpu.drop(buckets)
            archive.close()
            self._zip = None

    def _block(self, gpu: Gpu, hidden: Any, rows: int, layer: int, buckets: Any) -> Any:
        prefix = f"blocks.{layer}"
        normed = self._norm(gpu, hidden, rows, self._tensor(f"{prefix}.norm1.weight"))
        query = self._linear(gpu, normed, rows, self._tensor(f"{prefix}.attn.q.weight"))
        key = self._linear(gpu, normed, rows, self._tensor(f"{prefix}.attn.k.weight"))
        value = self._linear(gpu, normed, rows, self._tensor(f"{prefix}.attn.v.weight"))
        gpu.drop(normed)
        table = gpu.upload(self._tensor(f"{prefix}.pos_embedding.embedding.weight"))
        attn = gpu.alloc_f32(rows * _WIDTH)
        gpu.attend_bias(
            query, key, value, attn, buckets, table, rows, rows, _HEADS, _WIDTH // _HEADS
        )
        gpu.end()
        gpu.drop_many([query, key, value, table])
        gpu.add_f32(hidden, attn, rows * _WIDTH)
        gpu.end()
        gpu.drop(attn)
        normed = self._norm(gpu, hidden, rows, self._tensor(f"{prefix}.norm2.weight"))
        gate = self._linear(gpu, normed, rows, self._tensor(f"{prefix}.ffn.gate.0.weight"))
        gpu.pointwise("gelu_tanh", gate, rows, _FFN, _FFN)
        inner = self._linear(gpu, normed, rows, self._tensor(f"{prefix}.ffn.fc1.weight"))
        gpu.drop(normed)
        gpu.mul_f32(gate, inner, rows * _FFN)
        gpu.end()
        gpu.drop(inner)
        out = self._linear(gpu, gate, rows, self._tensor(f"{prefix}.ffn.fc2.weight"))
        gpu.drop(gate)
        gpu.add_f32(hidden, out, rows * _WIDTH)
        gpu.end()
        gpu.drop(out)
        return hidden

    def _linear(self, gpu: Gpu, source: Any, rows: int, weight: np.ndarray) -> Any:
        out, inner = int(weight.shape[0]), int(weight.shape[1])
        padded = pad_to(rows, TILE_M)
        half = gpu.empty(padded * inner * 2)
        gpu.cast_half(source, half, rows * inner)
        packed = gpu.upload(np.ascontiguousarray(weight, dtype=np.float16))
        dest = gpu.alloc_f32(padded * out)
        gpu.matmul_f16(half, packed, dest, padded, out, inner)
        gpu.end()
        gpu.drop_many([half, packed])
        return dest

    def _norm(self, gpu: Gpu, source: Any, rows: int, weight: np.ndarray) -> Any:
        width = int(weight.shape[0])
        scale = gpu.upload(np.ascontiguousarray(weight, dtype=np.float16))
        dest = gpu.alloc_f32(rows * width)
        gpu.rmsnorm(source, scale, dest, rows, width, float(_EPS))
        gpu.end()
        gpu.drop(scale)
        return dest

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

