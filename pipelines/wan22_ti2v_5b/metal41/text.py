"""umT5-XXL encoder. The packed record is the one I2V already stored.

Twenty-four blocks, width 4096, 64 heads of 64. The feed-forward is
gated GELU. Positions past the real tokens, including the end-of-sequence
id, stay zero.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from wan22_ti2v_5b.metal41.activations import gelu_tanh, merge_heads, rmsnorm, split_heads
from wan22_ti2v_5b.metal41.blob import copy_plane, load_planes, map_plane, resolve_named
from wan22_ti2v_5b.metal41.gpu import Gpu
from wan22_ti2v_5b.metal41.schedule import relative_bucket
from wan22_ti2v_5b.metal41.spec import TEXT_DIM, TEXT_LEN
from wan22_ti2v_5b.metal41.weights import apply, attend, load_matrix, matrix_from_planes

WEIGHTS = (
    "attn.q.weight",
    "attn.k.weight",
    "attn.v.weight",
    "attn.o.weight",
    "ffn.gate.0.weight",
    "ffn.fc1.weight",
    "ffn.fc2.weight",
)

_HEADS = 64
_LAYERS = 24
_BUCKETS = 32


class TextEncoder:
    def __init__(self, root: Path, gpu: Gpu, tokenizer: Path) -> None:
        self.root = root
        self.gpu = gpu
        self._kept: list[memoryview] = []
        self._span = None
        self._tokenizer = Tokenizer.from_file(str(tokenizer))
        buckets = _buckets(TEXT_LEN)
        blob = resolve_named(root, "weights.bin", "planes.json")
        if blob is None:
            self._embed = np.load(root / "embed.npy", mmap_mode="r")
            self._norm = np.load(root / "norm.npy")
            self._layers = [_layer(root, buckets, layer) for layer in range(_LAYERS)]
            return
        self._span = gpu.read_blob(blob)
        planes = load_planes(root / "planes.json")
        self._embed, embed_view = map_plane(self._span, _plane(planes, "embed"))
        self._kept.append(embed_view)
        self._norm = copy_plane(self._span, _plane(planes, "norm")).astype(np.float32).reshape(-1)
        self._layers = [
            _Layer(root, buckets, layer, planes=planes, span=self._span) for layer in range(_LAYERS)
        ]

    def close(self) -> None:
        self._kept.clear()
        self._span = None

    def encode(self, prompt: str) -> tuple[np.ndarray, np.ndarray]:
        if not prompt.strip():
            message = "prompt is empty"
            raise ValueError(message)
        ids = self._tokenizer.encode(prompt).ids
        if len(ids) > TEXT_LEN:
            message = f"prompt is longer than {TEXT_LEN} tokens"
            raise ValueError(message)
        hidden = np.zeros((TEXT_LEN, TEXT_DIM), dtype=np.float32)
        if ids:
            hidden[: len(ids)] = np.asarray(self._embed[ids], dtype=np.float32)
        mask = np.zeros(TEXT_LEN, dtype=bool)
        mask[: len(ids)] = True
        for layer in range(_LAYERS):
            hidden = self._block(hidden, mask, layer)
            hidden[~mask] = 0.0
        hidden = rmsnorm(hidden, self._norm, 1e-6)
        hidden[~mask] = 0.0
        return hidden, mask

    def _block(self, hidden: np.ndarray, mask: np.ndarray, layer: int) -> np.ndarray:
        block = self._layers[layer]
        normed = rmsnorm(hidden, block.norm1, 1e-6)
        query = apply(self.gpu, normed, block.q)
        key = apply(self.gpu, normed, block.k)
        value = apply(self.gpu, normed, block.v)
        mixed = attend(
            self.gpu,
            split_heads(query, _HEADS),
            split_heads(key, _HEADS),
            split_heads(value, _HEADS),
            bias=block.bias,
            key_mask=mask,
            scaled=False,
        )
        hidden = hidden + apply(self.gpu, merge_heads(mixed), block.o)
        hidden[~mask] = 0.0
        normed = rmsnorm(hidden, block.norm2, 1e-6)
        gate = gelu_tanh(apply(self.gpu, normed, block.gate))
        up = apply(self.gpu, normed, block.up)
        hidden = hidden + apply(self.gpu, gate * up, block.down)
        return hidden


class _Layer:
    def __init__(
        self,
        root: Path,
        buckets: np.ndarray,
        layer: int,
        planes: dict | None = None,
        span: object | None = None,
    ) -> None:
        stem = f"blocks.{layer}"
        if planes is None:
            self.norm1 = _vector(root / f"{stem}.norm1.npy")
            self.norm2 = _vector(root / f"{stem}.norm2.npy")
            for name, suffix in _FIELDS:
                setattr(self, name, load_matrix(root / f"{stem}.{suffix}.npz"))
            self.bias = _position_bias(_vector_2d(root / f"{stem}.pos.npy"), buckets)
            return
        self.norm1 = copy_plane(span, _plane(planes, f"{stem}.norm1")).astype(np.float32).reshape(-1)
        self.norm2 = copy_plane(span, _plane(planes, f"{stem}.norm2")).astype(np.float32).reshape(-1)
        for name, suffix in _FIELDS:
            setattr(
                self,
                name,
                matrix_from_planes(span, planes, f"{stem}.{suffix}", kind="int8"),
            )
        pos = copy_plane(span, _plane(planes, f"{stem}.pos")).astype(np.float32)
        self.bias = _position_bias(pos, buckets)


def _layer(root: Path, buckets: np.ndarray, layer: int) -> _Layer:
    return _Layer(root, buckets, layer)


def _plane(planes: dict, name: str) -> dict:
    found = planes.get(name)
    if found is None:
        message = f"umT5 pack is missing {name}"
        raise FileNotFoundError(message)
    return found


_FIELDS = (
    ("q", WEIGHTS[0]),
    ("k", WEIGHTS[1]),
    ("v", WEIGHTS[2]),
    ("o", WEIGHTS[3]),
    ("gate", WEIGHTS[4]),
    ("up", WEIGHTS[5]),
    ("down", WEIGHTS[6]),
)


def _vector(path: Path) -> np.ndarray:
    return np.load(path).astype(np.float32).reshape(-1)


def _vector_2d(path: Path) -> np.ndarray:
    return np.load(path).astype(np.float32)


def _buckets(length: int) -> np.ndarray:
    table = np.empty((length, length), dtype=np.int32)
    for query in range(length):
        for key in range(length):
            table[query, key] = relative_bucket(query, key, buckets=_BUCKETS)
    return table


def _position_bias(table: np.ndarray, buckets: np.ndarray) -> np.ndarray:
    """``table`` is ``[buckets, heads]``. The result is ``[heads, query, key]``."""
    return np.transpose(table[buckets], (2, 0, 1))
