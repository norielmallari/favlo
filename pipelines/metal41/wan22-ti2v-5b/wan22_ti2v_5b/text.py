"""umT5-XXL encoder. The packed record is the one I2V already stored.

Twenty-four blocks, width 4096, 64 heads of 64. The feed-forward is
gated GELU. Positions past the real tokens, including the end-of-sequence
id, stay zero.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from wan22_ti2v_5b.activations import gelu_tanh, merge_heads, rmsnorm, split_heads
from wan22_ti2v_5b.gpu import Gpu
from wan22_ti2v_5b.schedule import relative_bucket
from wan22_ti2v_5b.spec import TEXT_DIM, TEXT_LEN
from wan22_ti2v_5b.weights import apply, attend, load_matrix

_HEADS = 64
_LAYERS = 24
_BUCKETS = 32


class TextEncoder:
    def __init__(self, root: Path, gpu: Gpu, tokenizer: Path) -> None:
        self.root = root
        self.gpu = gpu
        self._embed = np.load(root / "embed.npy", mmap_mode="r")
        self._norm = np.load(root / "norm.npy")
        self._tokenizer = Tokenizer.from_file(str(tokenizer))
        buckets = _buckets(TEXT_LEN)
        self._layers = [_layer(root, buckets, layer) for layer in range(_LAYERS)]

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
    def __init__(self, root: Path, buckets: np.ndarray, layer: int) -> None:
        stem = f"blocks.{layer}"
        self.norm1 = _vector(root / f"{stem}.norm1.npy")
        self.norm2 = _vector(root / f"{stem}.norm2.npy")
        self.q = load_matrix(root / f"{stem}.attn.q.weight.npz")
        self.k = load_matrix(root / f"{stem}.attn.k.weight.npz")
        self.v = load_matrix(root / f"{stem}.attn.v.weight.npz")
        self.o = load_matrix(root / f"{stem}.attn.o.weight.npz")
        self.gate = load_matrix(root / f"{stem}.ffn.gate.0.weight.npz")
        self.up = load_matrix(root / f"{stem}.ffn.fc1.weight.npz")
        self.down = load_matrix(root / f"{stem}.ffn.fc2.weight.npz")
        self.bias = _position_bias(_vector_2d(root / f"{stem}.pos.npy"), buckets)


def _layer(root: Path, buckets: np.ndarray, layer: int) -> _Layer:
    return _Layer(root, buckets, layer)


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
