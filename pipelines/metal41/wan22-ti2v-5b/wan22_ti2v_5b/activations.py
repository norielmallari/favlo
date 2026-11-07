"""Norms, GELU, and RoPE. Matmuls stay in ``Gpu``."""

from __future__ import annotations

import math

import numpy as np

from wan22_ti2v_5b.schedule import rope_tables
from wan22_ti2v_5b.spec import EPS, HEAD, HEADS

_GELU = np.float32(0.7978845608)
_CUBE = np.float32(0.044715)


def rmsnorm(rows: np.ndarray, weight: np.ndarray | None, eps: float) -> np.ndarray:
    data = rows.astype(np.float32, copy=False)
    inv = np.reciprocal(np.sqrt(np.mean(data * data, axis=-1, keepdims=True) + np.float32(eps)))
    normed = data * inv
    if weight is not None:
        normed = normed * weight.astype(np.float32).reshape(1, -1)
    return normed


def layernorm(
    rows: np.ndarray,
    weight: np.ndarray | None = None,
    bias: np.ndarray | None = None,
    eps: float = EPS,
) -> np.ndarray:
    data = rows.astype(np.float32, copy=False)
    mean = np.mean(data, axis=-1, keepdims=True)
    var = np.mean((data - mean) ** 2, axis=-1, keepdims=True)
    normed = (data - mean) * np.reciprocal(np.sqrt(var + np.float32(eps)))
    if weight is not None:
        normed = normed * weight.astype(np.float32).reshape(1, -1)
    if bias is not None:
        normed = normed + bias.astype(np.float32).reshape(1, -1)
    return normed


def gelu_tanh(values: np.ndarray) -> np.ndarray:
    data = values.astype(np.float32, copy=False)
    inner = _GELU * (data + _CUBE * data * data * data)
    return np.float32(0.5) * data * (1.0 + np.tanh(inner))


def gelu_erf(values: np.ndarray) -> np.ndarray:
    data = values.astype(np.float64, copy=False)
    scaled = data / math.sqrt(2.0)
    applied = np.vectorize(math.erf, otypes=[np.float64])(scaled)
    return (0.5 * data * (1.0 + applied)).astype(np.float32)


def silu(values: np.ndarray) -> np.ndarray:
    data = values.astype(np.float32, copy=False)
    return data / (1.0 + np.exp(-np.clip(data, -80.0, 80.0)))


def softmax(scores: np.ndarray) -> np.ndarray:
    peak = np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores - peak)
    return weights / np.sum(weights, axis=-1, keepdims=True)


def split_heads(rows: np.ndarray, heads: int) -> np.ndarray:
    tokens, width = rows.shape
    return np.transpose(rows.reshape(tokens, heads, width // heads), (1, 0, 2))


def merge_heads(heads: np.ndarray) -> np.ndarray:
    return np.transpose(heads, (1, 0, 2)).reshape(heads.shape[1], -1)


def qk_norm(rows: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """RMS over the full width, or over each head when the weight is head-sized."""
    scale = weight.astype(np.float32).reshape(-1)
    if scale.shape[0] == rows.shape[1]:
        return rmsnorm(rows, scale, EPS)
    if scale.shape[0] != HEAD:
        message = f"qk norm width {scale.shape[0]} is not {rows.shape[1]} or {HEAD}"
        raise RuntimeError(message)
    tokens = rows.shape[0]
    grouped = rows.reshape(tokens, HEADS, HEAD)
    normed = rmsnorm(grouped.reshape(tokens * HEADS, HEAD), scale, EPS)
    return normed.reshape(tokens, HEADS * HEAD)


def apply_rope(rows: np.ndarray, height: int, width: int) -> np.ndarray:
    """Rotate pairs 0–21 in time, 22–42 in height, and 43–63 in width."""
    tokens, _heads, _dim = rows.shape[0], rows.shape[1] // HEAD, HEAD
    data = rows.reshape(tokens, _heads, 64, 2).astype(np.float32, copy=True)
    spatial = height * width
    if tokens % spatial != 0:
        message = f"{tokens} tokens do not fill a {height}x{width} grid"
        raise RuntimeError(message)
    index = np.arange(tokens)
    frame = index // spatial
    rem = index - frame * spatial
    y_pos = rem // width
    x_pos = rem - y_pos * width
    time_table, height_table, width_table = rope_tables()
    _rotate(data[:, :, 0:22], time_table[frame])
    _rotate(data[:, :, 22:43], height_table[y_pos])
    _rotate(data[:, :, 43:64], width_table[x_pos])
    return data.reshape(rows.shape)


def sinusoidal(position: float, width: int = 256) -> np.ndarray:
    half = width // 2
    freqs = np.exp(-np.log(10000.0) * np.arange(half, dtype=np.float64) / half)
    args = np.float64(position) * freqs
    wave = np.concatenate([np.cos(args), np.sin(args)])
    return wave.astype(np.float32)


def _rotate(pairs: np.ndarray, table: np.ndarray) -> None:
    count = pairs.shape[2]
    cosine = table[:, :count]
    sine = table[:, count:]
    # Copy first. The slices are views, and writing one half would change the other.
    even = np.array(pairs[:, :, :, 0], copy=True)
    odd = np.array(pairs[:, :, :, 1], copy=True)
    pairs[:, :, :, 0] = even * cosine[:, None, :] - odd * sine[:, None, :]
    pairs[:, :, :, 1] = even * sine[:, None, :] + odd * cosine[:, None, :]
