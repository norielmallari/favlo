"""RoPE tables. Text is rotate-half. The DiT repeats each axis angle on a pair."""

from __future__ import annotations

import numpy as np

from .constants import DIT_AXES, DIT_THETA, TEXT_HEAD_DIM, TEXT_THETA


def text_table(length: int) -> tuple[np.ndarray, np.ndarray]:
    """Cosine and sine, shape ``[length, head]``, for Qwen3 rotate-half."""
    half = TEXT_HEAD_DIM // 2
    inv = 1.0 / (TEXT_THETA ** (np.arange(0, TEXT_HEAD_DIM, 2, dtype=np.float64) / TEXT_HEAD_DIM))
    pos = np.arange(length, dtype=np.float64)[:, None]
    freqs = pos * inv[None, :]
    emb = np.concatenate([freqs, freqs], axis=1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def apply_text(values: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """``values`` is ``[seq, heads, dim]``."""
    half = values.shape[-1] // 2
    first = values[..., :half]
    second = values[..., half:]
    rotated = np.concatenate([-second, first], axis=-1)
    return values * cos[:, None, :] + rotated * sin[:, None, :]


def dit_table(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cosine and sine for ``ids`` of shape ``[seq, 4]``."""
    if ids.ndim != 2 or ids.shape[1] != 4:
        message = f"DiT ids want [seq, 4], got {ids.shape}"
        raise ValueError(message)
    cos_parts: list[np.ndarray] = []
    sin_parts: list[np.ndarray] = []
    pos = ids.astype(np.float64)
    for axis, dim in enumerate(DIT_AXES):
        inv = 1.0 / (DIT_THETA ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
        freqs = pos[:, axis][:, None] * inv[None, :]
        cos_parts.append(np.repeat(np.cos(freqs), 2, axis=1))
        sin_parts.append(np.repeat(np.sin(freqs), 2, axis=1))
    cos = np.concatenate(cos_parts, axis=1).astype(np.float32)
    sin = np.concatenate(sin_parts, axis=1).astype(np.float32)
    return cos, sin


def apply_dit(values: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Adjacent-pair rotation. ``values`` is ``[seq, heads, dim]``."""
    first = values[..., 0::2]
    second = values[..., 1::2]
    pair_cos = cos[:, None, 0::2]
    pair_sin = sin[:, None, 0::2]
    out = np.empty_like(values)
    out[..., 0::2] = first * pair_cos - second * pair_sin
    out[..., 1::2] = first * pair_sin + second * pair_cos
    return out


def text_ids(length: int) -> np.ndarray:
    """``(0, 0, 0, token)`` for each position."""
    ids = np.zeros((length, 4), dtype=np.int64)
    ids[:, 3] = np.arange(length)
    return ids


def latent_ids(height: int, width: int, time_index: int) -> np.ndarray:
    """``(T, row, col, 0)``, row-major. ``time_index`` is 0 for the target."""
    count = height * width
    rows = np.repeat(np.arange(height, dtype=np.int64), width)
    cols = np.tile(np.arange(width, dtype=np.int64), height)
    ids = np.zeros((count, 4), dtype=np.int64)
    ids[:, 0] = time_index
    ids[:, 1] = rows
    ids[:, 2] = cols
    return ids
