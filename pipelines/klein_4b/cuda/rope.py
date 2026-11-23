"""RoPE tables. Text is rotate-half. The DiT is adjacent-pair on four axes."""

from __future__ import annotations

import numpy as np

from klein_4b.cuda.constants import DIT_AXES, DIT_THETA, ROPE_SCALE, TEXT_HEAD_DIM, TEXT_THETA


def _pair_table(positions: np.ndarray, width: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    half = width // 2
    freqs = 1.0 / (theta ** (np.arange(half, dtype=np.float64) / half))
    angles = positions.astype(np.float64)[:, None] * freqs[None, :]
    cos = np.repeat(np.cos(angles), 2, axis=1)
    sin = np.repeat(np.sin(angles), 2, axis=1)
    return cos.astype(np.float32), sin.astype(np.float32)


def text_table(length: int) -> tuple[np.ndarray, np.ndarray]:
    positions = np.arange(length, dtype=np.float64)
    half = TEXT_HEAD_DIM // 2
    freqs = 1.0 / (TEXT_THETA ** (np.arange(half, dtype=np.float64) / half))
    angles = positions[:, None] * freqs[None, :]
    cos = np.cos(angles).astype(np.float32)
    sin = np.sin(angles).astype(np.float32)
    return np.concatenate([cos, cos], axis=1), np.concatenate([sin, sin], axis=1)


def apply_text(values: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """``values`` is ``[seq, dim]``. Rotate the two halves."""
    half = values.shape[-1] // 2
    first = values[..., :half]
    second = values[..., half:]
    out = np.empty_like(values)
    out[..., :half] = first * cos[..., :half] - second * sin[..., :half]
    out[..., half:] = second * cos[..., half:] + first * sin[..., half:]
    return out


def _axes(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    parts_cos = []
    parts_sin = []
    for axis, width in enumerate(DIT_AXES):
        cos, sin = _pair_table(ids[:, axis], width, DIT_THETA)
        parts_cos.append(cos)
        parts_sin.append(sin)
    return np.concatenate(parts_cos, axis=1), np.concatenate(parts_sin, axis=1)


def text_ids(length: int) -> np.ndarray:
    ids = np.zeros((length, 4), dtype=np.float64)
    ids[:, 3] = np.arange(length, dtype=np.float64)
    return ids


def latent_ids(grid_h: int, grid_w: int, time: float) -> np.ndarray:
    rows = np.repeat(np.arange(grid_h, dtype=np.float64), grid_w)
    cols = np.tile(np.arange(grid_w, dtype=np.float64), grid_h)
    ids = np.zeros((grid_h * grid_w, 4), dtype=np.float64)
    ids[:, 0] = time
    ids[:, 1] = rows
    ids[:, 2] = cols
    return ids


def image_ids(grids: list[tuple[int, int]]) -> np.ndarray:
    """References first, then the target. Reference ``i`` uses ``T = 10 + 10 i``."""
    pieces = []
    last = len(grids) - 1
    for index, (grid_h, grid_w) in enumerate(grids):
        time = 0.0 if index == last else ROPE_SCALE + ROPE_SCALE * index
        pieces.append(latent_ids(grid_h, grid_w, time))
    return np.concatenate(pieces, axis=0)


def dit_table(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return _axes(ids)


def apply_dit(values: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Adjacent-pair rotation. ``cos[..., 2i]`` is the pair cosine."""
    first = values[..., 0::2]
    second = values[..., 1::2]
    pair_cos = cos[..., 0::2]
    pair_sin = sin[..., 0::2]
    out = np.empty_like(values)
    out[..., 0::2] = first * pair_cos - second * pair_sin
    out[..., 1::2] = first * pair_sin + second * pair_cos
    return out
