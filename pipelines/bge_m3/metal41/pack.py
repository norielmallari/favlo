"""Symmetric int8 pack for one projection. The runtime does not import this."""

from __future__ import annotations

import numpy as np

TILE_M = 64
TILE_N = 128
K_ALIGN = 256


def pack_int8(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-output-channel codes and scales. ``weight`` is ``[N, K]`` float32.

    The scale is the row peak divided by 127. A zero row stores scale 0.
    Codes are round-half-to-even and clipped to ``-127..127``.
    """
    matrix = np.ascontiguousarray(weight, dtype=np.float32)
    if matrix.ndim != 2:
        message = f"matrix must be rank 2, got {matrix.shape}"
        raise ValueError(message)
    rows, cols = matrix.shape
    if rows % TILE_N != 0 or cols % K_ALIGN != 0:
        message = f"matrix {rows}x{cols} is not aligned to {TILE_N} by {K_ALIGN}"
        raise ValueError(message)
    peak = np.max(np.abs(matrix), axis=1)
    scale = np.zeros(rows, dtype=np.float32)
    inverse = np.zeros(rows, dtype=np.float32)
    live = peak > 0
    scale[live] = peak[live] / np.float32(127.0)
    inverse[live] = np.float32(127.0) / peak[live]
    codes = np.rint(matrix * inverse[:, None])
    codes = np.clip(codes, -127, 127).astype(np.int8)
    return np.ascontiguousarray(codes), np.ascontiguousarray(scale)


def restore_int8(codes: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Float32 matrix. Each row is ``code * scale``."""
    values = np.asarray(codes, dtype=np.float32)
    factors = np.asarray(scale, dtype=np.float32)
    return np.ascontiguousarray(values * factors[:, None])
