"""Symmetric int8. One float32 scale per output row. Codes stay in -127..127."""

from __future__ import annotations

import numpy as np

from .constants import GROUP


def pad_group(value: int) -> int:
    return ((int(value) + GROUP - 1) // GROUP) * GROUP


def pad_rows(value: int) -> int:
    from .constants import TILE_M

    return ((int(value) + TILE_M - 1) // TILE_M) * TILE_M


def pack_matrix(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pack ``[N, K]`` float. Extra rows and columns are zero codes and scale 0."""
    if weight.ndim != 2:
        message = f"matrix pack wants rank 2, got {weight.shape}"
        raise ValueError(message)
    rows, cols = weight.shape
    stored_n = pad_group(rows)
    stored_k = pad_group(cols)
    codes = np.zeros((stored_n, stored_k), dtype=np.int8)
    scales = np.zeros(stored_n, dtype=np.float32)
    if rows == 0 or cols == 0:
        return codes, scales
    values = np.ascontiguousarray(weight, dtype=np.float32)
    peak = np.max(np.abs(values), axis=1)
    live = peak > 0
    scales[:rows] = np.where(live, peak / np.float32(127), np.float32(0))
    inverse = np.zeros(rows, dtype=np.float32)
    inverse[live] = np.float32(127) / peak[live]
    quantized = np.rint(values * inverse[:, None]).astype(np.int32)
    codes[:rows, :cols] = np.clip(quantized, -127, 127).astype(np.int8)
    return codes, scales


def quantize_activation(values: np.ndarray, stored_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Group the resident float16 row by 128. A zero group stores scale 1."""
    if values.ndim != 2 or values.dtype != np.float16:
        message = f"activation quant wants float16 [rows, K], got {values.shape} {values.dtype}"
        raise ValueError(message)
    rows, cols = values.shape
    if stored_k % GROUP or cols > stored_k:
        message = f"stored K {stored_k} cannot hold {cols}"
        raise ValueError(message)
    groups = stored_k // GROUP
    codes = np.zeros((rows, stored_k), dtype=np.int8)
    scales = np.ones((rows, groups), dtype=np.float32)
    data = values.astype(np.float32)
    for group in range(groups):
        start = group * GROUP
        stop = min(start + GROUP, cols)
        if start >= cols:
            break
        block = data[:, start:stop]
        peak = np.max(np.abs(block), axis=1)
        live = peak > 0
        scales[:, group] = np.where(live, peak / np.float32(127), np.float32(1))
        inverse = np.zeros(rows, dtype=np.float32)
        inverse[live] = np.float32(127) / peak[live]
        quantized = np.rint(block * inverse[:, None]).astype(np.int32)
        codes[:, start:stop] = np.clip(quantized, -127, 127).astype(np.int8)
    return codes, scales


def reconstruct_row(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Widen one packed matrix with its per-row scale."""
    return codes.astype(np.float32) * scales.astype(np.float32)[:, None]


def reconstruct_groups(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Widen a grouped activation. ``scales`` is ``[rows, groups]``."""
    repeated = np.repeat(scales.astype(np.float32), GROUP, axis=1)
    width = codes.shape[1]
    return codes.astype(np.float32) * repeated[:, :width]
