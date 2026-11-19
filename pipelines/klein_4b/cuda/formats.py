"""int4 pack and the host dequant the kernel uses."""

from __future__ import annotations

import numpy as np

from favlo.quantize import quantize_int4
from klein_4b.cuda.constants import align_up


def pad_matrix(weight: np.ndarray) -> tuple[np.ndarray, int, int]:
    matrix = np.ascontiguousarray(weight, dtype=np.float32)
    rows, inner = matrix.shape
    stored_n = align_up(rows)
    stored_k = align_up(inner)
    if stored_n == rows and stored_k == inner:
        return matrix, rows, inner
    padded = np.zeros((stored_n, stored_k), dtype=np.float32)
    padded[:rows, :inner] = matrix
    return padded, rows, inner


def pack_int4(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    padded, rows, inner = pad_matrix(weight)
    codes, scales = quantize_int4(padded)
    return codes, scales.astype(np.float32), rows, inner


def restore_int4(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Low nibble is the even column. A nibble at least 8 is negative."""
    packed = np.ascontiguousarray(codes, dtype=np.uint8)
    low = packed & np.uint8(0x0F)
    high = packed >> np.uint8(4)
    even = np.where(low >= 8, low.astype(np.int16) - 16, low.astype(np.int16))
    odd = np.where(high >= 8, high.astype(np.int16) - 16, high.astype(np.int16))
    values = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.float32)
    values[:, 0::2] = even * scales[:, None]
    values[:, 1::2] = odd * scales[:, None]
    return values
