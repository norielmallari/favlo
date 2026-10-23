"""fp4 E2M1 pack and the tile constants for this checkpoint.

Stored K is the logical K rounded up to a multiple of 256 so a packed row is a
multiple of 128 bytes. The scale is E8M0, one byte per 32 values along K.
The power of two rounds up so the block peak fits the fp4 endpoint 6.
"""

from __future__ import annotations

import numpy as np

TILE_M = 64
TILE_N = 32
K_ALIGN = 256
FP4_LEVELS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
FP4_MAX = 6.0


def stored_k(logical: int) -> int:
    if logical <= 0:
        message = f"K must be positive, got {logical}"
        raise ValueError(message)
    return ((logical + K_ALIGN - 1) // K_ALIGN) * K_ALIGN


def pad_to(n: int, tile: int) -> int:
    if n < 0:
        message = f"extent must be non-negative, got {n}"
        raise ValueError(message)
    return ((n + tile - 1) // tile) * tile


def pack_fp4(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Pack ``[rows, K]`` float32 into nibbles and E8M0 scales.

    Low nibble is element 0. A zero block stores scale 0 and zero codes.
    Extra K columns are zero codes.
    """
    if weight.ndim != 2:
        message = f"fp4 matrix must be rank 2, got {weight.shape}"
        raise ValueError(message)
    matrix = np.ascontiguousarray(weight, dtype=np.float32)
    rows, logical = matrix.shape
    sk = stored_k(logical)
    packed = np.zeros((rows, sk // 2), dtype=np.uint8)
    scales = np.zeros((rows, sk // 32), dtype=np.uint8)
    step = 32
    for start in range(0, rows, step):
        stop = min(start + step, rows)
        block_rows = matrix[start:stop]
        if sk != logical:
            padded = np.zeros((stop - start, sk), dtype=np.float32)
            padded[:, :logical] = block_rows
            block_rows = padded
        groups = block_rows.reshape(stop - start, sk // 32, 32)
        peak = np.max(np.abs(groups), axis=-1)
        ratio = np.divide(peak, FP4_MAX, out=np.zeros_like(peak), where=peak > 0)
        exp = np.zeros_like(peak)
        live = peak > 0
        exp[live] = np.ceil(np.log2(ratio[live]))
        scale = np.exp2(exp)
        e = np.zeros(peak.shape, dtype=np.uint8)
        e[live] = np.clip(exp[live] + 127.0, 0.0, 254.0).astype(np.uint8)
        safe = np.where(live, scale, 1.0)
        scaled = groups / safe[..., None]
        distance = np.abs(np.abs(scaled)[..., None] - FP4_LEVELS)
        code = np.argmin(distance, axis=-1).astype(np.uint8)
        code = code | np.where(scaled < 0.0, np.uint8(8), np.uint8(0))
        flat = code.reshape(stop - start, sk)
        packed[start:stop] = flat[:, 0::2] | (flat[:, 1::2] << 4)
        scales[start:stop] = e
    return packed, scales, sk


def dequant_fp4(packed: np.ndarray, scales: np.ndarray, sk: int) -> np.ndarray:
    """Inverse of :func:`pack_fp4` for the kernels check."""
    rows = packed.shape[0]
    codes = np.empty((rows, sk), dtype=np.uint8)
    codes[:, 0::2] = packed & np.uint8(0x0F)
    codes[:, 1::2] = packed >> np.uint8(4)
    mag = FP4_LEVELS[codes & np.uint8(7)]
    sign = np.where(codes & np.uint8(8), -1.0, 1.0).astype(np.float32)
    values = (mag * sign).reshape(rows, sk // 32, 32)
    factor = np.exp2(scales.astype(np.float32) - 127.0)
    return (values * factor[..., None]).reshape(rows, sk)


def as_float32(array: np.ndarray, dtype_name: str) -> np.ndarray:
    """Checkpoint dtypes F32, F16, and BF16 as float32."""
    if dtype_name == "F32":
        return np.ascontiguousarray(array, dtype=np.float32)
    if dtype_name == "F16":
        return np.ascontiguousarray(array, dtype=np.float16).astype(np.float32)
    if dtype_name == "BF16":
        raw = np.ascontiguousarray(array).view(np.uint16)
        widened = raw.astype(np.uint32) << np.uint32(16)
        return widened.view(np.float32)
    message = f"unsupported checkpoint dtype {dtype_name}"
    raise ValueError(message)
