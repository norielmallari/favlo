"""Pack BF16 rows as fp4 E2M1 with an E8M0 scale. Checkpoint packing only."""

from __future__ import annotations

import numpy as np

from spark_x26_1_7b.layout import BLOCK

FP4_QMAX = 6.0
FP4_MAGNITUDE = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
_FP4_EDGE = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)


def bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    """Interpret a BF16 tensor as float32. A float32 array passes through."""
    if raw.dtype == np.float32:
        return np.ascontiguousarray(raw)
    if raw.dtype == np.float16:
        return raw.astype(np.float32)
    view = np.ascontiguousarray(raw).view(np.uint16)
    shifted = view.astype(np.uint32) << np.uint32(16)
    return shifted.view(np.float32)


def biased_e8m0(amax: np.ndarray, qmax: float) -> tuple[np.ndarray, np.ndarray]:
    """``2^ceil(log2(amax / qmax))`` so the block peak fits. A zero block stores 0."""
    positive = amax > 0.0
    ratio = np.ones(amax.shape, dtype=np.float64)
    np.divide(amax, qmax, out=ratio, where=positive)
    power = np.ceil(np.log2(np.maximum(ratio, np.finfo(np.float64).tiny)))
    biased = np.clip(power + 127.0, 0, 254).astype(np.uint8)
    biased = np.where(positive, biased, np.uint8(0))
    scale = np.ldexp(np.ones(amax.shape, dtype=np.float32), biased.astype(np.int32) - 127)
    return biased, scale


def quantize_fp4(weight: np.ndarray) -> tuple[bytes, bytes]:
    """Pack ``[rows, inner]`` float32. ``inner`` is a multiple of 32. Low nibble is element 0."""
    matrix = np.ascontiguousarray(weight, dtype=np.float32)
    rows, inner = matrix.shape
    if inner % BLOCK:
        message = f"inner dimension {inner} is not a multiple of {BLOCK}"
        raise ValueError(message)
    values = matrix.reshape(rows, inner // BLOCK, BLOCK)
    amax = np.max(np.abs(values), axis=-1)
    biased, scale = biased_e8m0(amax, FP4_QMAX)
    scaled = values / scale[..., None]
    choice = np.searchsorted(_FP4_EDGE, np.abs(scaled), side="left").astype(np.uint8)
    sign = np.signbit(scaled).astype(np.uint8) << np.uint8(3)
    nibbles = choice | sign
    packed = nibbles[..., 0::2] | (nibbles[..., 1::2] << np.uint8(4))
    return np.ascontiguousarray(packed).tobytes(), np.ascontiguousarray(biased).tobytes()


def dequant_fp4(packed: bytes, scales: bytes, rows: int, inner: int) -> np.ndarray:
    raw = np.frombuffer(packed, dtype=np.uint8).reshape(rows, inner // 2)
    codes = np.empty((rows, inner), dtype=np.uint8)
    codes[:, 0::2] = raw & np.uint8(0xF)
    codes[:, 1::2] = raw >> np.uint8(4)
    sign = np.where(codes & np.uint8(0x8), np.float32(-1.0), np.float32(1.0))
    magnitude = FP4_MAGNITUDE[codes & np.uint8(0x7)]
    factors = np.frombuffer(scales, dtype=np.uint8).astype(np.int32).reshape(rows, inner // BLOCK)
    restored = (sign * magnitude).reshape(rows, inner // BLOCK, BLOCK)
    restored = restored * np.ldexp(np.ones(factors.shape, dtype=np.float32), factors - 127)[..., None]
    return restored.reshape(rows, inner)


def f16_bytes(weight: np.ndarray) -> bytes:
    return np.ascontiguousarray(weight, dtype=np.float32).astype(np.float16).tobytes()
