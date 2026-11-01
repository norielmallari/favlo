"""Pack BF16 rows as fp8 E4M3 with an E8M0 scale. Checkpoint packing only."""

from __future__ import annotations

import numpy as np

from spark_x26_1_7b.layout import BLOCK

FP8_QMAX = 448.0


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


def enc_e4m3(values: np.ndarray) -> np.ndarray:
    """OCP E4M3FN codes. ``0x7E`` is 448 and ``0x7F`` is NaN."""
    original = np.ascontiguousarray(values, dtype=np.float32)
    bits = np.frombuffer(original.tobytes(), dtype=np.uint32).reshape(original.shape)
    sign = ((bits >> np.uint32(24)) & np.uint32(0x80)).astype(np.uint8)
    exp = ((bits >> np.uint32(23)) & np.uint32(0xFF)).astype(np.int32)
    mant = bits & np.uint32(0x7FFFFF)
    mag = np.abs(original)
    sub = np.minimum(np.rint(mag * np.float32(512.0)).astype(np.int32), 7).astype(np.uint8)
    out = sign.copy()
    out = np.where(mag >= np.float32(FP8_QMAX), sign | np.uint8(0x7E), out)
    tiny = (mag > np.float32(0.0)) & (mag < np.float32(FP8_QMAX)) & (
        (exp == 0) | (mag < np.float32(0.015625))
    )
    out = np.where(tiny, sign | sub, out)
    normal = (
        (mag >= np.float32(0.015625))
        & (mag < np.float32(FP8_QMAX))
        & (exp != 0)
        & (exp != 255)
    )
    top = (mant >> np.uint32(20)).astype(np.int32)
    rem = mant & np.uint32(0xFFFFF)
    round_up = (rem > np.uint32(0x80000)) | ((rem == np.uint32(0x80000)) & ((top & 1) != 0))
    top = top + round_up.astype(np.int32)
    biased = exp - 127 + 7
    carry = top >= 8
    top = np.where(carry, 0, top)
    biased = np.where(carry, biased + 1, biased)
    code = (sign.astype(np.int32) | ((biased & 15) << 3) | (top & 7)).astype(np.uint8)
    code = np.where(biased > 15, sign | np.uint8(0x7E), code)
    code = np.where((biased == 15) & (top > 6), sign | np.uint8(0x7E), code)
    code = np.where(biased <= 0, sign | sub, code)
    return np.where(normal, code, out).astype(np.uint8)


def dec_e4m3(codes: np.ndarray) -> np.ndarray:
    raw = np.asarray(codes, dtype=np.uint8)
    sign = np.where(raw & np.uint8(0x80), np.float32(-1.0), np.float32(1.0))
    exp = ((raw >> np.uint8(3)) & np.uint8(0xF)).astype(np.int32)
    mant = (raw & np.uint8(0x7)).astype(np.float32)
    normal = (np.float32(1.0) + mant / np.float32(8.0)) * np.ldexp(
        np.ones(raw.shape, dtype=np.float32), exp - 7
    )
    sub = mant * np.float32(2.0**-9)
    value = np.where(exp == 0, sub, normal).astype(np.float32)
    value = np.where((exp == 15) & (mant == np.float32(7.0)), np.float32(np.nan), value)
    return sign * value


def quantize_fp8(weight: np.ndarray) -> tuple[bytes, bytes]:
    """Pack ``[rows, inner]`` float32. ``inner`` is a multiple of 128."""
    matrix = np.ascontiguousarray(weight, dtype=np.float32)
    rows, inner = matrix.shape
    if inner % 128:
        message = f"inner dimension {inner} is not a multiple of 128"
        raise ValueError(message)
    values = matrix.reshape(rows, inner // BLOCK, BLOCK)
    amax = np.max(np.abs(values), axis=-1)
    biased, scale = biased_e8m0(amax, FP8_QMAX)
    scaled = (values / scale[..., None]).reshape(rows, inner)
    return np.ascontiguousarray(enc_e4m3(scaled)).tobytes(), np.ascontiguousarray(biased).tobytes()


def dequant_fp8(packed: bytes, scales: bytes, rows: int, inner: int) -> np.ndarray:
    codes = np.frombuffer(packed, dtype=np.uint8).reshape(rows, inner)
    factors = np.frombuffer(scales, dtype=np.uint8).astype(np.int32).reshape(rows, inner // BLOCK)
    restored = dec_e4m3(codes).reshape(rows, inner // BLOCK, BLOCK)
    restored = restored * np.ldexp(
        np.ones(factors.shape, dtype=np.float32), factors - 127
    )[..., None]
    return restored.reshape(rows, inner)


def f16_bytes(weight: np.ndarray) -> bytes:
    return np.ascontiguousarray(weight, dtype=np.float32).astype(np.float16).tobytes()
