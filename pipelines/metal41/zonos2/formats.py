"""fp4 E2M1 and fp8 E4M3 with a round-up E8M0 scale."""

from __future__ import annotations

import math

import numpy as np

BLOCK = 32
FP4_QMAX = 6.0
FP8_QMAX = 448.0
_FP4 = np.array((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), dtype=np.float32)


def promote_count(percent: float, units: int) -> int:
    """Largest whole count that does not exceed ``percent`` of ``units``."""
    if units < 1:
        message = f"unit count {units} is empty"
        raise ValueError(message)
    if not math.isfinite(percent) or percent < 0.0 or percent > 100.0:
        message = f"fp8 percentage {percent} is outside 0..100"
        raise ValueError(message)
    count = int(math.floor((percent * units / 100.0) + 1e-9))
    return min(count, units)


def _ceil_log2(value: float) -> int:
    mantissa, exponent = math.frexp(value)
    if mantissa == 0.5:
        return exponent - 1
    return exponent


def scale_exponent(peak: float, qmax: float) -> int:
    """Round-up E8M0 exponent. A zero block stores 0."""
    if peak <= 0.0 or not math.isfinite(peak):
        return 0
    power = _ceil_log2(peak / qmax)
    return min(254, max(0, power + 127))


def _fp4_codes(scaled: np.ndarray) -> np.ndarray:
    magnitude = np.abs(scaled)
    index = np.zeros(magnitude.shape, dtype=np.uint8)
    best = np.full(magnitude.shape, np.inf, dtype=np.float32)
    for code, level in enumerate(_FP4):
        error = np.abs(magnitude - level)
        take = error < best
        index = np.where(take, code, index).astype(np.uint8)
        best = np.minimum(best, error)
    sign = (scaled < 0.0).astype(np.uint8) << 3
    return (index + sign).astype(np.uint8)


def _exponents(peak: np.ndarray, qmax: float) -> np.ndarray:
    safe = np.where(peak > 0.0, peak, 1.0).astype(np.float32)
    mantissa, exponent = np.frexp(safe / np.float32(qmax))
    power = np.where(mantissa == np.float32(0.5), exponent - 1, exponent).astype(np.int32)
    stored = np.where(peak > 0.0, power + 127, 0)
    return np.clip(stored, 0, 254).astype(np.uint8)


def choose_promoted(scores: dict[int, float], percent: float) -> tuple[int, ...]:
    """Highest scores, up to the promoted count. An equal score keeps the lower index."""
    count = promote_count(percent, len(scores))
    ordered = sorted(scores, key=lambda layer: (-scores[layer], layer))
    return tuple(sorted(ordered[:count]))


def quantize_fp4(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pack K-innermost fp4. Returns data bytes ``[rows, K/2]`` and scales ``[rows, K/32]``."""
    matrix = np.ascontiguousarray(weight, dtype=np.float32)
    rows, inner = matrix.shape
    if inner % BLOCK != 0:
        message = f"K {inner} is not a multiple of {BLOCK}"
        raise ValueError(message)
    blocks = matrix.reshape(rows, inner // BLOCK, BLOCK)
    peak = np.max(np.abs(blocks), axis=2)
    exponents = _exponents(peak, FP4_QMAX)
    scale = np.ldexp(np.ones(exponents.shape, dtype=np.float32), exponents.astype(np.int32) - 127)
    codes = _fp4_codes(blocks / scale[..., None]).reshape(rows, inner)
    low = codes[:, 0::2]
    high = codes[:, 1::2]
    packed = (low | (high << 4)).astype(np.uint8)
    return packed, exponents


def dequant_fp4(data: np.ndarray, scales: np.ndarray, rows: int, inner: int) -> np.ndarray:
    packed = np.ascontiguousarray(data, dtype=np.uint8).reshape(rows, inner // 2)
    low = packed & 0x0F
    high = packed >> 4
    codes = np.empty((rows, inner), dtype=np.uint8)
    codes[:, 0::2] = low
    codes[:, 1::2] = high
    sign = np.where(codes & 0x08, -1.0, 1.0).astype(np.float32)
    magnitude = _FP4[codes & 0x07]
    values = (sign * magnitude).reshape(rows, inner // BLOCK, BLOCK)
    factors = np.ldexp(
        np.ones(scales.shape, dtype=np.float32),
        np.ascontiguousarray(scales, dtype=np.uint8).astype(np.int32) - 127,
    )
    return (values * factors.reshape(rows, inner // BLOCK, 1)).reshape(rows, inner)


def _e4m3_decode_table() -> np.ndarray:
    table = np.zeros(256, dtype=np.float32)
    for code in range(256):
        sign = -1.0 if code & 0x80 else 1.0
        exp = (code >> 3) & 0x0F
        mant = code & 0x07
        if exp == 0:
            table[code] = sign * (mant / 8.0) * 2.0**-6
        elif exp == 0x0F and mant == 0x07:
            table[code] = np.nan
        else:
            table[code] = sign * (1.0 + mant / 8.0) * 2.0 ** (exp - 7)
    return table


_E4M3 = _e4m3_decode_table()
_E4M3_FINITE = np.array([code for code in range(256) if np.isfinite(_E4M3[code])], dtype=np.int32)


def quantize_fp8(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pack K-innermost fp8 E4M3. Returns data ``[rows, K]`` and scales ``[rows, K/32]``."""
    matrix = np.ascontiguousarray(weight, dtype=np.float32)
    rows, inner = matrix.shape
    if inner % BLOCK != 0:
        message = f"K {inner} is not a multiple of {BLOCK}"
        raise ValueError(message)
    blocks = matrix.reshape(rows, inner // BLOCK, BLOCK)
    peak = np.max(np.abs(blocks), axis=2)
    exponents = _exponents(peak, FP8_QMAX)
    scale = np.ldexp(np.ones(exponents.shape, dtype=np.float32), exponents.astype(np.int32) - 127)
    scaled = blocks / scale[..., None]
    finite = _E4M3[_E4M3_FINITE]
    codes = np.empty((rows, inner), dtype=np.uint8)
    flat = scaled.reshape(rows, inner)
    for start in range(0, rows, 8):
        tile = flat[start : start + 8].reshape(-1)
        distance = np.abs(tile[:, None] - finite[None, :])
        codes[start : start + 8] = _E4M3_FINITE[np.argmin(distance, axis=1)].reshape(-1, inner)
    return codes, exponents


def dequant_fp8(data: np.ndarray, scales: np.ndarray, rows: int, inner: int) -> np.ndarray:
    codes = np.ascontiguousarray(data, dtype=np.uint8).reshape(rows, inner)
    values = _E4M3[codes].reshape(rows, inner // BLOCK, BLOCK)
    factors = np.ldexp(
        np.ones(scales.shape, dtype=np.float32),
        np.ascontiguousarray(scales, dtype=np.uint8).astype(np.int32) - 127,
    )
    return (values * factors.reshape(rows, inner // BLOCK, 1)).reshape(rows, inner)


def relative_error(weight: np.ndarray) -> float:
    """RMS of the source against its restored fp4 matrix, divided by the source RMS."""
    matrix = np.ascontiguousarray(weight, dtype=np.float32)
    if matrix.ndim != 2:
        message = "relative error scores a matrix"
        raise ValueError(message)
    numerator = 0.0
    denominator = 0.0
    for start in range(0, matrix.shape[0], 64):
        tile = matrix[start : start + 64]
        data, scales = quantize_fp4(tile)
        restored = dequant_fp4(data, scales, tile.shape[0], tile.shape[1])
        diff = tile.astype(np.float64) - restored.astype(np.float64)
        flat = tile.astype(np.float64).ravel()
        numerator += float(np.dot(diff.ravel(), diff.ravel()))
        denominator += float(np.dot(flat, flat))
    if denominator == 0.0:
        return 0.0
    return math.sqrt(numerator / denominator)
