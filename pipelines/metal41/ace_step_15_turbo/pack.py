"""fp4 E2M1 weights and fp8 E4M3 key/value blocks.

A scale byte is an E8M0 exponent. Zero means the block is zero. The fp4
endpoint is 6. The fp8 E4M3 endpoint is 448.
"""

from __future__ import annotations

import math

import numpy as np

FP4_MAGNITUDE = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
FP4_ENDPOINT = 6.0
FP8_ENDPOINT = 448.0
TILE_M = 64
TILE_N = 32
FP4_K = 256


def pad_pair(rows: int, cols: int) -> tuple[int, int]:
    """Pad a matmul to the fp4 tile. ``cols`` is K."""
    padded_rows = ((rows + TILE_N - 1) // TILE_N) * TILE_N
    padded_cols = ((cols + FP4_K - 1) // FP4_K) * FP4_K
    return padded_rows, padded_cols


def _e8m0(peak: float, endpoint: float) -> int:
    if not math.isfinite(peak) or peak <= 0.0:
        return 0
    ratio = peak / endpoint
    if ratio <= 0.0 or not math.isfinite(ratio):
        return 0
    power = math.ceil(math.log2(ratio))
    biased = power + 127
    if biased < 1:
        return 0
    if biased > 254:
        return 254
    return biased


def pack_fp4(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Pack ``[rows, cols]`` as fp4 with one scale per 32 columns.

    Returns packed nibbles, scales, padded rows, and padded columns.
    """
    matrix = np.ascontiguousarray(weight, dtype=np.float32)
    rows, cols = matrix.shape
    padded_rows, padded_cols = pad_pair(rows, cols)
    padded = np.zeros((padded_rows, padded_cols), dtype=np.float32)
    padded[:rows, :cols] = matrix
    blocks = padded_cols // 32
    slab = padded.reshape(padded_rows, blocks, 32)
    peak = np.max(np.abs(slab), axis=2)
    live = np.isfinite(peak) & (peak > 0)
    power = np.zeros_like(peak)
    power[live] = np.ceil(np.log2(peak[live] / FP4_ENDPOINT))
    biased = np.clip(power + 127, 0, 254)
    biased = np.where(live & (biased >= 1), biased, 0)
    scales = biased.astype(np.uint8)
    scale = np.zeros_like(peak)
    positive = scales > 0
    scale[positive] = np.ldexp(np.ones(int(positive.sum()), dtype=np.float32), scales[positive].astype(np.int32) - 127)
    quantized = np.zeros_like(slab)
    quantized[positive] = slab[positive] / scale[positive, None]
    nearest = np.abs(np.abs(quantized)[..., None] - FP4_MAGNITUDE.reshape(1, 1, 1, 8)).argmin(axis=-1)
    sign = np.where(quantized < 0, np.uint8(8), np.uint8(0))
    codes = ((sign | nearest.astype(np.uint8)) * positive[:, :, None]).reshape(padded_rows, padded_cols)
    nibbles = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return np.ascontiguousarray(nibbles), np.ascontiguousarray(scales), padded_rows, padded_cols


def dequant_fp4(nibbles: np.ndarray, scales: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Invert :func:`pack_fp4` for the leading ``rows`` by ``cols``."""
    padded_rows, padded_cols = scales.shape[0], scales.shape[1] * 32
    packed = np.zeros((padded_rows, padded_cols), dtype=np.uint8)
    flat = nibbles.reshape(padded_rows, -1)
    packed[:, 0::2] = flat & 0x0F
    packed[:, 1::2] = flat >> 4
    values = np.zeros((padded_rows, padded_cols), dtype=np.float32)
    for block in range(scales.shape[1]):
        exponent = scales[:, block].astype(np.int32)
        scale = np.zeros(padded_rows, dtype=np.float32)
        live = exponent > 0
        scale[live] = np.ldexp(1.0, exponent[live] - 127).astype(np.float32)
        code = packed[:, block * 32 : (block + 1) * 32]
        mag = FP4_MAGNITUDE[code & 7]
        sign = np.where((code & 8) != 0, -1.0, 1.0).astype(np.float32)
        values[:, block * 32 : (block + 1) * 32] = sign * mag * scale[:, None]
    return values[:rows, :cols]


def promote_count(percent: float, layers: int) -> int:
    """Largest whole layer count that does not exceed ``percent`` of ``layers``."""
    if layers < 1:
        message = f"layer count {layers} is empty"
        raise ValueError(message)
    if not math.isfinite(percent) or percent < 0.0 or percent > 100.0:
        message = f"fp8 percentage {percent} is outside 0..100"
        raise ValueError(message)
    count = int(math.floor((percent * layers / 100.0) + 1e-9))
    if count > layers:
        return layers
    return count


def promoted_ids(scores: dict[int, float], count: int) -> list[int]:
    """Highest scores. An equal score keeps the lower layer index."""
    if count < 0 or count > len(scores):
        message = f"promote count {count} does not fit {len(scores)} layers"
        raise ValueError(message)
    ranked = sorted(scores, key=lambda layer: (-float(scores[layer]), int(layer)))
    return sorted(int(layer) for layer in ranked[:count])


def relative_error(weight: np.ndarray) -> float:
    """RMS of the source against its restored fp4 matrix, divided by the source RMS."""
    matrix = np.ascontiguousarray(weight, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    elif matrix.ndim > 2:
        matrix = matrix.reshape(matrix.shape[0], -1)
    numerator = 0.0
    denominator = 0.0
    rows, cols = matrix.shape
    for start in range(0, rows, 512):
        tile = matrix[start : start + 512]
        nibbles, scales, _, _ = pack_fp4(tile)
        restored = dequant_fp4(nibbles, scales, tile.shape[0], cols)
        diff = tile.astype(np.float64) - restored.astype(np.float64)
        flat = tile.astype(np.float64).ravel()
        numerator += float(np.dot(diff.ravel(), diff.ravel()))
        denominator += float(np.dot(flat, flat))
    if denominator == 0.0:
        return 0.0
    return math.sqrt(numerator / denominator)


def _biased_e8m0(amax: np.ndarray, endpoint: float) -> tuple[np.ndarray, np.ndarray]:
    """Round ``2^ceil(log2(amax / endpoint))`` up. A zero block stores exponent 0."""
    ratio = np.maximum(amax / endpoint, np.finfo(np.float64).tiny)
    mantissa, exponent = np.frexp(ratio)
    power = exponent.astype(np.int32) - (mantissa == 0.5).astype(np.int32)
    power = np.where(amax <= 0.0, np.int32(-127), power)
    biased = np.clip(power + 127, 0, 254).astype(np.uint8)
    scale = np.ldexp(np.ones(amax.shape, dtype=np.float32), biased.astype(np.int32) - 127)
    return biased, scale


def _e4m3_levels() -> np.ndarray:
    codes = np.arange(127, dtype=np.uint32)
    sign = np.where(codes & 0x80, np.float32(-1.0), np.float32(1.0))
    exponent = (codes >> 3) & 0xF
    mantissa = (codes & 0x7).astype(np.float32)
    out = np.empty(codes.shape, dtype=np.float32)
    sub = exponent == 0
    normal = ~sub
    out[sub] = sign[sub] * (mantissa[sub] / np.float32(8.0)) * np.float32(2.0**-6)
    out[normal] = (
        sign[normal]
        * (np.float32(1.0) + mantissa[normal] / np.float32(8.0))
        * np.exp2(exponent[normal].astype(np.float32) - np.float32(7.0))
    )
    return out


_E4M3_EDGE = ((_e4m3_levels()[:-1] + _e4m3_levels()[1:]) * np.float32(0.5)).astype(np.float32)


def pack_fp8(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Pack ``[rows, cols]`` as E4M3 with one scale per 32 columns."""
    matrix = np.ascontiguousarray(weight, dtype=np.float32)
    rows, cols = matrix.shape
    padded_rows, padded_cols = pad_pair(rows, cols)
    blocks = padded_cols // 32
    codes = np.zeros((padded_rows, padded_cols), dtype=np.uint8)
    scales = np.zeros((padded_rows, blocks), dtype=np.uint8)
    for start in range(0, rows, 1024):
        stop = min(rows, start + 1024)
        tile = np.zeros((stop - start, padded_cols), dtype=np.float32)
        tile[:, :cols] = matrix[start:stop]
        values = tile.reshape(stop - start, blocks, 32)
        peak = np.max(np.abs(values), axis=-1)
        biased, scale = _biased_e8m0(peak, FP8_ENDPOINT)
        scaled = values / scale[..., None]
        magnitude = np.abs(scaled).astype(np.float32, copy=False)
        choice = np.searchsorted(_E4M3_EDGE, magnitude, side="left").astype(np.int16)
        choice = np.clip(choice, 0, 126)
        edge_at = np.minimum(choice, 125)
        on_edge = (choice < 126) & (magnitude == _E4M3_EDGE[edge_at])
        bump = on_edge & ((choice & 1) == 1)
        choice = np.where(bump, choice + 1, choice).astype(np.uint16)
        sign = np.signbit(scaled).astype(np.uint16) << np.uint16(7)
        codes[start:stop] = (choice | sign).astype(np.uint8).reshape(stop - start, padded_cols)
        scales[start:stop] = biased
    return np.ascontiguousarray(codes), np.ascontiguousarray(scales), padded_rows, padded_cols


def dequant_fp8(codes: np.ndarray, scales: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Invert :func:`pack_fp8` for the leading ``rows`` by ``cols``."""
    padded_rows, blocks = scales.shape
    padded_cols = blocks * 32
    raw = np.ascontiguousarray(codes, dtype=np.uint32).reshape(padded_rows, padded_cols)
    sign = np.where(raw & 0x80, np.float32(-1.0), np.float32(1.0))
    exponent = (raw >> 3) & 0xF
    mantissa = (raw & 0x7).astype(np.float32)
    values = np.empty(raw.shape, dtype=np.float32)
    sub = exponent == 0
    nan = (exponent == 0xF) & ((raw & 0x7) == 0x7)
    normal = ~(sub | nan)
    values[sub] = sign[sub] * (mantissa[sub] / np.float32(8.0)) * np.float32(2.0**-6)
    values[nan] = np.float32(np.nan)
    values[normal] = (
        sign[normal]
        * (np.float32(1.0) + mantissa[normal] / np.float32(8.0))
        * np.exp2(exponent[normal].astype(np.float32) - np.float32(7.0))
    )
    factors = np.zeros(scales.shape, dtype=np.float32)
    live = scales > 0
    factors[live] = np.ldexp(np.ones(int(live.sum()), dtype=np.float32), scales[live].astype(np.int32) - 127)
    restored = values.reshape(padded_rows, blocks, 32) * factors[..., None]
    return restored.reshape(padded_rows, padded_cols)[:rows, :cols]


def quant_fp8_blocks(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pack a trailing multiple of 32 as E4M3 with one E8M0 scale per block."""
    flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1, 32)
    scales = np.zeros(flat.shape[0], dtype=np.uint8)
    coded = np.zeros(flat.shape, dtype=np.uint8)
    for index, row in enumerate(flat):
        exponent = _e8m0(float(np.max(np.abs(row))), FP8_ENDPOINT)
        scales[index] = exponent
        if exponent == 0:
            continue
        scale = math.ldexp(1.0, exponent - 127)
        coded[index] = _encode_e4m3(row / scale)
    return coded.reshape(values.shape), scales.reshape(values.shape[:-1] + (values.shape[-1] // 32,))


def _encode_e4m3(values: np.ndarray) -> np.ndarray:
    out = np.zeros(values.shape, dtype=np.uint8)
    for index, value in enumerate(values.reshape(-1)):
        number = float(value)
        sign = 0x80 if math.copysign(1.0, number) < 0 and number != 0 else 0
        magnitude = abs(number)
        if magnitude == 0 or not math.isfinite(magnitude):
            out.reshape(-1)[index] = sign if math.isfinite(magnitude) else sign | 0x7F
            continue
        if magnitude >= FP8_ENDPOINT:
            out.reshape(-1)[index] = sign | 0x7E
            continue
        exponent = math.floor(math.log2(magnitude))
        biased = exponent + 7
        if biased <= 0:
            mantissa = int(round(magnitude / math.ldexp(1.0, -9)))
            mantissa = min(7, max(0, mantissa))
            out.reshape(-1)[index] = sign | mantissa
            continue
        if biased >= 15:
            out.reshape(-1)[index] = sign | 0x7E
            continue
        significand = magnitude / math.ldexp(1.0, exponent)
        mantissa = int(round((significand - 1.0) * 8.0))
        if mantissa == 8:
            mantissa = 0
            biased += 1
            if biased >= 15:
                out.reshape(-1)[index] = sign | 0x7E
                continue
        out.reshape(-1)[index] = sign | (biased << 3) | mantissa
    return out
