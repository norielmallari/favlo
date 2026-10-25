"""E8M0 block scales and fp4 E2M1 / fp8 E4M3 packs.

One K permutation groups columns of similar magnitude. The block score is the
maximum fp4 relative error of that matrix. A zero block stores scale 0.
"""

from __future__ import annotations

import numpy as np

_FP4 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
_FP4_ENDPOINT = np.float32(6.0)
_FP8_ENDPOINT = np.float32(448.0)


def _e4m3_table() -> tuple[np.ndarray, np.ndarray]:
    values: list[float] = []
    codes: list[int] = []
    for code in range(256):
        sign = code >> 7
        exp = (code >> 3) & 0xF
        man = code & 0x7
        if exp == 0xF and man == 0x7:
            continue
        if exp == 0:
            value = (man / 8.0) * (2.0**-6)
        else:
            value = (1.0 + man / 8.0) * (2.0 ** (exp - 7))
        if sign:
            value = -value
        values.append(value)
        codes.append(code)
    return np.array(values, dtype=np.float32), np.array(codes, dtype=np.uint8)


_E4M3_VALUES, _E4M3_CODES = _e4m3_table()


def permute_k(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort K by mean absolute value. ``stored[:, k] = weight[:, perm[k]]``."""
    score = np.mean(np.abs(weight), axis=0)
    perm = np.argsort(score, kind="stable").astype(np.uint32)
    return np.ascontiguousarray(weight[:, perm]), perm


def _scales(peak: np.ndarray, endpoint: float) -> np.ndarray:
    """E8M0 exponents. A zero block is 0. The power of two rounds up."""
    exponents = np.zeros(peak.shape, dtype=np.uint8)
    live = peak > 0
    if not np.any(live):
        return exponents
    ratio = np.maximum(peak[live] / np.float32(endpoint), np.float32(2.0**-126))
    power = np.ceil(np.log2(ratio)).astype(np.int32)
    stored = np.clip(power + 127, 0, 254).astype(np.uint8)
    exponents[live] = stored
    return exponents


def _decode_scale(exponents: np.ndarray) -> np.ndarray:
    return np.ldexp(np.ones(exponents.shape, dtype=np.float32), exponents.astype(np.int32) - 127)


def pack_fp4(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return packed bytes ``[N, K/2]``, scales ``[N, K/32]``, and the dequant."""
    rows, cols = weight.shape
    if cols % 32 != 0:
        message = f"fp4 K {cols} is not a multiple of 32"
        raise ValueError(message)
    blocks = weight.reshape(rows, cols // 32, 32)
    peak = np.max(np.abs(blocks), axis=-1)
    exponents = _scales(peak, float(_FP4_ENDPOINT))
    scale = _decode_scale(exponents)
    live = exponents != 0
    denom = np.where(live, scale, np.float32(1.0))
    scaled = blocks / denom[..., None]
    nearest = np.argmin(np.abs(np.abs(scaled)[..., None] - _FP4), axis=-1).astype(np.uint8)
    sign = np.where(scaled < 0, np.uint8(8), np.uint8(0))
    nibbles = (nearest | sign).reshape(rows, cols)
    packed = (nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)).astype(np.uint8)
    signed = _FP4[nearest] * np.where(scaled < 0, np.float32(-1.0), np.float32(1.0))
    dequant = (signed * np.where(live, scale, np.float32(0.0))[..., None]).reshape(rows, cols)
    return np.ascontiguousarray(packed), np.ascontiguousarray(exponents), dequant.astype(np.float32)


def pack_fp8(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return E4M3 bytes ``[N, K]``, scales ``[N, K/32]``, and the dequant."""
    rows, cols = weight.shape
    if cols % 32 != 0:
        message = f"fp8 K {cols} is not a multiple of 32"
        raise ValueError(message)
    blocks = weight.reshape(rows, cols // 32, 32)
    peak = np.max(np.abs(blocks), axis=-1)
    exponents = _scales(peak, float(_FP8_ENDPOINT))
    scale = _decode_scale(exponents)
    live = exponents != 0
    denom = np.where(live, scale, np.float32(1.0))
    scaled = (blocks / denom[..., None]).reshape(-1)
    codes_flat = np.empty(scaled.shape, dtype=np.uint8)
    values_flat = np.empty(scaled.shape, dtype=np.float32)
    step = 262144
    for start in range(0, scaled.shape[0], step):
        piece = scaled[start : start + step]
        nearest = np.argmin(np.abs(piece[:, None] - _E4M3_VALUES[None, :]), axis=1)
        codes_flat[start : start + step] = _E4M3_CODES[nearest]
        values_flat[start : start + step] = _E4M3_VALUES[nearest]
    codes = codes_flat.reshape(rows, cols)
    values = values_flat.reshape(rows, cols // 32, 32)
    dequant = values * np.where(live, scale, np.float32(0.0))[..., None]
    return np.ascontiguousarray(codes), np.ascontiguousarray(exponents), dequant.reshape(rows, cols)


def pack_int8(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Symmetric int8 codes, one float scale per row, and the dequant.

    ``N`` pads to 128 and ``K`` pads to 256. Extra rows and columns are zero.
    ``-128`` is unused. A zero row stores scale 0.
    """
    rows, cols = weight.shape
    pad_n = max(128, ((rows + 127) // 128) * 128)
    pad_k = max(256, ((cols + 255) // 256) * 256)
    padded = np.zeros((pad_n, pad_k), dtype=np.float32)
    padded[:rows, :cols] = np.ascontiguousarray(weight, dtype=np.float32)
    peak = np.max(np.abs(padded), axis=1)
    scale = np.zeros(pad_n, dtype=np.float32)
    live = peak > 0
    scale[live] = peak[live] / np.float32(127.0)
    inverse = np.zeros(pad_n, dtype=np.float32)
    inverse[live] = np.float32(127.0) / peak[live]
    codes = np.rint(padded * inverse[:, None])
    codes = np.clip(codes, -127, 127).astype(np.int8)
    dequant = codes.astype(np.float32) * scale[:, None]
    return np.ascontiguousarray(codes), np.ascontiguousarray(scale), dequant


def quantize_activation(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One int8 code per value and one float scale per row. ``rows`` is ``[M, K]``."""
    data = np.ascontiguousarray(rows, dtype=np.float32)
    peak = np.max(np.abs(data), axis=1)
    scale = np.zeros(peak.shape, dtype=np.float32)
    live = peak > 0
    scale[live] = peak[live] / np.float32(127.0)
    inverse = np.zeros(peak.shape, dtype=np.float32)
    inverse[live] = np.float32(127.0) / peak[live]
    codes = np.rint(data * inverse[:, None])
    codes = np.clip(codes, -127, 127).astype(np.int8)
    return np.ascontiguousarray(codes), np.ascontiguousarray(scale)


def relative_rms(weight: np.ndarray, dequant: np.ndarray) -> float:
    """RMS of the residual divided by RMS of the source row block."""
    rows, cols = weight.shape
    residual = dequant[:rows, :cols] - weight
    denom = float(np.sqrt(np.mean(np.square(weight))))
    numer = float(np.sqrt(np.mean(np.square(residual))))
    if denom == 0.0:
        return 0.0
    return numer / denom


def relative_error(weight: np.ndarray, dequant: np.ndarray) -> float:
    denom = np.maximum(np.abs(weight), np.float32(1e-8))
    return float(np.max(np.abs(dequant - weight) / denom))


def score_matrix(weight: np.ndarray) -> tuple[float, np.ndarray]:
    """fp4 relative error after one K permutation, and that permutation."""
    ordered, perm = permute_k(np.ascontiguousarray(weight, dtype=np.float32))
    _packed, _scales, dequant = pack_fp4(ordered)
    restored = np.empty_like(dequant)
    restored[:, perm] = dequant
    return relative_error(weight, restored), perm
