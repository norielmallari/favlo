"""fp4 E2M1 and fp8 E4M3 block scaling for this pipeline.

Block 32. The scale is the power of two rounded up so the peak fits, stored as
E8M0. Element codes use round-ties-to-even and then saturate. A zero block
stores exponent 0.
"""

from __future__ import annotations

import numpy as np

BLOCK = 32
FP4_QMAX = 6.0
FP8_QMAX = 448.0
FP4_MAGNITUDE = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)
PAGE = 16384
EXPERT_IMAGE = 1_671_168
# fp8 E4M3 gate-up is 2,097,152 data bytes plus 65,536 scale bytes.
# The down plane starts on the next 128-byte boundary.
EXPERT_FP8 = 3_244_032
FP8_GATE_SCALE = 2_097_152
FP8_DOWN_DATA = 2_162_688
FP8_DOWN_SCALE = 3_211_264


def stored_k(inner: int) -> int:
    """fp4 rows are a whole number of 128-byte groups, so K is a multiple of 256."""
    if inner < 1:
        message = f"inner dimension {inner} is empty"
        raise ValueError(message)
    return ((inner + 255) // 256) * 256


def biased_e8m0(amax: np.ndarray, qmax: float) -> tuple[np.ndarray, np.ndarray]:
    """Round ``2^ceil(log2(amax / qmax))`` up. A zero block stores exponent 0."""
    ratio = np.maximum(amax / qmax, np.finfo(np.float64).tiny)
    mantissa, exponent = np.frexp(ratio)
    power = exponent.astype(np.int32) - (mantissa == 0.5).astype(np.int32)
    power = np.where(amax <= 0.0, np.int32(-127), power)
    biased = np.clip(power + 127, 0, 254).astype(np.uint8)
    scale = np.ldexp(np.ones(amax.shape, dtype=np.float32), biased.astype(np.int32) - 127)
    return biased, scale


# Midpoints between the fp4 magnitudes. An exact midpoint whose lower code is odd
# moves to the even neighbor.
_FP4_EDGE = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)


def quantize_fp4(weight: np.ndarray) -> tuple[bytes, bytes]:
    """Pack ``[rows, inner]`` float32 as fp4. Inner is a multiple of 32.

    Element 0 is the low nibble. Bit 3 of the nibble is the sign. Rows are packed
    in tiles so a vocabulary matrix does not allocate one distance plane.
    """
    matrix = np.ascontiguousarray(weight, dtype=np.float32)
    _rows, inner = matrix.shape
    if inner % BLOCK:
        message = f"inner dimension {inner} is not a multiple of {BLOCK}"
        raise ValueError(message)
    data: list[bytes] = []
    scales: list[bytes] = []
    for start in range(0, matrix.shape[0], 4096):
        packed, biased = _quantize_fp4_tile(matrix[start : start + 4096])
        data.append(packed)
        scales.append(biased)
    return b"".join(data), b"".join(scales)


def _quantize_fp4_tile(matrix: np.ndarray) -> tuple[bytes, bytes]:
    rows, inner = matrix.shape
    values = matrix.reshape(rows, inner // BLOCK, BLOCK)
    amax = np.max(np.abs(values), axis=-1)
    biased, scale = biased_e8m0(amax, FP4_QMAX)
    scaled = values / scale[..., None]
    magnitude = np.abs(scaled).astype(np.float32, copy=False)
    choice = np.searchsorted(_FP4_EDGE, magnitude, side="left").astype(np.uint8)
    on_edge = (choice < 7) & (magnitude == _FP4_EDGE[np.minimum(choice, 6)])
    bump = on_edge & ((choice & 1) == 1)
    choice = np.where(bump, choice + 1, choice).astype(np.uint8)
    sign = np.signbit(scaled).astype(np.uint8) << np.uint8(3)
    nibbles = choice | sign
    packed = nibbles[..., 0::2] | (nibbles[..., 1::2] << np.uint8(4))
    return np.ascontiguousarray(packed).tobytes(), np.ascontiguousarray(biased).tobytes()


def dequant_fp4(packed: bytes, scales: bytes, rows: int, inner: int) -> np.ndarray:
    raw = np.frombuffer(packed, dtype=np.uint8).reshape(rows, inner // 2)
    codes = np.empty((rows, inner), dtype=np.uint8)
    codes[:, 0::2] = raw & 0xF
    codes[:, 1::2] = raw >> 4
    sign = np.where(codes & 0x8, np.float32(-1.0), np.float32(1.0))
    magnitude = FP4_MAGNITUDE[codes & 0x7].astype(np.float32)
    values = sign * magnitude
    factors = np.frombuffer(scales, dtype=np.uint8).astype(np.int32).reshape(rows, inner // BLOCK)
    restored = values.reshape(rows, inner // BLOCK, BLOCK)
    restored *= np.ldexp(np.ones(factors.shape, dtype=np.float32), factors - 127)[..., None]
    return restored.reshape(rows, inner)


def pack_fp4_matrix(
    weight: np.ndarray, *, rows: int | None = None, inner: int | None = None
) -> tuple[bytes, bytes, int, int]:
    """Pad K to a multiple of 256 and rows up to ``rows``.

    Returns the data plane, the scale plane, the stored row count, and the stored K.
    """
    matrix = np.ascontiguousarray(weight, dtype=np.float32)
    got_rows, got_inner = matrix.shape
    stored_inner = stored_k(got_inner if inner is None else inner)
    stored_rows = got_rows if rows is None else rows
    if stored_rows < got_rows or stored_inner < got_inner:
        message = "stored shape is smaller than the matrix"
        raise ValueError(message)
    if stored_rows != got_rows or stored_inner != got_inner:
        padded = np.zeros((stored_rows, stored_inner), dtype=np.float32)
        padded[:got_rows, :got_inner] = matrix
        matrix = padded
    data, scales = quantize_fp4(matrix)
    return data, scales, stored_rows, stored_inner


def fp4_image(weight: np.ndarray) -> bytes:
    """Data plane, then the E8M0 plane. Both start on a 128-byte boundary."""
    data, scales, _rows, _inner = pack_fp4_matrix(weight)
    pad = (128 - (len(data) % 128)) % 128
    return data + (b"\x00" * pad) + scales


def decode_e4m3(codes: np.ndarray) -> np.ndarray:
    """OCP E4M3FN. ``0x7E`` is 448. ``0x7F`` is NaN."""
    value = np.ascontiguousarray(codes, dtype=np.uint32)
    sign = np.where(value & 0x80, np.float32(-1.0), np.float32(1.0))
    exponent = (value >> 3) & 0xF
    mantissa = (value & 0x7).astype(np.float32)
    out = np.empty(value.shape, dtype=np.float32)
    sub = exponent == 0
    nan = (exponent == 0xF) & ((value & 0x7) == 0x7)
    normal = ~(sub | nan)
    out[sub] = sign[sub] * (mantissa[sub] / np.float32(8.0)) * np.float32(2.0**-6)
    out[nan] = np.float32(np.nan)
    out[normal] = (
        sign[normal]
        * (np.float32(1.0) + mantissa[normal] / np.float32(8.0))
        * np.exp2(exponent[normal].astype(np.float32) - np.float32(7.0))
    )
    return out


def dequant_e4m3_block(codes: np.ndarray, scale: int) -> np.ndarray:
    factor = np.ldexp(np.float32(1.0), int(scale) - 127)
    return decode_e4m3(codes).reshape(-1) * factor


def _e4m3_ladder() -> tuple[np.ndarray, np.ndarray]:
    codes = np.arange(256, dtype=np.uint8)
    values = decode_e4m3(codes)
    finite = np.isfinite(values)
    kept = codes[finite]
    ladder = values[finite].astype(np.float32)
    order = np.argsort(ladder, kind="mergesort")
    return kept[order], ladder[order]


_E4M3_CODE, _E4M3_VALUE = _e4m3_ladder()


def _nearest_e4m3(scaled: np.ndarray) -> np.ndarray:
    """Nearest finite E4M3FN code. An exact tie keeps the even mantissa."""
    flat = np.nan_to_num(
        np.ascontiguousarray(scaled, dtype=np.float32).reshape(-1),
        nan=0.0,
        posinf=FP8_QMAX,
        neginf=-FP8_QMAX,
    )
    flat = np.clip(flat, -FP8_QMAX, FP8_QMAX)
    pos = np.searchsorted(_E4M3_VALUE, flat, side="left")
    hi = np.clip(pos, 1, len(_E4M3_VALUE) - 1)
    lo = hi - 1
    dlo = np.abs(_E4M3_VALUE[lo] - flat)
    dhi = np.abs(_E4M3_VALUE[hi] - flat)
    clo = _E4M3_CODE[lo]
    chi = _E4M3_CODE[hi]
    prefer_lo = dlo < dhi
    tie = dlo == dhi
    lo_even = (clo & np.uint8(1)) == 0
    hi_even = (chi & np.uint8(1)) == 0
    both = lo_even == hi_even
    tie_lo = tie & ((lo_even & ~hi_even) | (both & (clo <= chi)))
    chosen = np.where(prefer_lo | tie_lo, clo, chi).astype(np.uint8)
    return chosen.reshape(scaled.shape)


def quantize_fp8(weight: np.ndarray) -> tuple[bytes, bytes]:
    """Pack ``[rows, inner]`` float32 as E4M3FN. Inner is a multiple of 32.

    ``0x7F`` is NaN and is not stored. Values past 448 saturate to 448.
    """
    matrix = np.ascontiguousarray(weight, dtype=np.float32)
    rows, inner = matrix.shape
    if inner % BLOCK:
        message = f"inner dimension {inner} is not a multiple of {BLOCK}"
        raise ValueError(message)
    values = matrix.reshape(rows, inner // BLOCK, BLOCK)
    amax = np.max(np.abs(values), axis=-1)
    biased, scale = biased_e8m0(amax, FP8_QMAX)
    codes = _nearest_e4m3(values / scale[..., None])
    data = np.ascontiguousarray(codes.reshape(rows, inner))
    return data.tobytes(), np.ascontiguousarray(biased).tobytes()


def dequant_fp8(data: bytes, scales: bytes, rows: int, inner: int) -> np.ndarray:
    codes = np.frombuffer(data, dtype=np.uint8).reshape(rows, inner)
    factors = np.frombuffer(scales, dtype=np.uint8).astype(np.int32).reshape(rows, inner // BLOCK)
    restored = decode_e4m3(codes).reshape(rows, inner // BLOCK, BLOCK)
    restored *= np.ldexp(np.ones(factors.shape, dtype=np.float32), factors - 127)[..., None]
    return restored.reshape(rows, inner)


def pack_fp8_matrix(weight: np.ndarray) -> tuple[bytes, bytes, int, int]:
    """Pad K to a multiple of 128. Returns data, scales, rows, and stored K."""
    matrix = np.ascontiguousarray(weight, dtype=np.float32)
    rows, inner = matrix.shape
    if inner < 1:
        message = f"inner dimension {inner} is empty"
        raise ValueError(message)
    stored = ((inner + 127) // 128) * 128
    if stored != inner:
        padded = np.zeros((rows, stored), dtype=np.float32)
        padded[:, :inner] = matrix
        matrix = padded
    data, scales = quantize_fp8(matrix)
    return data, scales, rows, stored


def fp8_image(weight: np.ndarray) -> bytes:
    """Data plane, then the E8M0 plane. Both start on a 128-byte boundary."""
    data, scales, _rows, _inner = pack_fp8_matrix(weight)
    pad = (128 - (len(data) % 128)) % 128
    return data + (b"\x00" * pad) + scales


def expert_bytes(width: int) -> int:
    """Byte length of one routed expert image."""
    if width == 4:
        return EXPERT_IMAGE
    if width == 8:
        return EXPERT_FP8
    message = f"expert width {width} is not 4 or 8"
    raise ValueError(message)


def expert_offsets(widths: np.ndarray) -> np.ndarray:
    """Pack each layer in expert-index order. Both image lengths divide by 128."""
    matrix = np.ascontiguousarray(widths, dtype=np.uint8)
    offsets = np.zeros(matrix.shape, dtype=np.uint64)
    for layer in range(matrix.shape[0]):
        cursor = 0
        for expert in range(matrix.shape[1]):
            offsets[layer, expert] = cursor
            cursor += expert_bytes(int(matrix[layer, expert]))
    return offsets


def expert_file_bytes(widths: np.ndarray, offsets: np.ndarray) -> list[int]:
    """Byte length of each layer file. Images stay 128-byte aligned and do not overlap."""
    matrix = np.ascontiguousarray(widths, dtype=np.uint8)
    places = np.ascontiguousarray(offsets, dtype=np.uint64)
    if matrix.shape != places.shape:
        message = "expert widths and offsets have different shapes"
        raise ValueError(message)
    sizes: list[int] = []
    for layer in range(matrix.shape[0]):
        spans: list[tuple[int, int]] = []
        end = 0
        for expert in range(matrix.shape[1]):
            off = int(places[layer, expert])
            if off % 128:
                message = f"expert {layer}:{expert} offset {off} is not 128-byte aligned"
                raise ValueError(message)
            size = expert_bytes(int(matrix[layer, expert]))
            spans.append((off, size))
            end = max(end, off + size)
        cursor = 0
        for off, size in sorted(spans):
            if off < cursor:
                message = f"layer {layer} expert images overlap"
                raise ValueError(message)
            cursor = off + size
        sizes.append(end)
    return sizes
