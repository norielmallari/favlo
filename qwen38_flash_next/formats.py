"""E8M0 scales, packed integers, and the int4 record image."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np

BLOCK = 32
SENTINEL = 0x5A
HOST_BYTES = 4096
PAGE = 16384
# statvfs f_bsize on this APFS volume, and the transfer size that fills the SSD.
PREFERRED_IO = 1 << 20
NGRAM_SLOT = 128
NGRAM_WIDTH = 160
IO_CHUNK = 256 * 1024 * 1024
RECORD_VERSION = 4
# Hugging Face model name, then -madmetal-, then each tier. s is the int8 pin,
# h is the int4 keep set, c is the int2 complement. Five int8 experts are lifted
# out of the 384 keep-set, so four layers have a shorter hot list.
DEFAULT_RECORD = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "Qwen3.8-Flash-Next-madmetal-s5-int8-h384-int4-c128-int2"
)

# MTLTensorDataType values from the macOS 27 SDK headers.
TENSOR_INT8 = 45
TENSOR_INT4 = 143
TENSOR_E8M0 = 145
TENSOR_E5M2 = 141
TENSOR_E4M3 = 142
TENSOR_INT2 = 150
PLANE_DATA = 0
PLANE_SCALES = 1

# fmt: off
INT4_ORDER_PATTERN = [0, 1, 2, 4, 8, 15, 7, 3, 5, 6, 9, 10, 11, 12, 13, 14] * 2
INT2_ORDER_PATTERN = [0, 1, 2, 3, 1, 0, 3, 2, 2, 2, 1, 1, 0, 3, 0, 1] * 2
# Packed bytes the device stores. Nibble and crumb order are not in this layout;
# the shader unpack decides which bits are element 0.
INT4_PACKED = bytes([
    0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
    0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE,
])
INT2_PACKED = bytes([0x1B, 0xE4, 0xA7, 0x5B, 0x0F, 0xF0, 0x36, 0xC9])
INT8_PATTERN = [
    -128, -127, -2, -1, 0, 1, 2, 126, 127, 50, -50, 64, -64, 7, -8, 15,
    -15, 32, -32, 100, -100, 3, -3, 9, -9, 16, -16, 40, -40, 5, -5, 8,
]
FP8_PATTERN = [
    0x00, 0x80, 0x38, 0x40, 0x3C, 0x3E, 0x48, 0xC8,
    0x1C, 0x9C, 0x04, 0x01, 0x7E, 0xFE, 0x7F, 0xFF,
    0x30, 0xB0, 0x28, 0xA8, 0x18, 0x98, 0x08, 0x88,
    0x44, 0xC4, 0x50, 0xD0, 0x60, 0xE0, 0x70, 0xF0,
]
# fmt: on


def pack_int4(codes: Sequence[int], *, low_nibble_first: bool) -> bytes:
    if len(codes) % 2:
        message = "int4 packing needs an even number of codes"
        raise ValueError(message)
    out = bytearray()
    for index in range(0, len(codes), 2):
        first = codes[index] & 0xF
        second = codes[index + 1] & 0xF
        if low_nibble_first:
            out.append(first | (second << 4))
        else:
            out.append((first << 4) | second)
    return bytes(out)


def unpack_int4(raw: bytes, *, low_nibble_first: bool, count: int = BLOCK) -> list[int]:
    codes: list[int] = []
    for byte in raw:
        if low_nibble_first:
            pair = (byte & 0xF, (byte >> 4) & 0xF)
        else:
            pair = ((byte >> 4) & 0xF, byte & 0xF)
        codes.extend(pair)
        if len(codes) >= count:
            return codes[:count]
    return codes


def pack_int2(codes: Sequence[int], *, low_bits_first: bool) -> bytes:
    if len(codes) % 4:
        message = "int2 packing needs a multiple of four codes"
        raise ValueError(message)
    out = bytearray()
    for index in range(0, len(codes), 4):
        group = [codes[index + offset] & 0x3 for offset in range(4)]
        byte = 0
        for offset, code in enumerate(group):
            shift = 2 * offset if low_bits_first else 6 - 2 * offset
            byte |= code << shift
        out.append(byte)
    return bytes(out)


def unpack_int2(raw: bytes, *, low_bits_first: bool, count: int = BLOCK) -> list[int]:
    codes: list[int] = []
    for byte in raw:
        for offset in range(4):
            shift = 2 * offset if low_bits_first else 6 - 2 * offset
            codes.append((byte >> shift) & 0x3)
            if len(codes) >= count:
                return codes[:count]
    return codes


def sign_extend(code: int, bits: int) -> int:
    sign = 1 << (bits - 1)
    masked = code & ((1 << bits) - 1)
    return masked - (1 << bits) if masked & sign else masked


def decode_e8m0(byte: int) -> float:
    if byte & 0xFF == 0xFF:
        return math.nan
    return 2.0 ** ((byte & 0xFF) - 127)


# Positive endpoint of the signed code. int4 is −8..7, int8 is −128..127, int2 is −2..1.
QMAX = {2: 1, 4: 7, 8: 127}


def ceil_log2(value: float) -> int:
    """Smallest integer p such that 2^p >= value, for value > 0."""
    mantissa, exponent = math.frexp(value)
    if mantissa == 0.5:
        return exponent - 1
    return exponent


def e8m0_scale_byte(amax: float, qmax: int) -> int:
    """Power of two at or above amax/qmax, stored with bias 127. Never 0xFF."""
    if qmax <= 0:
        message = "qmax must be positive"
        raise ValueError(message)
    if not math.isfinite(amax) or amax <= 0.0:
        return 0
    power = ceil_log2(amax / qmax)
    biased = power + 127
    if biased < 0:
        return 0
    if biased > 254:
        return 254
    return biased


def quantize_block(values: Sequence[float], bits: int) -> tuple[list[int], int]:
    """One block of 32. Returns signed codes and the E8M0 scale byte.

    The scale rounds up so every finite input fits in the signed range.
    Codes use round-ties-to-even, then saturate.
    """
    if bits not in QMAX:
        message = f"unsupported element width: {bits}"
        raise ValueError(message)
    if len(values) != BLOCK:
        message = f"a block is {BLOCK} values"
        raise ValueError(message)
    qmax = QMAX[bits]
    qmin = -(1 << (bits - 1))
    amax = 0.0
    for value in values:
        if not math.isfinite(value):
            message = "a block value must be finite"
            raise ValueError(message)
        magnitude = abs(value)
        if magnitude > amax:
            amax = magnitude
    scale_byte = e8m0_scale_byte(amax, qmax)
    scale = decode_e8m0(scale_byte)
    codes: list[int] = []
    for value in values:
        # Python 3 round is ties-to-even, which is the MX rounding mode.
        code = int(round(value / scale))
        if code < qmin:
            code = qmin
        elif code > qmax:
            code = qmax
        codes.append(code)
    return codes, scale_byte


def decode_e4m3(byte: int) -> float:
    """OCP E4M3FN. Exponent 15 is finite except mantissa 7, which is NaN."""
    value = byte & 0xFF
    sign = -1.0 if value & 0x80 else 1.0
    exponent = (value >> 3) & 0xF
    mantissa = value & 0x7
    if exponent == 0:
        return sign * (mantissa / 8.0) * 2.0**-6
    if exponent == 0xF and mantissa == 0x7:
        return math.copysign(math.nan, sign)
    return sign * (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)


def decode_e5m2(byte: int) -> float:
    """OCP E5M2. Exponent 31 is inf when the mantissa is 0, otherwise NaN."""
    value = byte & 0xFF
    sign = -1.0 if value & 0x80 else 1.0
    exponent = (value >> 2) & 0x1F
    mantissa = value & 0x3
    if exponent == 0:
        return sign * (mantissa / 4.0) * 2.0**-14
    if exponent == 0x1F:
        if mantissa == 0:
            return math.copysign(math.inf, sign)
        return math.copysign(math.nan, sign)
    return sign * (1.0 + mantissa / 4.0) * 2.0 ** (exponent - 15)


def _finite_equal(left: float, right: float) -> bool:
    if math.isnan(left) and math.isnan(right):
        return True
    if math.isinf(left) or math.isinf(right):
        return left == right
    return abs(left - right) <= 1e-4 * max(1.0, abs(right))


def json_number(value: float) -> float | str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


def quantize_matrix(weight: np.ndarray) -> tuple[bytes, bytes]:
    """Pack [rows, inner] float32 as int4, low nibble first, with one E8M0 byte per 32."""
    matrix = np.ascontiguousarray(weight, dtype=np.float32)
    rows, inner = matrix.shape
    if inner % BLOCK:
        message = f"inner dimension {inner} is not a multiple of {BLOCK}"
        raise ValueError(message)
    blocks = matrix.reshape(rows, inner // BLOCK, BLOCK).astype(np.float64)
    amax = np.max(np.abs(blocks), axis=-1)
    ratio = amax / QMAX[4]
    mantissa, exponent = np.frexp(np.maximum(ratio, np.finfo(np.float64).tiny))
    power = exponent.astype(np.int32) - (mantissa == 0.5).astype(np.int32)
    power = np.where(amax <= 0.0, np.int32(-127), power)
    biased = np.clip(power + 127, 0, 254).astype(np.uint8)
    scale = np.ldexp(np.ones(amax.shape, dtype=np.float64), biased.astype(np.int32) - 127)
    codes = np.rint(blocks / scale[..., None])
    codes = np.clip(codes, -8, 7).astype(np.int32)
    nibbles = (codes & 0xF).astype(np.uint8)
    packed = nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)
    return (
        np.ascontiguousarray(packed).tobytes(),
        np.ascontiguousarray(biased).tobytes(),
    )


def dequant_int8(packed: bytes, scales: bytes, rows: int, inner: int) -> np.ndarray:
    codes = np.frombuffer(packed, dtype=np.int8).reshape(rows, inner).astype(np.float32)
    factors = np.frombuffer(scales, dtype=np.uint8).astype(np.int32).reshape(rows, inner // BLOCK)
    restored = codes.reshape(rows, inner // BLOCK, BLOCK)
    restored *= np.ldexp(np.ones(factors.shape, dtype=np.float32), factors - 127)[..., None]
    return restored.reshape(rows, inner)


def dequant_int4(packed: bytes, scales: bytes, rows: int, inner: int) -> np.ndarray:
    raw = np.frombuffer(packed, dtype=np.uint8).reshape(rows, inner // 2)
    codes = np.empty((rows, inner), dtype=np.int32)
    codes[:, 0::2] = raw & 0xF
    codes[:, 1::2] = raw >> 4
    codes = np.where(codes > 7, codes - 16, codes)
    factors = np.frombuffer(scales, dtype=np.uint8).astype(np.int32).reshape(rows, inner // BLOCK)
    restored = codes.reshape(rows, inner // BLOCK, BLOCK).astype(np.float32)
    restored *= np.ldexp(np.ones(factors.shape, dtype=np.float32), factors - 127)[..., None]
    return restored.reshape(rows, inner)


def stored_k_int4(inner: int, *, align_k: bool) -> int:
    """int4 row bytes must be a multiple of 128, so stored K is a multiple of 256."""
    if inner % BLOCK:
        message = f"inner dimension {inner} is not a multiple of {BLOCK}"
        raise ValueError(message)
    if not align_k:
        return inner
    return ((inner + 255) // 256) * 256


def int4_plane_sizes(rows: int, stored_inner: int) -> tuple[int, int]:
    return rows * (stored_inner // 2), rows * (stored_inner // BLOCK)


def quantize_rows(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pack [rows, stored_k] float32. Same rule as ``quantize_matrix``."""
    rows, inner = block.shape
    blocks = block.reshape(rows, inner // BLOCK, BLOCK)
    amax = np.max(np.abs(blocks), axis=-1).astype(np.float64)
    ratio = amax / QMAX[4]
    mantissa, exponent = np.frexp(np.maximum(ratio, np.finfo(np.float64).tiny))
    power = exponent.astype(np.int32) - (mantissa == 0.5).astype(np.int32)
    power = np.where(amax <= 0.0, np.int32(-127), power)
    biased = np.clip(power + 127, 0, 254).astype(np.uint8)
    scale = np.ldexp(np.ones(amax.shape, dtype=np.float32), biased.astype(np.int32) - 127)
    codes = np.rint(blocks / scale[..., None])
    codes = np.clip(codes, -8, 7).astype(np.int32)
    nibbles = (codes & 0xF).astype(np.uint8)
    packed = nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)
    return np.ascontiguousarray(packed), np.ascontiguousarray(biased)


def quantize_int4_image(weight: np.ndarray, *, align_k: bool = True) -> tuple[bytes, int]:
    """Data plane, then tight E8M0 scales. ``align_k`` pads K out to a multiple of 256."""
    matrix = np.ascontiguousarray(weight, dtype=np.float32)
    rows, inner = matrix.shape
    stored = stored_k_int4(inner, align_k=align_k)
    if stored != inner:
        matrix = np.pad(matrix, ((0, 0), (0, stored - inner)))
    data_n, scale_n = int4_plane_sizes(rows, stored)
    image = bytearray(data_n + scale_n)
    step = 256
    data_row = stored // 2
    scale_row = stored // BLOCK
    for start in range(0, rows, step):
        stop = min(start + step, rows)
        packed, scales = quantize_rows(matrix[start:stop])
        count = stop - start
        image[start * data_row : stop * data_row] = packed.reshape(count, data_row).tobytes()
        base = data_n + start * scale_row
        image[base : base + count * scale_row] = scales.reshape(count, scale_row).tobytes()
    return bytes(image), stored


def ngram_slots(weight: np.ndarray) -> bytes:
    """One 128-byte slot per row: 80 int4 bytes, five E8M0 bytes, then padding."""
    rows, inner = weight.shape
    if inner != NGRAM_WIDTH:
        message = f"n-gram width {inner} is not {NGRAM_WIDTH}"
        raise ValueError(message)
    image, stored = quantize_int4_image(weight, align_k=False)
    if stored != NGRAM_WIDTH:
        message = f"n-gram stored K {stored} is not {NGRAM_WIDTH}"
        raise RuntimeError(message)
    data_n, _scale_n = int4_plane_sizes(rows, stored)
    data = np.frombuffer(image[:data_n], dtype=np.uint8).reshape(rows, 80)
    scales = np.frombuffer(image[data_n:], dtype=np.uint8).reshape(rows, 5)
    slots = np.zeros((rows, NGRAM_SLOT), dtype=np.uint8)
    slots[:, :80] = data
    slots[:, 80:85] = scales
    return np.ascontiguousarray(slots).tobytes()


def expert_live_bytes() -> int:
    """Middle-tier gate_up image followed by the down image."""
    gate_up = sum(int4_plane_sizes(1280, stored_k_int4(2560, align_k=True)))
    down = sum(int4_plane_sizes(2560, stored_k_int4(640, align_k=True)))
    return gate_up + down


def expert_slot_bytes() -> int:
    """On-disk stride of one hot expert. The live int4 image, so a run contains no hole."""
    return expert_live_bytes()


def stored_k_int2(inner: int) -> int:
    """int2 row bytes must be a multiple of 128, so stored K is a multiple of 512."""
    if inner % BLOCK:
        message = f"inner dimension {inner} is not a multiple of {BLOCK}"
        raise ValueError(message)
    return ((inner + 511) // 512) * 512


def int2_plane_sizes(rows: int, stored_inner: int) -> tuple[int, int]:
    return rows * (stored_inner // 4), rows * (stored_inner // BLOCK)


def quantize_int2_rows(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pack [rows, stored_k] float32 as int2, low bits first, one E8M0 byte per 32."""
    rows, inner = block.shape
    if inner % BLOCK:
        message = f"inner dimension {inner} is not a multiple of {BLOCK}"
        raise ValueError(message)
    blocks = block.reshape(rows, inner // BLOCK, BLOCK)
    amax = np.max(np.abs(blocks), axis=-1).astype(np.float64)
    ratio = amax / QMAX[2]
    mantissa, exponent = np.frexp(np.maximum(ratio, np.finfo(np.float64).tiny))
    power = exponent.astype(np.int32) - (mantissa == 0.5).astype(np.int32)
    power = np.where(amax <= 0.0, np.int32(-127), power)
    biased = np.clip(power + 127, 0, 254).astype(np.uint8)
    scale = np.ldexp(np.ones(amax.shape, dtype=np.float32), biased.astype(np.int32) - 127)
    codes = np.rint(blocks / scale[..., None])
    codes = np.clip(codes, -2, 1).astype(np.int32)
    bits = (codes & 0x3).astype(np.uint8)
    packed = (
        bits[..., 0::4]
        | (bits[..., 1::4] << 2)
        | (bits[..., 2::4] << 4)
        | (bits[..., 3::4] << 6)
    )
    return np.ascontiguousarray(packed), np.ascontiguousarray(biased)


def quantize_int2_image(weight: np.ndarray, *, align_k: bool = True) -> tuple[bytes, int]:
    """Data plane, then tight E8M0 scales. ``align_k`` pads K out to a multiple of 512."""
    matrix = np.ascontiguousarray(weight, dtype=np.float32)
    rows, inner = matrix.shape
    stored = stored_k_int2(inner) if align_k else inner
    if stored % BLOCK:
        message = f"stored K {stored} is not a multiple of {BLOCK}"
        raise ValueError(message)
    if stored != inner:
        matrix = np.pad(matrix, ((0, 0), (0, stored - inner)))
    data_n, scale_n = int2_plane_sizes(rows, stored)
    image = bytearray(data_n + scale_n)
    step = 256
    data_row = stored // 4
    scale_row = stored // BLOCK
    for start in range(0, rows, step):
        stop = min(start + step, rows)
        packed, scales = quantize_int2_rows(matrix[start:stop])
        count = stop - start
        image[start * data_row : stop * data_row] = packed.reshape(count, data_row).tobytes()
        base = data_n + start * scale_row
        image[base : base + count * scale_row] = scales.reshape(count, scale_row).tobytes()
    return bytes(image), stored


def dequant_int2(packed: bytes, scales: bytes, rows: int, inner: int) -> np.ndarray:
    raw = np.frombuffer(packed, dtype=np.uint8).reshape(rows, inner // 4)
    codes = np.empty((rows, inner), dtype=np.int32)
    codes[:, 0::4] = raw & 0x3
    codes[:, 1::4] = (raw >> 2) & 0x3
    codes[:, 2::4] = (raw >> 4) & 0x3
    codes[:, 3::4] = (raw >> 6) & 0x3
    codes = np.where(codes > 1, codes - 4, codes)
    factors = np.frombuffer(scales, dtype=np.uint8).astype(np.int32).reshape(rows, inner // BLOCK)
    restored = codes.reshape(rows, inner // BLOCK, BLOCK).astype(np.float32)
    restored *= np.ldexp(np.ones(factors.shape, dtype=np.float32), factors - 127)[..., None]
    return restored.reshape(rows, inner)


def cold_live_bytes() -> int:
    """Narrow gate_up image followed by the down image."""
    gate_up = sum(int2_plane_sizes(1280, stored_k_int2(2560)))
    down = sum(int2_plane_sizes(2560, stored_k_int2(640)))
    return gate_up + down


def cold_slot_bytes() -> int:
    """On-disk stride of one cold expert. The live image rounded up to a page."""
    live = cold_live_bytes()
    return (live + PAGE - 1) & ~(PAGE - 1)


def stored_k_int8(inner: int) -> int:
    """int8 row bytes must be a multiple of 128, so stored K is a multiple of 128."""
    if inner % BLOCK:
        message = f"inner dimension {inner} is not a multiple of {BLOCK}"
        raise ValueError(message)
    return ((inner + 127) // 128) * 128


def int8_plane_sizes(rows: int, stored_inner: int) -> tuple[int, int]:
    return rows * stored_inner, rows * (stored_inner // BLOCK)


def quantize_int8_rows(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pack [rows, stored_k] float32 as signed int8, one E8M0 byte per 32."""
    rows, inner = block.shape
    if inner % BLOCK:
        message = f"inner dimension {inner} is not a multiple of {BLOCK}"
        raise ValueError(message)
    blocks = block.reshape(rows, inner // BLOCK, BLOCK)
    amax = np.max(np.abs(blocks), axis=-1).astype(np.float64)
    ratio = amax / QMAX[8]
    mantissa, exponent = np.frexp(np.maximum(ratio, np.finfo(np.float64).tiny))
    power = exponent.astype(np.int32) - (mantissa == 0.5).astype(np.int32)
    power = np.where(amax <= 0.0, np.int32(-127), power)
    biased = np.clip(power + 127, 0, 254).astype(np.uint8)
    scale = np.ldexp(np.ones(amax.shape, dtype=np.float32), biased.astype(np.int32) - 127)
    codes = np.rint(blocks / scale[..., None])
    codes = np.clip(codes, -128, 127).astype(np.int8)
    return np.ascontiguousarray(codes.view(np.uint8)), np.ascontiguousarray(biased)


def quantize_int8_image(weight: np.ndarray, *, align_k: bool = True) -> tuple[bytes, int]:
    """Data plane, then tight E8M0 scales. ``align_k`` pads K out to a multiple of 128."""
    matrix = np.ascontiguousarray(weight, dtype=np.float32)
    rows, inner = matrix.shape
    stored = stored_k_int8(inner) if align_k else inner
    if stored % BLOCK:
        message = f"stored K {stored} is not a multiple of {BLOCK}"
        raise ValueError(message)
    if stored != inner:
        matrix = np.pad(matrix, ((0, 0), (0, stored - inner)))
    data_n, scale_n = int8_plane_sizes(rows, stored)
    image = bytearray(data_n + scale_n)
    step = 256
    for start in range(0, rows, step):
        stop = min(start + step, rows)
        packed, scales = quantize_int8_rows(matrix[start:stop])
        count = stop - start
        image[start * stored : stop * stored] = packed.reshape(count, stored).tobytes()
        base = data_n + start * (stored // BLOCK)
        image[base : base + count * (stored // BLOCK)] = scales.reshape(
            count, stored // BLOCK
        ).tobytes()
    return bytes(image), stored


def super_live_bytes() -> int:
    """Wide gate_up image followed by the down image."""
    gate_up = sum(int8_plane_sizes(1280, stored_k_int8(2560)))
    down = sum(int8_plane_sizes(2560, stored_k_int8(640)))
    return gate_up + down


def super_slot_bytes() -> int:
    """On-disk stride of one super expert. The live image rounded up to a page."""
    live = super_live_bytes()
    return (live + PAGE - 1) & ~(PAGE - 1)


class ExpertPlanes(NamedTuple):
    """Byte offsets of one concatenated gate_up + down image."""

    bits: int
    gate_rows: int
    gate_inner: int
    gate_scale: int
    down_inner: int
    down_weight: int
    down_scale: int


def expert_planes(bits: int, hidden: int = 2560, moe: int = 640) -> ExpertPlanes:
    """Layout of the image the expert shaders index. Gate data starts at 0."""
    gate_rows = 2 * moe
    if bits == 4:
        gate_inner = stored_k_int4(hidden, align_k=True)
        gate_data, gate_scales = int4_plane_sizes(gate_rows, gate_inner)
        down_inner = stored_k_int4(moe, align_k=True)
        down_data, _down_scales = int4_plane_sizes(hidden, down_inner)
    elif bits == 2:
        gate_inner = stored_k_int2(hidden)
        gate_data, gate_scales = int2_plane_sizes(gate_rows, gate_inner)
        down_inner = stored_k_int2(moe)
        down_data, _down_scales = int2_plane_sizes(hidden, down_inner)
    elif bits == 8:
        gate_inner = stored_k_int8(hidden)
        gate_data, gate_scales = int8_plane_sizes(gate_rows, gate_inner)
        down_inner = stored_k_int8(moe)
        down_data, _down_scales = int8_plane_sizes(hidden, down_inner)
    else:
        message = f"expert plane width {bits} is not 2, 4, or 8"
        raise ValueError(message)
    gate_image = gate_data + gate_scales
    return ExpertPlanes(
        bits,
        gate_rows,
        gate_inner,
        gate_data,
        down_inner,
        gate_image,
        gate_image + down_data,
    )
