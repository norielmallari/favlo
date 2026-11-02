"""Weight planes for Gemma 4.

Resident matrices, including the token embedding, are fp8 E4M3. Routed
experts are fp4 E2M1 or fp8 E4M3. Both planes store one E8M0 scale per 32
values. fp4 K is a multiple of 256 and fp8 K is a multiple of 128. The fp4
encoder tries the E8M0 exponents around the peak and keeps the one whose
nearest E2M1 codes have the smallest squared error. The fp8 encoder uses
the block peak. A zero block stores exponent 0. The scale row is packed
tightly, one byte per 32 K.

The q4_0 encoder remains for the lattice tests. A block is 32 values. The
scale is the QAT lattice step, stored with the ggml sign so the peak maps
to code −8, and rounded to bfloat16. A block that is not on that lattice
falls back to the signed peak divided by −8. Codes are −8..7. The low
nibble is element ``i`` and the high nibble is element ``i + 16``. The
packed record does not store q4_0.
"""

from __future__ import annotations

import numpy as np

from gemma4_26b_a4b.geometry import EXPERT_WIDTH, HIDDEN

FP4 = "fp4"
FP8 = "fp8"
Q40 = "q4_0"
Q40_BLOCK = 18
FP4_QMAX = 6.0
FP8_QMAX = 448.0
_FP4_MAGNITUDE = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
_FP4_EDGES = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float64)


def _fp8_tables() -> tuple[np.ndarray, np.ndarray]:
    codes: list[int] = []
    magnitudes: list[float] = []
    for code in range(127):
        exponent = (code >> 3) & 0xF
        mantissa = code & 0x7
        if exponent == 0:
            magnitude = (mantissa / 8.0) * 2.0**-6
        else:
            magnitude = (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)
        codes.append(code)
        magnitudes.append(magnitude)
    return np.array(codes, dtype=np.uint8), np.array(magnitudes, dtype=np.float64)


_FP8_CODE, _FP8_MAGNITUDE = _fp8_tables()


def align_k(k: int, plane: str) -> int:
    if plane == Q40:
        return ((k + 31) // 32) * 32
    if plane == FP8:
        return ((k + 127) // 128) * 128
    if plane == FP4:
        return ((k + 255) // 256) * 256
    message = f"unknown plane {plane}"
    raise ValueError(message)


def scale_stride(k_padded: int) -> int:
    """Bytes in one scale row. ``tensor_inline`` reads this tightly, one byte per 32 K."""
    return k_padded // 32


def matrix_nbytes(rows: int, k: int, plane: str) -> int:
    padded = align_k(k, plane)
    if plane == Q40:
        return rows * (padded // 32) * Q40_BLOCK
    data = rows * (padded // 2) if plane == FP4 else rows * padded
    return data + rows * scale_stride(padded)


def dequant_nbytes(rows: int, k: int) -> int:
    """Float16 bytes of one q4_0 matrix after the runtime expand."""
    return rows * align_k(k, Q40) * 2


def _ceil_log2(ratio: np.ndarray) -> np.ndarray:
    mantissa, exponent = np.frexp(ratio)
    return np.where(mantissa == 0.5, exponent - 1, exponent)


def _block_scale(weight: np.ndarray, qmax: float) -> tuple[np.ndarray, np.ndarray]:
    rows, padded = weight.shape
    blocks = weight.reshape(rows, padded // 32, 32)
    peak = np.max(np.abs(blocks), axis=-1)
    ratio = np.maximum(peak / qmax, np.finfo(np.float64).tiny)
    unbiased = np.where(peak == 0.0, -127, _ceil_log2(ratio.astype(np.float64)))
    unbiased = np.clip(unbiased, -127, 127).astype(np.int16)
    stored = (unbiased + 127).astype(np.uint8)
    scale = np.exp2(unbiased.astype(np.float64))
    return stored, scale


def _fp4_index(unit: np.ndarray) -> np.ndarray:
    """Nearest magnitude. An exact midpoint keeps the even index."""
    absolute = np.abs(unit).astype(np.float64, copy=False)
    index = np.digitize(absolute, _FP4_EDGES, right=True).astype(np.int16)
    for lower, edge in enumerate(_FP4_EDGES):
        if lower % 2 == 0:
            continue
        index = np.where(np.isclose(absolute, float(edge)), index + 1, index)
    return np.clip(index, 0, 7).astype(np.uint8)


def _fp8_code(unit: np.ndarray) -> np.ndarray:
    flat = np.abs(unit).reshape(-1).astype(np.float64, copy=False)
    flat = np.minimum(flat, FP8_QMAX)
    position = np.searchsorted(_FP8_MAGNITUDE, flat, side="left")
    position = np.clip(position, 1, len(_FP8_MAGNITUDE) - 1)
    left = position - 1
    right = np.minimum(position, len(_FP8_MAGNITUDE) - 1)
    left_distance = np.abs(flat - _FP8_MAGNITUDE[left])
    right_distance = np.abs(flat - _FP8_MAGNITUDE[right])
    left_even = (_FP8_CODE[left] & 1) == 0
    right_even = (_FP8_CODE[right] & 1) == 0
    tie = np.isclose(left_distance, right_distance)
    choose_left = np.where(tie, left_even >= right_even, left_distance < right_distance)
    chosen = np.where(choose_left, left, right)
    codes = _FP8_CODE[chosen]
    sign = np.where(unit.reshape(-1) < 0.0, np.uint8(0x80), np.uint8(0))
    return (codes | sign).reshape(unit.shape)


def _f32_to_bf16(values: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype(np.uint16)


def _bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    wide = np.ascontiguousarray(bits, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
    return wide.view(np.float32)


def _pack_q4_0(weight: np.ndarray) -> bytes:
    """Pack q4_0. A QAT block keeps the lattice step, not absmax/8."""
    rows, k = weight.shape
    padded = align_k(k, Q40)
    block = np.zeros((rows, padded), dtype=np.float32)
    block[:, :k] = weight
    groups = block.reshape(-1, 32)
    count = groups.shape[0]
    absmax = np.max(np.abs(groups), axis=1)
    taken = np.zeros(count, dtype=bool)
    scale = np.zeros(count, dtype=np.float32)
    signed = np.zeros((count, 32), dtype=np.float32)
    for extreme in range(8, 0, -1):
        step = absmax / np.float32(extreme)
        safe = np.where(step == 0.0, np.float32(1.0), step)
        trial = -safe
        codes = np.clip(np.rint(groups / trial[:, None]), -8, 7)
        recon = codes * trial[:, None]
        error = np.max(np.abs(groups - recon), axis=1)
        accept = (error <= (np.float32(0.075) * safe)) & ~taken & (absmax > 0.0)
        if not np.any(accept):
            continue
        numer = np.sum(codes * groups, axis=1)
        denom = np.sum(codes * codes, axis=1)
        refined = np.where(denom > 0.0, numer / np.maximum(denom, np.float32(1e-12)), trial)
        refined = _bf16_to_f32(_f32_to_bf16(refined))
        refined_safe = np.where(refined == 0.0, np.float32(1.0), refined)
        codes = np.clip(np.rint(groups / refined_safe[:, None]), -8, 7)
        scale = np.where(accept, refined, scale)
        signed = np.where(accept[:, None], codes, signed)
        taken |= accept
    missed = ~taken
    if np.any(missed):
        index = np.argmax(np.abs(groups), axis=1)
        peak = groups[np.arange(count), index]
        fallback = _bf16_to_f32(_f32_to_bf16(peak / np.float32(-8.0)))
        fallback_safe = np.where(fallback == 0.0, np.float32(1.0), fallback)
        unsigned = np.trunc(groups / fallback_safe[:, None] + np.float32(8.5))
        unsigned = np.clip(unsigned, 0, 15)
        scale = np.where(missed, fallback, scale)
        signed = np.where(missed[:, None], unsigned - np.float32(8.0), signed)
    scale = np.where(absmax == 0.0, np.float32(0.0), scale)
    signed = np.where((absmax == 0.0)[:, None], np.float32(0.0), signed)
    nibbles = np.clip(signed + np.float32(8.0), 0, 15).astype(np.uint8)
    nibbles = nibbles.reshape(rows, padded // 32, 32)
    scale = scale.reshape(rows, padded // 32)
    packed = np.empty((rows, padded // 32, Q40_BLOCK), dtype=np.uint8)
    packed[:, :, :2] = (
        np.ascontiguousarray(_f32_to_bf16(scale)).view(np.uint8).reshape(rows, padded // 32, 2)
    )
    packed[:, :, 2:] = nibbles[:, :, :16] | (nibbles[:, :, 16:] << 4)
    return packed.tobytes()


def _restore_q4_0(blob: bytes, rows: int, k: int) -> np.ndarray:
    padded = align_k(k, Q40)
    raw = np.frombuffer(blob, dtype=np.uint8).reshape(rows, padded // 32, Q40_BLOCK)
    scale_bits = raw[:, :, 0].astype(np.uint16) | (raw[:, :, 1].astype(np.uint16) << np.uint16(8))
    scale = _bf16_to_f32(scale_bits)
    qs = raw[:, :, 2:]
    codes = np.empty((rows, padded // 32, 32), dtype=np.float32)
    codes[:, :, :16] = (qs & 0x0F).astype(np.float32)
    codes[:, :, 16:] = (qs >> 4).astype(np.float32)
    values = (codes - np.float32(8.0)) * scale[..., None]
    return values.reshape(rows, padded)[:, :k].copy()


def fp4_parts(rows: int, k: int) -> tuple[int, int]:
    """Code bytes and scale bytes of one fp4 matrix."""
    padded = align_k(k, FP4)
    return rows * (padded // 2), rows * scale_stride(padded)


def fp8_parts(rows: int, k: int) -> tuple[int, int]:
    """Code bytes and scale bytes of one fp8 matrix."""
    padded = align_k(k, FP8)
    return rows * padded, rows * scale_stride(padded)


def expert_planes(plane: str) -> tuple[int, int, int, int, int]:
    """Gate data, gate scale, down data, down scale, and the image length.

    The image is one expert: gate-up, then down. Each plane starts on a
    128-byte boundary. ``plane`` is ``fp4`` or ``fp8``.
    """
    parts = fp8_parts if plane == FP8 else fp4_parts
    if plane not in {FP4, FP8}:
        message = f"unknown expert plane {plane}"
        raise ValueError(message)
    gate_data, gate_scale = parts(2 * EXPERT_WIDTH, HIDDEN)
    down_data, down_scale = parts(HIDDEN, EXPERT_WIDTH)
    total = gate_data + gate_scale + down_data + down_scale
    return gate_data, gate_scale, down_data, down_scale, total


def expert_bytes(width: int) -> int:
    """Byte length of one routed expert. ``4`` is fp4 and ``8`` is fp8."""
    if width == 4:
        return expert_planes(FP4)[4]
    if width == 8:
        return expert_planes(FP8)[4]
    message = f"expert width {width} is not 4 or 8"
    raise ValueError(message)


def _pack_fp4(weight: np.ndarray) -> bytes:
    """Pack fp4. The E8M0 exponent is the one with the smallest squared error."""
    rows, k = weight.shape
    padded = align_k(k, FP4)
    block = np.zeros((rows, padded), dtype=np.float64)
    block[:, :k] = weight
    groups = block.reshape(-1, 32)
    count = groups.shape[0]
    peak = np.max(np.abs(groups), axis=1)
    ratio = np.maximum(peak / FP4_QMAX, np.finfo(np.float64).tiny)
    base = np.where(peak == 0.0, np.int16(0), _ceil_log2(ratio).astype(np.int16))
    best_error = np.full(count, np.inf)
    best_offset = np.full(count, np.int16(99))
    best_unbiased = np.zeros(count, dtype=np.int16)
    best_nibble = np.zeros((count, 32), dtype=np.uint8)
    for offset in range(-6, 3):
        unbiased = np.clip(base + np.int16(offset), -126, 127).astype(np.int16)
        unbiased = np.where(peak == 0.0, np.int16(0), unbiased)
        scale = np.exp2(unbiased.astype(np.float64))
        unit = groups / scale[:, None]
        index = _fp4_index(unit)
        sign = np.where(unit < 0.0, np.uint8(8), np.uint8(0))
        magnitude = _FP4_MAGNITUDE[index]
        signed = np.where(unit < 0.0, -magnitude, magnitude)
        error = np.sum((groups - signed * scale[:, None]) ** 2, axis=1)
        nearer = np.isclose(error, best_error) & (abs(offset) < np.abs(best_offset))
        better = (error < best_error) | nearer
        best_error = np.where(better, error, best_error)
        best_offset = np.where(better, np.int16(offset), best_offset)
        best_unbiased = np.where(better, unbiased, best_unbiased)
        nibble = sign | index
        best_nibble = np.where(better[:, None], nibble, best_nibble)
    stored = (best_unbiased + np.int16(127)).astype(np.uint8).reshape(rows, padded // 32)
    nibble = best_nibble.reshape(rows, padded)
    data = (nibble[:, 0::2] | (nibble[:, 1::2] << 4)).astype(np.uint8)
    return data.tobytes() + stored.tobytes()


def pack_matrix(weight: np.ndarray, plane: str) -> bytes:
    """Pack ``[rows, K]``. Extra K columns are zero codes."""
    if weight.ndim != 2:
        message = "pack_matrix expects a rank-2 matrix"
        raise ValueError(message)
    if plane == Q40:
        return _pack_q4_0(weight)
    if plane == FP4:
        return _pack_fp4(weight)
    if plane != FP8:
        message = f"unknown plane {plane}"
        raise ValueError(message)
    rows, k = weight.shape
    padded = align_k(k, plane)
    block = np.zeros((rows, padded), dtype=np.float32)
    block[:, :k] = weight
    stored, scale = _block_scale(block, FP8_QMAX)
    expanded = np.repeat(scale.astype(np.float32), 32, axis=1)
    unit = block / expanded
    data = _fp8_code(unit.reshape(rows, padded))
    stride = scale_stride(padded)
    scale_plane = np.zeros((rows, stride), dtype=np.uint8)
    scale_plane[:, : stored.shape[1]] = stored
    return data.tobytes() + scale_plane.tobytes()


def restore_matrix(blob: bytes, rows: int, k: int, plane: str) -> np.ndarray:
    """Inverse of :func:`pack_matrix` for the original K columns."""
    if plane == Q40:
        return _restore_q4_0(blob, rows, k)
    padded = align_k(k, plane)
    data_bytes = rows * (padded // 2) if plane == FP4 else rows * padded
    raw = np.frombuffer(blob[:data_bytes], dtype=np.uint8)
    stride = scale_stride(padded)
    scales = np.frombuffer(blob[data_bytes:], dtype=np.uint8).reshape(rows, stride)
    stored = scales[:, : padded // 32].astype(np.int16)
    scale = np.exp2(stored - 127).astype(np.float32)
    if plane == FP4:
        packed = raw.reshape(rows, padded // 2)
        nibble = np.empty((rows, padded), dtype=np.uint8)
        nibble[:, 0::2] = packed & 0x0F
        nibble[:, 1::2] = packed >> 4
        magnitude = _FP4_MAGNITUDE[nibble & 7]
        sign = np.where((nibble >> 3) & 1, -1.0, 1.0).astype(np.float32)
        values = sign * magnitude
    else:
        codes = raw.reshape(rows, padded)
        finite = codes & 0x7F
        exponent = (finite >> 3) & 0xF
        mantissa = finite & 0x7
        subnormal = (mantissa.astype(np.float32) / 8.0) * np.float32(2.0**-6)
        normal = (1.0 + mantissa.astype(np.float32) / 8.0) * np.exp2(
            exponent.astype(np.float32) - 7.0
        )
        magnitude = np.where(exponent == 0, subnormal, normal).astype(np.float32)
        magnitude = np.where(finite == 0x7F, 0.0, magnitude)
        sign = np.where(codes & 0x80, -1.0, 1.0).astype(np.float32)
        values = sign * magnitude
    restored = values * np.repeat(scale, 32, axis=1)
    return restored[:, :k].copy()


def relative_rms(weight: np.ndarray, restored: np.ndarray) -> float:
    original = weight.astype(np.float64, copy=False)
    delta = restored.astype(np.float64, copy=False) - original
    denominator = float(np.linalg.norm(original))
    if denominator == 0.0:
        return 0.0
    return float(np.linalg.norm(delta) / denominator)


def restoration_rms(weight: np.ndarray, plane: str) -> float:
    restored = restore_matrix(pack_matrix(weight, plane), weight.shape[0], weight.shape[1], plane)
    return relative_rms(weight, restored)


def store_half(values: np.ndarray) -> bytes:
    return np.ascontiguousarray(values, dtype=np.float32).astype(np.float16).tobytes()
