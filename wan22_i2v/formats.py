"""fp4 and fp8 packs, the K permutation, and the block score."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TILE_M = 64
TILE_N = 32
K_ALIGN = 256
FP4_LEVELS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
FP4_MAX = np.float32(6.0)
FP8_MAX = np.float32(448.0)

BLOCK_MATRICES = (
    "self_attn.q.weight",
    "self_attn.k.weight",
    "self_attn.v.weight",
    "self_attn.o.weight",
    "cross_attn.q.weight",
    "cross_attn.k.weight",
    "cross_attn.v.weight",
    "cross_attn.o.weight",
    "ffn.0.weight",
    "ffn.2.weight",
)


def _e4m3_table() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values: list[float] = []
    codes: list[int] = []
    decode = np.zeros(256, dtype=np.float32)
    for code in range(256):
        sign = -1.0 if code & 0x80 else 1.0
        exp = (code >> 3) & 0xF
        mant = code & 0x7
        if exp == 15 and mant == 7:
            continue
        if exp == 0:
            mag = (mant / 8.0) * (2.0**-6)
        else:
            mag = (1.0 + mant / 8.0) * (2.0 ** (exp - 7))
        value = sign * mag
        values.append(value)
        codes.append(code)
        decode[code] = np.float32(value)
    return (
        np.asarray(values, dtype=np.float32),
        np.asarray(codes, dtype=np.uint8),
        decode,
    )


_FP8_VALUES, _FP8_CODES, _FP8_DECODE = _e4m3_table()


@dataclass(frozen=True)
class LinearPack:
    """One quantized matrix. ``perm[i]`` is the source column stored at packed column i."""

    kind: str
    packed: np.ndarray
    scales: np.ndarray
    bias: np.ndarray
    perm: np.ndarray
    logical_n: int
    logical_k: int
    padded_n: int
    stored_k: int


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


def permute_k(weight: np.ndarray, stored: int) -> tuple[np.ndarray, np.ndarray]:
    """One order for the whole matrix. Columns are ranked by the peak over rows."""
    if weight.ndim != 2:
        message = f"matrix must be rank 2, got {weight.shape}"
        raise ValueError(message)
    rows, logical = weight.shape
    score = np.max(np.abs(weight), axis=0)
    rank = np.argsort(-score, kind="stable").astype(np.uint32)
    perm = np.arange(stored, dtype=np.uint32)
    perm[:logical] = rank
    ordered = np.zeros((rows, stored), dtype=np.float32)
    ordered[:, :logical] = weight[:, rank]
    return ordered, perm


def _scale_exp(peak: np.ndarray, endpoint: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ratio = np.divide(peak, endpoint, out=np.zeros_like(peak), where=peak > 0)
    exp = np.zeros_like(peak)
    live = peak > 0
    exp[live] = np.ceil(np.log2(ratio[live]))
    scale = np.exp2(exp)
    encoded = np.zeros(peak.shape, dtype=np.uint8)
    encoded[live] = np.clip(exp[live] + 127.0, 0.0, 254.0).astype(np.uint8)
    return encoded, scale, live


def pack_fp4_matrix(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pack a K-aligned float32 matrix. Low nibble is element 0."""
    rows, sk = weight.shape
    if sk % K_ALIGN:
        message = f"fp4 K must be a multiple of {K_ALIGN}, got {sk}"
        raise ValueError(message)
    packed = np.empty((rows, sk // 2), dtype=np.uint8)
    encoded = np.empty((rows, sk // 32), dtype=np.uint8)
    for start, end in _row_spans(rows, sk, 8):
        groups = np.ascontiguousarray(weight[start:end]).reshape(end - start, sk // 32, 32)
        peak = np.max(np.abs(groups), axis=-1)
        enc, scale, live = _scale_exp(peak, float(FP4_MAX))
        safe = np.where(live, scale, 1.0)
        scaled = groups / safe[..., None]
        distance = np.abs(np.abs(scaled)[..., None] - FP4_LEVELS)
        code = np.argmin(distance, axis=-1).astype(np.uint8)
        code = code | np.where(scaled < 0.0, np.uint8(8), np.uint8(0))
        code = np.where(live[..., None], code, np.uint8(0))
        flat = code.reshape(end - start, sk)
        packed[start:end] = flat[:, 0::2] | (flat[:, 1::2] << 4)
        encoded[start:end] = enc
    return np.ascontiguousarray(packed), np.ascontiguousarray(encoded)


def dequant_fp4(packed: np.ndarray, scales: np.ndarray, sk: int) -> np.ndarray:
    rows = packed.shape[0]
    codes = np.empty((rows, sk), dtype=np.uint8)
    codes[:, 0::2] = packed & np.uint8(0x0F)
    codes[:, 1::2] = packed >> np.uint8(4)
    mag = FP4_LEVELS[codes & np.uint8(7)]
    sign = np.where(codes & np.uint8(8), -1.0, 1.0).astype(np.float32)
    values = (mag * sign).reshape(rows, sk // 32, 32)
    factor = np.exp2(scales.astype(np.float32) - 127.0)
    factor = np.where(scales == 0, 0.0, factor)
    return (values * factor[..., None]).reshape(rows, sk)


def pack_fp8_matrix(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pack a K-aligned float32 matrix as E4M3 with an E8M0 scale."""
    rows, sk = weight.shape
    if sk % 128:
        message = f"fp8 K must be a multiple of 128, got {sk}"
        raise ValueError(message)
    packed = np.empty((rows, sk), dtype=np.uint8)
    encoded = np.empty((rows, sk // 32), dtype=np.uint8)
    for start, end in _row_spans(rows, sk, _FP8_VALUES.shape[0]):
        groups = np.ascontiguousarray(weight[start:end]).reshape(end - start, sk // 32, 32)
        peak = np.max(np.abs(groups), axis=-1)
        enc, scale, live = _scale_exp(peak, float(FP8_MAX))
        safe = np.where(live, scale, 1.0)
        scaled = groups / safe[..., None]
        distance = np.abs(scaled[..., None] - _FP8_VALUES)
        nearest = np.argmin(distance, axis=-1)
        codes = _FP8_CODES[nearest]
        codes = np.where(live[..., None], codes, np.uint8(0))
        packed[start:end] = codes.reshape(end - start, sk)
        encoded[start:end] = enc
    return np.ascontiguousarray(packed), np.ascontiguousarray(encoded)


def _row_spans(rows: int, sk: int, levels: int) -> list[tuple[int, int]]:
    """Row tiles whose level-distance table stays near 256 MiB."""
    per_row = max(sk * levels * 4, 1)
    step = max(1, min(rows, (256 << 20) // per_row))
    return [(start, min(start + step, rows)) for start in range(0, rows, step)]


def dequant_fp8(packed: np.ndarray, scales: np.ndarray, sk: int) -> np.ndarray:
    values = _FP8_DECODE[packed].reshape(packed.shape[0], sk // 32, 32)
    factor = np.exp2(scales.astype(np.float32) - 127.0)
    factor = np.where(scales == 0, 0.0, factor)
    return (values * factor[..., None]).reshape(packed.shape[0], sk)


def pack_linear(weight: np.ndarray, bias: np.ndarray | None, kind: str) -> LinearPack:
    """Permute K, pad N to the matmul tile, then pack."""
    if kind not in {"fp4", "fp8"}:
        message = f"kind must be fp4 or fp8, got {kind}"
        raise ValueError(message)
    matrix = np.ascontiguousarray(weight, dtype=np.float32)
    logical_n, logical_k = matrix.shape
    sk = stored_k(logical_k)
    padded_n = pad_to(logical_n, TILE_N)
    padded = np.zeros((padded_n, logical_k), dtype=np.float32)
    padded[:logical_n] = matrix
    ordered, perm = permute_k(padded, sk)
    if kind == "fp4":
        packed, scales = pack_fp4_matrix(ordered)
    else:
        packed, scales = pack_fp8_matrix(ordered)
    if bias is None:
        stored_bias = np.zeros(logical_n, dtype=np.float16)
    else:
        stored_bias = np.ascontiguousarray(bias, dtype=np.float16).reshape(logical_n)
    return LinearPack(
        kind=kind,
        packed=packed,
        scales=scales,
        bias=stored_bias,
        perm=perm,
        logical_n=logical_n,
        logical_k=logical_k,
        padded_n=padded_n,
        stored_k=sk,
    )


def restore_linear(pack: LinearPack) -> np.ndarray:
    """Float32 logical matrix, K put back in source order."""
    if pack.kind == "fp4":
        restored = dequant_fp4(pack.packed, pack.scales, pack.stored_k)
    else:
        restored = dequant_fp8(pack.packed, pack.scales, pack.stored_k)
    back = np.zeros((pack.logical_n, pack.logical_k), dtype=np.float32)
    back[:, pack.perm[: pack.logical_k]] = restored[: pack.logical_n, : pack.logical_k]
    return back


def relative_error(restored: np.ndarray, source: np.ndarray) -> float:
    """RMS of the residual divided by RMS of the source. A zero source scores 0."""
    residual = restored.astype(np.float32) - source.astype(np.float32)
    denom = float(np.sqrt(np.mean(np.square(source.astype(np.float32)))))
    if denom == 0.0:
        return 0.0
    numer = float(np.sqrt(np.mean(np.square(residual))))
    return numer / denom


def block_winners(scores: dict[int, float], count: int) -> list[int]:
    """Highest scores. An equal score keeps the lower block index."""
    if count < 0 or count > len(scores):
        message = f"promote count {count} does not fit {len(scores)} blocks"
        raise ValueError(message)
    ranked = sorted(scores, key=lambda block: (-float(scores[block]), int(block)))
    return sorted(int(block) for block in ranked[:count])


def as_float32(array: np.ndarray) -> np.ndarray:
    """Checkpoint F32, F16, and BF16 as float32."""
    if array.dtype == np.float32:
        return np.ascontiguousarray(array)
    if array.dtype == np.float16:
        return np.ascontiguousarray(array).astype(np.float32)
    if array.dtype == np.uint16:
        widened = array.astype(np.uint32) << np.uint32(16)
        return widened.view(np.float32)
    message = f"unsupported checkpoint dtype {array.dtype}"
    raise ValueError(message)
