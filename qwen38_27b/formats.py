"""Shapes and block scaling for this pipeline.

Block 32. The scale is the power of two rounded up so the peak fits, stored as
E8M0. Element codes use round-ties-to-even and then saturate. A zero block
stores exponent 0.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BLOCK = 32
FP4_QMAX = 6.0
FP8_QMAX = 448.0
FP4_MAGNITUDE = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)


@dataclass(frozen=True)
class TextShapes:
    hidden: int
    layers: int
    vocab: int
    intermediate: int
    full_heads: int
    kv_heads: int
    head_dim: int
    linear_q_heads: int
    linear_v_heads: int
    linear_head_dim: int
    conv_kernel: int
    rotary_pairs: int
    context: int
    rms_eps: float
    rope_theta: float
    mrope_section: tuple[int, int, int]
    layer_types: tuple[str, ...]
    output_gate: str
    eos: tuple[int, ...]


def text_shapes(config: dict) -> TextShapes:
    text = config["text_config"]
    rope = text["rope_parameters"]
    section = tuple(int(v) for v in rope["mrope_section"])
    if len(section) != 3:
        message = "mrope_section must be three widths"
        raise ValueError(message)
    layers = tuple(text["layer_types"])
    if len(layers) != int(text["num_hidden_layers"]):
        message = "layer_types length does not match num_hidden_layers"
        raise ValueError(message)
    return TextShapes(
        hidden=int(text["hidden_size"]),
        layers=int(text["num_hidden_layers"]),
        vocab=int(text["vocab_size"]),
        intermediate=int(text["intermediate_size"]),
        full_heads=int(text["num_attention_heads"]),
        kv_heads=int(text["num_key_value_heads"]),
        head_dim=int(text["head_dim"]),
        linear_q_heads=int(text["linear_num_key_heads"]),
        linear_v_heads=int(text["linear_num_value_heads"]),
        linear_head_dim=int(text["linear_key_head_dim"]),
        conv_kernel=int(text["linear_conv_kernel_dim"]),
        rotary_pairs=sum(section),
        context=int(text["max_position_embeddings"]),
        rms_eps=float(text["rms_norm_eps"]),
        rope_theta=float(rope["rope_theta"]),
        mrope_section=(section[0], section[1], section[2]),
        layer_types=layers,
        output_gate=str(text["output_gate_type"]),
        eos=(),
    )


def load_text_shapes(source: Path) -> TextShapes:
    config = json.loads((source / "config.json").read_text())
    shapes = text_shapes(config)
    generation = json.loads((source / "generation_config.json").read_text())
    eos = generation["eos_token_id"]
    if isinstance(eos, int):
        eos = (eos,)
    else:
        eos = tuple(int(v) for v in eos)
    return TextShapes(
        hidden=shapes.hidden,
        layers=shapes.layers,
        vocab=shapes.vocab,
        intermediate=shapes.intermediate,
        full_heads=shapes.full_heads,
        kv_heads=shapes.kv_heads,
        head_dim=shapes.head_dim,
        linear_q_heads=shapes.linear_q_heads,
        linear_v_heads=shapes.linear_v_heads,
        linear_head_dim=shapes.linear_head_dim,
        conv_kernel=shapes.conv_kernel,
        rotary_pairs=shapes.rotary_pairs,
        context=shapes.context,
        rms_eps=shapes.rms_eps,
        rope_theta=shapes.rope_theta,
        mrope_section=shapes.mrope_section,
        layer_types=shapes.layer_types,
        output_gate=shapes.output_gate,
        eos=eos,
    )


def stored_k(inner: int) -> int:
    """An fp4 row is a whole number of 128-byte groups, so K is a multiple of 256."""
    if inner < 1:
        message = f"inner dimension {inner} is empty"
        raise ValueError(message)
    return ((inner + 255) // 256) * 256


def stored_rows(rows: int) -> int:
    """``matmul2d`` output tiles are 64 rows."""
    if rows < 1:
        message = f"row count {rows} is empty"
        raise ValueError(message)
    return ((rows + 63) // 64) * 64


def biased_e8m0(amax: np.ndarray, qmax: float) -> tuple[np.ndarray, np.ndarray]:
    """Round ``2^ceil(log2(amax / qmax))`` up. A zero block stores exponent 0."""
    ratio = np.maximum(amax / qmax, np.finfo(np.float64).tiny)
    mantissa, exponent = np.frexp(ratio)
    power = exponent.astype(np.int32) - (mantissa == 0.5).astype(np.int32)
    power = np.where(amax <= 0.0, np.int32(-127), power)
    biased = np.clip(power + 127, 0, 254).astype(np.uint8)
    scale = np.ldexp(np.ones(amax.shape, dtype=np.float32), biased.astype(np.int32) - 127)
    return biased, scale


_FP4_EDGE = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)


def quantize_fp4(weight: np.ndarray) -> tuple[bytes, bytes]:
    """Pack ``[rows, inner]`` float32 as fp4. Inner is a multiple of 32."""
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
    biased, _scale = biased_e8m0(amax, FP4_QMAX)
    scaled = values / _scale[..., None]
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
    """Pad K to a multiple of 256 and rows up to a multiple of 64 when asked."""
    matrix = np.ascontiguousarray(weight, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    got_rows, got_inner = matrix.shape
    stored_inner = stored_k(got_inner if inner is None else inner)
    stored = got_rows if rows is None else rows
    if stored < got_rows or stored_inner < got_inner:
        message = "stored shape is smaller than the matrix"
        raise ValueError(message)
    if stored != got_rows or stored_inner != got_inner:
        padded = np.zeros((stored, stored_inner), dtype=np.float32)
        padded[:got_rows, :got_inner] = matrix
        matrix = padded
    data, scales = quantize_fp4(matrix)
    return data, scales, stored, stored_inner


def fp4_bytes(numel: int) -> int:
    """Data bytes plus one E8M0 scale byte per 32 elements."""
    return numel // 2 + numel // 32


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


_E4M3_LEVEL = decode_e4m3(np.arange(127, dtype=np.uint8)).astype(np.float32)
_E4M3_EDGE = ((_E4M3_LEVEL[:-1] + _E4M3_LEVEL[1:]) * np.float32(0.5)).astype(np.float32)


def quantize_fp8(weight: np.ndarray) -> tuple[bytes, bytes]:
    """Pack ``[rows, inner]`` float32 as E4M3FN. Inner is a multiple of 32."""
    matrix = np.ascontiguousarray(weight, dtype=np.float32)
    _rows, inner = matrix.shape
    if inner % BLOCK:
        message = f"inner dimension {inner} is not a multiple of {BLOCK}"
        raise ValueError(message)
    data: list[bytes] = []
    scales: list[bytes] = []
    for start in range(0, matrix.shape[0], 4096):
        packed, biased = _quantize_fp8_tile(matrix[start : start + 4096])
        data.append(packed)
        scales.append(biased)
    return b"".join(data), b"".join(scales)


def _quantize_fp8_tile(matrix: np.ndarray) -> tuple[bytes, bytes]:
    rows, inner = matrix.shape
    values = matrix.reshape(rows, inner // BLOCK, BLOCK)
    amax = np.max(np.abs(values), axis=-1)
    biased, scale = biased_e8m0(amax, FP8_QMAX)
    scaled = values / scale[..., None]
    magnitude = np.abs(scaled).astype(np.float32, copy=False)
    choice = np.searchsorted(_E4M3_EDGE, magnitude, side="left").astype(np.int16)
    choice = np.clip(choice, 0, 126)
    edge_at = np.minimum(choice, 125)
    on_edge = (choice < 126) & (magnitude == _E4M3_EDGE[edge_at])
    bump = on_edge & ((choice & 1) == 1)
    choice = np.where(bump, choice + 1, choice).astype(np.uint8)
    sign = np.signbit(scaled).astype(np.uint8) << np.uint8(7)
    codes = choice | sign
    return np.ascontiguousarray(codes).tobytes(), np.ascontiguousarray(biased).tobytes()


def dequant_fp8(packed: bytes, scales: bytes, rows: int, inner: int) -> np.ndarray:
    codes = np.frombuffer(packed, dtype=np.uint8).reshape(rows, inner)
    values = decode_e4m3(codes)
    factors = np.frombuffer(scales, dtype=np.uint8).astype(np.int32).reshape(rows, inner // BLOCK)
    restored = values.reshape(rows, inner // BLOCK, BLOCK)
    restored *= np.ldexp(np.ones(factors.shape, dtype=np.float32), factors - 127)[..., None]
    return restored.reshape(rows, inner)


def pack_fp8_matrix(
    weight: np.ndarray, *, rows: int | None = None, inner: int | None = None
) -> tuple[bytes, bytes, int, int]:
    """Pad K to a multiple of 256 and rows up to a multiple of 64 when asked."""
    matrix = np.ascontiguousarray(weight, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    got_rows, got_inner = matrix.shape
    stored_inner = stored_k(got_inner if inner is None else inner)
    stored = got_rows if rows is None else rows
    if stored < got_rows or stored_inner < got_inner:
        message = "stored shape is smaller than the matrix"
        raise ValueError(message)
    if stored != got_rows or stored_inner != got_inner:
        padded = np.zeros((stored, stored_inner), dtype=np.float32)
        padded[:got_rows, :got_inner] = matrix
        matrix = padded
    data, scales = quantize_fp8(matrix)
    return data, scales, stored, stored_inner


def relative_error(weight: np.ndarray) -> float:
    """RMS of the BF16 row against its restored fp4 row, divided by the BF16 RMS."""
    matrix = np.ascontiguousarray(weight, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    elif matrix.ndim > 2:
        matrix = matrix.reshape(matrix.shape[0], -1)
    rows, inner = matrix.shape
    numerator = 0.0
    denominator = 0.0
    for start in range(0, rows, 4096):
        tile = matrix[start : start + 4096]
        data, scales = quantize_fp4(tile)
        restored = dequant_fp4(data, scales, tile.shape[0], inner)
        diff = tile.astype(np.float64) - restored.astype(np.float64)
        flat = tile.astype(np.float64).ravel()
        numerator += float(np.dot(diff.ravel(), diff.ravel()))
        denominator += float(np.dot(flat, flat))
    if denominator == 0.0:
        return 0.0
    return math.sqrt(numerator / denominator)


def ffn_promote_count(percent: float, layers: int) -> int:
    """Largest whole layer count that does not exceed ``percent`` of ``layers``."""
    if layers < 1:
        message = f"layer count {layers} is empty"
        raise ValueError(message)
    if not math.isfinite(percent) or percent < 0.0 or percent > 100.0:
        message = f"ffn fp8 percentage {percent} is outside 0..100"
        raise ValueError(message)
    count = int(math.floor((percent * layers / 100.0) + 1e-9))
    if count > layers:
        return layers
    return count


def promoted_layer_ids(scores: dict[int, float], count: int) -> list[int]:
    """Highest scores. An equal score keeps the lower layer index."""
    if count < 0 or count > len(scores):
        message = f"promote count {count} does not fit {len(scores)} layers"
        raise ValueError(message)
    ranked = sorted(scores, key=lambda layer: (-float(scores[layer]), int(layer)))
    return sorted(int(layer) for layer in ranked[:count])
