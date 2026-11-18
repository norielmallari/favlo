"""Qwen3.6-35B-A3B shapes and the CUDA opening.

Activation planes, Gated DeltaNet state, and the key/value record match the
Metal sizes. Weights do not: dense matrices are int8, and 10% of the routed
experts are int8. The rest are int4. The int8 experts stay in VRAM. The int4
experts stay in host RAM.
"""

from __future__ import annotations

import math

STEM = "qwen36_35b_a3b"
REPO = "Qwen/Qwen3.6-35B-A3B"
CHECKOUT = "Qwen3.6-35B-A3B-bf16"

HIDDEN = 2048
LAYERS = 40
EXPERTS = 256
TOP_K = 8
EXPERT_PERCENT = 10.0
LANGUAGE_EXPERTS = LAYERS * EXPERTS
VOCAB = 248_320
EOS = 248_044
IM_END = 248_046
HEAD_DIM = 256
KV_HEADS = 2
Q_HEADS = 16
Q_ROWS = 8192
KV_ROWS = 512
O_INNER = 4096
GATE_ROWS = 1024
GATE_INNER = 2048
DOWN_ROWS = 2048
DOWN_INNER = 512
LINEAR_KEY_HEADS = 16
LINEAR_VALUE_HEADS = 32
LINEAR_HEAD_DIM = 128
GDN_QKV = 8192
GDN_Z = 4096
GDN_LAYERS = 30
FULL_LAYERS = tuple(layer for layer in range(LAYERS) if layer % 4 == 3)
FULL = len(FULL_LAYERS)
BLOCK = 32
CONV_KERNEL = 4
RMS_EPS = 1e-6
THETA = 10_000_000.0
ATTN_SCALE = HEAD_DIM**-0.5
ROPE_DIM = 64
CONTEXT_FLOOR = 2048
CONTEXT_STEP = 2048
CONTEXT_CAP = 131_072
PREFILL_TILE = 512
INT8_QMAX = 127.0
INT4_QMAX = 7.0

KV_RECORD = HEAD_DIM + HEAD_DIM // BLOCK
KV_TOKEN_BYTES = FULL * 2 * KV_HEADS * KV_RECORD
GDN_STATE_BYTES = GDN_LAYERS * LINEAR_VALUE_HEADS * LINEAR_HEAD_DIM * LINEAR_HEAD_DIM * 4
CONV_BYTES = GDN_LAYERS * GDN_QKV * (CONV_KERNEL - 1) * 2


def is_full(layer: int) -> bool:
    return layer % 4 == 3


def gemm_bytes(plane: str, rows: int, inner: int) -> int:
    """Codes plus one fp32 scale per output row."""
    if plane == "int8":
        return rows * inner + rows * 4
    if plane == "int4":
        if inner % 2:
            message = f"int4 inner {inner} is odd"
            raise ValueError(message)
        return rows * (inner // 2) + rows * 4
    message = f"weight plane {plane} is not int8 or int4"
    raise ValueError(message)


def expert_image_bytes(plane: str) -> int:
    """One routed expert: gate-up ``[1024, 2048]`` then down ``[2048, 512]``."""
    return gemm_bytes(plane, GATE_ROWS, GATE_INNER) + gemm_bytes(plane, DOWN_ROWS, DOWN_INNER)


INT8_IMAGE = expert_image_bytes("int8")
INT4_IMAGE = expert_image_bytes("int4")


def expert_parts(plane: str) -> tuple[int, int, int, int]:
    """Offsets of gate codes, gate scales, down codes, and down scales."""
    gate_codes = gemm_bytes(plane, GATE_ROWS, GATE_INNER) - GATE_ROWS * 4
    gate_scales = gate_codes
    down_codes = gate_scales + GATE_ROWS * 4
    down_data = gemm_bytes(plane, DOWN_ROWS, DOWN_INNER) - DOWN_ROWS * 4
    return gate_codes, gate_scales, down_codes, down_codes + down_data


def promote_count(percent: float, pool: int = LANGUAGE_EXPERTS) -> int:
    """Largest whole count that does not exceed ``percent`` of ``pool``."""
    if pool < 1:
        message = f"expert pool {pool} is empty"
        raise ValueError(message)
    if not math.isfinite(percent) or percent < 0.0 or percent > 100.0:
        message = f"expert int8 percentage {percent} is outside 0..100"
        raise ValueError(message)
    count = int(math.floor((percent * pool / 100.0) + 1e-9))
    return min(count, pool)


def atlas_bytes(promoted: int | None = None) -> int:
    """Host bytes of every routed expert image."""
    count = promote_count(EXPERT_PERCENT) if promoted is None else int(promoted)
    if count < 0 or count > LANGUAGE_EXPERTS:
        message = f"promoted experts {count} are outside the pool"
        raise ValueError(message)
    rest = LANGUAGE_EXPERTS - count
    return count * INT8_IMAGE + rest * INT4_IMAGE


def pinned_expert_bytes() -> int:
    """VRAM for every int8 expert image. They stay resident for the open."""
    return promote_count(EXPERT_PERCENT) * INT8_IMAGE


def stage_bytes() -> int:
    """VRAM bytes for the eight int4 images copied on one layer."""
    return TOP_K * INT4_IMAGE


def cache_slots(room: int) -> int:
    """How many int8 expert images fit in ``room`` bytes of device memory."""
    if room < INT8_IMAGE:
        return 0
    return int(room) // INT8_IMAGE


def record_name(source_name: str, percent: float = EXPERT_PERCENT) -> str:
    """``<model>-int4``, or ``<model>-int8-e<P>`` when any expert is int8."""
    name = source_name[: -len("-bf16")] if source_name.endswith("-bf16") else source_name
    if percent <= 0:
        return f"{name}-int4"
    token = str(int(percent)) if float(percent) == int(percent) else f"{percent:g}"
    return f"{name}-int8-e{token}"


def _dense_int8() -> int:
    total = gemm_bytes("int8", VOCAB, HIDDEN)
    gdn = (
        gemm_bytes("int8", GDN_QKV, HIDDEN)
        + gemm_bytes("int8", GDN_Z, HIDDEN)
        + gemm_bytes("int8", LINEAR_VALUE_HEADS, HIDDEN) * 2
        + gemm_bytes("int8", HIDDEN, O_INNER)
    )
    attn = (
        gemm_bytes("int8", Q_ROWS, HIDDEN)
        + gemm_bytes("int8", KV_ROWS, HIDDEN) * 2
        + gemm_bytes("int8", HIDDEN, O_INNER)
    )
    shared = gemm_bytes("int8", GATE_ROWS, GATE_INNER) + gemm_bytes("int8", DOWN_ROWS, DOWN_INNER)
    return total + GDN_LAYERS * gdn + FULL * attn + LAYERS * shared


def _dense_fp16() -> int:
    router = LAYERS * EXPERTS * HIDDEN * 2
    coeff = LAYERS * HIDDEN * 2
    norms = (LAYERS * 2 + 1) * HIDDEN * 2
    qk = FULL * 2 * HEAD_DIM * 2
    conv = GDN_LAYERS * GDN_QKV * CONV_KERNEL * 2
    decay = GDN_LAYERS * LINEAR_VALUE_HEADS * 2 * 2
    gdn_norm = GDN_LAYERS * LINEAR_HEAD_DIM * 2
    return router + coeff + norms + qk + conv + decay + gdn_norm


def resident_bytes() -> int:
    """Language weights on the device. The expert atlas is not in this count."""
    return _dense_int8() + _dense_fp16()


def scratch_items() -> tuple[tuple[str, int], ...]:
    """Device planes. Token-parallel activations hold one prefill tile.

    Gated DeltaNet state, convolution history, and the logits stay one token.
    """
    groups = (VOCAB + 255) // 256
    tile = PREFILL_TILE
    return (
        ("hidden", HIDDEN * 4 * tile),
        ("normed", HIDDEN * 4 * tile),
        ("wide", GDN_QKV * 4 * tile),
        ("z", GDN_Z * 4 * tile),
        ("ab", LINEAR_VALUE_HEADS * 4 * tile),
        ("ab2", LINEAR_VALUE_HEADS * 4 * tile),
        ("mixed", GDN_QKV * 4 * tile),
        ("q_rep", GDN_Z * 4 * tile),
        ("k_rep", GDN_Z * 4 * tile),
        ("gdn_y", GDN_Z * 4 * tile),
        ("query", O_INNER * 4 * tile),
        ("gate", O_INNER * 4 * tile),
        ("kv", KV_ROWS * 4 * tile),
        ("vv", KV_ROWS * 4 * tile),
        ("attn", O_INNER * 4 * tile),
        ("proj", HIDDEN * 4 * tile),
        ("moe", HIDDEN * 4),
        ("up", GATE_ROWS * 4),
        ("silu", DOWN_INNER * 4),
        ("router", EXPERTS * 4),
        ("route_w", TOP_K * 4),
        ("coeff", 4),
        ("logits", VOCAB * 4),
        ("codes", GDN_QKV),
        ("ascales", (GDN_QKV // BLOCK) * 4),
        ("state", GDN_STATE_BYTES),
        ("history", CONV_BYTES),
        ("counts", VOCAB * 4),
        ("route_ids", TOP_K * 4),
        ("moe_rows", HIDDEN * 4 * tile),
        ("gather", HIDDEN * 4 * tile),
        ("up_rows", GATE_ROWS * 4 * tile),
        ("silu_rows", DOWN_INNER * 4 * tile),
        ("route_ids_tile", TOP_K * 4 * tile),
        ("route_w_tile", TOP_K * 4 * tile),
        ("group_token", 4 * tile),
        ("group_slot", 4 * tile),
        ("coeff_tile", 4 * tile),
        ("picked", 4),
        ("partial_v", groups * 4),
        ("partial_i", groups * 4),
    )


def scratch_bytes() -> int:
    """Bytes of ``scratch_items``."""
    return sum(size for _name, size in scratch_items())


def opening_bytes(tokens: int, *, resident: int | None = None) -> int:
    """VRAM for the language opening at ``tokens`` of int8 cache."""
    weights = resident_bytes() if resident is None else int(resident)
    return (
        weights
        + scratch_bytes()
        + stage_bytes()
        + pinned_expert_bytes()
        + int(tokens) * KV_TOKEN_BYTES
    )


def cache_tokens(budget: int, *, resident: int | None = None) -> int:
    """Largest multiple of 2048 int8-cache tokens that fit in ``budget``."""
    if budget < 1:
        return 0
    fixed = opening_bytes(0, resident=resident)
    if budget <= fixed:
        return 0
    room = (budget - fixed) // KV_TOKEN_BYTES
    capped = min(int(room), CONTEXT_CAP)
    return (capped // CONTEXT_STEP) * CONTEXT_STEP


def stream_spans(
    widths: list[int] | tuple[int, ...],
    offsets: list[int] | tuple[int, ...],
    experts: list[int] | tuple[int, ...],
) -> list[tuple[int, int, int]]:
    """``(file offset, nbytes, stage slot)`` for one layer's routed experts."""
    spans: list[tuple[int, int, int]] = []
    for slot, expert in enumerate(experts):
        width = int(widths[int(expert)])
        if width == 8:
            size = INT8_IMAGE
        elif width == 4:
            size = INT4_IMAGE
        else:
            message = f"expert width {width} is not 4 or 8"
            raise ValueError(message)
        spans.append((int(offsets[int(expert)]), size, slot))
    return spans
