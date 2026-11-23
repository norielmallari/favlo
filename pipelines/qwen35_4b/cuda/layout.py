"""Qwen3.5-4B shapes and the CUDA opening that fits a device budget.

Bytes are the resident planes. GEMM weights are the integer type the tensor
op consumes. The key/value cache is one int8 code plus one scale byte per
32 channels, on eight full-attention layers.
"""

from __future__ import annotations

from dataclasses import dataclass

STEM = "qwen35_4b"
REPO = "Qwen/Qwen3.5-4B"
CHECKOUT = "Qwen3.5-4B-bf16"
FOLDER = {"int8": "Qwen3.5-4B-int8", "int4": "Qwen3.5-4B-int4"}

HIDDEN = 2560
LAYERS = 32
INTERMEDIATE = 9216
VOCAB = 248_320
EOS = 248_044
IM_END = 248_046
FULL = 8
GDN_LAYERS = LAYERS - FULL
KV_HEADS = 4
HEAD_DIM = 256
ATTN_HEADS = 16
ATTN_SCALE = HEAD_DIM**-0.5
ROPE_DIM = 64
THETA = 10_000_000.0
RMS_EPS = 1e-6
LINEAR_KEY_HEADS = 16
LINEAR_VALUE_HEADS = 32
LINEAR_HEAD_DIM = 128
CONV_KERNEL = 4
GDN_QKV = LINEAR_KEY_HEADS * LINEAR_HEAD_DIM * 2 + LINEAR_VALUE_HEADS * LINEAR_HEAD_DIM
BLOCK = 32
VISION_DEPTH = 24
VISION_HIDDEN = 1024
VISION_HEADS = 16
VISION_INTERMEDIATE = 4096
VISION_POSITIONS = 2304
VISION_CHANNELS = 3
PATCH = 16
SPATIAL_MERGE = 2
TEMPORAL_PATCH = 2
PATCH_DIM = VISION_CHANNELS * TEMPORAL_PATCH * PATCH * PATCH
MERGER_DIM = VISION_HIDDEN * SPATIAL_MERGE * SPATIAL_MERGE
IMAGE_TOKEN = 248_056

FULL_LAYERS = tuple(range(3, LAYERS, 4))
FULL_SET = frozenset(FULL_LAYERS)
KV_TOKEN_BYTES = FULL * 2 * KV_HEADS * (HEAD_DIM + HEAD_DIM // BLOCK)
GDN_STATE_BYTES = GDN_LAYERS * 32 * 128 * 128 * 4
CONV_BYTES = GDN_LAYERS * 8192 * 3 * 4
CONTEXT_4GB = 65_536
CONTEXT_CAP = 131_072
VISION_OUT = HIDDEN


def is_full(layer: int) -> bool:
    return layer in FULL_SET


def _count(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def _language() -> list[tuple[str, tuple[int, ...]]]:
    rows: list[tuple[str, tuple[int, ...]]] = [("embedding", (VOCAB, HIDDEN))]
    for _ in range(FULL):
        rows.extend(
            (
                ("q", (8192, HIDDEN)),
                ("k", (1024, HIDDEN)),
                ("v", (1024, HIDDEN)),
                ("o", (HIDDEN, 4096)),
            )
        )
    for _ in range(GDN_LAYERS):
        rows.extend(
            (
                ("qkv", (8192, HIDDEN)),
                ("z", (4096, HIDDEN)),
                ("a", (32, HIDDEN)),
                ("b", (32, HIDDEN)),
                ("out", (HIDDEN, 4096)),
            )
        )
    for _ in range(LAYERS):
        rows.extend(
            (
                ("gate", (INTERMEDIATE, HIDDEN)),
                ("up", (INTERMEDIATE, HIDDEN)),
                ("down", (HIDDEN, INTERMEDIATE)),
            )
        )
    return rows


def _fp16_bytes() -> int:
    """Norms, convolution taps, A_log, dt_bias. Same on both weight planes."""
    total = 0
    total += FULL * 2 * 256 * 2
    total += GDN_LAYERS * (8192 * 4 + 32 + 32 + 128) * 2
    total += LAYERS * 2 * HIDDEN * 2
    total += HIDDEN * 2
    return total


def _vision_gemm() -> list[tuple[int, ...]]:
    rows = [(VISION_HIDDEN, PATCH_DIM), (VISION_POSITIONS, VISION_HIDDEN)]
    for _ in range(VISION_DEPTH):
        rows.extend(
            (
                (VISION_HIDDEN * 3, VISION_HIDDEN),
                (VISION_HIDDEN, VISION_HIDDEN),
                (VISION_INTERMEDIATE, VISION_HIDDEN),
                (VISION_HIDDEN, VISION_INTERMEDIATE),
            )
        )
    rows.extend(((MERGER_DIM, MERGER_DIM), (HIDDEN, MERGER_DIM)))
    return rows


def _vision_fp16() -> int:
    total = VISION_HIDDEN * 2
    block = VISION_HIDDEN * 3 + VISION_HIDDEN + VISION_INTERMEDIATE + VISION_HIDDEN
    total += VISION_DEPTH * block
    total += VISION_DEPTH * 4 * VISION_HIDDEN
    total += MERGER_DIM + HIDDEN + VISION_HIDDEN * 2
    return total * 2


def gemm_bytes(plane: str, shapes: list[tuple[int, ...]]) -> int:
    """int8 or int4 codes plus one fp32 scale per output row."""
    total = 0
    for shape in shapes:
        elements = _count(shape)
        rows = shape[0]
        if plane == "int8":
            total += elements + rows * 4
        elif plane == "int4":
            total += elements // 2 + rows * 4
        else:
            message = f"weight plane {plane} is not int8 or int4"
            raise ValueError(message)
    return total


def workspace_bytes(plane: str) -> int:
    """1024-token prefill workspace. It does not grow with context."""
    tile = 1024
    conv = 8192
    gdn_z = 4096
    widest = max(HIDDEN, INTERMEDIATE, conv, gdn_z)
    floats = (
        HIDDEN * 4
        + INTERMEDIATE * 3
        + conv
        + 32 * 128 * 2
        + 32 * 2
        + 16 * 256 * 3
        + 4 * 256 * 2
        + VOCAB
        + (VOCAB + 255) // 256 * 2
        + 2
        + tile
        + tile * HIDDEN * 3
        + tile * INTERMEDIATE * 3
        + tile * conv
        + tile * gdn_z
        + tile * 32 * 2
        + tile * 32 * 128
        + tile * 16 * 256 * 3
        + tile * 4 * 256 * 2
    )
    stage = max(INTERMEDIATE * HIDDEN, conv * HIDDEN, gdn_z * HIDDEN)
    if plane == "int4":
        stage //= 2
    stage += max(HIDDEN, INTERMEDIATE, conv, gdn_z) * 4
    stage += tile * max(HIDDEN, INTERMEDIATE, gdn_z) + tile * 4
    return floats * 4 + tile * widest * 2 + stage


def weight_bytes(plane: str, *, vision: bool) -> int:
    shapes = [shape for _name, shape in _language()]
    total = gemm_bytes(plane, shapes) + _fp16_bytes()
    if vision:
        total += gemm_bytes(plane, _vision_gemm()) + _vision_fp16()
    return total


def opening_bytes(plane: str, tokens: int, *, vision: bool) -> int:
    return (
        weight_bytes(plane, vision=vision)
        + GDN_STATE_BYTES
        + CONV_BYTES
        + workspace_bytes(plane)
        + tokens * KV_TOKEN_BYTES
    )


@dataclass(frozen=True)
class Pack:
    plane: str
    vision: bool
    context: int


def cache_tokens(budget: int, plane: str, *, vision: bool) -> int:
    """How many int8 cache tokens fit in ``budget`` after the resident opening."""
    fixed = opening_bytes(plane, 0, vision=vision)
    if budget <= fixed:
        return 0
    cap = CONTEXT_4GB if plane == "int4" else CONTEXT_CAP
    return int(min((budget - fixed) // KV_TOKEN_BYTES, cap))


def pack_for_budget(budget: int) -> Pack:
    """int8 when that opening fits, otherwise int4. Vision stays off on int4."""
    if budget < 1:
        message = f"device budget {budget} is empty"
        raise ValueError(message)
    int8_floor = opening_bytes("int8", 2048, vision=True)
    if budget >= int8_floor:
        return Pack("int8", True, cache_tokens(budget, "int8", vision=True))
    int8_language = opening_bytes("int8", 2048, vision=False)
    if budget >= int8_language:
        return Pack("int8", False, cache_tokens(budget, "int8", vision=False))
    int4_floor = opening_bytes("int4", 2048, vision=False)
    if budget < int4_floor:
        message = (
            f"device budget {budget} bytes is under the int4 opening {int4_floor}"
        )
        raise RuntimeError(message)
    tokens = cache_tokens(budget, "int4", vision=False)
    if tokens < 2048:
        message = f"device budget {budget} bytes cannot hold a 2048-token cache"
        raise RuntimeError(message)
    return Pack("int4", False, tokens)
