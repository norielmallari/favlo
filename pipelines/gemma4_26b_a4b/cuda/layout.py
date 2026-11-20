"""Gemma 4 26B-A4B shapes and the CUDA record.

Dense language matrices are int8. Routed experts are int4. Down K is stored
as 768. The hot path is fp16.
"""

from __future__ import annotations

import math

STEM = "gemma4_26b_a4b"
REPO = "google/gemma-4-26B-A4B-it"
FOLDER = "gemma-4-26B-A4B-it-int8-e10"
HIDDEN = 2816
LAYERS = 30
EXPERTS = 128
TOP_K = 8
EXPERT_PERCENT = 0.0
LANGUAGE_EXPERTS = LAYERS * EXPERTS
VOCAB = 262144
EOS = 1
EOT = 106
EXPERT_WIDTH = 704
EXPERT_K = 768
GATE_ROWS = EXPERT_WIDTH * 2
DENSE_WIDTH = 2112
WINDOW = 1024
CONTEXT = 131072
LOCAL_HEADS = 8
LOCAL_DIM = 256
GLOBAL_HEADS = 2
GLOBAL_DIM = 512
Q_HEADS = 16
GLOBAL_LAYERS = (5, 11, 17, 23, 29)
RMS_EPS = 1e-6
SOFTCAP = 30.0
ROPE_PARTIAL = 0.25
ROPE_LOCAL = 10_000.0
ROPE_GLOBAL = 1_000_000.0
PREFILL_TILE = 512
CONTEXT_STEP = 2048


def is_global(layer: int) -> bool:
    return layer in GLOBAL_LAYERS


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
    """Gate-up ``[1408, 2816]`` then down ``[2816, 768]``."""
    return gemm_bytes(plane, GATE_ROWS, HIDDEN) + gemm_bytes(plane, HIDDEN, EXPERT_K)


INT8_IMAGE = expert_image_bytes("int8")
INT4_IMAGE = expert_image_bytes("int4")


def expert_parts(plane: str) -> tuple[int, int, int, int]:
    """Offsets of gate codes, gate scales, down codes, and down scales."""
    gate_codes = gemm_bytes(plane, GATE_ROWS, HIDDEN) - GATE_ROWS * 4
    down_codes = gate_codes + GATE_ROWS * 4
    down_data = gemm_bytes(plane, HIDDEN, EXPERT_K) - HIDDEN * 4
    return gate_codes, gate_codes, down_codes, down_codes + down_data


def promote_count(percent: float = EXPERT_PERCENT, pool: int = LANGUAGE_EXPERTS) -> int:
    """Largest whole count that does not exceed ``percent`` of ``pool``."""
    if pool < 1:
        message = f"expert pool {pool} is empty"
        raise ValueError(message)
    if not math.isfinite(percent) or percent < 0.0 or percent > 100.0:
        message = f"expert int8 percentage {percent} is outside 0..100"
        raise ValueError(message)
    return min(int(math.floor((percent * pool / 100.0) + 1e-9)), pool)


def local_attn_bytes() -> int:
    q = gemm_bytes("int8", Q_HEADS * LOCAL_DIM, HIDDEN)
    k = gemm_bytes("int8", LOCAL_HEADS * LOCAL_DIM, HIDDEN)
    v = gemm_bytes("int8", LOCAL_HEADS * LOCAL_DIM, HIDDEN)
    o = gemm_bytes("int8", HIDDEN, Q_HEADS * LOCAL_DIM)
    return q + k + v + o


def global_attn_bytes() -> int:
    q = gemm_bytes("int8", Q_HEADS * GLOBAL_DIM, HIDDEN)
    k = gemm_bytes("int8", GLOBAL_HEADS * GLOBAL_DIM, HIDDEN)
    o = gemm_bytes("int8", HIDDEN, Q_HEADS * GLOBAL_DIM)
    return q + k + o


def mlp_bytes() -> int:
    return (
        gemm_bytes("int8", DENSE_WIDTH, HIDDEN) * 2
        + gemm_bytes("int8", HIDDEN, DENSE_WIDTH)
    )


def resident_bytes() -> int:
    """Language int8 weights. The expert atlas is not in this count."""
    embed = gemm_bytes("int8", VOCAB, HIDDEN)
    local = (LAYERS - len(GLOBAL_LAYERS)) * local_attn_bytes()
    glob = len(GLOBAL_LAYERS) * global_attn_bytes()
    return embed + local + glob + LAYERS * mlp_bytes()
