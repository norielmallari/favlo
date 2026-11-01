"""Shapes from the Spark-X2.6-1.7B config. No checkpoint is read here."""

from __future__ import annotations

STEM = "spark-x26-1-7b"
FOLDER = "Spark-X2.6-1.7B-fp4"
REPO = "XHToken/Spark-X2.6-1.7B"

HIDDEN = 2048
LAYERS = 28
HEADS = 8
KV_HEADS = 2
HEAD_DIM = 256
Q_DIM = HEADS * HEAD_DIM
KV_DIM = KV_HEADS * HEAD_DIM
INTERMEDIATE = 6656
VOCAB = 131072
WINDOW = 512
CONTEXT = 1_048_576
CHUNK = 64
BLOCK = 32
RMS_EPS = 1e-6
ROPE_FULL = 64
ROPE_SLIDE = 256
THETA_FULL = 5_000_000.0
THETA_SLIDE = 10_000.0
ATTN_SCALE = 0.0625
EOS = 1

FULL = (3, 7, 11, 15, 19, 23, 27)
FULL_SET = frozenset(FULL)


def is_full(layer: int) -> bool:
    return layer in FULL_SET


def kind_ordinal(layer: int) -> int:
    """Index of this layer among full layers, or among sliding layers."""
    if is_full(layer):
        return FULL.index(layer)
    return layer - sum(1 for item in FULL if item < layer)


def fp4_bytes(rows: int, inner: int) -> int:
    """Packed nibbles plus one E8M0 byte per 32 elements along K."""
    return rows * inner // 2 + rows * (inner // BLOCK)


def f16_bytes(count: int) -> int:
    return count * 2


def record_bytes() -> int:
    """The fp4 record, with ``g_proj`` and the RMSNorm gains left in float16."""
    embed = fp4_bytes(VOCAB, HIDDEN)
    qkv = fp4_bytes(Q_DIM + 2 * KV_DIM, HIDDEN)
    out = fp4_bytes(HIDDEN, HIDDEN)
    gate = fp4_bytes(INTERMEDIATE, HIDDEN)
    down = fp4_bytes(HIDDEN, INTERMEDIATE)
    layers = LAYERS * (qkv + out + 2 * gate + down)
    gates = LAYERS * f16_bytes(HEADS * HIDDEN)
    norms = f16_bytes((LAYERS * 2 + 1) * HIDDEN)
    return embed + layers + gates + norms


def full_token_bytes() -> int:
    """fp8 key and value planes of the seven full-attention layers, one token."""
    per_plane = KV_HEADS * (HEAD_DIM + HEAD_DIM // BLOCK)
    return len(FULL) * 2 * per_plane


def cache_bytes(tokens: int) -> int:
    return full_token_bytes() * tokens
