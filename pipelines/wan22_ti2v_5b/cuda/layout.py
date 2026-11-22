"""Resident bytes for the 6 GiB int4 opening.

Counts are the published widths. A kernels dispatch is what accepts a plane.
"""

from __future__ import annotations

DIM = 3072
FFN = 14336
LAYERS = 30
TEXT_LEN = 512
TEXT_DIM = 4096
HEADS = 24
HEAD = 128
FPS = 24
ROPE_LEN = 1024
BUDGET = 6 * 1024**3

# Published google/umt5-xxl widths. This package's encoder is 24 blocks of 4096.
UMT5_LAYERS = 24
UMT5_DIM = 4096
UMT5_FFN = 10240
UMT5_HEADS = 64
UMT5_BUCKETS = 32
UMT5_VOCAB = 256384


def dit_bytes() -> int:
    """Int4 codes plus fp16 scales, biases, and vectors."""
    block_elements = 8 * DIM * DIM + 2 * FFN * DIM
    codes = LAYERS * block_elements // 2
    rows = LAYERS * (8 * DIM + FFN + DIM)
    vectors = LAYERS * (6 * DIM + 2 * DIM + 4 * DIM) * 2
    stored_k = {
        "patch": 256,
        "text0": TEXT_DIM,
        "text2": DIM,
        "time0": 256,
        "time2": DIM,
        "time_proj": DIM,
        "head": DIM,
    }
    stored_n = {
        "patch": DIM,
        "text0": DIM,
        "text2": DIM,
        "time0": DIM,
        "time2": DIM,
        "time_proj": DIM * 6,
        "head": 192,
    }
    linear_codes = sum(stored_n[name] * stored_k[name] // 2 for name in stored_n)
    linear_rows = sum(stored_n.values())
    head_mod = 2 * DIM * 2
    return codes + rows * 4 + vectors + linear_codes + linear_rows * 4 + head_mod


def umt5_bytes() -> int:
    """Int4 body and token table, plus fp16 scales, norms, and buckets."""
    layer = 4 * UMT5_DIM * UMT5_DIM + 3 * UMT5_FFN * UMT5_DIM
    codes = UMT5_LAYERS * layer // 2
    rows = UMT5_LAYERS * (4 * UMT5_DIM + 3 * UMT5_FFN)
    norms = (UMT5_LAYERS * 2 * UMT5_DIM + UMT5_DIM) * 2
    buckets = UMT5_LAYERS * UMT5_BUCKETS * UMT5_HEADS * 2
    embed = UMT5_VOCAB * UMT5_DIM // 2 + UMT5_VOCAB * 2
    return codes + rows * 2 + norms + buckets + embed


def token_count(frames: int, height: int, width: int) -> int:
    if frames < 1 or (frames - 1) % 4 != 0:
        message = f"frames must be 4n+1, got {frames}"
        raise ValueError(message)
    if height % 32 != 0 or width % 32 != 0:
        message = f"edges must be multiples of 32, got {width}x{height}"
        raise ValueError(message)
    time = (frames - 1) // 4 + 1
    return time * (height // 32) * (width // 32)


def _padded(tokens: int) -> int:
    return ((tokens + 63) // 64) * 64


def scratch_bytes(frames: int, height: int, width: int) -> int:
    """fp16 residual, fp16 feed-forward hidden, context, and cross key/value."""
    tokens = _padded(token_count(frames, height, width))
    pair = tokens * (DIM + FFN) * 2
    context = TEXT_LEN * DIM * 2
    cross = LAYERS * 2 * TEXT_LEN * DIM * 2
    return pair + context + cross


def opening_bytes(frames: int, height: int, width: int) -> int:
    return dit_bytes() + scratch_bytes(frames, height, width)


def fits(frames: int, height: int, width: int, budget: int = BUDGET) -> bool:
    return opening_bytes(frames, height, width) <= budget


def longest(height: int, width: int, budget: int = BUDGET) -> tuple[int, int]:
    """Longest ``4n+1`` frame count and its latent time that still fit."""
    spatial = (height // 32) * (width // 32)
    if spatial < 1:
        message = f"edges must be multiples of 32, got {width}x{height}"
        raise ValueError(message)
    best = 1
    low = 1
    high = ROPE_LEN
    while low <= high:
        time = (low + high) // 2
        frames = 4 * (time - 1) + 1
        if fits(frames, height, width, budget):
            best = time
            low = time + 1
        else:
            high = time - 1
    return 4 * (best - 1) + 1, best
