"""Prompt length and the resident hot-expert budget.

The prompt limit is the tokens that fit, and at most 131,072. The indexer window is separate:
a query attends at most 512 complete blocks of 4, plus the unfinished tail.
"""

from __future__ import annotations

import ctypes
import math
import os
import struct
import sys
from collections.abc import Sequence

from qwen38_flash_next.formats import PAGE

# The dense attend kernel stores one score per visible token in scores[2052].
# 512 blocks of 4, plus a tail of at most 4, is the sparse window.
KEEP_BLOCKS = 512
BLOCK = 4
INDEX_STRIDE = KEEP_BLOCKS * BLOCK + BLOCK
DENSE_LIMIT = INDEX_STRIDE - 1
PREFILL_CHUNK = 2048
HEADROOM = 4 << 30
SCRATCH_FLOOR = 256 << 20
SCRATCH_PER_TOKEN = 2 << 20
RESIDENT_FLOOR = 8 << 30
RESIDENT_STEP = 4 << 30
DEFAULT_LADDER = (24 << 30, 20 << 30, 16 << 30, 12 << 30, 8 << 30)
DEFAULT_PROMPT_LIMIT = 128 << 10


def machine_memory() -> int:
    """Bytes of physical memory on this machine."""
    if sys.platform == "darwin":
        libc = ctypes.CDLL("/usr/lib/libSystem.dylib", use_errno=True)
        value = ctypes.c_uint64(0)
        size = ctypes.c_size_t(ctypes.sizeof(value))
        if libc.sysctlbyname(b"hw.memsize", ctypes.byref(value), ctypes.byref(size), None, 0) != 0:
            raise OSError(ctypes.get_errno(), "hw.memsize")
        return int(value.value)
    return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))


def kv_bytes_per_token(
    attention_layers: int, *, kv_width: int = 512, index_width: int = 128
) -> int:
    """Bytes of K, V, and indexer key the runtime allocates per token."""
    if attention_layers < 0:
        raise ValueError("attention_layers")
    return attention_layers * (kv_width + kv_width + index_width) * 4


def residual_bytes_per_token(branches: int, hidden: int) -> int:
    return branches * hidden * 4


def scratch_bytes(chunk: int) -> int:
    return max(SCRATCH_FLOOR, chunk * SCRATCH_PER_TOKEN)


def _page_down(nbytes: int) -> int:
    if nbytes < PAGE:
        return 0
    return nbytes - (nbytes % PAGE)


def resident_candidates(
    hot_bytes: int, percent: float | None, ladder: tuple[int, ...] = DEFAULT_LADDER
) -> tuple[int, ...]:
    """Atlas sizes to try, largest first.

    ``percent is None`` keeps the device ladder. A percent is that share of the
    hot-tier bytes, page-aligned, then 4 GiB smaller down to 8 GiB.
    """
    if percent is None:
        return ladder
    try:
        number = float(percent)
    except (TypeError, ValueError) as exc:
        message = f"resident percent must be in (0, 100], got {percent}"
        raise ValueError(message) from exc
    if isinstance(percent, bool) or not math.isfinite(number) or not 0 < number <= 100:
        message = f"resident percent must be in (0, 100], got {percent}"
        raise ValueError(message)
    if hot_bytes <= 0:
        raise ValueError("hot tier is empty")
    budget = _page_down(int(hot_bytes * (float(percent) / 100.0)))
    if budget < PAGE:
        raise ValueError("resident budget is below one page")
    if budget <= RESIDENT_FLOOR:
        return (budget,)
    steps = [budget]
    size = budget
    while True:
        nxt = _page_down(size - RESIDENT_STEP)
        if nxt >= RESIDENT_FLOOR and nxt < size:
            steps.append(nxt)
            size = nxt
            continue
        if size > RESIDENT_FLOOR:
            steps.append(RESIDENT_FLOOR)
        break
    return tuple(steps)


def prompt_token_limit(
    *,
    memory: int,
    resident_hot: int,
    kv_per_token: int,
    residual_per_token: int,
    scratch: int,
    staging: int,
    fixed: int,
    score_per_token: int = 0,
    cap: int = DEFAULT_PROMPT_LIMIT,
) -> int:
    """Tokens that fit in the prefill peak and in the decode peak, at most ``cap``.

    Prefill holds the cache, the residual, one chunk of block scores, scratch,
    and the staging buffers. Decode holds the cache, scratch, and the resident
    hot-expert budget. ``fixed`` is the weights and headroom mapped in both.
    The default cap is 131,072 tokens.
    """
    if kv_per_token <= 0:
        raise ValueError("kv_per_token")
    prefill_each = kv_per_token + max(residual_per_token, 0) + max(score_per_token, 0)
    if prefill_each <= 0:
        raise ValueError("prefill bytes per token")
    prefill_room = memory - fixed - scratch - staging
    decode_room = memory - fixed - scratch - resident_hot
    prefill_tokens = prefill_room // prefill_each if prefill_room > 0 else 0
    decode_tokens = decode_room // kv_per_token if decode_room > 0 else 0
    fitted = max(0, min(prefill_tokens, decode_tokens))
    return min(fitted, cap)


def score_bits(value: float) -> int:
    """IEEE bits of a non-negative finite score. Other values rank as zero."""
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        number = 0.0
    return struct.unpack("<I", struct.pack("<f", number))[0]


def top_blocks(scores: Sequence[float], keep: int = KEEP_BLOCKS) -> list[int]:
    """Exact top blocks. A tie keeps the lower block index.

    Radix on the IEEE bits of a non-negative float. The shader uses the same
    passes so a query's mask matches this function.
    """
    n = len(scores)
    if keep <= 0 or n == 0:
        return []
    if n <= keep:
        return list(range(n))
    bits = [score_bits(score) for score in scores]
    prefix: list[tuple[int, int]] = []

    def active(index: int) -> bool:
        value = bits[index]
        return all(((value >> shift) & 255) == byte for shift, byte in prefix)

    picked: list[int] = []
    need = keep
    for shift in (24, 16, 8, 0):
        hist = [0] * 256
        for index in range(n):
            if active(index):
                hist[(bits[index] >> shift) & 255] += 1
        if sum(hist) <= need:
            picked.extend(index for index in range(n) if active(index))
            break
        remaining = need
        cut = 0
        for bucket in range(255, -1, -1):
            if hist[bucket] >= remaining:
                cut = bucket
                break
            remaining -= hist[bucket]
        for index in range(n):
            if active(index) and ((bits[index] >> shift) & 255) > cut:
                picked.append(index)
        if shift == 0:
            taken = 0
            for index in range(n):
                if taken >= remaining:
                    break
                if active(index) and ((bits[index] >> shift) & 255) == cut:
                    picked.append(index)
                    taken += 1
            break
        if remaining == 0:
            break
        prefix.append((shift, cut))
        need = remaining
    if len(picked) != min(keep, n):
        message = f"selected {len(picked)} blocks, expected {min(keep, n)}"
        raise RuntimeError(message)
    return sorted(picked)


def select_query_tokens(pos: int, scores: Sequence[float], keep: int = KEEP_BLOCKS) -> list[int]:
    """Causal token ids for one query.

    Complete blocks are those that end before the unfinished tail. The tail is
    ``[(pos // 4) * 4, pos]`` and always includes this query. When the complete
    blocks fit in ``keep``, the query attends every token up to itself.
    """
    if pos < 0:
        raise ValueError("pos")
    tail_start = (pos // BLOCK) * BLOCK
    n_complete = tail_start // BLOCK
    if n_complete <= keep:
        return list(range(pos + 1))
    if len(scores) < n_complete:
        raise ValueError("block scores are short")
    chosen = top_blocks(scores[:n_complete], keep)
    tokens: list[int] = []
    for block in chosen:
        tokens.extend(range(block * BLOCK, block * BLOCK + BLOCK))
    tokens.extend(range(tail_start, pos + 1))
    if len(tokens) > INDEX_STRIDE:
        message = f"selection is {len(tokens)} tokens, the attend window is {INDEX_STRIDE}"
        raise RuntimeError(message)
    return tokens


def block_score(queries: Sequence[Sequence[float]], keys: Sequence[Sequence[float]]) -> float:
    """Mean of the block keys, then the sum of ReLU dots over the query heads."""
    if not keys or not queries:
        raise ValueError("block_score")
    dim = len(keys[0])
    pooled = [sum(key[col] for key in keys) / len(keys) for col in range(dim)]
    total = 0.0
    for head in queries:
        dot = sum(head[col] * pooled[col] for col in range(dim))
        if dot > 0.0:
            total += dot
    return total
