"""Resident expert budget.

The atlas takes the memory the live sequence does not. A shorter sequence leaves
a larger atlas. The indexer window is separate: a query attends at most 512
complete blocks of 4, plus the unfinished tail.
"""

from __future__ import annotations

import ctypes
import math
import os
import struct
import sys
from collections.abc import Sequence

from qwen38_flash_next.formats import (
    DEFAULT_KV_CACHE,
    INDEX_BF16_BYTES,
    KV_BF16_BYTES,
    KV_E4M3_BYTES,
    PAGE,
    kv_cache_name,
)

# The dense attend kernel stores one score per visible token in scores[2052].
# 512 blocks of 4, plus a tail of at most 4, is the sparse window.
KEEP_BLOCKS = 512
BLOCK = 4
INDEX_STRIDE = KEEP_BLOCKS * BLOCK + BLOCK
DENSE_LIMIT = INDEX_STRIDE - 1
PREFILL_CHUNK = 2048
# Physical memory kept for the rest of the machine. The prompt budget is the
# remainder.
HEADROOM = 6 << 30
SCRATCH_FLOOR = 256 << 20
SCRATCH_PER_TOKEN = 2 << 20
RESIDENT_FLOOR = 8 << 30
RESIDENT_STEP = 4 << 30
DEFAULT_LADDER = (24 << 30, 20 << 30, 16 << 30, 12 << 30, 8 << 30)
DEFAULT_PROMPT_LIMIT = 128 << 10


class _RUsageInfoV0(ctypes.Structure):
    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
        ("ri_child_user_time", ctypes.c_uint64),
        ("ri_child_system_time", ctypes.c_uint64),
        ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_child_interrupt_wkups", ctypes.c_uint64),
        ("ri_child_pageins", ctypes.c_uint64),
        ("ri_child_elapsed_abstime", ctypes.c_uint64),
        ("ri_diskio_bytesread", ctypes.c_uint64),
        ("ri_diskio_byteswritten", ctypes.c_uint64),
    ]


def process_footprint() -> int:
    """Physical footprint of this process.

    Jetsam uses this number. It includes the resident weights and the live
    cache, and it does not include an atlas that has not been allocated yet.
    """
    if sys.platform != "darwin":
        import resource

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    libc = ctypes.CDLL("/usr/lib/libSystem.dylib", use_errno=True)
    info = _RUsageInfoV0()
    if libc.proc_pid_rusage(os.getpid(), 0, ctypes.byref(info)) != 0:
        raise OSError(ctypes.get_errno(), "proc_pid_rusage")
    return int(info.ri_phys_footprint)


def cache_growth_bytes(tokens: int, per_token: int) -> int:
    """Bytes the next cache growth holds while the previous copy is still mapped.

    A short cache doubles until it passes one prefill chunk. Past that, the
    next growth keeps the live copy. The larger of those two is the reserve.
    """
    if per_token <= 0:
        raise ValueError("per_token")
    live = max(0, tokens) * per_token
    doubling = 2 * PREFILL_CHUNK * per_token
    return max(live, doubling)


def atlas_budget(
    *,
    memory: int,
    content_bytes: int,
    footprint: int,
    growth: int,
) -> int:
    """Largest atlas that leaves headroom, the live process, and one cache growth.

    ``content_bytes`` is the hot and cold expert files. The atlas is not larger
    than the bytes it can hold.
    """
    reserve = HEADROOM + max(0, growth)
    room = memory - max(0, footprint) - reserve
    if room < PAGE:
        return PAGE
    cap = _page_down(room)
    if content_bytes > 0:
        cap = min(cap, _page_down(content_bytes))
    if cap < PAGE:
        return PAGE
    return cap


def atlas_fallbacks(budget: int, *, floor: int = RESIDENT_FLOOR) -> tuple[int, ...]:
    """``budget`` first, then 1 GiB smaller, then 4 GiB steps down to ``floor``."""
    budget = _page_down(budget) if budget >= PAGE else PAGE
    if budget <= floor:
        return (max(budget, PAGE),)
    steps = [budget]
    size = budget
    step = 1 << 30
    while size > floor:
        nxt = _page_down(size - step)
        if floor <= nxt < size:
            steps.append(nxt)
            size = nxt
            step = RESIDENT_STEP
            continue
        steps.append(floor)
        break
    return tuple(steps)


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


def kv_bytes_per_token(attention_layers: int, *, cache: str = DEFAULT_KV_CACHE) -> int:
    """Bytes of K, V, and the indexer key stored per token.

    The E4M3 tail is at most three BF16 tokens and is not part of this count.
    """
    if attention_layers < 0:
        raise ValueError("attention_layers")
    name = kv_cache_name(cache)
    pair = KV_E4M3_BYTES * 2 if name == "e4m3" else KV_BF16_BYTES * 2
    return attention_layers * (pair + INDEX_BF16_BYTES)


def residual_bytes_per_token(branches: int, hidden: int) -> int:
    return branches * hidden * 4


def scratch_bytes(chunk: int) -> int:
    return max(SCRATCH_FLOOR, chunk * SCRATCH_PER_TOKEN)


def _page_down(nbytes: int) -> int:
    if nbytes < PAGE:
        return 0
    return nbytes - (nbytes % PAGE)


def _ladder_cap(content_bytes: int, memory: int) -> int:
    """Largest atlas that leaves room for the OS, both decode arenas, and weights.

    ``content_bytes`` is the hot and cold expert files. Both tiers are copied
    into the atlas, so the cap is that sum, not the hot files alone.
    """
    reserve = HEADROOM + 2 * scratch_bytes(PREFILL_CHUNK) + (4 << 30)
    cap = _page_down(memory - reserve) if memory > reserve else RESIDENT_FLOOR
    if content_bytes > 0:
        cap = min(cap, _page_down(content_bytes))
    if cap < PAGE:
        return PAGE
    return cap


def resident_candidates(
    hot_bytes: int,
    percent: float | None,
    ladder: tuple[int, ...] = DEFAULT_LADDER,
    memory: int | None = None,
    cold_bytes: int = 0,
) -> tuple[int, ...]:
    """Atlas sizes to try, largest first.

    ``percent is None`` keeps the device ladder, dropping a step larger than the
    hot and cold expert files together, or larger than the memory left after
    headroom, both decode scratch arenas, and the resident weight file. A
    percent is that share of the hot-tier bytes only, page-aligned, then 4 GiB
    smaller down to 8 GiB.
    """
    if percent is None:
        content = hot_bytes + max(0, cold_bytes)
        cap = _ladder_cap(content, machine_memory() if memory is None else memory)
        picked = tuple(size for size in ladder if size <= cap)
        if picked:
            return picked
        return (min(ladder[-1], cap),)
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
