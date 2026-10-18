"""fp8 attention on the key and value planes.

Keys are stored ``[head, token, channel]`` with an E8M0 scale every 32 channels.
Values are stored ``[head, channel, token]`` with an E8M0 scale every 32 tokens.
Prefill scores 2048 keys at a time on the 64 by 64 ``matmul2d``. Softmax writes
float32 weights. Each value-mix window divides by that row's peak, multiplies
in float16, and restores the peak. A one-token step packs eight query rows,
two of them zero, into one ``matmul2d``. Six of those rows are the queries
that share a key/value head.
"""

from __future__ import annotations

import struct
from typing import Any

from qwen38_27b.layout import CHUNK, FULL_LAYERS, GROUP, HEAD, KV_HEADS

TILE = 256
SCORE = 2048
BLOCK = 512
DECODE_ROWS = 8


def scratch_bytes(positions: int, slots: int = FULL_LAYERS) -> int:
    """Score, probability, and value-block scratch for one chunk."""
    rows = GROUP * CHUNK
    blocks = max(positions, BLOCK) // BLOCK
    return (
        rows * HEAD * 2
        + rows * SCORE * 4
        + rows * SCORE * 4
        + rows * SCORE * 2
        + rows * 4
        + rows * HEAD * 4
        + rows * HEAD * 4
        + rows * 4
        + rows * 4
        + rows * (SCORE // TILE) * 8
        + slots * KV_HEADS * 32 * HEAD * 4
        + blocks * DECODE_ROWS * HEAD * 4
    )


def bind_plane(runtime: Any) -> None:
    """Allocate the score, probability, and value-block scratch."""
    rows = GROUP * CHUNK
    zeros = runtime.session.zeros
    runtime.plane_q = zeros(rows * HEAD * 2)
    runtime.plane_scores = zeros(rows * SCORE * 4)
    runtime.plane_probs = zeros(rows * SCORE * 4)
    runtime.plane_prob_half = zeros(rows * SCORE * 2)
    runtime.plane_row_scale = zeros(rows * 4)
    blocks = max(runtime.max_pos, BLOCK) // BLOCK
    runtime.plane_parts = zeros(blocks * DECODE_ROWS * HEAD * 4)
    runtime.plane_partial = zeros(rows * HEAD * 4)
    runtime.plane_acc = zeros(rows * HEAD * 4)
    runtime.plane_max = zeros(rows * 4)
    runtime.plane_sum = zeros(rows * 4)
    slots = int(getattr(runtime, "kv_slots", FULL_LAYERS))
    runtime.plane_tail = zeros(slots * KV_HEADS * 32 * HEAD * 4)
    runtime.plane_chunk_max = zeros(rows * (SCORE // TILE) * 4)
    runtime.plane_chunk_sum = zeros(rows * (SCORE // TILE) * 4)


def attend_plane(
    runtime: Any,
    *,
    query: Any,
    key: Any,
    value: Any,
    gate: Any,
    dest: Any,
    pos0: int,
    count: int,
    slot: int,
) -> None:
    """Store this chunk on the fp8 plane and mix it with ``matmul2d``."""
    if runtime.max_pos % TILE:
        message = f"context {runtime.max_pos} is not a multiple of {TILE}"
        raise RuntimeError(message)
    if count < 1 or count > CHUNK:
        message = f"attention count {count} is outside 1..{CHUNK}"
        raise RuntimeError(message)
    _store_keys(runtime, key, pos0, count, slot)
    _store_values(runtime, value, pos0, count, slot)
    decode = count == 1
    pad = 1 if decode else (count + 31) // 32 * 32
    rows = DECODE_ROWS if decode else GROUP * pad
    for head in range(KV_HEADS):
        _encode(
            runtime,
            "kv_row_init",
            [(runtime.plane_max, 0), (runtime.plane_sum, 0), (runtime.plane_acc, 0)],
            [],
            (rows, 1, 1),
            (HEAD, 1, 1),
        )
        _encode(
            runtime,
            "kv_cast_q",
            [(query, 0), (runtime.plane_q, 0)],
            [_u32(pad), _u32(head)],
            (count, DECODE_ROWS if decode else GROUP, 1),
            (HEAD, 1, 1),
        )
        if decode:
            _mix_decode(runtime, head, rows, pad, pos0, count, slot)
        else:
            _mix_tiles(runtime, head, rows, pad, pos0, count, slot)
        _encode(
            runtime,
            "kv_attend_write",
            [
                (runtime.plane_acc, 0),
                (runtime.plane_sum, 0),
                (gate, 0),
                (dest, 0),
            ],
            [_u32(pad), _u32(count), _u32(head)],
            (count, DECODE_ROWS if decode else GROUP, 1),
            (HEAD, 1, 1),
        )


def _store_keys(runtime: Any, key: Any, pos0: int, count: int, slot: int) -> None:
    scale = _scale_base(runtime.max_pos)
    _encode(
        runtime,
        "kv_plane_k",
        [(runtime.k_cache[slot], 0), (runtime.k_cache[slot], scale), (key, 0)],
        [_u32(pos0), _u32(runtime.max_pos)],
        (count, KV_HEADS, 1),
        (HEAD, 1, 1),
    )


def _store_values(runtime: Any, value: Any, pos0: int, count: int, slot: int) -> None:
    scale = _scale_base(runtime.max_pos)
    tail = slot * KV_HEADS * 32 * HEAD * 4
    src = 0
    pos = pos0
    while src < count:
        room = 32 - (pos % 32)
        take = min(room, count - src)
        block = pos - (pos % 32)
        _encode(
            runtime,
            "kv_tail_store",
            [(value, 0), (runtime.plane_tail, tail)],
            [_u32(src), _u32(pos % 32)],
            (take, KV_HEADS, 1),
            (HEAD, 1, 1),
        )
        _encode(
            runtime,
            "kv_plane_v",
            [(runtime.v_cache[slot], 0), (runtime.v_cache[slot], scale), (runtime.plane_tail, tail)],
            [_u32(block), _u32(pos % 32 + take), _u32(runtime.max_pos)],
            (HEAD, KV_HEADS, 1),
            (32, 1, 1),
        )
        src += take
        pos += take


def _mix_tiles(
    runtime: Any, head: int, rows: int, pad: int, pos0: int, count: int, slot: int
) -> None:
    limit = pos0 + count
    start = 0
    while start < limit:
        if limit - start >= SCORE:
            _mix_score(runtime, head, rows, pad, pos0, count, start, slot)
            start += SCORE
            continue
        _mix_one_tile(runtime, head, rows, pad, pos0, count, start, slot)
        start += TILE


def _mix_score(
    runtime: Any, head: int, rows: int, pad: int, pos0: int, count: int, start: int, slot: int
) -> None:
    data_off, scale_off = _key_tile(runtime, head, start)
    _matmul(
        runtime,
        "prefill_fp8_n64",
        runtime.plane_q,
        runtime.k_cache[slot],
        data_off,
        scale_off,
        runtime.plane_scores,
        rows,
        SCORE,
        HEAD,
    )
    _softmax_rows(runtime, SCORE, pad, count, pos0, start, rows)
    data_off, scale_off = _value_window(runtime, head, start)
    _matmul_stride(
        runtime,
        runtime.plane_probs,
        runtime.v_cache[slot],
        data_off,
        scale_off,
        runtime.plane_partial,
        rows,
        HEAD,
        SCORE,
        runtime.max_pos,
    )
    _acc_add(runtime, rows)


def _mix_one_tile(
    runtime: Any, head: int, rows: int, pad: int, pos0: int, count: int, start: int, slot: int
) -> None:
    data_off, scale_off = _key_tile(runtime, head, start)
    _matmul(
        runtime,
        "prefill_fp8_n64",
        runtime.plane_q,
        runtime.k_cache[slot],
        data_off,
        scale_off,
        runtime.plane_scores,
        rows,
        TILE,
        HEAD,
    )
    _softmax_tile(runtime, pad, count, pos0, start, rows)
    data_off, scale_off = _value_window(runtime, head, start)
    _matmul_stride(
        runtime,
        runtime.plane_probs,
        runtime.v_cache[slot],
        data_off,
        scale_off,
        runtime.plane_partial,
        rows,
        HEAD,
        TILE,
        runtime.max_pos,
    )
    _acc_add(runtime, rows)


def _mix_decode(
    runtime: Any, head: int, rows: int, pad: int, pos0: int, count: int, slot: int
) -> None:
    """One score product per span. Values are reduced in blocks of 512 keys."""
    del pad
    limit = pos0 + count
    aligned = (limit + BLOCK - 1) // BLOCK * BLOCK
    if aligned > runtime.max_pos:
        message = f"attention span {limit} does not fit context {runtime.max_pos}"
        raise RuntimeError(message)
    room = (GROUP * CHUNK * SCORE) // rows
    room = room // BLOCK * BLOCK
    if room < BLOCK:
        message = f"score buffer cannot hold {rows} rows"
        raise RuntimeError(message)
    start = 0
    while start < aligned:
        span = min(room, aligned - start)
        if span % TILE:
            message = f"decode span {span} is not a multiple of {TILE}"
            raise RuntimeError(message)
        data_off, scale_off = _key_tile(runtime, head, start)
        _matmul_m8(
            runtime,
            runtime.plane_q,
            runtime.k_cache[slot],
            data_off,
            scale_off,
            runtime.plane_scores,
            rows,
            span,
            HEAD,
        )
        _softmax8(runtime, span, pos0, start, rows)
        data_off, scale_off = _value_window(runtime, head, start)
        _pv_split8(runtime, data_off, scale_off, rows, span, slot)
        start += span


def _softmax_rows(
    runtime: Any, cols: int, pad: int, count: int, pos0: int, start: int, rows: int
) -> None:
    chunks = cols // TILE
    _encode(
        runtime,
        "kv_softmax_max_rows",
        [(runtime.plane_scores, 0), (runtime.plane_chunk_max, 0)],
        [_u32(cols), _u32(chunks), _u32(pad), _u32(count), _u32(pos0), _u32(start)],
        (chunks, rows, 1),
        (32, 1, 1),
    )
    _encode(
        runtime,
        "kv_softmax_prob_rows",
        [
            (runtime.plane_scores, 0),
            (runtime.plane_probs, 0),
            (runtime.plane_chunk_max, 0),
            (runtime.plane_max, 0),
            (runtime.plane_chunk_sum, 0),
        ],
        [_u32(cols), _u32(chunks), _u32(pad), _u32(count), _u32(pos0), _u32(start)],
        (chunks, rows, 1),
        (32, 1, 1),
    )
    _encode(
        runtime,
        "kv_softmax_close",
        [
            (runtime.plane_acc, 0),
            (runtime.plane_max, 0),
            (runtime.plane_sum, 0),
            (runtime.plane_chunk_max, 0),
            (runtime.plane_chunk_sum, 0),
        ],
        [_u32(chunks)],
        (rows, 1, 1),
        (32, 1, 1),
    )


def _softmax_tile(runtime: Any, pad: int, count: int, pos0: int, start: int, rows: int) -> None:
    _encode(
        runtime,
        "kv_softmax_tile",
        [
            (runtime.plane_scores, 0),
            (runtime.plane_probs, 0),
            (runtime.plane_acc, 0),
            (runtime.plane_max, 0),
            (runtime.plane_sum, 0),
        ],
        [_u32(pad), _u32(count), _u32(pos0), _u32(start)],
        (rows, 1, 1),
        (32, 1, 1),
    )


def _softmax8(runtime: Any, cols: int, pos0: int, start: int, rows: int) -> None:
    chunks = cols // TILE
    _encode(
        runtime,
        "kv_softmax_max",
        [(runtime.plane_scores, 0), (runtime.plane_chunk_max, 0)],
        [_u32(cols), _u32(chunks), _u32(pos0), _u32(start)],
        (chunks, rows, 1),
        (32, 1, 1),
    )
    _encode(
        runtime,
        "kv_softmax_prob",
        [
            (runtime.plane_scores, 0),
            (runtime.plane_probs, 0),
            (runtime.plane_chunk_max, 0),
            (runtime.plane_max, 0),
            (runtime.plane_chunk_sum, 0),
        ],
        [_u32(cols), _u32(chunks), _u32(pos0), _u32(start)],
        (chunks, rows, 1),
        (32, 1, 1),
    )
    _encode(
        runtime,
        "kv_softmax_close",
        [
            (runtime.plane_acc, 0),
            (runtime.plane_max, 0),
            (runtime.plane_sum, 0),
            (runtime.plane_chunk_max, 0),
            (runtime.plane_chunk_sum, 0),
        ],
        [_u32(chunks)],
        (rows, 1, 1),
        (32, 1, 1),
    )


def _acc_add(runtime: Any, rows: int) -> None:
    _encode(
        runtime,
        "kv_acc_add",
        [(runtime.plane_acc, 0), (runtime.plane_partial, 0)],
        [],
        (rows, 1, 1),
        (HEAD, 1, 1),
    )


def _key_tile(runtime: Any, head: int, start: int) -> tuple[int, int]:
    token = head * runtime.max_pos + start
    data = token * HEAD
    scale = _scale_base(runtime.max_pos) + token * (HEAD // 32)
    return data, scale


def _value_window(runtime: Any, head: int, start: int) -> tuple[int, int]:
    if start % 32:
        message = f"value window {start} is not a multiple of 32"
        raise RuntimeError(message)
    origin = (head * HEAD) * runtime.max_pos + start
    scale = (
        _scale_base(runtime.max_pos)
        + (head * HEAD) * (runtime.max_pos // 32)
        + start // 32
    )
    return origin, scale


def _scale_base(max_pos: int) -> int:
    return KV_HEADS * max_pos * HEAD


def _matmul(
    runtime: Any,
    name: str,
    left: Any,
    right: Any,
    data_off: int,
    scale_off: int,
    dest: Any,
    rows: int,
    cols: int,
    inner: int,
) -> None:
    width = int(runtime.session.pipes[name].threadExecutionWidth())
    _encode(
        runtime,
        name,
        [(left, 0), (right, data_off), (right, scale_off), (dest, 0)],
        [_u32(rows), _u32(cols), _u32(inner)],
        (cols // 64, rows // 64, 1),
        (width * 4, 1, 1),
    )


def _matmul_m8(
    runtime: Any,
    left: Any,
    right: Any,
    data_off: int,
    scale_off: int,
    dest: Any,
    rows: int,
    cols: int,
    inner: int,
) -> None:
    name = "prefill_fp8_m8"
    width = int(runtime.session.pipes[name].threadExecutionWidth())
    _encode(
        runtime,
        name,
        [(left, 0), (right, data_off), (right, scale_off), (dest, 0)],
        [_u32(rows), _u32(cols), _u32(inner)],
        (cols // TILE, 1, 1),
        (width * 2, 1, 1),
    )


def _matmul_stride(
    runtime: Any,
    left: Any,
    right: Any,
    data_off: int,
    scale_off: int,
    dest: Any,
    rows: int,
    cols: int,
    inner: int,
    stride: int,
) -> None:
    name = "prefill_fp8_n64_stride"
    _cast_probs(runtime, left, rows, inner, inner, 0)
    width = int(runtime.session.pipes[name].threadExecutionWidth())
    _encode(
        runtime,
        name,
        [(runtime.plane_prob_half, 0), (right, data_off), (right, scale_off), (dest, 0)],
        [_u32(rows), _u32(cols), _u32(inner), _u32(stride)],
        (cols // 64, rows // 64, 1),
        (width * 4, 1, 1),
    )
    _scale_partial(runtime, dest, rows, cols)


def _pv_split8(
    runtime: Any, data_off: int, scale_off: int, rows: int, span: int, slot: int
) -> None:
    """Mix every 512-key block of this head in one dispatch."""
    blocks = span // BLOCK
    _cast_probs(runtime, runtime.plane_probs, rows, BLOCK, span, 0, blocks)
    name = "kv_pv_split8"
    width = int(runtime.session.pipes[name].threadExecutionWidth())
    _encode(
        runtime,
        name,
        [
            (runtime.plane_prob_half, 0),
            (runtime.v_cache[slot], data_off),
            (runtime.v_cache[slot], scale_off),
            (runtime.plane_parts, 0),
            (runtime.plane_row_scale, 0),
        ],
        [
            _u32(rows),
            _u32(HEAD),
            _u32(BLOCK),
            _u32(BLOCK),
            _u32(runtime.max_pos),
        ],
        (HEAD // 64, blocks, 1),
        (width * 4, 1, 1),
    )
    _encode(
        runtime,
        "kv_part_sum",
        [(runtime.plane_parts, 0), (runtime.plane_acc, 0)],
        [_u32(blocks), _u32(rows * HEAD)],
        ((rows * HEAD + 255) // 256, 1, 1),
        (256, 1, 1),
    )


def _cast_probs(
    runtime: Any,
    source: Any,
    rows: int,
    cols: int,
    src_stride: int,
    src_col: int,
    blocks: int = 1,
) -> None:
    """Pack float32 weights into float16, one scale per row of each block."""
    _encode(
        runtime,
        "kv_cast_probs",
        [
            (source, 0),
            (runtime.plane_prob_half, 0),
            (runtime.plane_row_scale, 0),
        ],
        [_u32(cols), _u32(src_stride), _u32(src_col), _u32(rows)],
        (rows, blocks, 1),
        (32, 1, 1),
    )


def _scale_partial(
    runtime: Any, dest: Any, rows: int, channels: int, blocks: int = 1
) -> None:
    """Undo the per-row scale after the float16 value mix."""
    _encode(
        runtime,
        "kv_scale_partial",
        [(dest, 0), (runtime.plane_row_scale, 0)],
        [_u32(channels), _u32(rows)],
        (rows, blocks, 1),
        (channels, 1, 1),
    )


def _encode(
    runtime: Any,
    name: str,
    bindings: list[tuple[Any, int]],
    constants: list[bytes],
    groups: tuple[int, int, int],
    threads: tuple[int, int, int],
) -> None:
    runtime.session.encode(name, bindings, constants, groups, threads)


def _u32(value: int) -> bytes:
    return struct.pack("<I", value)
