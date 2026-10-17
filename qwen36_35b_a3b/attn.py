"""Attention on the fp8 or fp4 plane.

Keys are stored ``[head, token, channel]`` with an E8M0 scale every 32 channels.
Values are stored ``[head, channel, token]`` with an E8M0 scale every 32 tokens.
Prefill scores up to 4096 keys at a time: one 64 by 64 ``matmul2d`` for the scores,
a parallel softmax over 256-key chunks, and one strided value mix.
A one-token fp8 step packs the eight live query rows into one 8-row
``matmul2d``, softmaxes 256-key chunks in parallel, and splits the value mix
into blocks of 512 keys.
The score product is a dense ``matmul2d``. The value mix is a ``matmul2d`` with
the context length as its token stride. Softmax runs on the score tile.
"""

from __future__ import annotations

import struct
from typing import Any

TILE = 256
SCORE = 4096
BLOCK = 512
HEAD = 256
CHUNK = 2048
GROUP = 8
LAYERS = 10
_KV = 2
_SWEEP = 160


def plane_bytes(positions: int = 262144) -> int:
    """Scratch for one chunk of plane attention, reserved with the atlas."""
    rows = GROUP * CHUNK
    blocks = max(positions, BLOCK) // BLOCK
    return (
        rows * HEAD * 2
        + rows * SCORE * 4
        + rows * SCORE * 2
        + rows * HEAD * 4
        + rows * HEAD * 4
        + rows * 4
        + rows * 4
        + LAYERS * _KV * 32 * HEAD * 4
        + blocks * (GROUP * GROUP) * HEAD * 4
        + rows * (SCORE // TILE) * 4 * 2
    )


def bind_plane(runtime: Any) -> None:
    """Allocate the score, probability, and value-block scratch."""
    rows = GROUP * CHUNK
    zeros = runtime.session.zeros
    runtime.plane_q = zeros(rows * HEAD * 2)
    runtime.plane_scores = zeros(rows * SCORE * 4)
    runtime.plane_probs = zeros(rows * SCORE * 2)
    blocks = max(runtime.max_pos, BLOCK) // BLOCK
    runtime.plane_parts = zeros(blocks * (GROUP * GROUP) * HEAD * 4)
    runtime.plane_partial = zeros(rows * HEAD * 4)
    runtime.plane_acc = zeros(rows * HEAD * 4)
    runtime.plane_max = zeros(rows * 4)
    runtime.plane_sum = zeros(rows * 4)
    runtime.plane_tail = zeros(LAYERS * _KV * 32 * HEAD * 4)
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
    """Store this chunk on the plane and mix it with ``matmul2d``."""
    if runtime.max_pos % TILE:
        message = f"context {runtime.max_pos} is not a multiple of {TILE}"
        raise RuntimeError(message)
    if count < 1 or count > CHUNK:
        message = f"attention count {count} is outside 1..{CHUNK}"
        raise RuntimeError(message)
    mode = 0 if runtime.kv_cache == "e4m3" else 1
    _store_keys(runtime, key, pos0, count, mode, slot)
    _store_values(runtime, value, pos0, count, mode, slot)
    decode8 = count == 1 and mode == 0
    pad = 1 if decode8 else (count + GROUP - 1) // GROUP * GROUP
    rows = GROUP if decode8 else GROUP * pad
    dense = "prefill_fp8_n64" if mode == 0 else "prefill_fp4_n64"
    strided = "prefill_fp8_n64_stride" if mode == 0 else "prefill_fp4_n64_stride"
    for head in range(2):
        _zero(runtime, runtime.plane_q, rows * HEAD // 2)
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
            (count, GROUP, 1),
            (HEAD, 1, 1),
        )
        if count == 1:
            _mix_decode(runtime, dense, strided, head, rows, pad, pos0, count, mode, slot)
        else:
            _mix_tiles(runtime, dense, strided, head, rows, pad, pos0, count, mode, slot)
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
            (count, GROUP, 1),
            (HEAD, 1, 1),
        )


def _store_keys(
    runtime: Any, key: Any, pos0: int, count: int, mode: int, slot: int
) -> None:
    scale = _scale_base(runtime.max_pos, mode)
    _encode(
        runtime,
        "kv_plane_k",
        [
            (runtime.k_cache[slot], 0),
            (runtime.k_cache[slot], scale),
            (key, 0),
        ],
        [_u32(pos0), _u32(mode), _u32(runtime.max_pos)],
        (count, 2, 1),
        (HEAD, 1, 1),
    )


def _store_values(
    runtime: Any, value: Any, pos0: int, count: int, mode: int, slot: int
) -> None:
    tail = slot * _KV * 32 * HEAD * 4
    scale = _scale_base(runtime.max_pos, mode)
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
            (take, 2, 1),
            (HEAD, 1, 1),
        )
        _encode(
            runtime,
            "kv_plane_v",
            [
                (runtime.v_cache[slot], 0),
                (runtime.v_cache[slot], scale),
                (runtime.plane_tail, tail),
            ],
            [
                _u32(block),
                _u32(pos % 32 + take),
                _u32(mode),
                _u32(runtime.max_pos),
            ],
            (HEAD, 2, 1),
            (32, 1, 1),
        )
        src += take
        pos += take


def _mix_tiles(
    runtime: Any,
    dense: str,
    strided: str,
    head: int,
    rows: int,
    pad: int,
    pos0: int,
    count: int,
    mode: int,
    slot: int,
) -> None:
    limit = pos0 + count
    start = 0
    while start < limit:
        if mode == 0:
            width = min(SCORE, limit - start) // TILE * TILE
            if width >= TILE:
                _mix_score(
                    runtime, strided, head, rows, pad, pos0, count, mode, slot, start, width
                )
                start += width
                continue
        _mix_one_tile(runtime, dense, strided, head, rows, pad, pos0, count, mode, slot, start)
        start += TILE


def _mix_score(
    runtime: Any,
    strided: str,
    head: int,
    rows: int,
    pad: int,
    pos0: int,
    count: int,
    mode: int,
    slot: int,
    start: int,
    cols: int,
) -> None:
    data_off, scale_off = _key_tile(runtime, head, start, mode)
    _matmul(
        runtime,
        "prefill_fp8_n64",
        runtime.plane_q,
        runtime.k_cache[slot],
        data_off,
        scale_off,
        runtime.plane_scores,
        rows,
        cols,
        HEAD,
    )
    _softmax_rows(runtime, cols, pad, count, pos0, start)
    data_off, scale_off = _value_window(runtime, head, start, mode)
    _matmul_stride(
        runtime,
        strided,
        runtime.plane_probs,
        runtime.v_cache[slot],
        data_off,
        scale_off,
        runtime.plane_partial,
        rows,
        HEAD,
        cols,
        runtime.max_pos,
    )
    _acc_add(runtime, rows)


def _mix_one_tile(
    runtime: Any,
    dense: str,
    strided: str,
    head: int,
    rows: int,
    pad: int,
    pos0: int,
    count: int,
    mode: int,
    slot: int,
    start: int,
) -> None:
    data_off, scale_off = _key_tile(runtime, head, start, mode)
    _matmul(
        runtime,
        dense,
        runtime.plane_q,
        runtime.k_cache[slot],
        data_off,
        scale_off,
        runtime.plane_scores,
        rows,
        TILE,
        HEAD,
    )
    _softmax_tile(runtime, pad, count, pos0, start)
    data_off, scale_off = _value_window(runtime, head, start, mode)
    _matmul_stride(
        runtime,
        strided,
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
    runtime: Any,
    dense: str,
    strided: str,
    head: int,
    rows: int,
    pad: int,
    pos0: int,
    count: int,
    mode: int,
    slot: int,
) -> None:
    """One score product per span. fp8 values are reduced in blocks of 512 keys."""
    limit = pos0 + count
    aligned = (limit + BLOCK - 1) // BLOCK * BLOCK
    if aligned > runtime.max_pos or aligned < limit:
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
        data_off, scale_off = _key_tile(runtime, head, start, mode)
        if mode == 0 and rows == GROUP:
            if span % TILE:
                message = f"decode span {span} is not a multiple of {TILE}"
                raise RuntimeError(message)
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
            _softmax8(runtime, span, pos0, start)
            data_off, scale_off = _value_window(runtime, head, start, mode)
            _pv_split8(runtime, data_off, scale_off, slot, rows, span)
            start += span
            continue
        if mode == 0 and rows == GROUP * GROUP:
            _matmul_sweep(
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
        else:
            _matmul(
                runtime,
                dense,
                runtime.plane_q,
                runtime.k_cache[slot],
                data_off,
                scale_off,
                runtime.plane_scores,
                rows,
                span,
                HEAD,
            )
        _softmax_span(runtime, span, pad, count, pos0, start)
        data_off, scale_off = _value_window(runtime, head, start, mode)
        if mode == 0 and rows == GROUP * GROUP and span % BLOCK == 0:
            _pv_split(runtime, data_off, scale_off, slot, rows, span)
        else:
            _matmul_stride(
                runtime,
                strided,
                runtime.plane_probs,
                runtime.v_cache[slot],
                data_off,
                scale_off,
                runtime.plane_partial,
                rows,
                HEAD,
                span,
                runtime.max_pos,
            )
            _acc_add(runtime, rows)
        start += span


def _softmax_rows(
    runtime: Any, cols: int, pad: int, count: int, pos0: int, start: int
) -> None:
    chunks = cols // TILE
    _encode(
        runtime,
        "kv_softmax_max_rows",
        [(runtime.plane_scores, 0), (runtime.plane_chunk_max, 0)],
        [
            _u32(cols),
            _u32(chunks),
            _u32(pad),
            _u32(count),
            _u32(pos0),
            _u32(start),
        ],
        (chunks, GROUP * pad, 1),
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
        [
            _u32(cols),
            _u32(chunks),
            _u32(pad),
            _u32(count),
            _u32(pos0),
            _u32(start),
        ],
        (chunks, GROUP * pad, 1),
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
        (GROUP * pad, 1, 1),
        (32, 1, 1),
    )


def _softmax_tile(runtime: Any, pad: int, count: int, pos0: int, start: int) -> None:
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
        (GROUP * pad, 1, 1),
        (32, 1, 1),
    )


def _softmax_span(
    runtime: Any, cols: int, pad: int, count: int, pos0: int, start: int
) -> None:
    _encode(
        runtime,
        "kv_softmax_span",
        [
            (runtime.plane_scores, 0),
            (runtime.plane_probs, 0),
            (runtime.plane_acc, 0),
            (runtime.plane_max, 0),
            (runtime.plane_sum, 0),
        ],
        [_u32(cols), _u32(pad), _u32(count), _u32(pos0), _u32(start)],
        (GROUP * pad, 1, 1),
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


def _key_tile(runtime: Any, head: int, start: int, mode: int) -> tuple[int, int]:
    token = head * runtime.max_pos + start
    width = 256 if mode == 0 else 128
    data = token * width
    scale = _scale_base(runtime.max_pos, mode) + token * 8
    return data, scale


def _value_window(runtime: Any, head: int, start: int, mode: int) -> tuple[int, int]:
    if start % 32:
        message = f"value window {start} is not a multiple of 32"
        raise RuntimeError(message)
    origin = (head * HEAD) * runtime.max_pos + start
    data = origin if mode == 0 else origin // 2
    scale = (
        _scale_base(runtime.max_pos, mode)
        + (head * HEAD) * (runtime.max_pos // 32)
        + start // 32
    )
    return data, scale


def _scale_base(max_pos: int, mode: int) -> int:
    width = 256 if mode == 0 else 128
    return _KV * max_pos * width


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
    groups = cols // 64 if name.endswith("n64") else cols // 32
    _encode(
        runtime,
        name,
        [(left, 0), (right, data_off), (right, scale_off), (dest, 0)],
        [_u32(rows), _u32(cols), _u32(inner)],
        (groups, rows // 64, 1),
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


def _softmax8(runtime: Any, cols: int, pos0: int, start: int) -> None:
    chunks = cols // TILE
    _encode(
        runtime,
        "kv_softmax_max",
        [(runtime.plane_scores, 0), (runtime.plane_chunk_max, 0)],
        [_u32(cols), _u32(chunks), _u32(pos0), _u32(start)],
        (chunks, GROUP, 1),
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
        (chunks, GROUP, 1),
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
        (GROUP, 1, 1),
        (32, 1, 1),
    )


def _matmul_sweep(
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
    name = "prefill_fp8_n64_sweep"
    width = int(runtime.session.pipes[name].threadExecutionWidth())
    ntiles = cols // 64
    groups = ntiles if ntiles < _SWEEP else _SWEEP
    _encode(
        runtime,
        name,
        [(left, 0), (right, data_off), (right, scale_off), (dest, 0)],
        [_u32(rows), _u32(cols), _u32(inner), _u32(groups)],
        (groups, 1, 1),
        (width * 4, 1, 1),
    )


def _pv_split8(
    runtime: Any,
    data_off: int,
    scale_off: int,
    slot: int,
    rows: int,
    span: int,
) -> None:
    name = "kv_pv_split8"
    width = int(runtime.session.pipes[name].threadExecutionWidth())
    blocks = span // BLOCK
    _encode(
        runtime,
        name,
        [
            (runtime.plane_probs, 0),
            (runtime.v_cache[slot], data_off),
            (runtime.v_cache[slot], scale_off),
            (runtime.plane_parts, 0),
        ],
        [
            _u32(rows),
            _u32(HEAD),
            _u32(BLOCK),
            _u32(span),
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
        ((rows * HEAD) // 256, 1, 1),
        (256, 1, 1),
    )


def _pv_split(
    runtime: Any,
    data_off: int,
    scale_off: int,
    slot: int,
    rows: int,
    span: int,
) -> None:
    name = "kv_pv_split"
    width = int(runtime.session.pipes[name].threadExecutionWidth())
    blocks = span // BLOCK
    _encode(
        runtime,
        name,
        [
            (runtime.plane_probs, 0),
            (runtime.v_cache[slot], data_off),
            (runtime.v_cache[slot], scale_off),
            (runtime.plane_parts, 0),
        ],
        [
            _u32(rows),
            _u32(HEAD),
            _u32(BLOCK),
            _u32(span),
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
        ((rows * HEAD) // 256, 1, 1),
        (256, 1, 1),
    )


def _matmul_stride(
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
    stride: int,
) -> None:
    width = int(runtime.session.pipes[name].threadExecutionWidth())
    _encode(
        runtime,
        name,
        [(left, 0), (right, data_off), (right, scale_off), (dest, 0)],
        [_u32(rows), _u32(cols), _u32(inner), _u32(stride)],
        (cols // 64, rows // 64, 1),
        (width * 4, 1, 1),
    )


def _zero(runtime: Any, buf: Any, n: int) -> None:
    _encode(runtime, "zero_f32", [(buf, 0)], [], (n // 256, 1, 1), (256, 1, 1))


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
