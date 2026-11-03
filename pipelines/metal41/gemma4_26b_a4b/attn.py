"""fp8 attention.

Prefill dequantizes every key head of a tile and scores that plane with
``matmul_half_64_batch``. Decode scores every key head with ``matmul_half_batch``.
The value mix is the same product. Scores are the raw dot. A prefill chunk
commits every 1024 keys so the window server can run. A decode step keeps its
dispatches in the layer buffer.
"""

from __future__ import annotations

from typing import Any

from gemma4_26b_a4b.geometry import GLOBAL_DIM, Q_HEADS

_TILE = 256
_ROWS = 1024 * 8
# Eight local heads of a 1024-token chunk, or two global heads, at a 256-key tile.
_SCORE = 8 * 2048 * _TILE
_DECODE = 32768


def bind_attention(runtime: Any) -> None:
    """Scratch for one fp8 attention tile. Rows cover a 1024-token global chunk."""
    zeros = runtime.session.zeros
    runtime.attn_q = zeros(2 * _ROWS * GLOBAL_DIM * 2)
    runtime.attn_k = zeros(2 * _DECODE * GLOBAL_DIM * 2)
    runtime.attn_v = zeros(2 * _DECODE * GLOBAL_DIM * 2)
    runtime.attn_scores = zeros(_SCORE * 4)
    runtime.attn_probs = zeros(_SCORE * 2)
    runtime.attn_part = zeros(2 * _ROWS * GLOBAL_DIM * 4)


def attend_fp8(
    runtime: Any,
    queries: object,
    keys: object,
    values: object,
    destination: object,
    *,
    pos0: int,
    count: int,
    kv_heads: int,
    dim: int,
    window: int,
) -> None:
    """Attend ``count`` queries starting at ``pos0``. One key head at a time."""
    if count < 1 or count > 1024:
        message = f"attention count {count} is outside 1..1024"
        raise RuntimeError(message)
    if count == 1:
        _decode_matmul(runtime, queries, keys, values, destination, pos0, kv_heads, dim, window)
        return
    group = Q_HEADS // kv_heads
    length = pos0 + count
    earliest = 0
    if window and pos0 + 1 > window:
        earliest = pos0 + 1 - window
    rows = count * group
    rows_pad = (rows + 63) // 64 * 64
    span = kv_heads * rows_pad
    tile = min(_TILE, (_SCORE // span) // 64 * 64)
    if tile < 64:
        message = f"attention tile {tile} does not fit {kv_heads} heads"
        raise RuntimeError(message)
    runtime._zero(runtime.fp8_acc, count * Q_HEADS * dim * 4)
    for head in range(kv_heads):
        _encode(
            runtime,
            "kv_cast_q",
            [queries, runtime.attn_q],
            [
                _u32(runtime, count),
                _u32(runtime, group),
                _u32(runtime, dim),
                _u32(runtime, head),
                _u32(runtime, Q_HEADS),
                _u32(runtime, rows_pad),
            ],
            (rows_pad, 1, 1),
            (dim, 1, 1),
            origins=[0, head * rows_pad * dim * 2],
        )
    key = earliest
    first = 1
    walked = 0
    while key < length:
        cols = min(tile, length - key)
        cols_pad = (cols + 63) // 64 * 64
        _dequant_heads(runtime, keys, kv_heads, dim, key, cols, cols_pad, length, 0)
        _matmul64_batch(
            runtime,
            runtime.attn_q,
            runtime.attn_k,
            runtime.attn_scores,
            rows_pad,
            cols_pad,
            dim,
            kv_heads,
        )
        for head in range(kv_heads):
            plane = head * rows_pad * cols_pad
            _softmax(
                runtime,
                count,
                group,
                dim,
                head,
                cols,
                cols_pad,
                pos0,
                key,
                window,
                first,
                rows_pad=rows_pad,
                origins=[plane * 4, plane * 2, 0, 0],
            )
        _dequant_heads(runtime, values, kv_heads, dim, key, cols, cols_pad, length, 1)
        _matmul64_batch(
            runtime,
            runtime.attn_probs,
            runtime.attn_v,
            runtime.attn_part,
            rows_pad,
            dim,
            cols_pad,
            kv_heads,
        )
        for head in range(kv_heads):
            _encode(
                runtime,
                "kv_acc_add",
                [runtime.fp8_acc, runtime.attn_part],
                [
                    _u32(runtime, count),
                    _u32(runtime, group),
                    _u32(runtime, dim),
                    _u32(runtime, head),
                    _u32(runtime, Q_HEADS),
                ],
                (rows, 1, 1),
                (dim, 1, 1),
                origins=[0, head * rows_pad * dim * 4],
            )
        first = 0
        key += cols
        walked += cols
        if key < length and walked >= 1024:
            runtime._yield_gpu()
            walked = 0
    _encode(
        runtime,
        "kv_attend_store",
        [runtime.fp8_acc, runtime.fp8_stats, destination],
        [
            _u32(runtime, count),
            _u32(runtime, dim),
            _u32(runtime, Q_HEADS),
        ],
        (count, Q_HEADS, 1),
        (dim, 1, 1),
    )


def _decode_matmul(
    runtime: Any,
    queries: object,
    keys: object,
    values: object,
    destination: object,
    pos0: int,
    kv_heads: int,
    dim: int,
    window: int,
) -> None:
    """One token. Every key head shares one score product and one value product."""
    length = pos0 + 1
    start = 0
    if window and length > window:
        start = length - window
    group = Q_HEADS // kv_heads
    rows_pad = 8
    runtime._zero(runtime.fp8_acc, Q_HEADS * dim * 4)
    for head in range(kv_heads):
        _encode(
            runtime,
            "kv_cast_q",
            [queries, runtime.attn_q],
            [
                _u32(runtime, 1),
                _u32(runtime, group),
                _u32(runtime, dim),
                _u32(runtime, head),
                _u32(runtime, Q_HEADS),
                _u32(runtime, rows_pad),
            ],
            (rows_pad, 1, 1),
            (dim, 1, 1),
            origins=[0, head * rows_pad * dim * 2],
        )
    key = start
    first = 1
    while key < length:
        cols = min(_DECODE, length - key)
        cols_pad = (cols + 31) // 32 * 32
        _encode(
            runtime,
            "kv_dequant_heads",
            [keys, runtime.attn_k],
            [
                _u32(runtime, kv_heads),
                _u32(runtime, dim),
                _u32(runtime, key),
                _u32(runtime, cols),
                _u32(runtime, cols_pad),
                _u32(runtime, length),
                _u32(runtime, 0),
            ],
            (cols_pad, kv_heads, 1),
            (32, 1, 1),
        )
        _matmul_batch(
            runtime, runtime.attn_q, runtime.attn_k, runtime.attn_scores, rows_pad, cols_pad, dim, kv_heads
        )
        for head in range(kv_heads):
            plane = head * rows_pad * cols_pad
            _softmax(
                runtime,
                1,
                group,
                dim,
                head,
                cols,
                cols_pad,
                pos0,
                key,
                window,
                first,
                rows_pad=rows_pad,
                origins=[plane * 4, plane * 2, 0, 0],
            )
        _encode(
            runtime,
            "kv_dequant_heads",
            [values, runtime.attn_v],
            [
                _u32(runtime, kv_heads),
                _u32(runtime, dim),
                _u32(runtime, key),
                _u32(runtime, cols),
                _u32(runtime, cols_pad),
                _u32(runtime, length),
                _u32(runtime, 1),
            ],
            (cols_pad, kv_heads, 1),
            (32, 1, 1),
        )
        _matmul_batch(
            runtime, runtime.attn_probs, runtime.attn_v, runtime.attn_part, rows_pad, dim, cols_pad, kv_heads
        )
        for head in range(kv_heads):
            _encode(
                runtime,
                "kv_acc_add",
                [runtime.fp8_acc, runtime.attn_part],
                [
                    _u32(runtime, 1),
                    _u32(runtime, group),
                    _u32(runtime, dim),
                    _u32(runtime, head),
                    _u32(runtime, Q_HEADS),
                ],
                (group, 1, 1),
                (dim, 1, 1),
                origins=[0, head * rows_pad * dim * 4],
            )
        first = 0
        key += cols
    _encode(
        runtime,
        "kv_attend_store",
        [runtime.fp8_acc, runtime.fp8_stats, destination],
        [_u32(runtime, 1), _u32(runtime, dim), _u32(runtime, Q_HEADS)],
        (1, Q_HEADS, 1),
        (dim, 1, 1),
    )


def _dequant_heads(
    runtime: Any,
    cache: object,
    heads: int,
    dim: int,
    key: int,
    cols: int,
    cols_pad: int,
    length: int,
    transposed: int,
) -> None:
    dest = runtime.attn_k if transposed == 0 else runtime.attn_v
    _encode(
        runtime,
        "kv_dequant_heads",
        [cache, dest],
        [
            _u32(runtime, heads),
            _u32(runtime, dim),
            _u32(runtime, key),
            _u32(runtime, cols),
            _u32(runtime, cols_pad),
            _u32(runtime, length),
            _u32(runtime, transposed),
        ],
        (cols_pad, heads, 1),
        (32, 1, 1),
    )


def _softmax(
    runtime: Any,
    count: int,
    group: int,
    dim: int,
    head: int,
    cols: int,
    cols_pad: int,
    pos0: int,
    key: int,
    window: int,
    first: int,
    rows_pad: int | None = None,
    origins: list[int] | None = None,
) -> None:
    if rows_pad is None:
        rows_pad = (count * group + 63) // 64 * 64
    _encode(
        runtime,
        "kv_softmax_raw",
        [runtime.attn_scores, runtime.attn_probs, runtime.fp8_acc, runtime.fp8_stats],
        [
            _u32(runtime, count),
            _u32(runtime, group),
            _u32(runtime, dim),
            _u32(runtime, head),
            _u32(runtime, Q_HEADS),
            _u32(runtime, cols),
            _u32(runtime, cols_pad),
            _u32(runtime, pos0),
            _u32(runtime, key),
            _u32(runtime, window),
            _u32(runtime, first),
        ],
        (rows_pad, 1, 1),
        (256, 1, 1),
        origins=origins,
    )


def _matmul64_batch(
    runtime: Any,
    left: object,
    right: object,
    dest: object,
    rows: int,
    cols: int,
    inner: int,
    heads: int,
) -> None:
    pipe = runtime.session.pipes["matmul_half_64_batch"]
    width = int(pipe.threadExecutionWidth())
    _encode(
        runtime,
        "matmul_half_64_batch",
        [left, right, dest],
        [_u32(runtime, rows), _u32(runtime, cols), _u32(runtime, inner)],
        (cols // 64, rows // 64, heads),
        (width * 4, 1, 1),
    )


def _matmul_batch(
    runtime: Any,
    left: object,
    right: object,
    dest: object,
    rows: int,
    cols: int,
    inner: int,
    heads: int,
) -> None:
    pipe = runtime.session.pipes["matmul_half_batch"]
    width = int(pipe.threadExecutionWidth())
    _encode(
        runtime,
        "matmul_half_batch",
        [left, right, dest],
        [_u32(runtime, rows), _u32(runtime, cols), _u32(runtime, inner)],
        (cols // 32, rows // 8, heads),
        (width * 4, 1, 1),
    )


def _encode(
    runtime: Any,
    name: str,
    handles: list[object],
    constants: list[bytes],
    groups: tuple[int, int, int],
    threads: tuple[int, int, int],
    origins: list[int] | None = None,
) -> None:
    runtime.session.encode(name, handles, constants, groups, threads, origins=origins)


def _u32(runtime: Any, value: int) -> bytes:
    return runtime.session._u32(value)
