"""Greedy MTP verify for this pipeline.

The head drafts ``mtp_depth`` tokens from the current residual. Absent, that
depth is three. The first three feeds share one fp4 weight read when three
are present. Two feeds share one read. Each later feed is a serial step. The DeltaNet state
and the convolution history are copied first into device snap buffers. A rejected
tail restores that snap and replays the recorded convolution and recurrence rows
for the kept tokens. The language weights are not read again. Temperature 0 keeps a draft that matches
the target argmax. A sampler keeps a draft that matches its draw. The
request flag ``mtp`` turns the verify off.
"""

from __future__ import annotations

import struct
from typing import Any

from qwen38_27b.layout import HIDDEN, LINEAR_LAYERS, LINEAR_V, MIX

DEPTH = 3
_STATE_BYTES = LINEAR_LAYERS * LINEAR_V * 128 * 128 * 4
_HISTORY_BYTES = LINEAR_LAYERS * 3 * MIX * 2


def draft_window(
    chosen: int, drafts: list[int], stop: set[int], room: int
) -> tuple[list[int], list[int]]:
    """Inputs to feed, and the draft token each input should predict.

    A stop id can be a prediction. It is not fed. ``room`` is the number of
    draft tokens the caller can still emit.
    """
    usable: list[int] = []
    for token in drafts[: max(room, 0)]:
        usable.append(int(token))
        if int(token) in stop:
            break
    if not usable:
        return [int(chosen)], []
    return [int(chosen), *usable[:-1]], usable


def propose(runtime: Any, token: int, anchor: int, depth: int = DEPTH) -> list[int]:
    """Draft ``depth`` tokens. ``hidden`` is the residual at ``anchor``."""
    drafts: list[int] = []
    current = int(token)
    at = int(anchor)
    for _ in range(depth):
        if at + 1 >= int(runtime.max_pos):
            break
        drafts.append(int(runtime._draft(current, at, backbone=not drafts)))
        current = drafts[-1]
        at += 1
    return drafts


def verify(
    runtime: Any,
    pos: int,
    feeds: list[int],
    expected: list[int],
    *,
    draw: Any = None,
    commit: Any = None,
) -> tuple[int, int, int]:
    """Score ``feeds`` at ``pos``. Return accepted drafts, the next token, and feeds kept.

    ``draw`` samples the row just copied into ``hidden``. Absent, the row's
    argmax is the target token. ``commit`` records a draft that matched before
    the next row is drawn. The next token is the first target token that missed
    its draft, or the last target token when every draft matched. The state
    includes the feeds that produced those tokens and does not include the
    returned token.
    """
    if len(feeds) != len(expected):
        message = f"{len(feeds)} feeds and {len(expected)} drafts"
        raise RuntimeError(message)
    _extend_rope(runtime, pos, len(feeds))
    from qwen38_27b.prefill import backbone_ids
    from qwen38_27b.span import (
        load_row,
        load_span_logits,
        prime_span_logits,
        replay_kept,
        score_span_argmax,
        span_pair,
        span_triple,
    )

    try:
        span_n = 3 if len(feeds) >= 3 else (2 if len(feeds) >= 2 else 0)
        workspace = None
        targets: list[int] | None = None
        if span_n == 3:
            # Snap copy is the first encode inside the span command buffer.
            span_triple(runtime, int(feeds[0]), int(feeds[1]), int(feeds[2]), pos)
            if draw is None:
                targets = score_span_argmax(runtime, span_n)
            else:
                prime_span_logits(runtime, span_n)
        elif span_n == 2:
            span_pair(runtime, int(feeds[0]), int(feeds[1]), pos)
            if draw is None:
                targets = score_span_argmax(runtime, span_n)
            else:
                prime_span_logits(runtime, span_n)
        else:
            _capture(runtime)
            workspace = backbone_ids(runtime, feeds, pos)
        if targets is not None:
            nxt = int(targets[0])
            accepted = 0
            while accepted < len(expected) and int(nxt) == int(expected[accepted]):
                accepted += 1
                if accepted == len(expected):
                    break
                if commit is not None:
                    commit(int(expected[accepted - 1]))
                nxt = int(targets[accepted])
        else:
            nxt = _score(
                runtime,
                workspace,
                0,
                draw,
                feeds,
                pos,
                span_n,
                primed=span_n > 0,
                load_span_logits=load_span_logits,
            )
            accepted = 0
            while accepted < len(expected) and int(nxt) == int(expected[accepted]):
                accepted += 1
                if accepted == len(expected):
                    break
                if commit is not None:
                    commit(int(expected[accepted - 1]))
                nxt = _score(
                    runtime,
                    workspace,
                    accepted,
                    draw,
                    feeds,
                    pos,
                    span_n,
                    primed=span_n > 0,
                    load_span_logits=load_span_logits,
                )
    except Exception:
        _restore(runtime)
        raise
    keep = len(feeds) if accepted == len(feeds) else accepted + 1
    if span_n and keep < span_n:
        _restore(runtime)
        replay_kept(runtime, keep)
        runtime.filled = pos + keep
        load_row(runtime, keep - 1)
    elif span_n == 0 and keep < len(feeds):
        _restore(runtime)
        for offset, token in enumerate(feeds[:keep]):
            runtime.step(int(token), pos + offset, logits=False, rope=_rope_at(runtime, pos + offset))
    else:
        runtime.filled = pos + keep
        if workspace is not None:
            _copy_hidden(runtime, workspace, keep - 1)
        elif span_n and keep <= span_n:
            load_row(runtime, keep - 1)
    cursor = getattr(runtime, "_rope_next", None)
    if cursor is not None:
        runtime._rope_next = int(cursor) + keep
    return accepted, int(nxt), keep


def _score(
    runtime: Any,
    workspace: Any,
    index: int,
    draw: Any,
    feeds: list[int],
    pos: int,
    span_n: int,
    *,
    primed: bool = False,
    load_span_logits: Any = None,
) -> int:
    if span_n and index < span_n:
        from qwen38_27b.span import load_row

        load_row(runtime, index)
        if primed and draw is not None and load_span_logits is not None:
            load_span_logits(runtime, index)
            return int(draw())
    elif span_n:
        runtime.step(
            int(feeds[index]),
            pos + index,
            logits=False,
            rope=_rope_at(runtime, pos + index),
        )
    else:
        _copy_hidden(runtime, workspace, index)
    if draw is None:
        return int(runtime._argmax_hidden())
    return int(draw())


def _copy_hidden(runtime: Any, workspace: Any, index: int) -> None:
    runtime.session.begin()
    try:
        runtime.session.encode(
            "copy_f32",
            [(workspace.hidden, index * HIDDEN * 4), (runtime.hidden, 0)],
            [struct.pack("<I", HIDDEN)],
            (HIDDEN // 256, 1, 1),
            (256, 1, 1),
        )
        runtime.session.end()
    except Exception:
        runtime.session.finish()
        raise


def _capture(runtime: Any) -> None:
    """Copy the DeltaNet state and the convolution history into the device snap."""
    state = getattr(runtime, "_mtp_state", None)
    history = getattr(runtime, "_mtp_history", None)
    if state is None or history is None:
        message = "mtp snap buffers are missing"
        raise RuntimeError(message)
    _copy_words(runtime, runtime.state, 0, state, 0, _STATE_BYTES // 4)
    _copy_words(runtime, runtime.history, 0, history, 0, _HISTORY_BYTES // 4)


def _restore(runtime: Any) -> None:
    """Copy the device snap back over the live state and history."""
    state = getattr(runtime, "_mtp_state", None)
    history = getattr(runtime, "_mtp_history", None)
    if state is None or history is None:
        message = "mtp snap buffers are missing"
        raise RuntimeError(message)
    _copy_words(runtime, state, 0, runtime.state, 0, _STATE_BYTES // 4)
    _copy_words(runtime, history, 0, runtime.history, 0, _HISTORY_BYTES // 4)


def _copy_words(
    runtime: Any, src: Any, src_off: int, dest: Any, dest_off: int, count: int
) -> None:
    runtime.session.begin()
    try:
        runtime.session.encode(
            "copy_f32",
            [(src, src_off), (dest, dest_off)],
            [struct.pack("<I", int(count))],
            ((int(count) + 255) // 256, 1, 1),
            (256, 1, 1),
        )
        runtime.session.end()
    except Exception:
        runtime.session.finish()
        raise


def _extend_rope(runtime: Any, pos: int, count: int) -> None:
    rope = getattr(runtime, "_rope_ids", None)
    cursor = getattr(runtime, "_rope_next", None)
    if not rope or cursor is None:
        return
    last = rope[-1]
    have = int(last[0]) + int(last[1])
    need = pos + count
    if have >= need:
        return
    extra = (have, need - have, int(cursor), 0)
    runtime._rope_ids = (*rope, extra)


def _rope_at(runtime: Any, index: int) -> tuple[int, int, int] | None:
    rope = getattr(runtime, "_rope_ids", None)
    if not rope:
        return None
    from qwen38_27b.vision import position_at

    return position_at(rope, index)
