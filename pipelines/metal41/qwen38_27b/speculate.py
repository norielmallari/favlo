"""Greedy MTP verify for this pipeline.

The head drafts ``mtp_depth`` tokens from the current residual. Absent, that
depth is three. The first three feeds share one fp4 weight read when three
are present. Two feeds share one read. Each later feed is a serial step. The DeltaNet state
and the convolution history are copied first. A rejected tail restores that copy
and replays the recorded convolution and recurrence rows for the kept tokens.
The language weights are not read again. Temperature 0 keeps a draft that matches
the target argmax. A host sampler keeps a draft that matches its draw. The
request flag ``mtp`` turns the verify off.
"""

from __future__ import annotations

from typing import Any

import numpy as np

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
    snap = _capture(runtime)
    from qwen38_27b.prefill import backbone_ids
    from qwen38_27b.span import load_row, replay_kept, span_pair, span_triple

    try:
        span_n = 3 if len(feeds) >= 3 else (2 if len(feeds) >= 2 else 0)
        workspace = None
        if span_n == 3:
            span_triple(runtime, int(feeds[0]), int(feeds[1]), int(feeds[2]), pos)
        elif span_n == 2:
            span_pair(runtime, int(feeds[0]), int(feeds[1]), pos)
        else:
            workspace = backbone_ids(runtime, feeds, pos)
        nxt = _score(runtime, workspace, 0, draw, feeds, pos, span_n)
        accepted = 0
        while accepted < len(expected) and int(nxt) == int(expected[accepted]):
            accepted += 1
            if accepted == len(expected):
                break
            if commit is not None:
                commit(int(expected[accepted - 1]))
            nxt = _score(runtime, workspace, accepted, draw, feeds, pos, span_n)
    except Exception:
        _restore(runtime, snap)
        raise
    keep = len(feeds) if accepted == len(feeds) else accepted + 1
    if span_n and keep < span_n:
        _restore(runtime, snap)
        replay_kept(runtime, keep)
        runtime.filled = pos + keep
        load_row(runtime, keep - 1)
    elif span_n == 0 and keep < len(feeds):
        _restore(runtime, snap)
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
) -> int:
    if span_n and index < span_n:
        from qwen38_27b.span import load_row

        load_row(runtime, index)
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
    start = index * HIDDEN
    row = runtime.session.download_f32(workspace.hidden, (index + 1) * HIDDEN)[start : start + HIDDEN]
    view = memoryview(runtime.hidden.contents().as_buffer(HIDDEN * 4))
    view[:] = np.ascontiguousarray(row, dtype=np.float32).tobytes()


def _capture(runtime: Any) -> tuple[bytes, bytes]:
    session = runtime.session
    return session.download(runtime.state, _STATE_BYTES), session.download(runtime.history, _HISTORY_BYTES)


def _restore(runtime: Any, snap: tuple[bytes, bytes]) -> None:
    state, history = snap
    _put(runtime.state, state)
    _put(runtime.history, history)


def _put(buffer: Any, blob: bytes) -> None:
    view = memoryview(buffer.contents().as_buffer(len(blob)))
    view[:] = blob


def _extend_rope(runtime: Any, pos: int, count: int) -> None:
    rope = getattr(runtime, "_rope_ids", None)
    cursor = getattr(runtime, "_rope_next", None)
    if rope is None or cursor is None:
        return
    have = int(len(rope))
    need = pos + count
    if have >= need:
        return
    angle = int(cursor)
    extra = np.zeros((need - have, 3), dtype=np.uint32)
    for row in range(need - have):
        extra[row] = angle
        angle += 1
    runtime._rope_ids = np.concatenate((np.asarray(rope), extra))


def _rope_at(runtime: Any, index: int) -> tuple[int, int, int] | None:
    rope = getattr(runtime, "_rope_ids", None)
    if rope is None or index >= len(rope):
        return None
    row = rope[index]
    return int(row[0]), int(row[1]), int(row[2])
