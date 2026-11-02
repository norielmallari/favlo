"""Dense MTP draft and greedy serial verify for Qwen3.5-4B.

The official head predicts the next-next token from the backbone residual and
the embedding of the just-chosen token. Weights are Metal ``half``. The live
MTP attention plane uses the same Metal ``half`` / ``fp8_e4m3`` cache mode as
the language layers. Greedy verify only steps confirmed tokens into the
language cache, so a rejected draft needs no DeltaNet restore.
"""

from __future__ import annotations

from typing import Any

DEPTH = 3


def draft_window(
    chosen: int, drafts: list[int], stop: set[int], room: int
) -> tuple[list[int], list[int]]:
    """Inputs to feed, and the draft token each input should predict."""
    usable: list[int] = []
    for token in drafts[: max(room, 0)]:
        usable.append(int(token))
        if int(token) in stop:
            break
    if not usable:
        return [int(chosen)], []
    return [int(chosen), *usable[:-1]], usable


def propose(runtime: Any, token: int, anchor: int, depth: int = DEPTH) -> list[int]:
    """Draft ``depth`` tokens. ``hidden`` holds the backbone residual at ``anchor``."""
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
) -> tuple[int, int]:
    """Greedy verify. Returns accepted draft count and the next backbone token.

    ``feeds[0]`` is the confirmed token. Each accepted draft is stepped before
    the next sample. ``pos`` is the language position of ``feeds[0]``.
    """
    del pos
    if len(feeds) != len(expected) and expected:
        if len(feeds) != len(expected):
            message = f"{len(feeds)} feeds and {len(expected)} drafts"
            raise RuntimeError(message)
    runtime._step(int(feeds[0]))
    nxt = int(runtime._sample())
    if not expected:
        return 0, nxt
    accepted = 0
    while accepted < len(expected) and nxt == int(expected[accepted]):
        accepted += 1
        runtime._step(int(expected[accepted - 1]))
        nxt = int(runtime._sample())
    return accepted, nxt
