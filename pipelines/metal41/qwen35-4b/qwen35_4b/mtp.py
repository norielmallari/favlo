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
    """Inputs to feed, and the draft token each input should predict.

    A stop id can be a prediction. It is not fed.
    """
    usable: list[int] = []
    for token in drafts[: max(room, 0)]:
        usable.append(int(token))
        if int(token) in stop:
            break
    if not usable:
        return [int(chosen)], []
    return [int(chosen), *usable[:-1]], usable


def kept_drafts(
    expected: list[int], accepted: int, stop: set[int]
) -> tuple[list[int], bool]:
    """Drafts to append, and whether a stop id ended the window.

    A confirmed stop id is not appended. Tokens after it are not appended.
    The caller does not continue from the token sampled past that stop.
    """
    kept: list[int] = []
    for token in expected[: max(accepted, 0)]:
        if int(token) in stop:
            return kept, True
        kept.append(int(token))
    return kept, False


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
    stop: set[int],
) -> tuple[int, int]:
    """Greedy verify. Returns accepted draft count and the next backbone token.

    ``feeds[0]`` is the confirmed token. Each accepted draft is stepped before
    the next sample, except a stop id. That id ends the window and stays out of
    the cache, so the next turn continues from the last kept token. ``pos`` is
    the language position of ``feeds[0]``.
    """
    del pos
    if len(feeds) != len(expected) and expected:
        if len(feeds) != len(expected):
            message = f"{len(feeds)} feeds and {len(expected)} drafts"
            raise RuntimeError(message)
    nxt = int(runtime._step(int(feeds[0]), sample=True))
    if not expected:
        return 0, nxt
    accepted = 0
    while accepted < len(expected) and nxt == int(expected[accepted]):
        accepted += 1
        if int(nxt) in stop:
            break
        nxt = int(runtime._step(int(expected[accepted - 1]), sample=True))
    return accepted, nxt
