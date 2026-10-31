"""MTP draft and verify for this pipeline.

The paired step drafts one token and reads each dense language weight once
for that token and the draft. ``mtp_depth`` above one does not widen the row.
Temperature 0 keeps the draft when it is the target argmax. A sampled request
keeps the draft when a uniform draw falls under the target probability divided
by the draft probability, and a miss draws from the positive remainder. The
request flag ``mtp`` turns the verify off. Sampling settings do not.
"""

from __future__ import annotations

from typing import Any

import numpy as np

DEPTH = 3
_STATE_BYTES = 30 * 32 * 128 * 128 * 4
_HISTORY_BYTES = 30 * 3 * 8192 * 2


def penalized(
    logits: np.ndarray,
    presence: float,
    seen: set[int],
    banned: set[int] | None,
    frequency: float,
    counts: dict[int, int] | None,
) -> np.ndarray:
    """Copy ``logits`` when a penalty or a ban applies. Otherwise return them."""
    counted = counts or {}
    if not ((presence and seen) or (frequency and counted) or banned):
        return logits
    scores = np.array(logits, copy=True)
    if presence and seen:
        penalty = np.float32(presence)
        for token in seen:
            scores[token] -= penalty
    if frequency and counted:
        scale = np.float32(frequency)
        for token, count in counted.items():
            if count > 0:
                scores[token] -= scale * count
    if banned:
        for token in banned:
            scores[token] = np.float32(-np.inf)
    return scores


def candidate_probs(
    scores: np.ndarray,
    temperature: float,
    top_p: float,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Token ids and probabilities after temperature, top-k, and top-p.

    Temperature 0 is the argmax with probability 1.
    """
    if temperature == 0.0:
        winner = int(np.argmax(scores))
        return np.asarray([winner], dtype=np.int64), np.asarray([1.0], dtype=np.float64)
    if top_k > 0 and top_k < scores.size:
        keep = np.argpartition(scores, -top_k)[-top_k:]
        chosen = np.asarray(scores[keep], dtype=np.float64)
        local, probs = _distribution(chosen, temperature, top_p)
        return np.asarray(keep, dtype=np.int64)[local], probs
    chosen = np.asarray(scores, dtype=np.float64)
    local, probs = _distribution(chosen, temperature, top_p)
    return local, probs


def probs_from_top(
    ids: np.ndarray,
    values: np.ndarray,
    temperature: float,
    top_p: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Temperature and top-p over ids that are already the top-k."""
    local, probs = _distribution(np.asarray(values, dtype=np.float64), temperature, top_p)
    return np.asarray(ids, dtype=np.int64)[local], probs


def merge_top(
    values: np.ndarray,
    ids: np.ndarray,
    k: int,
    temperature: float,
    top_p: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse per-thread top lists, then apply temperature and top-p."""
    flat_v = np.asarray(values, dtype=np.float64).reshape(-1)
    flat_i = np.asarray(ids, dtype=np.int64).reshape(-1)
    if k > 0 and flat_v.size > k:
        pick = np.argpartition(flat_v, -k)[-k:]
        flat_v = flat_v[pick]
        flat_i = flat_i[pick]
    return probs_from_top(flat_i, flat_v, temperature, top_p)


def keep_draft(
    draft: int,
    p_idx: np.ndarray,
    p_pr: np.ndarray,
    q_idx: np.ndarray,
    q_pr: np.ndarray,
) -> tuple[bool, int]:
    """Keep ``draft`` when a uniform draw falls under ``p(draft) / q(draft)``.

    A miss returns a draw from the positive part of ``p - q``.
    """
    chance = _at(p_idx, p_pr, draft)
    proposal = _at(q_idx, q_pr, draft)
    if proposal > 0.0 and float(np.random.random()) < min(1.0, chance / proposal):
        return True, int(draft)
    return False, _residual(p_idx, p_pr, q_idx, q_pr)


def _distribution(
    chosen: np.ndarray, temperature: float, top_p: float
) -> tuple[np.ndarray, np.ndarray]:
    scaled = chosen / temperature
    scaled -= np.max(scaled)
    probs = np.exp(scaled)
    total = float(np.sum(probs))
    if not np.isfinite(total) or total <= 0.0:
        winner = int(np.argmax(chosen))
        return np.asarray([winner], dtype=np.int64), np.asarray([1.0], dtype=np.float64)
    probs /= total
    if 0.0 < top_p < 1.0:
        order = np.argsort(probs)[::-1]
        running = np.cumsum(probs[order])
        cut = int(np.searchsorted(running, top_p) + 1)
        order = np.asarray(order[:cut], dtype=np.int64)
        kept = probs[order]
        kept = kept / float(np.sum(kept))
        return order, kept
    return np.arange(chosen.size, dtype=np.int64), probs


def _at(idx: np.ndarray, probs: np.ndarray, token: int) -> float:
    hit = np.nonzero(idx == int(token))[0]
    if hit.size == 0:
        return 0.0
    return float(probs[int(hit[0])])


def _residual(
    p_idx: np.ndarray, p_pr: np.ndarray, q_idx: np.ndarray, q_pr: np.ndarray
) -> int:
    keys = np.union1d(p_idx, q_idx).astype(np.int64, copy=False)
    target = np.zeros(keys.size, dtype=np.float64)
    draft = np.zeros(keys.size, dtype=np.float64)
    for index, token in enumerate(keys):
        target[index] = _at(p_idx, p_pr, int(token))
        draft[index] = _at(q_idx, q_pr, int(token))
    resid = np.maximum(0.0, target - draft)
    total = float(np.sum(resid))
    if not np.isfinite(total) or total <= 0.0:
        return int(p_idx[int(np.argmax(p_pr))])
    resid /= total
    return int(keys[int(np.random.choice(keys.size, p=resid))])


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
    """Draft ``depth`` tokens. ``hidden`` is the residual at ``anchor``.

    A recorded next step stays recorded. Its embed is submitted after the
    draft, so the draft's write to ``hidden`` is finished before those layers run.
    """
    return runtime._draft_chain(int(token), int(anchor), int(depth))


def verify(
    runtime: Any,
    pos: int,
    feeds: list[int],
    expected: list[int],
    *,
    draw: Any,
    commit: Any,
    primed: Any = None,
) -> tuple[int, int, int]:
    """Score ``feeds`` at ``pos``. Return accepted drafts, the next token, and feeds kept.

    ``draw`` samples the logits of the row just stepped. ``commit`` records a
    draft that matched before the next row is drawn. The next token is the
    first target that missed its draft, or the last target when every draft
    matched. The state includes the feeds that produced those tokens and does
    not include the returned token. ``primed`` is the logit row of ``feeds[0]``
    when that step was already queued behind the draft.
    """
    if len(feeds) != len(expected):
        message = f"{len(feeds)} feeds and {len(expected)} drafts"
        raise RuntimeError(message)
    snap = _capture(runtime)
    preds: list[int] = []
    try:
        if primed is not None:
            preds.append(int(draw(primed)))
        for index, feed in enumerate(feeds):
            if index < len(preds):
                if preds[index] != int(expected[index]):
                    break
                if commit is not None and index + 1 < len(expected):
                    commit(int(expected[index]))
                continue
            logits = runtime.step(int(feed), pos + index, logits=True)
            take = getattr(runtime, "_take_routes", None)
            if take is not None:
                take(prefill=False)
            if logits is None:
                message = "verify step returned no logits"
                raise RuntimeError(message)
            pred = int(draw(logits))
            preds.append(pred)
            if pred != int(expected[index]):
                break
            if commit is not None and index + 1 < len(expected):
                commit(int(expected[index]))
    except Exception:
        _restore(runtime, snap)
        raise
    accepted = 0
    while (
        accepted < len(expected)
        and accepted < len(preds)
        and preds[accepted] == int(expected[accepted])
    ):
        accepted += 1
    keep = len(feeds) if accepted == len(expected) else accepted + 1
    if keep < len(feeds):
        _restore(runtime, snap)
        runtime._drop_pending()
        for offset, feed in enumerate(feeds[:keep]):
            runtime.step(int(feed), pos + offset, logits=False)
    runtime.filled = pos + keep
    runtime._mtp_snapped = False
    return accepted, int(preds[keep - 1]), keep


def _capture(runtime: Any) -> tuple[bytes, bytes] | None:
    if getattr(runtime, "_mtp_snapped", False):
        return None
    session = runtime.session
    return (
        session.download(runtime.state, _STATE_BYTES),
        session.download(runtime.history, _HISTORY_BYTES),
    )


def _restore(runtime: Any, snap: tuple[bytes, bytes] | None) -> None:
    if snap is None:
        runtime._restore_mtp_snap()
        return
    state, history = snap
    _put(runtime.state, state)
    _put(runtime.history, history)


def _put(buffer: Any, blob: bytes) -> None:
    view = memoryview(buffer.contents().as_buffer(len(blob)))
    view[:] = blob
