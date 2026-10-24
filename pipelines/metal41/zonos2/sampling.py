"""Published inference sampler. The card fields are the registry contract."""

from __future__ import annotations

import numpy as np

CONTEXT_LENGTH = 6144

VARIANTS = {
    "default": {
        "temperature": 1.15,
        "top_p": 0.0,
        "top_k": 106,
        "presence_penalty": 0.0,
    }
}
DEFAULT_VARIANT = "default"

MIN_P = 0.18
REPETITION_PENALTY = 1.2
REPETITION_WINDOW = 50
REPETITION_CODEBOOKS = 8


def sample_code(
    logits: np.ndarray,
    seen: list[int],
    rng: np.random.Generator,
    *,
    allow_eoa: bool,
) -> int:
    """One codebook. End-of-audio and pad stay available when ``allow_eoa`` is set."""
    card = VARIANTS[DEFAULT_VARIANT]
    scores = np.array(logits, dtype=np.float64, copy=True)
    scores[(1026 if allow_eoa else 1024) :] = -1.0e9
    for code in set(seen[-REPETITION_WINDOW:]):
        if code >= scores.size:
            continue
        if scores[code] > 0.0:
            scores[code] /= REPETITION_PENALTY
        else:
            scores[code] *= REPETITION_PENALTY
    scores /= float(card["temperature"])
    finite = np.flatnonzero(np.isfinite(scores) & (scores > -1.0e8))
    if finite.size == 0:
        message = "every code was masked"
        raise RuntimeError(message)
    if int(card["top_k"]) > 0 and finite.size > int(card["top_k"]):
        order = finite[np.argpartition(scores[finite], -int(card["top_k"]))[-int(card["top_k"]) :]]
        masked = np.full(scores.shape, -np.inf)
        masked[order] = scores[order]
        scores = masked
    scores = np.where(np.isfinite(scores), scores, -1.0e9)
    scores -= np.max(scores)
    probs = np.exp(scores)
    total = probs.sum()
    if total == 0.0:
        message = "the codebook distribution is empty"
        raise RuntimeError(message)
    probs /= total
    floor = MIN_P * float(probs.max())
    kept = probs >= floor
    if not np.any(kept):
        kept[int(np.argmax(probs))] = True
    probs = np.where(kept, probs, 0.0)
    probs /= probs.sum()
    return int(rng.choice(scores.size, p=probs))
