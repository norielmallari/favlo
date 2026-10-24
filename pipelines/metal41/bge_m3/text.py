"""Host math for one BGE-M3 sequence.

Position ids follow the XLM-RoBERTa rule: a non-pad token's id is
``padding_idx + cumsum(not pad)``, and a pad stays at ``padding_idx``.
The dense vector is the L2 norm of hidden row 0. The lexical weight is
``ReLU(W_lex^T h)``, and a repeated token id keeps the max. Chen et al.,
arXiv:2402.03216, section 3.2.
"""

from __future__ import annotations

import numpy as np

PAD_ID = 1
HIDDEN = 1024
BUCKETS = (128, 256, 512, 1024, 2048, 4096)


def bucket_for(count: int, buckets: tuple[int, ...] = BUCKETS) -> int:
    """The static function length that covers ``count`` tokens."""
    if count < 1:
        message = "input is empty"
        raise ValueError(message)
    for size in buckets:
        if count <= size:
            return size
    limit = buckets[-1] if buckets else 0
    message = f"input is {count} tokens and the limit is {limit}"
    raise ValueError(message)


def lengths_from_names(names: list[str]) -> tuple[int, ...]:
    """``encode_N`` names, in increasing length. The record lists the packs it has."""
    lengths: list[int] = []
    for name in names:
        if not name.startswith("encode_"):
            message = f"{name} is not an encode function"
            raise ValueError(message)
        text = name.removeprefix("encode_")
        if not text.isdecimal():
            message = f"{name} is not an encode function"
            raise ValueError(message)
        lengths.append(int(text))
    if not lengths or lengths != sorted(set(lengths)):
        message = "encode functions are not an increasing set"
        raise ValueError(message)
    return tuple(lengths)


def position_ids(token_ids: list[int]) -> list[int]:
    """One position id per token. Pads stay at 1. The first real token is 2."""
    positions: list[int] = []
    seen = 0
    for token in token_ids:
        if token == PAD_ID:
            positions.append(PAD_ID)
            continue
        seen += 1
        positions.append(seen + PAD_ID)
    return positions


def key_mask(token_ids: list[int]) -> np.ndarray:
    """``[1, 1, length]`` float16. Pads are -40000 and other tokens are 0."""
    mask = np.zeros((1, 1, len(token_ids)), dtype=np.float16)
    fill = np.float16(-40000.0)
    for index, token in enumerate(token_ids):
        if token == PAD_ID:
            mask[0, 0, index] = fill
    return mask


def pad_ids(token_ids: list[int], length: int) -> list[int]:
    """Pad with the pad id up to a bucket length."""
    if len(token_ids) > length:
        message = f"input is {len(token_ids)} tokens and the bucket is {length}"
        raise ValueError(message)
    return token_ids + [PAD_ID] * (length - len(token_ids))


def l2_normalize(row: np.ndarray) -> np.ndarray:
    """Unit vector. A zero row is an error."""
    vector = np.asarray(row, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        message = "embedding norm is zero"
        raise ValueError(message)
    return (vector / np.float32(norm)).astype(np.float32)


def lexical_max(
    rows: np.ndarray,
    token_ids: list[int],
    weight: np.ndarray,
    bias: float,
) -> dict[int, float]:
    """Max ReLU weight for each non-pad token id."""
    hidden = np.asarray(rows, dtype=np.float32)
    if hidden.shape[0] != len(token_ids):
        message = "lexical rows do not match the token ids"
        raise ValueError(message)
    scale = np.asarray(weight, dtype=np.float32).reshape(-1)
    scores = hidden @ scale + np.float32(bias)
    scores = np.maximum(scores, np.float32(0))
    best: dict[int, float] = {}
    for token, score in zip(token_ids, scores, strict=True):
        if token == PAD_ID:
            continue
        value = float(score)
        current = best.get(int(token))
        if current is None or value > current:
            best[int(token)] = value
    return best
