"""Harness checks for embeddings responses. The listener is not started."""

from __future__ import annotations

import math

import pytest
from harness.embeddings_agent import HIDDEN, SENTENCES, check_payload, check_scores


def _vector() -> list[float]:
    value = 1.0 / math.sqrt(HIDDEN)
    return [value] * HIDDEN


def test_a_unit_vector_passes() -> None:
    check_payload(
        {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": _vector()}],
            "usage": {"prompt_tokens": 6, "total_tokens": 6},
        }
    )


def test_a_short_vector_fails() -> None:
    with pytest.raises(RuntimeError, match="length"):
        check_payload(
            {
                "object": "list",
                "data": [{"object": "embedding", "embedding": [1.0]}],
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            }
        )


def _unit(*pairs: tuple[int, float]) -> list[float]:
    row = [0.0] * HIDDEN
    for index, value in pairs:
        row[index] = value
    norm = math.sqrt(sum(value * value for value in row))
    return [value / norm for value in row]


def test_paraphrases_outrank_other_topics() -> None:
    vectors = {
        "bridge": _unit((0, 1.0)),
        "bridge_near": _unit((0, 0.95), (1, 0.05)),
        "cat": _unit((2, 1.0)),
        "cat_near": _unit((2, 0.95), (3, 0.05)),
        "orbit": _unit((4, 1.0)),
        "wheat": _unit((5, 1.0)),
    }
    rows = check_scores(vectors)
    assert {kind for kind, _left, _right, _score in rows} == {"near", "far"}


def test_tied_scores_fail() -> None:
    one = _unit((0, 1.0))
    vectors = {name: list(one) for name, _text in SENTENCES}
    with pytest.raises(RuntimeError, match="near minimum"):
        check_scores(vectors)


def test_a_cross_topic_pair_above_a_paraphrase_fails() -> None:
    vectors = {
        "bridge": _unit((0, 1.0)),
        "bridge_near": _unit((1, 1.0)),
        "cat": _unit((2, 1.0)),
        "cat_near": _unit((2, 0.9), (3, 0.1)),
        "orbit": _unit((0, 1.0)),
        "wheat": _unit((5, 1.0)),
    }
    with pytest.raises(RuntimeError, match="near minimum"):
        check_scores(vectors)
