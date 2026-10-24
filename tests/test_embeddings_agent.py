"""Harness check for one embeddings response. The listener is not started."""

from __future__ import annotations

import math

import pytest
from harness.embeddings_agent import HIDDEN, check_payload


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
