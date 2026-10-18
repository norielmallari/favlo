"""Host sampling. No GPU."""

from __future__ import annotations

import numpy as np
from harness.benchmarks.serve import Request, apply_sampling, card_model_id, select_sampling
from qwen38_27b.forward import sample_token
from qwen38_flash_next.runtime import Qwen38Runtime


def test_temperature_zero_is_the_argmax() -> None:
    logits = np.array([0.1, 3.0, 2.0], dtype=np.float32)
    drawn = sample_token(logits, 0.0, 1.0, 0, 0.0, 0.0, {}, {}, np.random.default_rng(0))
    assert drawn == 1


def test_presence_moves_a_close_logit() -> None:
    logits = np.array([1.0, 1.2, 0.0], dtype=np.float32)
    drawn = sample_token(logits, 0.0, 1.0, 0, 0.5, 0.0, {1: 1}, {}, np.random.default_rng(0))
    assert drawn == 0


def test_frequency_scales_with_count() -> None:
    logits = np.array([1.0, 2.0], dtype=np.float32)
    once = sample_token(logits, 0.0, 1.0, 0, 0.0, 0.4, {1: 1}, {}, np.random.default_rng(0))
    twice = sample_token(logits, 0.0, 1.0, 0, 0.0, 0.6, {1: 2}, {}, np.random.default_rng(0))
    assert once == 1
    assert twice == 0


def test_top_k_excludes_the_tail() -> None:
    logits = np.array([5.0, 0.0, 4.0], dtype=np.float32)
    rng = np.random.default_rng(0)
    for _ in range(20):
        drawn = sample_token(logits, 1.0, 1.0, 2, 0.0, 0.0, {}, {}, rng)
        assert drawn in (0, 2)


def test_flash_applies_presence_and_frequency_together() -> None:
    runtime = Qwen38Runtime.__new__(Qwen38Runtime)
    runtime.tool_names = None
    runtime._frequency_penalty = 0.5
    runtime._presence_penalty = 1.0
    runtime._penalty_window = 20
    logits = np.array([3.0, 0.0], dtype=np.float32)
    adjusted = runtime._prepare(logits, 0.0, [0, 0])
    assert float(adjusted[0]) == 3.0 - 0.5 - 0.5 - 1.0
    assert float(adjusted[1]) == 0.0


def test_card_fills_omitted_fields_and_keeps_overrides() -> None:
    assert card_model_id("qwen38-27b", thinking=False) == "qwen38-27b:instruct"
    assert card_model_id("qwen38-27b:thinking", thinking=False) == "qwen38-27b:thinking"
    filled = apply_sampling(
        Request(temperature=0.0, enable_thinking=False),
        {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5},
    )
    assert filled.temperature == 0.0
    assert filled.top_p == 0.8
    assert filled.top_k == 20
    assert filled.presence_penalty == 1.5
    card = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}
    rows = [
        {
            "id": "folder:instruct",
            "pipeline": "qwen38-flash-next",
            "variant": "instruct",
            "sampling": card,
        }
    ]
    assert select_sampling(rows, "qwen38-flash-next:instruct") == card
