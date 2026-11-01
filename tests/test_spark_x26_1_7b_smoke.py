"""Carrot completions for Spark-X2.6-1.7B. Skips when the record is not installed.

Greedy instruct is temperature 0. The none card is instruct. high and low
change the template and share the sampling card. The reply is 6 carrots.
"""

from __future__ import annotations

import re

import pytest

from spark_x26_1_7b.chat import decode_text, encode_chat, split_thinking
from spark_x26_1_7b.commands import CARROT_PROMPT, installed_record
from spark_x26_1_7b.forward import SparkRuntime

_INSTRUCT_TOKENS = 384
_THINK_TOKENS = 768
_ANSWER = re.compile(
    r"(?:\*\*|__)?(?:6|six)(?:\*\*|__)?(?:\s+\w+){0,3}\s+carrots?\b",
    re.IGNORECASE,
)


@pytest.fixture(scope="module")
def carrot_runtime():
    root = installed_record()
    if root is None:
        pytest.skip("quantized record is not installed")
    runtime = SparkRuntime(root)
    try:
        yield runtime
    finally:
        runtime.close()


def _complete(runtime: SparkRuntime, *, thinking: bool, effort: str | None, temperature: float) -> str:
    runtime.reset()
    prompt = encode_chat(
        runtime.root,
        [{"role": "user", "content": CARROT_PROMPT}],
        None,
        thinking,
        effort,
    )
    limit = _THINK_TOKENS if thinking else _INSTRUCT_TOKENS
    tokens = runtime.generate(
        prompt,
        limit,
        {1},
        temperature=temperature,
        top_p=1.0 if temperature == 0.0 else 0.95,
        top_k=0,
        seed=0,
    )
    return decode_text(runtime.root, tokens)


def test_greedy_instruct(carrot_runtime: SparkRuntime) -> None:
    text = _complete(carrot_runtime, thinking=False, effort=None, temperature=0.0)
    assert _ANSWER.search(text)


def test_none_card(carrot_runtime: SparkRuntime) -> None:
    text = _complete(carrot_runtime, thinking=False, effort=None, temperature=1.0)
    assert _ANSWER.search(text)


def test_high_card(carrot_runtime: SparkRuntime) -> None:
    text = _complete(carrot_runtime, thinking=True, effort="high", temperature=1.0)
    reasoning, visible = split_thinking(text, True)
    assert reasoning
    assert _ANSWER.search(visible)


def test_low_card(carrot_runtime: SparkRuntime) -> None:
    text = _complete(carrot_runtime, thinking=True, effort="low", temperature=1.0)
    reasoning, visible = split_thinking(text, True)
    assert reasoning
    assert _ANSWER.search(visible)
