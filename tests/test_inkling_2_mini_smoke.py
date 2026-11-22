"""Carrot completions for Inkling 2 Mini. Skips when the checkpoint is absent.

Greedy instruct is temperature 0 and effort none. The thinking card uses the
template default. The reply is 6 carrots.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from inkling_2_mini.metal41.chat import STOP, decode_text, encode_chat
from inkling_2_mini.metal41.commands import CARROT_PROMPT
from inkling_2_mini.metal41.forward import InklingRuntime
from inkling_2_mini.metal41.sampling import EFFORT_ORDER

_ROOT = Path("models/Inkling-2-Mini")
_ANSWER = re.compile(
    r"(?:\*\*|__)?(?:6|six)(?:\*\*|__)?(?:\s+\w+){0,3}\s+carrots?\b",
    re.IGNORECASE,
)


@pytest.fixture(scope="module")
def carrot_runtime():
    if not (_ROOT / "model-00001-of-00001.safetensors").is_file():
        pytest.skip("Inkling checkpoint is not installed")
    runtime = InklingRuntime(_ROOT)
    try:
        yield runtime
    finally:
        runtime.close()


def _complete(runtime: InklingRuntime, *, effort: str, temperature: float) -> str:
    np.random.seed(0)
    runtime.reset()
    prompt = encode_chat(
        runtime.root,
        [{"role": "user", "content": CARROT_PROMPT}],
        None,
        effort != "none",
        effort=effort,
    )
    tokens = runtime.generate(prompt, 256, temperature, 1.0, 0, 0.0, STOP)
    return decode_text(runtime.root, tokens)


def test_carrot_instruct_greedy(carrot_runtime: InklingRuntime) -> None:
    text = _complete(carrot_runtime, effort="none", temperature=0.0)
    print(f"instruct greedy: {text[:160]}")
    if _ANSWER.search(text) is None:
        raise AssertionError("carrot reply did not collect 6 carrots: " + text[-400:])


def test_carrot_instruct_card(carrot_runtime: InklingRuntime) -> None:
    text = _complete(carrot_runtime, effort="none", temperature=1.0)
    print(f"instruct: {text[:160]}")
    if _ANSWER.search(text) is None:
        raise AssertionError("carrot reply did not collect 6 carrots: " + text[-400:])


def test_carrot_thinking_card(carrot_runtime: InklingRuntime) -> None:
    text = _complete(carrot_runtime, effort="high", temperature=1.0)
    print(f"thinking: {text[:160]}")
    if _ANSWER.search(text) is None:
        raise AssertionError("carrot reply did not collect 6 carrots: " + text[-400:])


@pytest.mark.parametrize("effort", [name for name in EFFORT_ORDER if name != "high"])
def test_carrot_effort(carrot_runtime: InklingRuntime, effort: str) -> None:
    text = _complete(carrot_runtime, effort=effort, temperature=1.0)
    print(f"{effort}: {text[:160]}")
    if _ANSWER.search(text) is None:
        raise AssertionError("carrot reply did not collect 6 carrots: " + text[-400:])
