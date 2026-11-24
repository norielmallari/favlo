"""Carrot completions for Inkling 2 Mini. Skips when the checkpoint is absent.

Greedy instruct is temperature 0 and effort none. The thinking card uses the
template default. The reply is 6 carrots.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from favlo.registry import choose_complete, load_registry
from inkling_2_mini.portable import open_runtime
from inkling_2_mini.portable.chat import STOP, decode_text, encode_chat
from inkling_2_mini.portable.record import CARROT_PROMPT
from inkling_2_mini.portable.sampling import EFFORT_ORDER

_PIPELINE = "inkling_2_mini"
_ANSWER = re.compile(
    r"(?:\*\*|__)?(?:6|six)(?:\*\*|__)?(?:\s+\w+){0,3}\s+carrots?\b",
    re.IGNORECASE,
)


@pytest.fixture(scope="module")
def carrot_runtime():
    try:
        path, selection = choose_complete(
            model=None, record=None, pipeline=_PIPELINE
        )
    except LookupError:
        pytest.skip("Inkling checkpoint is not installed")
    assert selection is not None
    row = load_registry()["models"][selection.folder]
    source = Path(str(row.get("source") or path))
    marker = "model-00001-of-00001.safetensors"
    if not (source / marker).is_file() and not (path / marker).is_file():
        pytest.skip("Inkling checkpoint is not installed")
    runtime = open_runtime(path)
    try:
        yield runtime
    finally:
        runtime.close()


def _complete(runtime: Any, *, effort: str, temperature: float) -> str:
    np.random.seed(0)
    runtime.reset()
    prompt = encode_chat(
        runtime.root,
        [{"role": "user", "content": CARROT_PROMPT}],
        None,
        effort != "none",
        effort=effort,
    )
    tokens = runtime.generate(
        prompt,
        256,
        STOP,
        temperature=temperature,
        top_p=1.0,
        top_k=0,
        presence=0.0,
    )
    return decode_text(runtime.root, tokens)


def test_carrot_instruct_greedy(carrot_runtime: Any) -> None:
    text = _complete(carrot_runtime, effort="none", temperature=0.0)
    print(f"instruct greedy: {text[:160]}")
    if _ANSWER.search(text) is None:
        raise AssertionError("carrot reply did not collect 6 carrots: " + text[-400:])


def test_carrot_instruct_card(carrot_runtime: Any) -> None:
    text = _complete(carrot_runtime, effort="none", temperature=1.0)
    print(f"instruct: {text[:160]}")
    if _ANSWER.search(text) is None:
        raise AssertionError("carrot reply did not collect 6 carrots: " + text[-400:])


def test_carrot_thinking_card(carrot_runtime: Any) -> None:
    text = _complete(carrot_runtime, effort="high", temperature=1.0)
    print(f"thinking: {text[:160]}")
    if _ANSWER.search(text) is None:
        raise AssertionError("carrot reply did not collect 6 carrots: " + text[-400:])


@pytest.mark.parametrize("effort", [name for name in EFFORT_ORDER if name != "high"])
def test_carrot_effort(carrot_runtime: Any, effort: str) -> None:
    text = _complete(carrot_runtime, effort=effort, temperature=1.0)
    print(f"{effort}: {text[:160]}")
    if _ANSWER.search(text) is None:
        raise AssertionError("carrot reply did not collect 6 carrots: " + text[-400:])
