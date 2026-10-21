"""Carrot completions for Qwen3.8-27B. Skips when the record is not installed.

Greedy instruct is temperature 0. The recommended instruct and thinking cases
use the sampling card. A reasoning effort runs only when that template names
it. The reply is 6 carrots.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from qwen38_27b.chat import STOP, decode_text, encode_chat
from qwen38_27b.commands import CARROT_PROMPT, installed_record
from qwen38_27b.forward import Qwen38Runtime
from qwen38_27b.sampling import VARIANTS
from qwen38_27b.tools import split_thinking
_INSTRUCT_TOKENS = 384
_THINK_TOKENS = 768
_ANSWER = re.compile(
    r"(?:\*\*|__)?(?:6|six)(?:\*\*|__)?(?:\s+\w+){0,3}\s+carrots?\b",
    re.IGNORECASE,
)


def _template(root: Path) -> str:
    jinja = root / "chat_template.jinja"
    if jinja.is_file():
        return jinja.read_text()
    config = root / "tokenizer_config.json"
    if not config.is_file():
        return ""
    text = json.loads(config.read_text()).get("chat_template")
    return text if isinstance(text, str) else ""


def _efforts(template: str) -> tuple[str, ...]:
    if "reasoning_effort" not in template:
        return ()
    return tuple(name for name in ("low", "medium", "xhigh") if f"'{name}'" in template)


@pytest.fixture(scope="module")
def carrot_runtime():
    root = installed_record()
    if root is None:
        pytest.skip("quantized record is not installed")
    runtime = Qwen38Runtime(root)
    try:
        yield runtime
    finally:
        runtime.close()


def _complete(
    runtime: Qwen38Runtime,
    *,
    thinking: bool,
    effort: str | None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    presence: float = 0.0,
    seed: int = 0,
) -> str:
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
        STOP,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        presence=presence,
        rng=np.random.default_rng(seed),
    )
    text = decode_text(runtime.root, tokens).rstrip()
    _expect_six(text, thinking=thinking)
    return text


def _card(name: str) -> tuple[float, float, int, float]:
    card = VARIANTS[name]
    return (
        float(card["temperature"]),
        float(card["top_p"]),
        int(card["top_k"]),
        float(card["presence_penalty"]),
    )


def _expect_six(text: str, *, thinking: bool) -> None:
    _reasoning, reply = split_thinking(text, thinking)
    body = reply if thinking else text
    if _ANSWER.search(body) is None:
        message = "carrot reply did not collect 6 carrots: " + text[-400:]
        raise AssertionError(message)


def test_carrot_instruct_greedy(carrot_runtime: Qwen38Runtime) -> None:
    text = _complete(carrot_runtime, thinking=False, effort=None)
    print(f"instruct greedy: {text[:160]}")


def test_carrot_instruct_recommended(carrot_runtime: Qwen38Runtime) -> None:
    temperature, top_p, top_k, presence = _card("instruct")
    text = _complete(
        carrot_runtime,
        thinking=False,
        effort=None,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        presence=presence,
    )
    print(f"instruct card: {text[:160]}")


def test_carrot_thinking_recommended(carrot_runtime: Qwen38Runtime) -> None:
    if "enable_thinking" not in _template(carrot_runtime.root):
        pytest.skip("template does not open a think block")
    temperature, top_p, top_k, presence = _card("thinking")
    text = _complete(
        carrot_runtime,
        thinking=True,
        effort=None,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        presence=presence,
    )
    print(f"thinking card: {text[:160]}")


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "carrot_effort" not in metafunc.fixturenames:
        return
    root = installed_record()
    template = _template(root) if root is not None else ""
    efforts = _efforts(template) if "enable_thinking" in template else ()
    if not efforts:
        metafunc.parametrize(
            "carrot_effort",
            [
                pytest.param(
                    "none",
                    marks=pytest.mark.skip(reason="template names no reasoning effort"),
                )
            ],
        )
        return
    metafunc.parametrize("carrot_effort", efforts)


def test_carrot_thinking_effort(carrot_runtime: Qwen38Runtime, carrot_effort: str) -> None:
    temperature, top_p, top_k, presence = _card("thinking")
    text = _complete(
        carrot_runtime,
        thinking=True,
        effort=carrot_effort,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        presence=presence,
    )
    print(f"thinking {carrot_effort}: {text[:160]}")
