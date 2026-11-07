"""Carrot completions for qwen35-4b. Skips when the record is not installed."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1] / "pipelines" / "metal41" / "qwen35-4b"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from qwen35_4b.chat import STOP, decode_text, encode_chat, split_thinking  # noqa: E402
from qwen35_4b.commands import CARROT_PROMPT, installed_record  # noqa: E402
from qwen35_4b.forward import Qwen35Runtime  # noqa: E402
from qwen35_4b.sampling import VARIANTS  # noqa: E402

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
    runtime = Qwen35Runtime(root)
    try:
        yield runtime
    finally:
        runtime.close()


def _complete(
    runtime: Qwen35Runtime,
    *,
    thinking: bool,
    temperature: float,
    mtp: bool = False,
) -> str:
    prompt = encode_chat(
        runtime.root,
        [{"role": "user", "content": CARROT_PROMPT}],
        None,
        thinking,
    )
    card = VARIANTS["thinking" if thinking else "instruct"]
    limit = _THINK_TOKENS if thinking else _INSTRUCT_TOKENS
    tokens = runtime.generate(
        prompt,
        limit,
        STOP,
        temperature=temperature,
        top_p=float(card["top_p"]) if temperature else 1.0,
        top_k=int(card["top_k"]) if temperature else 0,
        presence=float(card["presence_penalty"]) if temperature else 0.0,
        mtp=mtp,
        mtp_depth=3,
    )
    return decode_text(runtime.root, tokens)


def test_greedy_instruct(carrot_runtime: Qwen35Runtime) -> None:
    text = _complete(carrot_runtime, thinking=False, temperature=0.0)
    assert _ANSWER.search(text)


def test_greedy_mtp(carrot_runtime: Qwen35Runtime) -> None:
    if not carrot_runtime._mtp_ready:
        pytest.skip("mtp half is not packed into the record")
    text = _complete(carrot_runtime, thinking=False, temperature=0.0, mtp=True)
    assert _ANSWER.search(text)
    assert carrot_runtime._mtp_tries > 0


def test_instruct_card(carrot_runtime: Qwen35Runtime) -> None:
    text = _complete(carrot_runtime, thinking=False, temperature=0.7)
    assert _ANSWER.search(text)


def test_thinking_card(carrot_runtime: Qwen35Runtime) -> None:
    text = _complete(carrot_runtime, thinking=True, temperature=1.0)
    reasoning, visible = split_thinking(text, True)
    assert reasoning
    assert _ANSWER.search(visible)
