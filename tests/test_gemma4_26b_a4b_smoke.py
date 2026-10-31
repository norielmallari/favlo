"""Carrot completions for Gemma 4 26B-A4B. Skips when the record is not installed.

Greedy instruct is temperature 0. The instruct and thinking cases use this
model's sampling card. The reply collects 6 carrots.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from gemma4_26b_a4b.chat import decode_text, encode_chat, reply_text
from gemma4_26b_a4b.commands import CARROT_PROMPT
from gemma4_26b_a4b.forward import Gemma4Runtime
from gemma4_26b_a4b.sampling import VARIANTS

_ROOT = Path("models/gemma-4-26B-A4B-it-q4_0")
_INSTRUCT_TOKENS = 384
_THINK_TOKENS = 768
_ANSWER = re.compile(
    r"(?:\*\*|__)?(?:6|six)(?:\*\*|__)?(?:\s+\w+){0,3}\s+carrots?\b",
    re.IGNORECASE,
)


@pytest.fixture(scope="module")
def carrot_runtime():
    if not (_ROOT / "manifest.json").is_file():
        pytest.skip("quantized record is not installed")
    runtime = Gemma4Runtime(_ROOT, owner="gemma4-26b-a4b:smoke")
    try:
        yield runtime
    finally:
        runtime.close()


def _complete(
    runtime: Gemma4Runtime,
    *,
    thinking: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    presence: float,
    seed: int = 0,
) -> str:
    prompt = encode_chat(
        runtime.root,
        [{"role": "user", "content": CARROT_PROMPT}],
        thinking,
    )
    limit = _THINK_TOKENS if thinking else _INSTRUCT_TOKENS
    tokens, prefill_line, decode_line = runtime.generate(
        prompt,
        limit,
        temperature,
        top_p,
        top_k,
        presence,
        seed=seed,
    )
    print(prefill_line, flush=True)
    print(decode_line, flush=True)
    text = decode_text(runtime.root, tokens)
    _reasoning, answer = reply_text(text)
    return answer if thinking else text


def _expect_carrot(text: str) -> None:
    if _ANSWER.search(text) is None:
        message = "carrot reply did not collect 6 carrots: " + text[-400:]
        raise AssertionError(message)


def test_carrot_instruct_greedy(carrot_runtime: Gemma4Runtime) -> None:
    text = _complete(
        carrot_runtime,
        thinking=False,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        presence=0.0,
    )
    _expect_carrot(text)
    print(f"instruct greedy: {text[:160]}")


def test_carrot_instruct_recommended(carrot_runtime: Gemma4Runtime) -> None:
    card = VARIANTS["instruct"]
    text = _complete(
        carrot_runtime,
        thinking=False,
        temperature=float(card["temperature"]),
        top_p=float(card["top_p"]),
        top_k=int(card["top_k"]),
        presence=float(card["presence_penalty"]),
    )
    _expect_carrot(text)
    print(f"instruct card: {text[:160]}")


def test_carrot_thinking_recommended(carrot_runtime: Gemma4Runtime) -> None:
    card = VARIANTS["thinking"]
    misses: list[str] = []
    for seed in range(4):
        text = _complete(
            carrot_runtime,
            thinking=True,
            temperature=float(card["temperature"]),
            top_p=float(card["top_p"]),
            top_k=int(card["top_k"]),
            presence=float(card["presence_penalty"]),
            seed=seed,
        )
        try:
            _expect_carrot(text)
        except AssertionError as exc:
            misses.append(str(exc)[-180:])
            continue
        print(f"thinking card seed {seed}: {text[:160]}")
        return
    raise AssertionError("carrot reply did not collect 6 carrots\n" + "\n".join(misses))
