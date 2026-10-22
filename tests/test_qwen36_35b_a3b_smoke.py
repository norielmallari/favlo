"""Carrot completions for Qwen3.6-35B-A3B. Skips when the record is not installed.

Greedy instruct is temperature 0. The recommended instruct and thinking cases
use this model's sampling card. Thinking runs when the checkpoint template
opens a think block. A passed reasoning effort is not rendered.
The reply is 6 carrots.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from qwen36_35b_a3b.chat import (
    STOP,
    decode_text,
    encode_chat,
    encode_text,
    load_tokenizer,
    think_close_ids,
    think_open_id,
)
from qwen36_35b_a3b.commands import CARROT_PROMPT
from qwen36_35b_a3b.forward import Qwen36Runtime, letter_token_ids, long_token_ids
from qwen36_35b_a3b.sampling import VARIANTS
from qwen36_35b_a3b.tools import split_thinking

_ROOT = Path("models/Qwen3.6-35B-A3B-fp8-e20")
_INSTRUCT_TOKENS = 384
_THINK_TOKENS = 768
# The test forces the close so the carrot reply is in the scored text.
_THINK_BUDGET = 512
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


@pytest.fixture(scope="module")
def carrot_runtime():
    if not (_ROOT / "manifest.json").is_file():
        pytest.skip("quantized record is not installed")
    runtime = Qwen36Runtime(_ROOT)
    try:
        yield runtime
    finally:
        runtime.close()


def _complete(
    runtime: Qwen36Runtime,
    *,
    thinking: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    presence: float,
    seed: int = 0,
    check: bool = True,
) -> str:
    np.random.seed(seed)
    runtime.reset()
    prompt = encode_chat(
        runtime.root,
        [{"role": "user", "content": CARROT_PROMPT}],
        None,
        thinking,
    )
    close: list[int] | None = None
    end = None
    tool_open = None
    long_ids = None
    word_ids = None
    forbid = None if thinking else {think_open_id(runtime.root)}
    if thinking:
        close, end = think_close_ids(runtime.root)
        tool_open = encode_text(runtime.root, "<tool_call>")[0]
        vocab = load_tokenizer(runtime.root).get_vocab()
        long_ids = long_token_ids(vocab)
        word_ids = letter_token_ids(vocab)
    limit = _THINK_TOKENS if thinking else _INSTRUCT_TOKENS
    tokens = runtime.generate(
        prompt,
        limit,
        temperature,
        top_p,
        top_k,
        presence,
        STOP,
        think_close=close,
        think_end=end,
        thinking_budget=_THINK_BUDGET if thinking else 0,
        tool_open=tool_open,
        long_ids=long_ids,
        forbid=forbid,
        word_ids=word_ids,
    )
    text = decode_text(runtime.root, tokens).rstrip()
    if check:
        _expect_carrot(text, thinking=thinking)
    return text


def _expect_carrot(text: str, *, thinking: bool) -> None:
    _reasoning, reply = split_thinking(text, thinking)
    body = reply if thinking else text
    if _ANSWER.search(body) is None:
        message = "carrot reply did not collect 6 carrots: " + text[-400:]
        raise AssertionError(message)


def test_carrot_instruct_greedy(carrot_runtime: Qwen36Runtime) -> None:
    text = _complete(
        carrot_runtime,
        thinking=False,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        presence=0.0,
    )
    print(f"instruct greedy: {text[:160]}")


def test_carrot_instruct_recommended(carrot_runtime: Qwen36Runtime) -> None:
    card = VARIANTS["instruct"]
    text = _complete(
        carrot_runtime,
        thinking=False,
        temperature=float(card["temperature"]),
        top_p=float(card["top_p"]),
        top_k=int(card["top_k"]),
        presence=float(card["presence_penalty"]),
    )
    print(f"instruct card: {text[:160]}")


def test_carrot_thinking_recommended(carrot_runtime: Qwen36Runtime) -> None:
    if "enable_thinking" not in _template(_ROOT):
        pytest.skip("template does not open a think block")
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
            check=False,
        )
        try:
            _expect_carrot(text, thinking=True)
        except AssertionError as exc:
            misses.append(str(exc)[-180:])
            continue
        print(f"thinking card seed {seed}: {text[:160]}")
        return
    raise AssertionError("carrot reply did not collect 6 carrots\n" + "\n".join(misses))

