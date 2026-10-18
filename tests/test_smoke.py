"""Carrot completions for this pipeline. Skips when the record is not installed.

The stored file is greedy instruct. The other cases use this model's instruct
and thinking cards. A reasoning effort runs only when the checkpoint template
names it. The reply is 6 carrots.
"""

import json
import re
import time
from pathlib import Path

import numpy as np
import pytest
from qwen38_flash_next.carrot import CARROT_PROMPT
from qwen38_flash_next.sampling import VARIANTS
from qwen38_flash_next.tools import split_thinking

from decode_rate import billed_decode
from test_packing import pipeline

_THINK_TOKENS = 768
_ANSWER = re.compile(
    r"(?:\*\*|__)?(?:6|six)(?:\*\*|__)?(?:\s+\w+){0,3}\s+carrots?\b",
    re.IGNORECASE,
)


def test_carrot_matches_the_record() -> None:
    root = pipeline.DEFAULT_RECORD
    if not (root / "manifest.json").is_file():
        pytest.skip("quantized record is not installed")
    stored = pipeline.load_carrot(root)
    runtime = pipeline.Qwen38Runtime(root)
    try:
        _assert_carrot(runtime, stored)
    finally:
        runtime.close()


def _assert_carrot(runtime: pipeline.Qwen38Runtime, stored: dict) -> None:
    io = {"expert": 0.0, "experts": 0, "ngram": 0.0, "rows": 0}
    expert_image = runtime.files.expert_image
    ngram_vector = runtime.files.ngram_vector

    def timed_expert(layer: int, expert: int) -> bytes:
        started = time.perf_counter()
        blob = expert_image(layer, expert)
        io["expert"] += time.perf_counter() - started
        io["experts"] += 1
        return blob

    def timed_ngram(row: int):
        started = time.perf_counter()
        vector = ngram_vector(row)
        io["ngram"] += time.perf_counter() - started
        io["rows"] += 1
        return vector

    runtime.files.expert_image = timed_expert
    runtime.files.ngram_vector = timed_ngram
    prompt = runtime.encode_chat(
        [{"role": "user", "content": stored["prompt"]}], None, bool(stored["thinking"])
    )
    started = time.perf_counter()
    logits = runtime.forward(prompt)
    produced = list(
        runtime._new_tokens(
            logits,
            max_tokens=int(stored["max_tokens"]),
            temperature=float(stored["temperature"]),
            top_p=float(stored["top_p"]),
            top_k=int(stored["top_k"]),
        )
    )
    runtime._log_decode()
    decode_rate = billed_decode(runtime)
    wall = time.perf_counter() - started
    text = runtime.decode(produced).rstrip()
    gpu = runtime.gpu.gpu_seconds
    encode = runtime.gpu.encode_seconds
    launches = runtime.gpu.launch_count
    commands = runtime.gpu.command_count
    print(
        f"wall {wall:.2f}s gpu {gpu:.2f}s ({100 * gpu / wall:.1f}%) "
        f"launch-wait {encode:.2f}s dispatches {launches} commands {commands}"
    )
    print(
        f"io {runtime.gpu.io_loads} loads in {runtime.gpu.io_seconds:.2f}s"
    )
    print(
        f"experts {io['experts']} in {io['expert']:.2f}s "
        f"ngram-rows {io['rows']} in {io['ngram']:.2f}s"
    )
    steps = runtime._decode_steps
    full = runtime._mtp_full if hasattr(runtime, "_mtp_full") else 0
    partial = runtime._mtp_partial if hasattr(runtime, "_mtp_partial") else 0
    first = steps[0][2] if steps else 0
    last = steps[-1][2] if steps else 0
    print(f"mtp full {full} partial {partial} steps {len(steps)} resident {first}→{last}")
    assert text == stored["output"]
    if _ANSWER.search(text) is None:
        message = "carrot reply did not collect 6 carrots: " + text[-400:]
        raise AssertionError(message)
    assert decode_rate > 0


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
    root = pipeline.DEFAULT_RECORD
    if not (root / "manifest.json").is_file():
        pytest.skip("quantized record is not installed")
    runtime = pipeline.Qwen38Runtime(root)
    try:
        yield runtime
    finally:
        runtime.close()


def _card_completion(runtime: pipeline.Qwen38Runtime, *, thinking: bool, effort: str | None) -> str:
    card = VARIANTS["thinking" if thinking else "instruct"]
    runtime._presence_penalty = float(card["presence_penalty"])
    runtime._frequency_penalty = 0.0
    runtime.rng = np.random.default_rng(0)
    prompt = runtime.encode_chat(
        [{"role": "user", "content": CARROT_PROMPT}],
        None,
        thinking,
        effort,
    )
    limit = _THINK_TOKENS if thinking else 384
    tokens = runtime.generate(
        prompt,
        limit,
        float(card["temperature"]),
        float(card["top_p"]),
        int(card["top_k"]),
    )
    text = runtime.decode(tokens).rstrip()
    _expect_six(text, thinking=thinking)
    return text


def _expect_six(text: str, *, thinking: bool) -> None:
    _reasoning, reply = split_thinking(text, thinking)
    body = reply if thinking else text
    if _ANSWER.search(body) is None:
        message = "carrot reply did not collect 6 carrots: " + text[-400:]
        raise AssertionError(message)


def test_carrot_instruct_uses_the_card(carrot_runtime: pipeline.Qwen38Runtime) -> None:
    text = _card_completion(carrot_runtime, thinking=False, effort=None)
    print(f"instruct card: {text[:160]}")


def test_carrot_thinking_uses_the_card(carrot_runtime: pipeline.Qwen38Runtime) -> None:
    template = _template(pipeline.DEFAULT_RECORD)
    if "enable_thinking" not in template:
        pytest.skip("template does not open a think block")
    text = _card_completion(carrot_runtime, thinking=True, effort=None)
    print(f"thinking card: {text[:160]}")


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "carrot_effort" not in metafunc.fixturenames:
        return
    template = _template(pipeline.DEFAULT_RECORD)
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


def test_carrot_thinking_effort(
    carrot_runtime: pipeline.Qwen38Runtime, carrot_effort: str
) -> None:
    text = _card_completion(carrot_runtime, thinking=True, effort=carrot_effort)
    print(f"thinking {carrot_effort}: {text[:160]}")
