"""A 65536-token prompt through the quantized runtime.

Skipped unless MADMETAL_LONG_CONTEXT=1, so the default suite stays the short smoke.
"""

import os

import pytest
from qwen38_flash_next.formats import DEFAULT_RECORD
from qwen38_flash_next.runtime import Qwen38Runtime

from decode_rate import billed_decode

PROMPT = 65536


@pytest.mark.skipif(
    os.environ.get("MADMETAL_LONG_CONTEXT") != "1",
    reason="set MADMETAL_LONG_CONTEXT=1",
)
def test_dummy_64k_prompt_generates_one_token() -> None:
    if not (DEFAULT_RECORD / "manifest.json").is_file():
        pytest.skip("quantized record is not installed")
    runtime = Qwen38Runtime(DEFAULT_RECORD)
    prompt = [100] * PROMPT
    logits = runtime.forward(prompt)
    token = runtime._pick(logits, 0.0, 1.0, 0, [])
    if token in runtime.stop_ids:
        token = 15 if 15 not in runtime.stop_ids else 17
    del logits
    runtime.last_logits = None
    runtime.forward([token], decode=True)
    decode_rate = billed_decode(runtime)
    assert runtime.tokens[:PROMPT] == prompt
    assert runtime.tokens[-1] == token
    assert runtime.atlas is not None
    assert runtime.resident_budget > 0
    assert decode_rate > 0
