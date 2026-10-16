"""Billed decode rate with every draft kept, then with every draft rejected.

Skips when the quantized record is not installed. The acceptance decision is
forced. The forwards are the real backbone.
"""

import pytest

from test_packing import pipeline

TOKENS = 32


def _run(runtime: pipeline.Qwen38Runtime, prompt: list[int], *, accept: bool) -> float:
    runtime.reset()
    runtime._mtp_full = 0
    runtime._mtp_partial = 0
    logits = runtime.forward(prompt)
    original = runtime.gpu.spec_accept

    def forced(
        target: object,
        draft_logits: object,
        draft: int,
        temperature: float,
        top_p: float,
        top_k: int,
        rng: object,
    ) -> tuple[int, bool]:
        if accept:
            return int(draft), True
        other = int(draft) + 1
        if other in runtime.stop_ids or other >= runtime.vocab:
            other = 15
        return other, False

    runtime.gpu.spec_accept = forced
    try:
        produced = list(runtime._new_tokens(logits, TOKENS, 0.0, 1.0, 0))
    finally:
        runtime.gpu.spec_accept = original
    if runtime._decode_seconds <= 0 or not produced:
        message = "acceptance smoke produced no billed decode"
        raise RuntimeError(message)
    return runtime._decode_tokens / runtime._decode_seconds


def test_higher_acceptance_raises_decode_rate() -> None:
    root = pipeline.DEFAULT_RECORD
    if not (root / "manifest.json").is_file():
        pytest.skip("quantized record is not installed")
    stored = pipeline.load_grove(root)
    runtime = pipeline.Qwen38Runtime(root)
    try:
        prompt = runtime.encode_chat(
            [{"role": "user", "content": stored["prompt"]}],
            None,
            bool(stored["thinking"]),
        )
        kept = _run(runtime, prompt, accept=True)
        kept_full = runtime._mtp_full
        kept_partial = runtime._mtp_partial
        rejected = _run(runtime, prompt, accept=False)
        rejected_full = runtime._mtp_full
        rejected_partial = runtime._mtp_partial
    finally:
        runtime.close()
    print(
        f"kept {kept:.2f} tok/s full {kept_full} partial {kept_partial} "
        f"rejected {rejected:.2f} tok/s full {rejected_full} partial {rejected_partial}"
    )
    assert kept_full > 0 and kept_partial == 0
    assert rejected_partial > 0 and rejected_full == 0
    assert kept > 0
    assert rejected > 0
    assert kept > rejected
