"""Billed decode above 10 tok/s on a kept span whose experts are already resident.

Skips when the quantized record is not installed. Drafts are forced to the
token that filled the atlas, and every draft is kept. The forwards are the
real backbone, including draft time.
"""

import pytest

from test_packing import pipeline

WARM = 4
MEASURED = 6


def test_resident_kept_span_exceeds_10_tok_s() -> None:
    root = pipeline.DEFAULT_RECORD
    if not (root / "manifest.json").is_file():
        pytest.skip("quantized record is not installed")
    runtime = pipeline.Qwen38Runtime(root)
    try:
        prompt = runtime.encode_chat(
            [{"role": "user", "content": "Count to four."}],
            None,
            False,
        )
        logits = runtime.forward(prompt)
        token = runtime._pick(logits, 0.0, 1.0, 0, [])
        if token in runtime.stop_ids:
            message = "prefill argmax is a stop token"
            raise RuntimeError(message)
        original_sample = runtime.gpu.sample
        original_accept = runtime.gpu.spec_accept

        def same_token(
            logits: object,
            temperature: float,
            top_p: float,
            top_k: int,
            rng: object,
        ) -> int:
            del logits, temperature, top_p, top_k, rng
            return token

        def keep(
            target: object,
            draft_logits: object,
            draft: int,
            temperature: float,
            top_p: float,
            top_k: int,
            rng: object,
        ) -> tuple[int, bool]:
            del target, draft_logits, temperature, top_p, top_k, rng
            return int(draft), True

        runtime.gpu.sample = same_token
        runtime.gpu.spec_accept = keep
        runtime._mtp_full = 0
        runtime._mtp_partial = 0
        try:
            for _cycle in range(WARM):
                runtime._mtp_cycle(token, [], 4, 0.0, 1.0, 0)
            resident = runtime.atlas.resident_count if runtime.atlas is not None else 0
            runtime._decode_tokens = 0
            runtime._decode_seconds = 0.0
            runtime._mtp_full = 0
            runtime._mtp_partial = 0
            for _cycle in range(MEASURED):
                runtime._mtp_cycle(token, [], 4, 0.0, 1.0, 0)
        finally:
            runtime.gpu.sample = original_sample
            runtime.gpu.spec_accept = original_accept
        billed_tokens = runtime._decode_tokens
        billed_seconds = runtime._decode_seconds
        full = runtime._mtp_full
        partial = runtime._mtp_partial
        rate = billed_tokens / billed_seconds
        after = runtime.atlas.resident_count if runtime.atlas is not None else 0
    finally:
        runtime.close()
    print(
        f"resident span {rate:.2f} tok/s "
        f"tokens {billed_tokens} seconds {billed_seconds:.2f} "
        f"full {full} partial {partial} "
        f"atlas {resident} -> {after}"
    )
    assert full == MEASURED
    assert partial == 0
    assert billed_tokens == MEASURED * 4
    assert billed_seconds > 0
    assert rate > 10
