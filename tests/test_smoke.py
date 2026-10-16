"""One quantized chat completion. Skips when the record is not installed."""

import time

import pytest

from decode_rate import billed_decode
from test_packing import pipeline


def test_grove_matches_the_record() -> None:
    root = pipeline.DEFAULT_RECORD
    if not (root / "manifest.json").is_file():
        pytest.skip("quantized record is not installed")
    stored = pipeline.load_grove(root)
    runtime = pipeline.Qwen38Runtime(root)
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
    assert text == stored["output"]
    assert decode_rate > 0
