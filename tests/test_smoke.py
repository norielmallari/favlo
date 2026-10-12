"""One quantized chat completion. Skips when the record is not installed."""

import time

import pytest

from test_packing import pipeline

GROVE = (
    "There are 15 trees in the grove. Grove workers will plant trees in the grove today. "
    "After they are done, there will be 21 trees. How many trees did the grove workers plant today?"
)

GROVE_OUTPUT = (
    "To find out how many trees the grove workers planted, "
    "we can subtract the initial number of trees from the final number of trees.\n"
    "\n"
    "1.  **Initial number of trees:** 15\n"
    "2.  **Final number of trees:** 21\n"
    "3.  **Calculation:** $21 - 15 = 6$\n"
    "\n"
    "The grove workers planted **6** trees today."
)


def test_grove_completion_says_six_trees() -> None:
    root = pipeline.DEFAULT_RECORD
    if not (root / "manifest.json").is_file():
        pytest.skip("quantized record is not installed")
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
    prompt = runtime.encode_chat([{"role": "user", "content": GROVE}], None, False)
    started = time.perf_counter()
    logits = runtime.forward(prompt)
    produced = list(
        runtime._new_tokens(logits, max_tokens=128, temperature=0.0, top_p=1.0, top_k=0)
    )
    runtime._log_decode()
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
    assert text == GROVE_OUTPUT
