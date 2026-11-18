# Synthetic prompt, thinking

Measurement note for the CUDA runtime of `qwen35_4b`. The serving contract is [../ARCHITECTURE.md](../ARCHITECTURE.md). This file records one synthetic-prompt run. It is a recorded run, not a format bound. The fp8 instruct run is in [synthetic-prompt.md](synthetic-prompt.md).

## Remainder

- On a GeForce RTX 3070 Laptop GPU with 8192 MiB, `Qwen3.5-4B-int8` with int8 keys and values and MTP off passed a cold thinking prompt.
- A 512-token prompt prefilled at 2244.58 tok/s and decoded 128 tokens at 66.10 tok/s.
- A 1024-token prompt prefilled at 2248.86 tok/s and decoded 128 tokens at 64.93 tok/s.
- A 4096-token prompt prefilled at 2011.74 tok/s and decoded 128 tokens at 57.59 tok/s.
- A 10000-token prompt prefilled at 1544.01 tok/s and decoded 128 tokens at 49.20 tok/s.
- The end-to-end rates, prefill plus decode, were 294.93, 474.07, 991.55, and 1115.42 tok/s.
- A search on 18 November 2026 did not find that table.

## Predecessors

- Method. [llama-bench](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md) defaults to a 512-token prompt and 128 generated tokens. [`mlx_lm.benchmark`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/benchmark.py) uses the same prompt default. [oMLX](https://omlx.ai/benchmarks/performance/x0t6bie9) reports 1024-token and 4096-token sessions. [Artificial Analysis](https://artificialanalysis.ai/methodology/performance-benchmarking) times a prompt of about 10000 tokens and reports time to first token, not prefill tok/s.
- Model. [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B). The card publishes task scores and says throughput varies by framework. It does not publish this harness rate table.
- Rate. [llamabench.ai](https://llamabench.ai/models/qwen3-5-4b) lists an RTX 3060 at 2417 tok/s prefill and 81.3 tok/s decode for UD-Q4_K_XL. [A formula estimate for an RTX 3070](https://localai.computer/can/rtx-3070/run/qwen-qwen3.5-4b) quotes about 109 tok/s at Q4 and about 67 tok/s at Q8, and labels both as estimates. [llama.cpp on an RTX 5060 Ti](https://dev.to/elcances/qwen35-4b-on-a-16-gb-rtx-5060-ti-823-tokens-per-second-for-16-users-6i) reports 5050 tok/s prompt processing and 114 tok/s generation at a 512-token context. [llama.cpp NVFP4 on an RTX 5090](https://github.com/ggml-org/llama.cpp/pull/21074) reports 13064.37 tok/s at pp512 and 203.58 tok/s at tg128. The fp8 instruct table in [synthetic-prompt.md](synthetic-prompt.md) recorded 1322.13 / 40.37 tok/s at 512 tokens and 874.93 / 13.61 tok/s at 10000 tokens on an M5 Pro with 48 GB. None of them is this int8 thinking table.

## Rule

- MTP is off.
- Thinking is on, so the request uses the thinking card.
- Temperature is 0.
- The banner prints the loaded plane.
- Each measured prompt is one user message and encodes to that length under the chat template, including the template tokens.
- A 64-token warmup runs before the measured prompts.
- The measured prompts start with different labels, so each prefill is a cache miss.
- Decode is 128 tokens at every length.
- The end-to-end rate adds the prefill tokens to the decode tokens and adds those two clocks.
- The banner printed depth 3. That depth is unused while MTP is off.
- A banner that prints `thinking off`, `fp8_e4m3`, or `mtp on` is a different row.

## Command

```text
uv run python -m harness.synthetic --model qwen35_4b --thinking on --mtp off
```

The banner prints the loaded plane, MTP, thinking, and the decode cap.

## Table

| Prompt tokens | Prefill | Decode | End to end |
|---:|---|---|---|
| 512 | 2244.58 tok/s (512 tok in 0.23 s) | 66.10 tok/s (128 tok in 1.94 s) | 294.93 tok/s (640 tok in 2.17 s) |
| 1024 | 2248.86 tok/s (1024 tok in 0.46 s) | 64.93 tok/s (128 tok in 1.97 s) | 474.07 tok/s (1152 tok in 2.43 s) |
| 4096 | 2011.74 tok/s (4096 tok in 2.04 s) | 57.59 tok/s (128 tok in 2.22 s) | 991.55 tok/s (4224 tok in 4.26 s) |
| 10000 | 1544.01 tok/s (10000 tok in 6.48 s) | 49.20 tok/s (128 tok in 2.60 s) | 1115.42 tok/s (10128 tok in 9.08 s) |

## Search

18 November 2026.

- Queries: "2244.58 OR 2011.74 OR 1544.01 OR 1115.42 Qwen3.5-4B", and "Qwen3.5-4B RTX 3070 prefill tokens per second pp512 tg128 CUDA int8 benchmark 2026".
- The predecessors in this note are the overlaps those queries returned.
- The [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) card publishes task scores and says throughput varies by framework. It does not publish 2244.58, 2248.86, 2011.74, or 1544.01.
- The search did not find 2244.58 and 66.10, 2248.86 and 64.93, 2011.74 and 57.59, or 1544.01 and 49.20 in those sources.

## Corrections

Later finds are appended here. The sections above stay as committed.
