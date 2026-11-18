# Synthetic prompt

Measurement note for the CUDA runtime of `qwen35_4b`. The serving contract is [../ARCHITECTURE.md](../ARCHITECTURE.md). This file records one synthetic-prompt run. It is a recorded run, not a format bound. The fp8 instruct run is in [synthetic-prompt.md](synthetic-prompt.md). The thinking-on int8 run is in [synthetic-prompt-thinking-cuda.md](synthetic-prompt-thinking-cuda.md).

## Remainder

- On a GeForce RTX 3070 Laptop GPU with 8192 MiB, `Qwen3.5-4B-int8` with int8 keys and values and MTP off passed a cold instruct prompt.
- A 512-token prompt prefilled at 2272.71 tok/s and decoded 128 tokens at 67.86 tok/s.
- A 1024-token prompt prefilled at 2312.39 tok/s and decoded 128 tokens at 65.57 tok/s.
- A 4096-token prompt prefilled at 1906.79 tok/s and decoded 128 tokens at 59.00 tok/s.
- A 10000-token prompt prefilled at 1509.82 tok/s and decoded 128 tokens at 49.12 tok/s.
- The end-to-end rates, prefill plus decode, were 301.89, 482.01, 977.78, and 1097.29 tok/s.
- A search on 18 November 2026 did not find that table.

## Predecessors

- Method. [llama-bench](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md) defaults to a 512-token prompt and 128 generated tokens. [`mlx_lm.benchmark`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/benchmark.py) uses the same prompt default. [oMLX](https://omlx.ai/benchmarks/performance/x0t6bie9) reports 1024-token and 4096-token sessions. [Artificial Analysis](https://artificialanalysis.ai/methodology/performance-benchmarking) times a prompt of about 10000 tokens and reports time to first token, not prefill tok/s.
- Model. [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B). The card publishes task scores and says throughput varies by framework. It does not publish this harness rate table.
- Rate. [llamabench.ai](https://llamabench.ai/models/qwen3-5-4b) lists an RTX 3060 at 2417 tok/s prefill and 81.3 tok/s decode for UD-Q4_K_XL. [A formula estimate for an RTX 3070](https://localai.computer/can/rtx-3070/run/qwen-qwen3.5-4b) quotes about 109 tok/s at Q4 and about 67 tok/s at Q8, and labels both as estimates. [llama.cpp on an RTX 5060 Ti](https://dev.to/elcances/qwen35-4b-on-a-16-gb-rtx-5060-ti-823-tokens-per-second-for-16-users-6i) reports 5050 tok/s prompt processing and 114 tok/s generation at a 512-token context. [llama.cpp NVFP4 on an RTX 5090](https://github.com/ggml-org/llama.cpp/pull/21074) reports 13064.37 tok/s at pp512 and 203.58 tok/s at tg128. [oMLX Qwen3.5-4B-OptiQ-4bit on a 10-core M5](https://omlx.ai/benchmarks/performance/299yxolf) reports 1414 tok/s prefill and 35.5 tok/s decode at 1024 tokens, and 1525 tok/s prefill and 34.6 tok/s decode at 4096 tokens. The fp8 instruct table in [synthetic-prompt.md](synthetic-prompt.md) recorded 1322.13 / 40.37 tok/s at 512 tokens and 874.93 / 13.61 tok/s at 10000 tokens on an M5 Pro with 48 GB. None of them is this int8 instruct table.

## Rule

- MTP is off.
- Thinking is off, so the request uses the instruct card and `reasoning.effort` `none`.
- Temperature is 0.
- The banner prints the loaded plane.
- Each measured prompt is one user message and encodes to that length under the chat template, including the template tokens.
- A 64-token warmup runs before the measured prompts.
- The measured prompts start with different labels, so each prefill is a cache miss.
- Decode is 128 tokens at every length. `favlo:fill_decode` keeps sampling through stop ids until that cap.
- The end-to-end rate adds the prefill tokens to the decode tokens and adds those two clocks.
- The banner printed depth 3. That depth is unused while MTP is off.
- A banner that prints `thinking on`, `fp8_e4m3`, or `mtp on` is a different row.

## Command

```text
uv run python -m harness.synthetic --model qwen35_4b --thinking off --mtp off
```

The banner prints the loaded plane, MTP, thinking, and the decode cap.

## Table

| Prompt tokens | Prefill | Decode | End to end |
|---:|---|---|---|
| 512 | 2272.71 tok/s (512 tok in 0.23 s) | 67.86 tok/s (128 tok in 1.89 s) | 301.89 tok/s (640 tok in 2.12 s) |
| 1024 | 2312.39 tok/s (1024 tok in 0.44 s) | 65.57 tok/s (128 tok in 1.95 s) | 482.01 tok/s (1152 tok in 2.39 s) |
| 4096 | 1906.79 tok/s (4096 tok in 2.15 s) | 59.00 tok/s (128 tok in 2.17 s) | 977.78 tok/s (4224 tok in 4.32 s) |
| 10000 | 1509.82 tok/s (10000 tok in 6.62 s) | 49.12 tok/s (128 tok in 2.61 s) | 1097.29 tok/s (10128 tok in 9.23 s) |

## Search

18 November 2026.

- Queries: "2272.71 OR 2312.39 OR 1906.79 OR 1509.82 Qwen3.5-4B tok/s", and "Qwen3.5-4B RTX 3070 prefill tokens per second pp512 tg128 CUDA int8 benchmark 2026".
- The predecessors in this note are the overlaps those queries returned.
- The [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) card publishes task scores and says throughput varies by framework. It does not publish 2272.71, 2312.39, 1906.79, or 1509.82.
- The search did not find 2272.71 and 67.86, 2312.39 and 65.57, 1906.79 and 59.00, or 1509.82 and 49.12 in those sources.

## Corrections

Later finds are appended here. The sections above stay as committed.
