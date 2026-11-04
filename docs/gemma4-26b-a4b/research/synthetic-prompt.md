# Synthetic prompt

Measurement note for the `gemma4-26b-a4b` pipeline. The serving contract is [../README.md](../README.md). This file records one synthetic-prompt run. It is a recorded run, not a format bound.

## Remainder

On an M5 Pro with 48 GB, Gemma 4 26B-A4B with `fp8_e4m3` keys and values and MTP off passed a cold instruct prompt. A 512-token prompt prefilled at 1190.77 tok/s and decoded 108 tokens at 26.36 tok/s. A 10000-token prompt prefilled at 1133.84 tok/s and decoded 128 tokens at 22.99 tok/s. The end-to-end rates, prefill plus decode, were 136.87 tok/s and 703.82 tok/s. A search on 4 November 2026 did not find that table.

## Predecessors

- Method. [llama-bench](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md) defaults to a 512-token prompt and 128 generated tokens. [`mlx_lm.benchmark`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/benchmark.py) uses the same prompt default. [Artificial Analysis](https://artificialanalysis.ai/methodology/performance-benchmarking) times a prompt of about 10000 tokens and reports time to first token, not prefill tok/s.
- Model. [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it). The card's published scores are not this harness rate.
- Rate. A [llama.cpp UD-Q4_K_XL recipe on an M5 Pro with 24 GB](https://local-ai-registry.vercel.app/recipes/gemma-4-26b-a4b-it-ud-q4-k-xl-apple-m5-pro-24gb-llama-cpp-tp1) reports 1147.5 tok/s prefill and 55 tok/s decode. [oMLX 4-bit on an M5 Pro](https://omlx.ai/benchmarks/performance/1k0xdolc) reports 1810 tok/s prefill and 71.0 tok/s generation at 4k. [oMLX QAT 4-bit on an M5 Pro with 48 GB](https://omlx.ai/benchmarks/performance/94mao60k) reports 1802 tok/s prefill and 89.6 tok/s generation at 4k. [Token Atlas](https://tokatlas.github.io/models/gemma-4-26b-a4b/) lists 77.8 tok/s decode and 2418 tok/s prefill for 8-bit oMLX on an M5 Pro at 1024 tokens. None of them is this `fp8_e4m3` table.

## Rule

MTP is off. The banner also prints depth 3; that depth is unused while MTP is off. Thinking is off. This pipeline maps `reasoning.effort` `none` onto the instruct card (`minimal`). The banner prints the loaded plane. Each measured prompt is one user message and encodes to that length under the chat template, including the template tokens. A 64-token warmup runs first. The measured prompts start with different labels, so each prefill is a cache miss. Decode is capped at 128 tokens. The 512-token prompt stopped at 108 sampled tokens. The 10000-token prompt sampled the cap. The end-to-end rate adds the prefill tokens to the decode tokens and adds those two clocks. A banner that prints `half` or `thinking on` is a different row.

## Command

```text
uv run python -m harness.synthetic --model gemma4-26b-a4b
```

The banner prints the loaded plane, MTP, thinking, and the decode cap.

## Table

| Prompt tokens | Prefill | Decode | End to end |
|---:|---|---|---|
| 512 | 1190.77 tok/s (512 tok in 0.43 s) | 26.36 tok/s (108 tok in 4.10 s) | 136.87 tok/s (620 tok in 4.53 s) |
| 10000 | 1133.84 tok/s (10000 tok in 8.82 s) | 22.99 tok/s (128 tok in 5.57 s) | 703.82 tok/s (10128 tok in 14.39 s) |

## Search

4 November 2026. Queries: "Gemma 4 26B A4B pp512 tg128 prefill tok/s M5 Pro OR MLX OR llama.cpp 2026", and "1190.77 OR 1133.84 OR 136.87 OR 703.82 OR 26.36 gemma-4-26B-A4B prefill tok/s". The predecessors in this note are the overlaps those queries returned. The search did not find 1190.77 and 26.36, or 1133.84 and 22.99, in those sources.

## Corrections

Later finds are appended here. The sections above stay as committed.
