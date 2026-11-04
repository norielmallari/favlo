# Synthetic prompt

Measurement note for the `qwen35-4b` pipeline. The serving contract is [../README.md](../README.md). This file records one synthetic-prompt run. It is a recorded run, not a format bound.

## Remainder

On an M5 Pro with 48 GB, `Qwen3.5-4B-fp8` with `fp8_e4m3` keys and values and MTP off passed a cold instruct prompt. A 512-token prompt prefilled at 2239.26 tok/s and decoded 61 tokens at 40.79 tok/s. A 10000-token prompt prefilled at 991.90 tok/s and decoded 128 tokens at 14.28 tok/s. The end-to-end rates, prefill plus decode, were 331.21 tok/s and 531.93 tok/s. A search on 4 November 2026 did not find that table.

## Predecessors

- Method. [llama-bench](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md) defaults to a 512-token prompt and 128 generated tokens. [`mlx_lm.benchmark`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/benchmark.py) uses the same prompt default. [Artificial Analysis](https://artificialanalysis.ai/methodology/performance-benchmarking) times a prompt of about 10000 tokens and reports time to first token, not prefill tok/s.
- Model. [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B). The card does not publish this harness rate table.
- Rate. [mac-llm-bench on base M5](https://github.com/enescingoz/mac-llm-bench/blob/main/results/m5/base/speed/README.md) reports 1074.2 tok/s at pp512 and 48.7 tok/s at tg128 for Qwen3.5-4B-4bit under MLX, and 628.0 tok/s at pp512 and 29.4 tok/s at tg128 for Q4_K_M under llama.cpp. [oMLX Qwen3.5-4B-8bit on an M5 Pro with 48 GB](https://omlx.ai/benchmarks/performance/x0t6bie9) reports 2643 tok/s prefill and 55.4 tok/s decode at 4096 context. [llmcheck](https://llmcheck.net/models/qwen-35-4b-on-m5-pro/) reports about 66 tok/s decode for llama.cpp Q4_K_M on an M5 Pro. None of them is this `fp8_e4m3` table.

## Rule

MTP is off. Thinking is off, so the request uses the instruct card and `reasoning.effort` `none`. The banner prints the loaded plane. Each measured prompt is one user message and encodes to that length under the chat template, including the template tokens. A 64-token warmup runs first. The measured prompts start with different labels, so each prefill is a cache miss. Decode is capped at 128 tokens. A stop before that cap is the kept length. The end-to-end rate adds the prefill tokens to the decode tokens and adds those two clocks. A banner that prints `half`, `mtp on`, or `thinking on` is a different row.

## Command

```text
uv run python -m harness.synthetic --model qwen35-4b
```

The banner prints the loaded plane, MTP, thinking, and the decode cap.

## Table

| Prompt tokens | Prefill | Decode | End to end |
|---:|---|---|---|
| 512 | 2239.26 tok/s (512 tok in 0.23 s) | 40.79 tok/s (61 tok in 1.50 s) | 331.21 tok/s (573 tok in 1.73 s) |
| 10000 | 991.90 tok/s (10000 tok in 10.08 s) | 14.28 tok/s (128 tok in 8.96 s) | 531.93 tok/s (10128 tok in 19.04 s) |

## Search

4 November 2026. Queries: "Qwen3.5-4B pp512 tg128 prefill tok/s Metal OR MLX OR llama.cpp M5 Pro 2026", and "2239.26 OR 991.90 OR 331.21 OR 531.93 Qwen3.5-4B synthetic prompt fp8_e4m3". The predecessors in this note are the overlaps those queries returned. The search did not find 2239.26 and 40.79, or 991.90 and 14.28, in those sources.

## Corrections

Later finds are appended here. The sections above stay as committed.
