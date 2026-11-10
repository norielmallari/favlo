# Synthetic prompt

Measurement note for the `qwen36_35b_a3b` pipeline. The serving contract is [../README.md](../README.md). This file records one synthetic-prompt run. It is a recorded run, not a format bound.

## Remainder

On an M5 Pro with 48 GB, `Qwen3.6-35B-A3B-fp8-e20` with `fp8_e4m3` keys and values and MTP off passed a cold instruct prompt. A 512-token prompt prefilled at 825.26 tok/s and decoded 128 tokens at 67.97 tok/s. A 10000-token prompt prefilled at 1272.73 tok/s and decoded 128 tokens at 67.79 tok/s. The end-to-end rates, prefill plus decode, were 256.00 tok/s and 1038.77 tok/s. A search on 4 November 2026 did not find that table.

## Predecessors

- Method. [llama-bench](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md) defaults to a 512-token prompt and 128 generated tokens. [`mlx_lm.benchmark`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/benchmark.py) uses the same prompt default. [Artificial Analysis](https://artificialanalysis.ai/methodology/performance-benchmarking) times a prompt of about 10000 tokens and reports time to first token, not prefill tok/s.
- Model. [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) and [Qwen/Qwen3.6-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8). The card reports SWE-bench Verified 73.4 for this checkpoint. That score is not this harness rate.
- Rate. [llama-bench on one R9700](https://deepindeepsea.github.io/r9700-vllm-blogs/part13-r9700-the-35b-only-one-runtime-could-load.html) reports 2752.05 tok/s at pp512 and 78.48 tok/s at tg128 for a Q4_K_M file. [An M1 Max llama.cpp and MLX comparison](https://bishal.io/blog/six-local-models-m1-max-llama-cpp-vs-mlx/) reports 891.7 and 636.3 tok/s prefill, and 54.5 and 67.8 tok/s generation, on a 512-token prompt and 128 generated tokens, Q4_K_M and 4-bit. [oMLX on an M3 Ultra](https://omlx.ai/benchmarks/3hbboanp) reports 1473 tok/s prefill and 85.1 tok/s generation at 1k, 8-bit. [How To Spark](https://howtospark.com/recipes/qwen3-6-35b-a3b-nvfp4) reports about 7.1k tok/s prefill and 73.3 tok/s decode at 8k for an NVFP4 build on one DGX Spark. None of them is this `fp8_e4m3` table.

## Rule

MTP is off. Thinking is off, so the request uses the instruct card and `reasoning.effort` `none`. The banner prints the loaded plane. Each measured prompt is one user message and encodes to that length under the chat template, including the template tokens. A 64-token warmup runs first. The measured prompts start with different labels, so each prefill is a cache miss. Decode is capped at 128 tokens. The end-to-end rate adds the prefill tokens to the decode tokens and adds those two clocks. A banner that prints `half`, `e2m1`, `f32`, or `thinking on` is a different row.

## Command

```text
uv run python -m harness.synthetic --model qwen36-35b-a3b
```

The banner prints the loaded plane, MTP, thinking, and the decode cap.

## Table

| Prompt tokens | Prefill | Decode | End to end |
|---:|---|---|---|
| 512 | 825.26 tok/s (512 tok in 0.62 s) | 67.97 tok/s (128 tok in 1.88 s) | 256.00 tok/s (640 tok in 2.50 s) |
| 10000 | 1272.73 tok/s (10000 tok in 7.86 s) | 67.79 tok/s (128 tok in 1.89 s) | 1038.77 tok/s (10128 tok in 9.75 s) |

## Search

4 November 2026. Queries: "825.26 OR 1272.73 OR 1038.77 OR 67.97 Qwen3.6-35B-A3B prefill tok/s", and "Qwen3.6-35B-A3B pp512 OR 512 tokens prefill tok/s Metal OR MLX OR llama.cpp 10000 prompt". The predecessors in this note are the overlaps those queries returned. The search did not find 825.26 and 67.97, or 1272.73 and 67.79, in those sources.

## Corrections

Later finds are appended here. The sections above stay as committed.
