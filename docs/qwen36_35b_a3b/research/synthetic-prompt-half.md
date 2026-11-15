# Synthetic prompt, half cache

Measurement note for the `qwen36_35b_a3b` pipeline. The serving contract is [../ARCHITECTURE.md](../ARCHITECTURE.md). This file records one synthetic-prompt run on `half`. It is a recorded run, not a format bound. The `fp8_e4m3` table is [synthetic-prompt.md](synthetic-prompt.md).

## Remainder

On an M5 Pro with 48 GB, `Qwen3.6-35B-A3B-fp8-e30-24h` with `half` keys and values and MTP off passed a cold instruct prompt.

- A 512-token prompt prefilled at 691.10 tok/s and decoded 128 tokens at 71.36 tok/s.
- A 1024-token prompt prefilled at 998.83 tok/s and decoded 128 tokens at 71.08 tok/s.
- A 4096-token prompt prefilled at 1337.47 tok/s and decoded 128 tokens at 69.31 tok/s.
- A 10000-token prompt prefilled at 1317.11 tok/s and decoded 128 tokens at 67.50 tok/s.
- The end-to-end rates, prefill plus decode, were 252.96, 407.07, 860.29, and 1067.23 tok/s.

A search on 15 November 2026 did not find that table.

## Predecessors

- Method. [llama-bench](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md) defaults to a 512-token prompt and 128 generated tokens. [`mlx_lm.benchmark`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/benchmark.py) uses the same prompt default. [Artificial Analysis](https://artificialanalysis.ai/methodology/performance-benchmarking) times a prompt of about 10000 tokens and reports time to first token, not prefill tok/s.
- Model. [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) and [Qwen/Qwen3.6-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8). The card reports SWE-bench Verified 73.4 for this checkpoint. That score is not this harness rate.
- Rate. The same harness on `fp8_e4m3` keys and values recorded 825.26 tok/s prefill and 67.97 tok/s decode for 512 tokens, and 1272.73 tok/s prefill and 67.79 tok/s decode for 10000 tokens, on 4 November 2026. That table is [synthetic-prompt.md](synthetic-prompt.md).
- Rate. [llama-bench on one R9700](https://deepindeepsea.github.io/r9700-vllm-blogs/part13-r9700-the-35b-only-one-runtime-could-load.html) reports 2752.05 tok/s at pp512, 2531.20 tok/s at pp4096, and 78.48 tok/s at tg128 for a Q4_K_M file.
- Rate. [oMLX FP8 on an M5 Pro with 64 GB](https://omlx.ai/benchmarks/performance/sxg8da3t) reports 1631 tok/s prefill and 56.7 tok/s generation at 1024 tokens, with KV compression off and MTP off. The same session at 4096 tokens is [xmg0pbh7](https://omlx.ai/benchmarks/performance/xmg0pbh7), 2362 tok/s prefill and 57.1 tok/s generation.
- Rate. [oMLX 4-bit on an M5 Pro with 48 GB](https://omlx.ai/benchmarks/performance/0sxj28v0) reports 2896 tok/s prefill and 104.8 tok/s generation at 4096 tokens.
- Rate. No published Qwen3.6-35B-A3B row on this chip uses a 10000-token prompt.
- Rate. None of them is this `half` table.

## Rule

MTP is off. The banner also prints depth 3; that depth is unused while MTP is off. Thinking is off, so the request uses the instruct card and `reasoning.effort` `none`. Temperature is 0. The banner prints the loaded plane.

- Each measured prompt is one user message and encodes to that length under the chat template, including the template tokens.
- A 64-token warmup runs first.
- The measured prompts start with different labels, so each prefill is a cache miss.
- Decode is filled to 128 tokens at every length.
- The end-to-end rate adds the prefill tokens to the decode tokens and adds those two clocks.
- A banner that prints `fp8_e4m3` or `thinking on` is a different row.

## Command

```text
uv run python -m harness.synthetic --model qwen36_35b_a3b --thinking off --mtp off
```

The banner prints the loaded plane, MTP, thinking, and the decode cap.

## Table

The packed folder is `Qwen3.6-35B-A3B-fp8-e30-24h`. Language projections, the token embedding, `lm_head`, the vision tower, and the shared expert are fp8. Of the 10,240 routed experts, 3,072 are fp8 and 7,168 are fp4, chosen by the firing rank of a 24-hour log. Keys and values are `half`.

| Prompt tokens | Prefill | Decode | End to end |
|---:|---|---|---|
| 512 | 691.10 tok/s (512 tok in 0.74 s) | 71.36 tok/s (128 tok in 1.79 s) | 252.96 tok/s (640 tok in 2.53 s) |
| 1024 | 998.83 tok/s (1024 tok in 1.03 s) | 71.08 tok/s (128 tok in 1.80 s) | 407.07 tok/s (1152 tok in 2.83 s) |
| 4096 | 1337.47 tok/s (4096 tok in 3.06 s) | 69.31 tok/s (128 tok in 1.85 s) | 860.29 tok/s (4224 tok in 4.91 s) |
| 10000 | 1317.11 tok/s (10000 tok in 7.59 s) | 67.50 tok/s (128 tok in 1.90 s) | 1067.23 tok/s (10128 tok in 9.49 s) |

## Search

15 November 2026. Queries: "691.10 OR 998.83 OR 1337.47 OR 1317.11 Qwen3.6-35B-A3B prefill tok/s", and "Qwen3.6-35B-A3B M5 Pro pp512 OR pp1024 OR pp4096 tg128 prefill tok/s half KV OR fp8 2026". The predecessors in this note are the overlaps those queries returned. The search did not find 691.10, 998.83, 1337.47, or 1317.11.

## Corrections

Later finds are appended here. The sections above stay as committed.
