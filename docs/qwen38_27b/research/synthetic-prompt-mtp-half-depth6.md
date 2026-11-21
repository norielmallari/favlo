# Synthetic prompt, half cache, MTP depth 6

Measurement note for the `qwen38_27b` pipeline. The serving contract is [../ARCHITECTURE.md](../ARCHITECTURE.md). This file records one synthetic-prompt run on `half` with MTP on at depth 6. It is a recorded run, not a format bound. The depth-3 table on this plane is [synthetic-prompt-mtp-half.md](synthetic-prompt-mtp-half.md). The depth-4 table is [synthetic-prompt-mtp-half-depth4.md](synthetic-prompt-mtp-half-depth4.md).

## Remainder

On an M5 Pro with 48 GB, `Qwen3.8-27B-fp8-e20` with `half` keys and values and MTP on at depth 6 passed a cold instruct prompt.

- A 512-token prompt prefilled at 299.19 tok/s and decoded 128 tokens at 13.66 tok/s.
- A 1024-token prompt prefilled at 396.18 tok/s and decoded 128 tokens at 15.93 tok/s.
- A 4096-token prompt prefilled at 482.19 tok/s and decoded 128 tokens at 14.19 tok/s.
- A 10000-token prompt prefilled at 467.52 tok/s and decoded 128 tokens at 10.87 tok/s.
- The end-to-end rates, prefill plus decode, were 57.76, 108.58, 241.23, and 305.34 tok/s.

A search on 19 November 2026 did not find that table.

## Predecessors

- Method. [llama-bench](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md) defaults to a 512-token prompt and 128 generated tokens. [`mlx_lm.benchmark`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/benchmark.py) uses the same prompt default. [Artificial Analysis](https://artificialanalysis.ai/methodology/performance-benchmarking) times a prompt of about 10000 tokens and reports time to first token, not prefill tok/s.
- Model. [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). The card does not publish this harness rate table.
- Rate. The same harness on `half` with MTP on at depth 3 recorded 298.20 tok/s prefill and 15.99 tok/s decode at 512 tokens, and 467.52 tok/s prefill and 15.33 tok/s decode at 10000 tokens, on 19 November 2026. That table is [synthetic-prompt-mtp-half.md](synthetic-prompt-mtp-half.md). The 10000-token prefill rate matches this run. The decode rate does not.
- Rate. [oMLX oQ4e with adaptive MTP depth 6 on an M5 Pro with 48 GB](https://omlx.ai/benchmarks/performance/q8w6wc0a) reports 522.3 tok/s prefill and 30.5 tok/s generation at 4,096 tokens, and [516.1 tok/s prefill and 48.8 tok/s generation at 16,384 tokens](https://omlx.ai/benchmarks/performance/f93j58j3).
- Rate. [An RTX 3080 draft-depth sweep](https://huggingface.co/Davidmg0815/Qwen3.8-27B-abliterated-RTX3080-MTP-benchmarks) reports 59.8 tok/s code and 38.3 tok/s prose at draft depth 6, and about 889 tok/s prefill on a 35,000-token prompt.
- Rate. [MTP on an RTX 4090](https://llm-speed.com/blog/mtp-speculative-decoding-rtx-4090) reports chat-short prefill falling from 428.6 to 299.3 tok/s when MTP turns on, with decode rising from 43.06 to 84.51 tok/s. The page does not record the draft depth.
- None of them is this `half` depth-6 table.

## Rule

MTP is on at depth 6. Thinking is off, so the request uses the instruct card and `reasoning.effort` `none`. Temperature is 0. The banner prints the loaded plane. The loaded folder is `Qwen3.8-27B-fp8-e20`.

- Each measured prompt is one user message and encodes to that length under the chat template, including the template tokens.
- A 64-token warmup runs first.
- The measured prompts start with different labels, so each prefill is a cache miss.
- Decode is filled to 128 tokens at every length.
- The end-to-end rate adds the prefill tokens to the decode tokens and adds those two clocks.
- A banner that prints `fp8_e4m3`, `mtp off`, `thinking on`, or another depth is a different row.

## Command

```text
uv run python -m harness.synthetic --model qwen38_27b --thinking off --mtp on --mtp-depth 6
```

The banner prints the loaded plane, MTP, thinking, and the decode cap.

## Table

The packed folder is `Qwen3.8-27B-fp8-e20`. Twelve of the 64 language feed-forward layers are fp8. The embedding, the projections, `lm_head`, the vision tower, and the other feed-forward layers stay fp4. MTP matrices are Metal `half`. Keys and values are `half`. MTP depth is 6.

| Prompt tokens | Prefill | Decode | End to end |
|---:|---|---|---|
| 512 | 299.19 tok/s (512 tok in 1.71 s) | 13.66 tok/s (128 tok in 9.37 s) | 57.76 tok/s (640 tok in 11.08 s) |
| 1024 | 396.18 tok/s (1024 tok in 2.58 s) | 15.93 tok/s (128 tok in 8.03 s) | 108.58 tok/s (1152 tok in 10.61 s) |
| 4096 | 482.19 tok/s (4096 tok in 8.49 s) | 14.19 tok/s (128 tok in 9.02 s) | 241.23 tok/s (4224 tok in 17.51 s) |
| 10000 | 467.52 tok/s (10000 tok in 21.39 s) | 10.87 tok/s (128 tok in 11.78 s) | 305.34 tok/s (10128 tok in 33.17 s) |

## Search

19 November 2026. Queries: "299.19 OR 396.18 OR 482.19 OR 467.52 Qwen3.8-27B prefill tok/s MTP depth", and "Qwen3.8-27B MTP depth 4 OR depth 6 OR k=4 OR k=6 prefill tok/s M5 Pro oMLX". The predecessors in this note are the overlaps those queries returned. The search did not find 299.19, 396.18, 482.19, or 467.52.

## Corrections

Later finds are appended here. The sections above stay as committed.
