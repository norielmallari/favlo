# Synthetic prompt, half cache, MTP depth 2

Measurement note for the `qwen38_27b` pipeline. The serving contract is [../ARCHITECTURE.md](../ARCHITECTURE.md). This file records one synthetic-prompt run on `half` with MTP on at depth 2. It is a recorded run, not a format bound. The depth-3 table on this plane is [synthetic-prompt-mtp-half.md](synthetic-prompt-mtp-half.md).

## Remainder

On an M5 Pro with 48 GB, `Qwen3.8-27B-fp8-e20` with `half` keys and values and MTP on at depth 2 passed a cold instruct prompt.

- A 512-token prompt prefilled at 297.60 tok/s and decoded 128 tokens at 18.22 tok/s.
- A 1024-token prompt prefilled at 392.69 tok/s and decoded 128 tokens at 15.60 tok/s.
- A 4096-token prompt prefilled at 483.37 tok/s and decoded 128 tokens at 15.90 tok/s.
- A 10000-token prompt prefilled at 467.77 tok/s and decoded 128 tokens at 15.41 tok/s.
- The end-to-end rates, prefill plus decode, were 73.23, 106.57, 255.69, and 341.12 tok/s.

A search on 19 November 2026 did not find that table.

## Predecessors

- Method. [llama-bench](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md) defaults to a 512-token prompt and 128 generated tokens. [`mlx_lm.benchmark`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/benchmark.py) uses the same prompt default. [Artificial Analysis](https://artificialanalysis.ai/methodology/performance-benchmarking) times a prompt of about 10000 tokens and reports time to first token, not prefill tok/s.
- Model. [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). The card does not publish this harness rate table.
- Rate. The same harness on `half` with MTP on at depth 3 recorded 298.20 tok/s prefill and 15.99 tok/s decode at 512 tokens, and 467.52 tok/s prefill and 15.33 tok/s decode at 10000 tokens, on 19 November 2026. That table is [synthetic-prompt-mtp-half.md](synthetic-prompt-mtp-half.md).
- Rate. [An RTX 3080 draft-depth sweep](https://huggingface.co/Davidmg0815/Qwen3.8-27B-abliterated-RTX3080-MTP-benchmarks) reports 64.6 tok/s code and 55.1 tok/s prose at draft depth 2, and 891 tok/s prefill on a 35,000-token prompt.
- Rate. [A llama.cpp MTP tune on two Tesla P40s](https://github.com/lyesrock/qwen-mtp-tuning-guide/tree/main/) reports 21.1 tok/s decode for Qwen3.8-27B Q5 at draft depth 2, and about 300 to 375 tok/s prefill on a 20,000-token prompt.
- Rate. [An M5 Max measurement of an abliterated pack](https://huggingface.co/PocketAiHub/Qwen3.8-27B-Abliterated-MTPLX-Optimized-Speed) reports 51.55 tok/s decode at depth 2.
- None of them is this `half` depth-2 table.

## Rule

MTP is on at depth 2. Thinking is off, so the request uses the instruct card and `reasoning.effort` `none`. Temperature is 0. The banner prints the loaded plane. The loaded folder is `Qwen3.8-27B-fp8-e20`.

- Each measured prompt is one user message and encodes to that length under the chat template, including the template tokens.
- A 64-token warmup runs first.
- The measured prompts start with different labels, so each prefill is a cache miss.
- Decode is filled to 128 tokens at every length.
- The end-to-end rate adds the prefill tokens to the decode tokens and adds those two clocks.
- A banner that prints `fp8_e4m3`, `mtp off`, `thinking on`, or another depth is a different row.

## Command

```text
uv run python -m harness.synthetic --model qwen38_27b --thinking off --mtp on --mtp-depth 2
```

The banner prints the loaded plane, MTP, thinking, and the decode cap.

## Table

The packed folder is `Qwen3.8-27B-fp8-e20`. Twelve of the 64 language feed-forward layers are fp8. The embedding, the projections, `lm_head`, the vision tower, and the other feed-forward layers stay fp4. MTP matrices are Metal `half`. Keys and values are `half`. MTP depth is 2.

| Prompt tokens | Prefill | Decode | End to end |
|---:|---|---|---|
| 512 | 297.60 tok/s (512 tok in 1.72 s) | 18.22 tok/s (128 tok in 7.02 s) | 73.23 tok/s (640 tok in 8.74 s) |
| 1024 | 392.69 tok/s (1024 tok in 2.61 s) | 15.60 tok/s (128 tok in 8.20 s) | 106.57 tok/s (1152 tok in 10.81 s) |
| 4096 | 483.37 tok/s (4096 tok in 8.47 s) | 15.90 tok/s (128 tok in 8.05 s) | 255.69 tok/s (4224 tok in 16.52 s) |
| 10000 | 467.77 tok/s (10000 tok in 21.38 s) | 15.41 tok/s (128 tok in 8.31 s) | 341.12 tok/s (10128 tok in 29.69 s) |

## Search

19 November 2026. Query: "297.60 OR 392.69 OR 483.37 OR 467.77 Qwen3.8-27B prefill tok/s MTP depth 2". The predecessors in this note are the overlaps that query returned. The search did not find 297.60, 392.69, 483.37, or 467.77.

## Corrections

Later finds are appended here. The sections above stay as committed.
