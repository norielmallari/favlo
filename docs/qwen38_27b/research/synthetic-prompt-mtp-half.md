# Synthetic prompt, half cache, MTP on

Measurement note for the `qwen38_27b` pipeline. The serving contract is [../ARCHITECTURE.md](../ARCHITECTURE.md). This file records one synthetic-prompt run on `half` with MTP on. It is a recorded run, not a format bound. The MTP-off table on this plane is [synthetic-prompt-half.md](synthetic-prompt-half.md). The `fp8_e4m3` MTP-on table is [synthetic-prompt-mtp.md](synthetic-prompt-mtp.md).

## Remainder

On an M5 Pro with 48 GB, `Qwen3.8-27B-fp8-e20` with `half` keys and values and MTP on at depth 3 passed a cold instruct prompt.

- A 512-token prompt prefilled at 298.20 tok/s and decoded 128 tokens at 15.99 tok/s.
- A 1024-token prompt prefilled at 396.65 tok/s and decoded 128 tokens at 17.27 tok/s.
- A 4096-token prompt prefilled at 487.60 tok/s and decoded 128 tokens at 15.80 tok/s.
- A 10000-token prompt prefilled at 467.52 tok/s and decoded 128 tokens at 15.33 tok/s.
- The end-to-end rates, prefill plus decode, were 65.78, 115.32, 256.00, and 340.55 tok/s.

A search on 19 November 2026 did not find that table.

## Predecessors

- Method. [llama-bench](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md) defaults to a 512-token prompt and 128 generated tokens. [`mlx_lm.benchmark`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/benchmark.py) uses the same prompt default. [Artificial Analysis](https://artificialanalysis.ai/methodology/performance-benchmarking) times a prompt of about 10000 tokens and reports time to first token, not prefill tok/s.
- Model. [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). The card does not publish this harness rate table.
- Rate. The same harness on `half` with MTP off recorded 293.68 tok/s prefill and 14.37 tok/s decode at 512 tokens, and 467.09 tok/s prefill and 13.77 tok/s decode at 10000 tokens, on 16 November 2026. That table is [synthetic-prompt-half.md](synthetic-prompt-half.md).
- Rate. The same harness on `fp8_e4m3` with MTP on at depth 3 recorded 359.96 tok/s prefill and 14.97 tok/s decode at 512 tokens, and 468.03 tok/s prefill and 15.17 tok/s decode at 10000 tokens, on 4 November 2026. The 10000-token decode stopped at 92 tokens. That table is [synthetic-prompt-mtp.md](synthetic-prompt-mtp.md).
- Rate. [oMLX oQ4e with MTP on an M5 Pro with 64 GB](https://omlx.ai/benchmarks/performance/0uwpq1jj) reports 464.4 tok/s prefill and 36.7 tok/s generation at 1,024 tokens. [The same pack at 4,096 tokens](https://omlx.ai/benchmarks/performance/606lrmx0) reports 497.3 tok/s prefill and 31.7 tok/s generation. [oMLX oQ4e with adaptive MTP depth 3 and 4-bit TurboQuant keys](https://omlx.ai/benchmarks/performance/qfnkym9u) reports 598.9 tok/s prefill and 34.8 tok/s generation at 1,024 tokens, and 538.3 tok/s prefill and 32.1 tok/s generation at 4,096 tokens, on an M5 Pro with 64 GB.
- Rate. The [oMLX community table](https://omlx.ai/benchmarks/performance) lists `Qwen3.8-27B-MLX` 4-bit on an M5 Pro with 48 GB at 482.1 tok/s prefill and 25.4 tok/s generation at 1,024 tokens, dated 25 August 2026. The same table lists `Qwen3.8-27B-oQ4e-mtp` 4-bit on an M5 Pro with 48 GB at 476.0 tok/s prefill and 17.5 tok/s generation at 1,024 tokens, and 492.7 tok/s prefill and 16.8 tok/s generation at 4,096 tokens, dated 28 September 2026.
- Rate. [llama.cpp IQ4_XS on an M5 Pro Mac mini](https://singula2045.hatenablog.com/entry/2026/10/03/000000) reports 381.62 tok/s at pp512 and 18.45 tok/s at tg128.
- Rate. [One model on four machines](https://baem1n.dev/en/posts/qwen3-8-27b-four-device-benchmark/) reports MLX on an M5 Max at 335 tok/s prefill and 25.45 tok/s decode for a 512-token prompt, and llama.cpp Metal on that machine at 306 and 17.23 tok/s.
- Rate. [Int8 on two RTX 3090s](https://github.com/noonghunna/club-3090/issues/1102) reports 2466.32 tok/s prefill on a 10000-token prompt.
- None of them is this `half` MTP-on table.

## Rule

MTP is on at depth 3. Thinking is off, so the request uses the instruct card and `reasoning.effort` `none`. Temperature is 0. The banner prints the loaded plane. The loaded folder is `Qwen3.8-27B-fp8-e20`.

- Each measured prompt is one user message and encodes to that length under the chat template, including the template tokens.
- A 64-token warmup runs first.
- The measured prompts start with different labels, so each prefill is a cache miss.
- Decode is filled to 128 tokens at every length.
- The end-to-end rate adds the prefill tokens to the decode tokens and adds those two clocks.
- A banner that prints `fp8_e4m3`, `mtp off`, or `thinking on` is a different row.

## Command

```text
uv run python -m harness.synthetic --model qwen38_27b --thinking off --mtp on
```

The banner prints the loaded plane, MTP, thinking, and the decode cap.

## Table

The packed folder is `Qwen3.8-27B-fp8-e20`. Twelve of the 64 language feed-forward layers are fp8. The embedding, the projections, `lm_head`, the vision tower, and the other feed-forward layers stay fp4. MTP matrices are Metal `half`. Keys and values are `half`. MTP depth is 3.

| Prompt tokens | Prefill | Decode | End to end |
|---:|---|---|---|
| 512 | 298.20 tok/s (512 tok in 1.72 s) | 15.99 tok/s (128 tok in 8.01 s) | 65.78 tok/s (640 tok in 9.73 s) |
| 1024 | 396.65 tok/s (1024 tok in 2.58 s) | 17.27 tok/s (128 tok in 7.41 s) | 115.32 tok/s (1152 tok in 9.99 s) |
| 4096 | 487.60 tok/s (4096 tok in 8.40 s) | 15.80 tok/s (128 tok in 8.10 s) | 256.00 tok/s (4224 tok in 16.50 s) |
| 10000 | 467.52 tok/s (10000 tok in 21.39 s) | 15.33 tok/s (128 tok in 8.35 s) | 340.55 tok/s (10128 tok in 29.74 s) |

## Search

19 November 2026. Queries: "298.20 OR 396.65 OR 487.60 OR 467.52 Qwen3.8-27B prefill tok/s", "Qwen3.8-27B M5 Pro pp512 OR pp1024 OR pp4096 tg128 prefill tok/s half KV OR MTP 2026", "Qwen3.8-27B oMLX benchmark M5 Pro 1024 4096 prefill generation tok/s", "Qwen3.8-27B llama-bench pp512 tg128 M5 Pro OR M5 Max MLX prefill", and "Qwen3.8-27B 10000 OR 10k tokens prefill tok/s M5 Artificial Analysis time to first token". The predecessors in this note are the overlaps those queries returned. The search did not find 298.20, 396.65, 487.60, or 467.52.

## Corrections

Later finds are appended here. The sections above stay as committed.
