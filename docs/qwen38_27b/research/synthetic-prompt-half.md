# Synthetic prompt, half cache

Measurement note for the `qwen38_27b` pipeline. The serving contract is [../ARCHITECTURE.md](../ARCHITECTURE.md). This file records one synthetic-prompt run on `half`. It is a recorded run, not a format bound. The `fp8_e4m3` table is [synthetic-prompt.md](synthetic-prompt.md).

## Remainder

On an M5 Pro with 48 GB, `Qwen3.8-27B-fp8-e20` with `half` keys and values and MTP off passed a cold instruct prompt.

- A 512-token prompt prefilled at 295.35 tok/s and decoded 128 tokens at 15.13 tok/s.
- A 1024-token prompt prefilled at 392.19 tok/s and decoded 128 tokens at 15.07 tok/s.
- A 4096-token prompt prefilled at 477.30 tok/s and decoded 128 tokens at 14.56 tok/s.
- A 10000-token prompt prefilled at 466.55 tok/s and decoded 128 tokens at 13.37 tok/s.
- The end-to-end rates, prefill plus decode, were 62.81, 103.78, 243.18, and 326.71 tok/s.

A search on 15 November 2026 did not find that table.

## Predecessors

- Method. [llama-bench](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md) defaults to a 512-token prompt and 128 generated tokens. [`mlx_lm.benchmark`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/benchmark.py) uses the same prompt default. [Artificial Analysis](https://artificialanalysis.ai/methodology/performance-benchmarking) times a prompt of about 10000 tokens and reports time to first token, not prefill tok/s.
- Model. [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). The card does not publish this harness rate table.
- Rate. The same harness on `fp8_e4m3` keys and values recorded 427.59 tok/s prefill and 14.61 tok/s decode for 512 tokens, and 483.79 tok/s prefill and 13.19 tok/s decode for 10000 tokens, on 4 November 2026. The 10000-token decode stopped at 20 tokens. That table is [synthetic-prompt.md](synthetic-prompt.md).
- Rate. [FlagOS W4A8 on an M5 Pro with 64 GB, CPU only](https://huggingface.co/FlagRelease/Qwen3.8-27B-W4A8-arm-FlagOS-Express) reports 74.93 tok/s at pp512 and 12.73 tok/s at tg128. Metal and the GPU are off.
- Rate. The [oMLX community table](https://omlx.ai/benchmarks/performance) lists `Qwen3.8-27B-MLX` 4-bit on an M5 Pro with 48 GB at 482.1 tok/s prefill and 25.4 tok/s generation at 1,024 tokens, dated 25 August 2026. [oMLX 4-bit on an M5 Pro with 48 GB](https://omlx.ai/benchmarks/performance/0z2uuqeo), MTP off and KV compression off, reports 465.1 tok/s prefill and 16.9 tok/s generation at 8,192 tokens.
- Rate. [One model on four machines](https://baem1n.dev/en/posts/qwen3-8-27b-four-device-benchmark/) reports MLX on an M5 Max at 335 tok/s prefill and 25.45 tok/s decode for a 512-token prompt, and llama.cpp Metal on that machine at 306 and 17.23 tok/s. The lengths are 512, 2048, 8192, and 32768.
- Rate. [llama.cpp Q4_K_M on a DGX Spark](https://github.com/vcruz305/Qwen3.8-27B-DGX-Spark) reports 844.11 tok/s at pp512 and 12.38 tok/s at tg128.
- Rate. [Int8 on two RTX 3090s](https://github.com/noonghunna/club-3090/issues/1102) reports 2466.32 tok/s prefill on a 10000-token prompt. [Artificial Analysis](https://artificialanalysis.ai/models/qwen3-8-27b) reports hosted output speed and time to first token for this checkpoint, not prefill tok/s.
- None of them is this `half` table.

## Rule

MTP is off. The banner also prints depth 3; that depth is unused while MTP is off. Thinking is off, so the request uses the instruct card and `reasoning.effort` `none`. Temperature is 0. The banner prints the loaded plane. The loaded folder is `Qwen3.8-27B-fp8-e20`.

- Each measured prompt is one user message and encodes to that length under the chat template, including the template tokens.
- A 64-token warmup runs first.
- The measured prompts start with different labels, so each prefill is a cache miss.
- Decode is filled to 128 tokens at every length.
- The end-to-end rate adds the prefill tokens to the decode tokens and adds those two clocks.
- A banner that prints `fp8_e4m3` or `thinking on` is a different row.

## Command

```text
uv run python -m harness.synthetic --model qwen38_27b --thinking off --mtp off
```

The banner prints the loaded plane, MTP, thinking, and the decode cap.

## Table

The packed folder is `Qwen3.8-27B-fp8-e20`. Twelve of the 64 language feed-forward layers are fp8. The embedding, the projections, `lm_head`, the vision tower, and the other feed-forward layers stay fp4. MTP matrices are Metal `half`. Keys and values are `half`.

| Prompt tokens | Prefill | Decode | End to end |
|---:|---|---|---|
| 512 | 295.35 tok/s (512 tok in 1.73 s) | 15.13 tok/s (128 tok in 8.46 s) | 62.81 tok/s (640 tok in 10.19 s) |
| 1024 | 392.19 tok/s (1024 tok in 2.61 s) | 15.07 tok/s (128 tok in 8.49 s) | 103.78 tok/s (1152 tok in 11.10 s) |
| 4096 | 477.30 tok/s (4096 tok in 8.58 s) | 14.56 tok/s (128 tok in 8.79 s) | 243.18 tok/s (4224 tok in 17.37 s) |
| 10000 | 466.55 tok/s (10000 tok in 21.43 s) | 13.37 tok/s (128 tok in 9.57 s) | 326.71 tok/s (10128 tok in 31.00 s) |

## Search

15 November 2026. Queries: "295.35 OR 392.19 OR 477.30 OR 466.55 Qwen3.8-27B prefill tok/s", "Qwen3.8-27B M5 Pro pp512 OR pp1024 OR pp4096 tg128 prefill tok/s half KV OR fp8 2026", "Qwen3.8-27B oMLX benchmark M5 Pro 1024 4096 prefill generation tok/s", "Qwen3.8-27B llama-bench pp512 tg128 M5 Pro OR M5 Max MLX prefill", and "Qwen3.8-27B 10000 OR 10k tokens prefill tok/s M5 Artificial Analysis time to first token". The predecessors in this note are the overlaps those queries returned. The search did not find 295.35, 392.19, 477.30, or 466.55.

## Corrections

Later finds are appended here. The sections above stay as committed.

16 November 2026. Gate and up store the scaled int8 product as half, and SwiGLU reads that half. The same protocol run recorded:

| Prompt tokens | Prefill | Decode | End to end |
|---:|---|---|---|
| 512 | 293.68 tok/s (512 tok in 1.74 s) | 14.37 tok/s (128 tok in 8.91 s) | 60.09 tok/s (640 tok in 10.65 s) |
| 1024 | 403.33 tok/s (1024 tok in 2.54 s) | 14.82 tok/s (128 tok in 8.63 s) | 103.13 tok/s (1152 tok in 11.17 s) |
| 4096 | 446.84 tok/s (4096 tok in 9.17 s) | 14.00 tok/s (128 tok in 9.14 s) | 230.69 tok/s (4224 tok in 18.31 s) |
| 10000 | 467.09 tok/s (10000 tok in 21.41 s) | 13.77 tok/s (128 tok in 9.29 s) | 329.90 tok/s (10128 tok in 30.70 s) |

The banner was `kv half mtp off depth 3 thinking off decode 128`, and every measured prefill was a cache miss. A search the same day for "293.68 OR 403.33 OR 446.84 OR 467.09 Qwen3.8-27B prefill tok/s" did not find those four prefill rates. The published rows named above are different packs and different machines.
