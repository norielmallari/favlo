# Synthetic prompt, full fp4

Measurement note for the `qwen38_27b` pipeline. The serving contract is [../ARCHITECTURE.md](../ARCHITECTURE.md). This file records one synthetic-prompt run on the full-fp4 GPTQ-FOEM record with `half` keys and values. It is a recorded run, not a format bound. The fp8-e20 table on this plane is [synthetic-prompt-half.md](synthetic-prompt-half.md).

## Remainder

On an M5 Pro with 48 GB, `Qwen3.8-27B-fp4` with `half` keys and values and MTP off passed a cold instruct prompt.

- A 512-token prompt prefilled at 310.49 tok/s and decoded 128 tokens at 16.70 tok/s.
- A 1024-token prompt prefilled at 410.11 tok/s and decoded 128 tokens at 16.48 tok/s.
- A 4096-token prompt prefilled at 498.10 tok/s and decoded 128 tokens at 15.79 tok/s.
- A 10000-token prompt prefilled at 476.25 tok/s and decoded 128 tokens at 14.86 tok/s.
- The end-to-end rates, prefill plus decode, were 68.74, 112.17, 258.67, and 342.05 tok/s.

A search on 23 November 2026 did not find that table.

## Predecessors

- Method. [llama-bench](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md) defaults to a 512-token prompt and 128 generated tokens. [`mlx_lm.benchmark`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/benchmark.py) uses the same prompt default. [Artificial Analysis](https://artificialanalysis.ai/methodology/performance-benchmarking) times a prompt of about 10000 tokens and reports time to first token, not prefill tok/s.
- Model. [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). The card does not publish this harness rate table.
- Rate. The same harness on `Qwen3.8-27B-fp8-e20` with `half` keys and values recorded 293.68 tok/s prefill and 14.37 tok/s decode at 512 tokens, and 467.09 tok/s prefill and 13.77 tok/s decode at 10000 tokens, on 16 November 2026. That table is [synthetic-prompt-half.md](synthetic-prompt-half.md).
- Rate. [FlagOS W4A8 on an M5 Pro with 64 GB, CPU only](https://huggingface.co/FlagRelease/Qwen3.8-27B-W4A8-arm-FlagOS-Express) reports 74.93 tok/s at pp512 and 12.73 tok/s at tg128. Metal and the GPU are off.
- Rate. [oMLX 4-bit on an M5 Pro with 48 GB and 16 GPU cores](https://omlx.ai/benchmarks/performance/gsu2btxi) reports 401.0 tok/s prefill and 22.9 tok/s generation at 4,096 tokens. The [oMLX community table](https://omlx.ai/benchmarks/performance) lists `Qwen3.8-27B-MLX` 4-bit on an M5 Pro with 48 GB at 482.1 tok/s prefill and 25.4 tok/s generation at 1,024 tokens, dated 25 August 2026.
- Rate. [AWQ INT4 on one RTX 5090](https://huggingface.co/abihsoro/Qwen3.8-27B-AWQ-INT4) reports 3,349 tok/s at pp512 and 74.9 tok/s at tg128.
- None of them is this full-fp4 table.

## Rule

MTP is off. The banner also prints depth 3; that depth is unused while MTP is off. Thinking is off, so the request uses the instruct card and `reasoning.effort` `none`. Temperature is 0. The banner prints the loaded plane. The loaded folder is `Qwen3.8-27B-fp4`.

- Each measured prompt is one user message and encodes to that length under the chat template, including the template tokens.
- A 64-token warmup runs first.
- The measured prompts start with different labels, so each prefill is a cache miss.
- Decode is filled to 128 tokens at every length.
- The end-to-end rate adds the prefill tokens to the decode tokens and adds those two clocks.
- A banner that prints `fp8_e4m3`, another weight mix, or `thinking on` is a different row.

## Command

```text
uv run python -m harness.synthetic --model qwen38_27b --thinking off --mtp off
```

The banner prints the loaded plane, MTP, thinking, and the decode cap.

## Table

The packed folder is `Qwen3.8-27B-fp4`. Language projections are GPTQ with the FOEM column update at β = 0.1, stored as fp4 E2M1 with an E8M0 scale every 32. The embedding and the vision tower stay round-to-nearest fp4. MTP matrices are Metal `half`. Keys and values are `half`.

| Prompt tokens | Prefill | Decode | End to end |
|---:|---|---|---|
| 512 | 310.49 tok/s (512 tok in 1.65 s) | 16.70 tok/s (128 tok in 7.66 s) | 68.74 tok/s (640 tok in 9.31 s) |
| 1024 | 410.11 tok/s (1024 tok in 2.50 s) | 16.48 tok/s (128 tok in 7.77 s) | 112.17 tok/s (1152 tok in 10.27 s) |
| 4096 | 498.10 tok/s (4096 tok in 8.22 s) | 15.79 tok/s (128 tok in 8.11 s) | 258.67 tok/s (4224 tok in 16.33 s) |
| 10000 | 476.25 tok/s (10000 tok in 21.00 s) | 14.86 tok/s (128 tok in 8.61 s) | 342.05 tok/s (10128 tok in 29.61 s) |

## Search

23 November 2026. Queries: "310.49 OR 410.11 OR 498.10 OR 476.25 Qwen3.8-27B prefill tok/s", "\"310.49\" \"410.11\" Qwen3.8-27B tok/s", and "Qwen3.8-27B GPTQ fp4 OR W4A16 M5 Pro prefill tok/s pp512 tg128 2026". The predecessors in this note are the overlaps those queries returned. The search did not find 310.49, 410.11, 498.10, or 476.25 as prefill rates for this model.

## Corrections

Later finds are appended here. The sections above stay as committed.
