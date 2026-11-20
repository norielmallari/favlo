# Synthetic prompt, CUDA

Measurement note for the CUDA runtime of `gemma4-26b-a4b`. The serving contract is [../ARCHITECTURE.md](../ARCHITECTURE.md). This file records one protocol-complete synthetic-prompt run on half keys and values with MTP off. It is a recorded run, not a format bound. The Metal half table is [synthetic-prompt-half.md](synthetic-prompt-half.md).

## Remainder

On a GeForce RTX 3070 Laptop GPU with 8192 MiB, Gemma 4 26B-A4B with int8 dense weights, int8 for 384 of 3840 experts and int4 for the rest, half keys and values, and MTP off passed a cold instruct prompt.

- A 512-token prompt prefilled at 66.58 tok/s and decoded 128 tokens at 10.99 tok/s.
- A 1024-token prompt prefilled at 30.81 tok/s and decoded 128 tokens at 7.91 tok/s.
- A 4096-token prompt prefilled at 12.31 tok/s and decoded 128 tokens at 5.24 tok/s.
- A 10000-token prompt prefilled at 7.44 tok/s and decoded 128 tokens at 3.10 tok/s.
- The end-to-end rates, prefill plus decode, were 33.11, 23.32, 11.82, and 7.32 tok/s.

A search on 19 November 2026 did not find that table.

## Predecessors

- Method. [llama-bench](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md) defaults to a 512-token prompt and 128 generated tokens. [Artificial Analysis](https://artificialanalysis.ai/methodology/performance-benchmarking) times a prompt of about 10000 tokens and reports time to first token, not prefill tok/s.
- Model. [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it). The card publishes task scores. It does not publish this harness rate.
- Rate. [GPUfit](https://gpufit.tech/can-i-run/gemma-4-26b-a4b-it/on/rtx-3070/) estimates 18–25 tok/s for UD-Q4_K_M with expert offload on an RTX 3070 and says that figure is not a benchmark. [llm-bench.io](https://llm-bench.io/benchmarks/cmusmf3af00p601oxfmjf6ihe) records 570.7 tok/s prefill and 57.7 tok/s decode for UD-Q4_K_XL on an RTX 3070 Ti. [Caleb Coffie](https://calebcoffie.com/benchmarks/gemma-4-26b-a4b-it-gguf) records 1229.9 tok/s prefill and 116.3 tok/s decode for an agent shape on an RTX 3090.
- Rate. The same harness on the fp8-e20 pack with half keys and values, on an M5 Pro, recorded 920.39 / 32.80, 947.85 / 32.87, 1047.66 / 30.67, and 977.06 / 28.25 tok/s. That table is [synthetic-prompt-half.md](synthetic-prompt-half.md).
- None of those is this int8-e10 half row.

## Rule

- MTP is off.
- The banner also prints depth 3; that depth is unused while MTP is off.
- Thinking is off.
- This pipeline maps `reasoning.effort` `none` onto the instruct card (`minimal`).
- The banner prints the loaded plane.
- Each measured prompt is one user message and encodes to that length under the chat template, including the template tokens.
- A 64-token warmup runs first.
- The measured prompts start with different labels, so each prefill is a cache miss.
- Decode is capped at 128 tokens. Every measured prompt sampled 128 tokens.
- The end-to-end rate adds the prefill tokens to the decode tokens and adds those two clocks.
- A banner that prints `fp8_e4m3` or `thinking on` is a different row.

## Command

```text
uv run python -m harness.synthetic --model gemma4_26b_a4b --thinking off
```

The banner prints the loaded plane, MTP, thinking, and the decode cap.

## Table

| Prompt tokens | Prefill | Decode | End to end |
|---:|---|---|---|
| 512 | 66.58 tok/s (512 tok in 7.69 s) | 10.99 tok/s (128 tok in 11.64 s) | 33.11 tok/s (640 tok in 19.33 s) |
| 1024 | 30.81 tok/s (1024 tok in 33.23 s) | 7.91 tok/s (128 tok in 16.18 s) | 23.32 tok/s (1152 tok in 49.41 s) |
| 4096 | 12.31 tok/s (4096 tok in 332.81 s) | 5.24 tok/s (128 tok in 24.42 s) | 11.82 tok/s (4224 tok in 357.23 s) |
| 10000 | 7.44 tok/s (10000 tok in 1343.26 s) | 3.10 tok/s (128 tok in 41.28 s) | 7.32 tok/s (10128 tok in 1384.54 s) |

## Search

19 November 2026. Queries: "Gemma 4 26B-A4B tokens per second CUDA RTX 3070 benchmark tok/s" and "66.58 OR 30.81 OR 12.31 OR 7.44 gemma-4-26B-A4B prefill tok/s int8". The predecessors in this note are the overlaps those queries returned. The search did not find 66.58, 30.81, 12.31, and 7.44 as this int8-e10 half table.

## Corrections

Later finds are appended here. The sections above stay as committed.

20 November 2026. The same command, nonce 15a820c3, replaced the table above. A 64-token warmup prefilled at 103.23 tok/s and decoded 16 tokens at 30.30 tok/s. The measured prompts:

| Prompt tokens | Prefill | Decode | End to end |
|---:|---|---|---|
| 512 | 413.36 tok/s (512 tok in 1.24 s) | 19.46 tok/s (128 tok in 6.58 s) | 81.84 tok/s (640 tok in 7.82 s) |
| 1024 | 23.06 tok/s (1024 tok in 44.40 s) | 9.35 tok/s (128 tok in 13.69 s) | 19.83 tok/s (1152 tok in 58.09 s) |
| 4096 | 257.41 tok/s (4096 tok in 15.91 s) | 10.44 tok/s (128 tok in 12.26 s) | 149.95 tok/s (4224 tok in 28.17 s) |
| 10000 | 191.18 tok/s (10000 tok in 52.31 s) | 5.75 tok/s (128 tok in 22.27 s) | 135.80 tok/s (10128 tok in 74.58 s) |

Queries that day: "Gemma 4 26B-A4B tokens per second CUDA RTX 3070 benchmark tok/s prefill" and "413.36 OR 257.41 OR 191.18 gemma-4-26B-A4B prefill tok/s". The search did not find 413.36, 23.06, 257.41, and 191.18 as this int8-e10 half table. [llm-bench.io](https://llm-bench.io/benchmarks/cmusmf3af00p601oxfmjf6ihe) records 570.7 tok/s prefill and 57.7 tok/s decode for UD-Q4_K_XL on an RTX 3070 Ti. [Harris Oldroyd](https://harrisoldroyd.com/benchmarks/gemma-4-26b-a4b/1-rtx-3090-live-llama-server-ud-q4-k-xl-mtp-off/) records a GPU-resident UD-Q4_K_XL sweep on an RTX 3090, 4,565.84 tok/s prefill and 145.11 tok/s decode at an empty cache. Neither is this 8192 MiB int8-e10 half row.

20 November 2026. The same command, nonce 9204cf98, ran after every routed expert was int4 and decode attention split the key span across the multiprocessors. Dense matrices stayed int8. The packed folder name stayed `gemma-4-26B-A4B-it-int8-e10`. The banner printed `kv half`, `mtp off`, and `thinking off`. A 64-token warmup prefilled at 103.52 tok/s and decoded 16 tokens at 33.75 tok/s. Every measured prompt sampled 128 tokens.

| Prompt tokens | Prefill | Decode | End to end |
|---:|---|---|---|
| 512 | 436.25 tok/s (512 tok in 1.17 s) | 33.68 tok/s (128 tok in 3.80 s) | 128.77 tok/s (640 tok in 4.97 s) |
| 1024 | 411.48 tok/s (1024 tok in 2.49 s) | 33.29 tok/s (128 tok in 3.84 s) | 181.99 tok/s (1152 tok in 6.33 s) |
| 4096 | 297.50 tok/s (4096 tok in 13.77 s) | 30.19 tok/s (128 tok in 4.24 s) | 234.54 tok/s (4224 tok in 18.01 s) |
| 10000 | 204.29 tok/s (10000 tok in 48.95 s) | 26.52 tok/s (128 tok in 4.83 s) | 188.32 tok/s (10128 tok in 53.78 s) |

Queries that day: "436.25 OR 33.68 OR 204.29 OR 26.52 Gemma 4 26B prefill tok/s" and "Gemma 4 26B-A4B tokens per second CUDA RTX 3070 prefill int4 2026". The search did not find 436.25 / 33.68, 411.48 / 33.29, 297.50 / 30.19, or 204.29 / 26.52 as this half int4-expert table. [QA.BLOG](https://blog.lucad.cloud/articles/llama-cpp-turboquant-gemma4-26b-moe-8gb-vram) records about 83 tok/s prompt processing and about 27 tok/s generation for a TurboQuant build on an RTX 3070. That is not this 8192 MiB half row.
