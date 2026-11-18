# Synthetic prompt, half cache

Measurement note for the `gemma4_26b_a4b` pipeline. The serving contract is [../ARCHITECTURE.md](../ARCHITECTURE.md). This file records one synthetic-prompt run on `half`. It is a recorded run, not a format bound. The `fp8_e4m3` table is [synthetic-prompt.md](synthetic-prompt.md).

## Remainder

On an M5 Pro with 48 GB, Gemma 4 26B-A4B with `half` keys and values and MTP off passed a cold instruct prompt.

- A 512-token prompt prefilled at 869.04 tok/s and decoded 113 tokens at 31.92 tok/s.
- A 10000-token prompt prefilled at 481.99 tok/s and decoded 102 tokens at 27.41 tok/s.
- The end-to-end rates, prefill plus decode, were 151.33 tok/s and 412.83 tok/s.

A search on 15 November 2026 did not find that table.

## Predecessors

- Method. [llama-bench](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md) defaults to a 512-token prompt and 128 generated tokens. [`mlx_lm.benchmark`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/benchmark.py) uses the same prompt default. [Artificial Analysis](https://artificialanalysis.ai/methodology/performance-benchmarking) times a prompt of about 10000 tokens and reports time to first token, not prefill tok/s.
- Model. [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it). The card's published scores are not this harness rate.
- Rate. The same harness on `fp8_e4m3` keys and values recorded 1190.77 tok/s prefill and 26.36 tok/s decode for 512 tokens, and 1133.84 tok/s prefill and 22.99 tok/s decode for 10000 tokens, on 4 November 2026. That table is [synthetic-prompt.md](synthetic-prompt.md).
- Rate. [oMLX 4-bit on an M5 Pro with 64 GB](https://omlx.ai/benchmarks/performance/4x7iexzb) reports 2,571 tok/s prefill and 99.4 tok/s generation at 1,024 tokens, and 2,461 tok/s prefill and 77.4 tok/s generation at 4,096 tokens.
- Rate. [Token Atlas](https://tokatlas.github.io/models/gemma-4-26b-a4b/) lists an 8-bit oMLX row on an M5 Pro at 110.4 tok/s decode and 2,373 tok/s prefill at 4,096 tokens.
- Rate. [A llama.cpp UD-Q4_K_XL recipe on an M5 Pro with 24 GB](https://local-ai-registry.vercel.app/recipes/gemma-4-26b-a4b-it-ud-q4-k-xl-apple-m5-pro-24gb-llama-cpp-tp1) reports 1147.5 tok/s prefill and 55 tok/s decode.
- None of them is this `half` table.

## Rule

- MTP is off.
- The banner also prints depth 3; that depth is unused while MTP is off.
- Thinking is off.
- This pipeline maps `reasoning.effort` `none` onto the instruct card (`minimal`).
- The banner prints the loaded plane.
- Each measured prompt is one user message and encodes to that length under the chat template, including the template tokens.
- A 64-token warmup runs first.
- The measured prompts start with different labels, so each prefill is a cache miss.
- Decode is capped at 128 tokens.
- The 512-token prompt stopped at 113 sampled tokens.
- The 10000-token prompt stopped at 102 sampled tokens.
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
| 512 | 869.04 tok/s (512 tok in 0.59 s) | 31.92 tok/s (113 tok in 3.54 s) | 151.33 tok/s (625 tok in 4.13 s) |
| 10000 | 481.99 tok/s (10000 tok in 20.75 s) | 27.41 tok/s (102 tok in 3.72 s) | 412.83 tok/s (10102 tok in 24.47 s) |

## Search

15 November 2026. Queries: "869.04 OR 481.99 OR 151.33 OR 412.83 OR 31.92 gemma-4-26B-A4B half prefill tok/s" and "gemma-4-26B-A4B-it M5 Pro pp512 tg128 prefill tok/s half KV 2026". The predecessors in this note are the overlaps those queries returned. The search did not find 869.04 and 31.92, or 481.99 and 27.41, in those sources.

## Corrections

Later finds are appended here. The sections above stay as committed.

15 November 2026. The harness now samples at temperature 0, fills 128 tokens, and puts a fresh nonce in the label so the prefill is cold. The same `half` MTP-off instruct run recorded:

| Prompt tokens | Prefill | Decode | End to end |
|---:|---|---|---|
| 512 | 940.96 tok/s (512 tok in 0.54 s) | 32.53 tok/s (128 tok in 3.94 s) | 142.86 tok/s (640 tok in 4.48 s) |
| 1024 | 974.84 tok/s (1024 tok in 1.05 s) | 32.38 tok/s (128 tok in 3.95 s) | 230.40 tok/s (1152 tok in 5.00 s) |
| 4096 | 700.91 tok/s (4096 tok in 5.84 s) | 29.59 tok/s (128 tok in 4.33 s) | 415.34 tok/s (4224 tok in 10.17 s) |
| 10000 | 487.04 tok/s (10000 tok in 20.53 s) | 26.09 tok/s (128 tok in 4.91 s) | 398.11 tok/s (10128 tok in 25.44 s) |

The command is `uv run python -m harness.synthetic --model gemma4_26b_a4b --thinking off`. A search the same day for "940.96 OR 974.84 OR 700.91 OR 487.04 gemma-4-26B-A4B prefill tok/s half" did not find those four prefill rates. The table above this correction used the instruct card and stopped before 128 tokens.

15 November 2026. The aligned table uses the packed folder `gemma-4-26B-A4B-it-fp8-e20`. Resident matrices are fp8. Of the 3,840 routed experts, 768 are fp8 and 3,072 are fp4. Keys and values are `half`.

Queries the same day: "gemma-4-26B-A4B oMLX M5 Pro 48GB 1721 74.1 4096 site:omlx.ai", and a fetch of the llama.cpp recipe page. A direct fetch of the oMLX pages returned 504. The search result for [6w533fgx](https://omlx.ai/benchmarks/performance/6w533fgx) included the table and `turboquant_kv_enabled` false.

- [llama.cpp UD-Q4_K_XL on an M5 Pro with 24 GB](https://local-ai-registry.vercel.app/recipes/gemma-4-26b-a4b-it-ud-q4-k-xl-apple-m5-pro-24gb-llama-cpp-tp1) runs `llama-bench` with `-pg 512,128` and `-d 3584`. The measured row is context 4,224, prefill 1,147.5 tok/s, and decode 55 tok/s. The 512-token prompt is timed after a depth of 3,584 tokens.
- [oMLX 4-bit on an M5 Pro with 48 GB](https://omlx.ai/benchmarks/performance/6w533fgx), KV compression off and MTP off, reports 1,721 tok/s prefill and 74.1 tok/s generation at 4,096 tokens, and 1,855 tok/s prefill and 62.5 tok/s generation at 16,384 tokens.
- [oMLX 4-bit on an M5 Pro with 48 GB](https://omlx.ai/benchmarks/performance/6hg5kmhw) reports 2,082 tok/s prefill and 76.9 tok/s generation at 1,024 tokens, and 2,012 tok/s prefill and 74.0 tok/s generation at 4,096 tokens. The returned page text does not list the KV-compression flag.
- No published Gemma row on this chip uses a 10,000-token prompt. The 16,384-token row above is the nearest longer step on the same machine.
- The search did not find 940.96, 974.84, 700.91, or 487.04.

15 November 2026. A 640-token half span now gathers the cache and scores it with `matmul_half_64_batch`. Other half spans stay on `attend_span`. The same protocol run recorded:

| Prompt tokens | Prefill | Decode | End to end |
|---:|---|---|---|
| 512 | 920.39 tok/s (512 tok in 0.56 s) | 32.80 tok/s (128 tok in 3.90 s) | 143.50 tok/s (640 tok in 4.46 s) |
| 1024 | 947.85 tok/s (1024 tok in 1.08 s) | 32.87 tok/s (128 tok in 3.89 s) | 231.79 tok/s (1152 tok in 4.97 s) |
| 4096 | 1047.66 tok/s (4096 tok in 3.91 s) | 30.67 tok/s (128 tok in 4.17 s) | 522.77 tok/s (4224 tok in 8.08 s) |
| 10000 | 977.06 tok/s (10000 tok in 10.23 s) | 28.25 tok/s (128 tok in 4.53 s) | 686.18 tok/s (10128 tok in 14.76 s) |

The banner was `kv half mtp off depth 3 thinking off decode 128`, and every measured prefill was a cache miss. A search the same day for "920.39 OR 947.85 OR 1047.66 OR 977.06 gemma-4-26B-A4B prefill tok/s half" did not find those four prefill rates. The published rows named in the correction above are different packs and different prompt lengths.

18 November 2026. The same protocol run, nonce `d51201c1`, recorded:

| Prompt tokens | Prefill | Decode | End to end |
|---:|---|---|---|
| 512 | 911.63 tok/s (512 tok in 0.56 s) | 31.19 tok/s (128 tok in 4.10 s) | 137.34 tok/s (640 tok in 4.66 s) |
| 1024 | 937.42 tok/s (1024 tok in 1.09 s) | 32.01 tok/s (128 tok in 4.00 s) | 226.33 tok/s (1152 tok in 5.09 s) |
| 4096 | 975.64 tok/s (4096 tok in 4.20 s) | 30.37 tok/s (128 tok in 4.21 s) | 502.26 tok/s (4224 tok in 8.41 s) |
| 10000 | 812.87 tok/s (10000 tok in 12.30 s) | 25.41 tok/s (128 tok in 5.04 s) | 584.08 tok/s (10128 tok in 17.34 s) |

The banner was `kv half mtp off depth 3 thinking off decode 128`, and every measured prefill was a cache miss. A search the same day for "911.63 OR 937.42 OR 975.64 OR 812.87 gemma-4-26B-A4B prefill tok/s half" did not find those four prefill rates. The published rows are different packs, machines, and prompt lengths.

19 November 2026. Decode projections launch `gemv_fp8_vec`. The same protocol run, nonce `7d0eba7f`, recorded:

| Prompt tokens | Prefill | Decode | End to end |
|---:|---|---|---|
| 512 | 935.02 tok/s (512 tok in 0.55 s) | 33.74 tok/s (128 tok in 3.79 s) | 147.47 tok/s (640 tok in 4.34 s) |
| 1024 | 938.28 tok/s (1024 tok in 1.09 s) | 33.85 tok/s (128 tok in 3.78 s) | 236.55 tok/s (1152 tok in 4.87 s) |
| 4096 | 1046.11 tok/s (4096 tok in 3.92 s) | 32.97 tok/s (128 tok in 3.88 s) | 541.54 tok/s (4224 tok in 7.80 s) |
| 10000 | 884.93 tok/s (10000 tok in 11.30 s) | 27.38 tok/s (128 tok in 4.68 s) | 633.79 tok/s (10128 tok in 15.98 s) |

The banner was `kv half mtp off depth 3 thinking off decode 128`, and every measured prefill was a cache miss. A search the same day for "935.02 OR 938.28 OR 1046.11 OR 884.93 gemma-4-26B-A4B prefill tok/s half" did not find those four prefill rates. The published rows are different packs, machines, and prompt lengths.

19 November 2026. `half` keys and values are `[head, capacity, dim]`. The same protocol run, nonce `f19a7c20`, recorded:

| Prompt tokens | Prefill | Decode | End to end |
|---:|---|---|---|
| 512 | 860.83 tok/s (512 tok in 0.59 s) | 33.75 tok/s (128 tok in 3.79 s) | 146.12 tok/s (640 tok in 4.38 s) |
| 1024 | 898.65 tok/s (1024 tok in 1.14 s) | 33.92 tok/s (128 tok in 3.77 s) | 234.62 tok/s (1152 tok in 4.91 s) |
| 4096 | 956.07 tok/s (4096 tok in 4.28 s) | 33.27 tok/s (128 tok in 3.85 s) | 519.56 tok/s (4224 tok in 8.13 s) |
| 10000 | 902.08 tok/s (10000 tok in 11.09 s) | 29.28 tok/s (128 tok in 4.37 s) | 655.11 tok/s (10128 tok in 15.46 s) |

The banner was `kv half mtp off depth 3 thinking off decode 128`, and every measured prefill was a cache miss. A search the same day for "860.83 OR 898.65 OR 956.07 OR 902.08 gemma-4-26B-A4B prefill tok/s half" did not find those four prefill rates. The published rows are different packs, machines, and prompt lengths.
