# Synthetic prompt, fp8 attention

Measurement note for the `inkling_2_mini` pipeline. The serving contract is [../ARCHITECTURE.md](../ARCHITECTURE.md). This file records one synthetic-prompt run on the FOEM fp8 attention pack with `half` keys and values. It is a recorded run, not a format bound.

## Remainder

On an M5 Pro with 48 GB, `Inkling-2-Mini` with FOEM E4M3 attention and output projections, a float16 head, `half` keys and values, and MTP off passed a cold instruct prompt.

- A 512-token prompt prefilled at 1197.33 tok/s and decoded 128 tokens at 51.51 tok/s.
- A 1024-token prompt prefilled at 1401.14 tok/s and decoded 128 tokens at 50.43 tok/s.
- A 4096-token prompt prefilled at 1440.07 tok/s and decoded 128 tokens at 49.90 tok/s.
- A 10000-token prompt prefilled at 1349.14 tok/s and decoded 128 tokens at 47.87 tok/s.
- The end-to-end rates, prefill plus decode, were 219.93, 352.29, 782.22, and 1004.76 tok/s.

A search on 23 November 2026 did not find that table.

## Predecessors

- Method. [llama-bench](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md) defaults to a 512-token prompt and 128 generated tokens. [`mlx_lm.benchmark`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/benchmark.py) uses the same prompt default. [Artificial Analysis](https://artificialanalysis.ai/methodology/performance-benchmarking) times a prompt of about 10000 tokens and reports time to first token, not prefill tok/s.
- Model. [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini), Apache-2.0. The [model card](https://thinkingmachines.ai/model-card/inkling-2-mini/) reports 26.4B parameters, 3.4B active per pass, and SWE-bench Verified (bash only) 79.2%. That score is not this harness rate.
- Rate. [llama.cpp on a 64 GB M5 Pro Mac mini](https://huggingface.co/datasets/pratikgx/inkling-2-mini-m5-pro-mac-mini-bench) with the official IQ4_NL GGUF reports 490.8 tok/s prompt and 29.95 tok/s generation at 512 tokens, and 357.5 and 28.42 tok/s at 4096 tokens. It generates 256 tokens per cell at temperature 0.
- Rate. [An MLX nvfp4 and mxfp8 repack on an M5 Pro with 48 GB](https://huggingface.co/baytizzel/Inkling-2-Mini-MLX-M5Pro-48gb) reports 26.6 tok/s decode, and 750 tok/s prefill for a 3.3K-token prompt.
- Rate. [A Metal engine on an M5 Pro with 48 GB](https://github.com/Eriskii/erisinky) reading a Q4_K_M GGUF reports 2200 tok/s prefill on a 2394-token prompt and 63.0 tok/s short-context decode.
- Rate. [oMLX on an M5 Max with 128 GB](https://omlx.ai/benchmarks/performance/1gf0zsb5), pack `Inkling-2-Mini-oQ4e-mtp`, reports 1,579 tok/s prefill and 57.8 tok/s generation at 1k tokens, and 1,841 and 63.9 tok/s at 4k tokens.
- Rate. None of them is this fp8 table. The earlier GPTQ dense fp4 table is [synthetic-prompt-half.md](synthetic-prompt-half.md).

## Rule

MTP is off. The banner also prints depth 3; that depth is unused while MTP is off. Thinking is off, so the request uses the instruct card and `reasoning.effort` `none`. Temperature is 0. The banner prints the loaded plane.

- Each measured prompt is one user message and encodes to that length under the chat template, including the template tokens and the instruct opening `encode_chat` appends.
- A 64-token warmup runs first.
- The measured prompts start with different labels, so each prefill is a cache miss.
- Decode is filled to 128 tokens at every length.
- The end-to-end rate adds the prefill tokens to the decode tokens and adds those two clocks.
- A banner that prints another plane or `thinking on` is a different row.

## Command

```text
uv run python -m harness.synthetic --model inkling_2_mini --thinking off
```

The banner prints the loaded plane, MTP, thinking, and the decode cap.

## Table

The fused attention plane and the output projection are E4M3 with one E8M0 scale per 32, from the FOEM update on the saved calibration grams. The head and the embedding are float16. Experts in layers 0–19 are NVFP4 requantized to Metal fp4. Experts in layers 20–23 are MXFP8. Keys and values are `half`.

| Prompt tokens | Prefill | Decode | End to end |
|---:|---|---|---|
| 512 | 1197.33 tok/s (512 tok in 0.43 s) | 51.51 tok/s (128 tok in 2.48 s) | 219.93 tok/s (640 tok in 2.91 s) |
| 1024 | 1401.14 tok/s (1024 tok in 0.73 s) | 50.43 tok/s (128 tok in 2.54 s) | 352.29 tok/s (1152 tok in 3.27 s) |
| 4096 | 1440.07 tok/s (4096 tok in 2.84 s) | 49.90 tok/s (128 tok in 2.56 s) | 782.22 tok/s (4224 tok in 5.40 s) |
| 10000 | 1349.14 tok/s (10000 tok in 7.41 s) | 47.87 tok/s (128 tok in 2.67 s) | 1004.76 tok/s (10128 tok in 10.08 s) |

## Search

23 November 2026. Queries: "Inkling-2-Mini 1197.33 OR 48.48 OR 787.76 tok/s prefill M5 Pro" and "Inkling-2-Mini fp8 E4M3 FOEM prefill decode tok/s Apple M5 Pro 48GB 2026". The predecessors in this note are the overlaps those queries returned. The search did not find 1197.33, 1401.14, 1440.07, or 1349.14.

## Corrections

Later finds are appended here. The sections above stay as committed.
