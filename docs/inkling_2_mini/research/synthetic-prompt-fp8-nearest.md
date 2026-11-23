# Synthetic prompt, nearest fp8 attention

Measurement note for the `inkling_2_mini` pipeline. The serving contract is [../ARCHITECTURE.md](../ARCHITECTURE.md). This file records one synthetic-prompt run on nearest E4M3 attention with GPTQ-FOEM Metal fp4 experts and `half` keys and values. It is a recorded run, not a format bound.

## Remainder

On an M5 Pro with 48 GB, `Inkling-2-Mini` with nearest E4M3 attention and output projections, a float16 head, GPTQ-FOEM Metal fp4 experts in layers 0–19, MXFP8 experts in layers 20–23, `half` keys and values, and MTP off passed a cold instruct prompt.

- A 512-token prompt prefilled at 1188.57 tok/s and decoded 128 tokens at 50.72 tok/s.
- A 1024-token prompt prefilled at 1390.37 tok/s and decoded 128 tokens at 50.06 tok/s.
- A 4096-token prompt prefilled at 1425.40 tok/s and decoded 128 tokens at 48.87 tok/s.
- A 10000-token prompt prefilled at 1332.01 tok/s and decoded 128 tokens at 47.38 tok/s.
- The end-to-end rates, prefill plus decode, were 216.95, 349.09, 769.40, and 991.97 tok/s.

A search on 23 November 2026 did not find that table.

## Predecessors

- Method. [llama-bench](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md) defaults to a 512-token prompt and 128 generated tokens. [`mlx_lm.benchmark`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/benchmark.py) uses the same prompt default. [Artificial Analysis](https://artificialanalysis.ai/methodology/performance-benchmarking) times a prompt of about 10000 tokens and reports time to first token, not prefill tok/s.
- Model. [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini), Apache-2.0. The [model card](https://thinkingmachines.ai/model-card/inkling-2-mini/) reports 26.4B parameters, 3.4B active per pass, and SWE-bench Verified (bash only) 79.2%. That score is not this harness rate.
- Rate. [llama.cpp on a 64 GB M5 Pro Mac mini](https://huggingface.co/datasets/pratikgx/inkling-2-mini-m5-pro-mac-mini-bench) with the official IQ4_NL GGUF reports 490.8 tok/s prompt and 29.95 tok/s generation at 512 tokens, and 357.5 and 28.42 tok/s at 4096 tokens. It generates 256 tokens per cell at temperature 0.
- Rate. [oMLX on an M5 Pro with 48 GB](https://omlx.ai/benchmarks/performance/5173qvsb), pack `Inkling-2-Mini-oQ4e-mtp` with Lightning MTP on, reports 1,034 tok/s prefill and 37.7 tok/s generation at 1k tokens, and 1,112 and 50.1 tok/s at 4k tokens.
- Rate. [An MLX nvfp4 and mxfp8 repack on an M5 Pro with 48 GB](https://huggingface.co/baytizzel/Inkling-2-Mini-MLX-M5Pro-48gb) reports 26.6 tok/s decode, and 750 tok/s prefill for a 3.3K-token prompt.
- Rate. [A Metal engine on an M5 Pro with 48 GB](https://github.com/Eriskii/erisinky) reading a Q4_K_M GGUF reports 2200 tok/s prefill on a 2394-token prompt and 63.0 tok/s short-context decode.
- Rate. The same harness on the FOEM fp8 attention pack recorded 1197.33 / 51.51, 1401.14 / 50.43, 1440.07 / 49.90, and 1349.14 / 47.87 tok/s on 23 November 2026. That row is in [synthetic-prompt-fp8.md](synthetic-prompt-fp8.md).
- Rate. None of them is this nearest-fp8 table. The earlier GPTQ dense fp4 table is [synthetic-prompt-half.md](synthetic-prompt-half.md).

## Rule

MTP is off. The banner also prints depth 3; that depth is unused while MTP is off. Thinking is off, so the request uses the instruct card and `reasoning.effort` `none`. Temperature is 0. The banner prints the loaded plane.

- Each measured prompt is one user message and encodes to that length under the chat template, including the template tokens and the instruct opening `encode_chat` appends.
- A 64-token warmup runs first.
- The measured prompts start with different labels, so each prefill is a cache miss.
- Decode is filled to 128 tokens at every length.
- The end-to-end rate adds the prefill tokens to the decode tokens and adds those two clocks.
- A banner that prints another plane or `thinking on` is a different row.
- Attention and the output projection are nearest E4M3. The head and the embedding are float16. Experts in layers 0–19 are Metal fp4 from GPTQ-FOEM. Experts in layers 20–23 are MXFP8.

## Command

```text
uv run python -m harness.synthetic --model inkling_2_mini --thinking off
```

The banner prints the loaded plane, MTP, thinking, and the decode cap.

## Table

Keys and values are `half`. The banner printed `kv half`, `mtp off`, and `thinking off`.

| Prompt tokens | Prefill | Decode | End to end |
|---:|---|---|---|
| 512 | 1188.57 tok/s (512 tok in 0.43 s) | 50.72 tok/s (128 tok in 2.52 s) | 216.95 tok/s (640 tok in 2.95 s) |
| 1024 | 1390.37 tok/s (1024 tok in 0.74 s) | 50.06 tok/s (128 tok in 2.56 s) | 349.09 tok/s (1152 tok in 3.30 s) |
| 4096 | 1425.40 tok/s (4096 tok in 2.87 s) | 48.87 tok/s (128 tok in 2.62 s) | 769.40 tok/s (4224 tok in 5.49 s) |
| 10000 | 1332.01 tok/s (10000 tok in 7.51 s) | 47.38 tok/s (128 tok in 2.70 s) | 991.97 tok/s (10128 tok in 10.21 s) |

## Search

23 November 2026. Queries: "Inkling-2-Mini 1188.57 OR 1390.37 OR 1425.40 OR 1332.01 tok/s prefill M5 Pro 2026". The predecessors in this note are the overlaps that query returned. The search did not find 1188.57, 1390.37, 1425.40, or 1332.01.

## Corrections

Later finds are appended here. The sections above stay as committed.
