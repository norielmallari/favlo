# Synthetic prompt, half cache

Measurement note for the `inkling_2_mini` pipeline. The serving contract is [../ARCHITECTURE.md](../ARCHITECTURE.md). This file records one synthetic-prompt run on `half`. It is a recorded run, not a format bound.

## Remainder

On an M5 Pro with 48 GB, `Inkling-2-Mini` with the GPTQ dense fp4 pack, `half` keys and values, and MTP off passed a cold instruct prompt.

- A 512-token prompt prefilled at 1221.30 tok/s and decoded 128 tokens at 68.89 tok/s.
- A 1024-token prompt prefilled at 1388.06 tok/s and decoded 128 tokens at 67.60 tok/s.
- A 4096-token prompt prefilled at 1430.63 tok/s and decoded 128 tokens at 65.41 tok/s.
- A 10000-token prompt prefilled at 1328.57 tok/s and decoded 128 tokens at 62.56 tok/s.
- The end-to-end rates, prefill plus decode, were 280.70, 438.02, 876.35, and 1057.20 tok/s.

A search on 22 November 2026 did not find that table.

## Predecessors

- Method. [llama-bench](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md) defaults to a 512-token prompt and 128 generated tokens. [`mlx_lm.benchmark`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/benchmark.py) uses the same prompt default. [Artificial Analysis](https://artificialanalysis.ai/methodology/performance-benchmarking) times a prompt of about 10000 tokens and reports time to first token, not prefill tok/s.
- Model. [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini), Apache-2.0. The [model card](https://thinkingmachines.ai/model-card/inkling-2-mini/) reports 26.4B parameters, 3.4B active per pass, and SWE-bench Verified (bash only) 79.2%. That score is not this harness rate.
- Rate. [llama.cpp on a 64 GB M5 Pro Mac mini](https://huggingface.co/datasets/pratikgx/inkling-2-mini-m5-pro-mac-mini-bench) with the official IQ4_NL GGUF reports 490.8 tok/s prompt and 29.95 tok/s generation at 512 tokens, and 357.5 and 28.42 tok/s at 4096 tokens. It generates 256 tokens per cell at temperature 0.
- Rate. [modelfit.io on an M5 Pro with 48 GB](https://modelfit.io/blog/inkling-2-mini-mac-benchmark/), an MLX 4/8-bit build, reports 123 to 144 tok/s prefill and 27 tok/s decode on 38 to 62-token prompts.
- Rate. [An MLX nvfp4 and mxfp8 repack on an M5 Pro with 48 GB](https://huggingface.co/baytizzel/Inkling-2-Mini-MLX-M5Pro-48gb) reports 26.6 tok/s decode, and 750 tok/s prefill for a 3.3K-token prompt.
- Rate. [The NetProjects MLX 4/8-bit build on an M5 Max with 128 GB](https://huggingface.co/netprojects/Inkling-2-Mini-MLX-4-8bit) reports 1752 tok/s prompt processing at 1K tokens, 1658 tok/s at 8K, and 36.6 tok/s generation.
- Rate. [oMLX on an M5 Max with 128 GB](https://omlx.ai/benchmarks/performance/1gf0zsb5), pack `Inkling-2-Mini-oQ4e-mtp`, reports 1,579 tok/s prefill and 57.8 tok/s generation at 1k tokens, and 1,841 and 63.9 tok/s at 4k tokens.
- Rate. [oMLX on an M5 with 32 GB](https://omlx.ai/benchmarks/performance/30f36yjk) reports 372.2 tok/s prefill and 18.4 tok/s generation at 1k tokens.
- Rate. None of these sources reports an M5 Pro row at a 10000-token prompt.
- Rate. None of them is this `half` table.

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

The served folder is `Inkling-2-Mini`. The fused attention input plane, the attention output projection, and the head are Metal fp4 from `dense-fp4.safetensors`, rounded with GPTQ. The router, the sink, the dynamic convolution taps, the message gates, and the embedding are half. Experts in layers 0–19 are NVFP4 requantized to Metal fp4 at open. Experts in layers 20–23 are MXFP8. Keys and values are `half`.

| Prompt tokens | Prefill | Decode | End to end |
|---:|---|---|---|
| 512 | 1221.30 tok/s (512 tok in 0.42 s) | 68.89 tok/s (128 tok in 1.86 s) | 280.70 tok/s (640 tok in 2.28 s) |
| 1024 | 1388.06 tok/s (1024 tok in 0.74 s) | 67.60 tok/s (128 tok in 1.89 s) | 438.02 tok/s (1152 tok in 2.63 s) |
| 4096 | 1430.63 tok/s (4096 tok in 2.86 s) | 65.41 tok/s (128 tok in 1.96 s) | 876.35 tok/s (4224 tok in 4.82 s) |
| 10000 | 1328.57 tok/s (10000 tok in 7.53 s) | 62.56 tok/s (128 tok in 2.05 s) | 1057.20 tok/s (10128 tok in 9.58 s) |

## Search

22 November 2026. Queries: "Inkling-2-Mini tok/s prefill decode Apple M5 OR M4 llama.cpp OR MLX benchmark pp512 tg128", "Inkling-2-Mini MLX nvfp4 mxfp8 M5 Pro 48GB prefill decode tok/s", and "thinkingmachines Inkling-2-Mini model card benchmark SWE-bench active parameters". The predecessors in this note are the overlaps those queries returned. The search did not find 1221.30, 1388.06, 1430.63, or 1328.57.

## Corrections

Later finds are appended here. The sections above stay as committed.

22 November 2026. A later protocol-complete run on the same pack, plane, and MTP setting replaced the Active models row. Thinking stayed off. Decode stayed 128. Each measured prefill was a cache miss.

- A 512-token prompt prefilled at 1198.92 tok/s and decoded 128 tokens at 68.53 tok/s.
- A 1024-token prompt prefilled at 1394.74 tok/s and decoded 128 tokens at 67.31 tok/s.
- A 4096-token prompt prefilled at 1414.29 tok/s and decoded 128 tokens at 64.95 tok/s.
- A 10000-token prompt prefilled at 1323.82 tok/s and decoded 128 tokens at 62.44 tok/s.
- The end-to-end rates, prefill plus decode, were 278.26, 438.02, 867.35, and 1055.00 tok/s.

| Prompt tokens | Prefill | Decode | End to end |
|---:|---|---|---|
| 512 | 1198.92 tok/s (512 tok in 0.43 s) | 68.53 tok/s (128 tok in 1.87 s) | 278.26 tok/s (640 tok in 2.30 s) |
| 1024 | 1394.74 tok/s (1024 tok in 0.73 s) | 67.31 tok/s (128 tok in 1.90 s) | 438.02 tok/s (1152 tok in 2.63 s) |
| 4096 | 1414.29 tok/s (4096 tok in 2.90 s) | 64.95 tok/s (128 tok in 1.97 s) | 867.35 tok/s (4224 tok in 4.87 s) |
| 10000 | 1323.82 tok/s (10000 tok in 7.55 s) | 62.44 tok/s (128 tok in 2.05 s) | 1055.00 tok/s (10128 tok in 9.60 s) |

A search the same day did not find 1198.92, 1394.74, 1414.29, or 1323.82. [oMLX on an M5 Pro with 48 GB](https://omlx.ai/benchmarks/performance/twcgm64y), pack `Inkling-2-Mini-oQ4e-mtp`, reports 813.8 tok/s prefill and 40.0 tok/s generation at 1k tokens, and 826.3 and 39.4 tok/s at 4k tokens. That row is not this table. Queries: "Inkling-2-Mini 1198.92 OR 1394.74 OR 1414.29 OR 1323.82 tok/s prefill M5", and "Inkling-2-Mini oMLX OR llama.cpp OR MLX M5 Pro 48GB prefill tok/s 512 1024 4096".
