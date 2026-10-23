# Mixed-record agent forward

Measurement note for the `qwen38-27b` pipeline. The serving contract is [../README.md](../README.md). This file records one run. The uniform fp4 sums are [recorded-forward.md](recorded-forward.md).

## Remainder

On `Qwen3.8-27B-fp8-fp4`, on an M5 Pro with 20 shader cores and 48 GB, the file agent recorded 179.56 tok/s prefill over 91802 tokens and 11.80 tok/s decode over 142 tokens, after the live length grew from 2048 to 92160. A search on 22 October 2026 did not find that run on that machine.

## Predecessors

- Method. The forward is this pipeline's shader library. The live length starts at 2048 and grows by 2048. [MTPLX](https://mtplx.com/benchmarks/) and [oMLX issue 2795](https://github.com/jundot/omlx/issues/2795) already publish long-context ladders for this checkpoint, on an M5 Max with 128 GB and an M3 Max with 128 GB.
- Model. [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). [lribeiro/Qwen3.8-27B-nvfp4-v6](https://huggingface.co/lribeiro/Qwen3.8-27B-nvfp4-v6) publishes a different mixed split, 16 MLP layers at NVFP4 and the rest at fp8, with a 2k prefill of 7756 tok/s on an NVIDIA GPU. This record keeps 12 language feed-forward layers at fp8 and the rest at fp4.
- Measurement. MTPLX reports 535 prefill tok/s and 30.4 decode tok/s at 88k on an M5 Max with 128 GB. Its 48 GB row is a simulated seat at 42k, 645 prefill tok/s and 33 decode tok/s, not an M5 Pro run. oMLX issue 2795 reports 231.7 prefill tok/s and 27.0 decode tok/s at 16k on an M3 Max with 128 GB. On an M5 Pro with 20 GPU cores and 48 GB, [oMLX v0.6.3rc2](https://omlx.ai/benchmarks/performance/e78ir7ct) reports 251.5 prefill tok/s and 17.7 decode tok/s at 64k, [v0.6.3rc3](https://omlx.ai/benchmarks/performance/2djm3rah) reports 1131 and 33.4 at 128k, and [v0.7.0.dev4](https://omlx.ai/benchmarks/performance/mu14ro3q) reports 383.4 and 10.8 at 128k. An M5 Pro with 16 GPU cores is a different core count. None of those runs is this harness table.
- Mechanism. The load line is context 2048, weights 16126543872, cache 73531392, with no `mtp off` and no feed-forward count, so the MTP head and all 64 language feed-forward layers stayed resident. The file agent then logged context 47104 and, after the server restart, context 92160, each before the counted prefill. The coding agent stayed inside 2048.

## Rule

Prefill is the summed prompt time. Decode is the summed sampled-token time. The file agent sets `enable_thinking` false. The coding agent sets `enable_thinking` true. Omitted sampling fields come from that card. The growth lines are not inside the prefill seconds: each counted prefill starts after the context line.

## Command

```text
uv run python -m harness --model qwen38-27b
uv run python -m harness.coding --model qwen38-27b
```

Those commands start the series. No committed command reprints the rows below.

## Table

| Run | Prefill | Decode |
|---|---|---|
| File agent | 179.56 tok/s (91802 tok in 511.27 s) | 11.80 tok/s (142 tok in 12.03 s) |
| Coding agent | 299.54 tok/s (653 tok in 2.18 s) | 14.73 tok/s (309 tok in 20.98 s) |

| File-agent request | Prompt | Prefill | Decode |
|---|---|---|---|
| 1, cache miss | 361 | 301.94 tok/s (361 tok in 1.20 s) | 15.08 tok/s (31 tok in 2.06 s) |
| 2, cache hit 392, length 47104 | 46425 | 231.90 tok/s (46033 tok in 198.50 s) | 10.81 tok/s (31 tok in 2.87 s) |
| 3, prefix load 46456, length 47104 | 46493 | 8.46 tok/s (37 tok in 4.37 s) | 12.79 tok/s (40 tok in 3.13 s) |
| 4, cache hit 46533, length 92160 | 91904 | 147.69 tok/s (45371 tok in 307.20 s) | 10.07 tok/s (40 tok in 3.97 s) |

| Coding-agent request | Prompt | Prefill | Decode |
|---|---|---|---|
| 1, cache miss | 437 | 316.27 tok/s (437 tok in 1.38 s) | 14.73 tok/s (196 tok in 13.30 s) |
| 2, prefix load 437 | 653 | 268.93 tok/s (216 tok in 0.80 s) | 14.71 tok/s (113 tok in 7.68 s) |

## Search

22 October 2026. Queries: "Qwen3.8-27B inference tokens per second prefill decode Apple Silicon", "Qwen3.8-27B MTPLX tok/s Gated DeltaNet MLX", "Qwen3.8-27B fp8 fp4 mixed feed-forward prefill tok/s file agent 92160", "Qwen3.8-27B M5 Pro 48GB tok/s prefill decode", and "MTPLX Qwen 3.8 27B 535 tok/s 88k M5 Max". The predecessors in this note are the overlaps those queries returned. The 535 and 30.4 tok/s pair is an M5 Max with 128 GB. The 231.7 and 27.0 tok/s pair is an M3 Max with 128 GB. The search did not find the 179.56 tok/s prefill over 91802 tokens or the 11.80 tok/s decode over 142 tokens on an M5 Pro with 48 GB.

## Corrections

Later finds are appended here. The sections above stay as committed.

23 October 2026. The same two commands, after the resident DeltaNet row, the 32×128 down tile, and the four-layer submit. The file agent recorded 189.11 tok/s prefill over 101579 tokens and 10.99 tok/s decode over 3093 tokens. The last prompt was 102388 tokens. The coding agent recorded 329.34 tok/s prefill over 853 tokens and 14.78 tok/s decode over 479 tokens.

| Run | Prefill | Decode |
|---|---|---|
| File agent | 189.11 tok/s (101579 tok in 537.13 s) | 10.99 tok/s (3093 tok in 281.41 s) |
| Coding agent | 329.34 tok/s (853 tok in 2.59 s) | 14.78 tok/s (479 tok in 32.40 s) |

| File-agent request | Prompt | Prefill | Decode |
|---|---|---|---|
| 1, cache miss | 361 | 298.88 tok/s (361 tok in 1.21 s) | 14.69 tok/s (31 tok in 2.11 s) |
| 2, cache hit 392 | 46901 | 259.10 tok/s (46509 tok in 179.50 s) | 12.83 tok/s (738 tok in 57.50 s) |
| 3, prefix load 47639, cache hit 47639 | 47676 | 41.90 tok/s (37 tok in 0.88 s) | 10.21 tok/s (40 tok in 3.92 s) |
| 4, cache hit 47716 | 102388 | 153.77 tok/s (54672 tok in 355.54 s) | 10.48 tok/s (2284 tok in 217.88 s) |

| Coding-agent request | Prompt | Prefill | Decode |
|---|---|---|---|
| 1, cache miss | 437 | 319.39 tok/s (437 tok in 1.37 s) | 14.80 tok/s (396 tok in 26.75 s) |
| 2, prefix load 437, cache hit 437 | 853 | 340.67 tok/s (416 tok in 1.22 s) | 14.70 tok/s (83 tok in 5.65 s) |
