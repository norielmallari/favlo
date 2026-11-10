# Kernel ceilings

Measurement note for the `spark-x26-1-7b` pipeline. The serving contract is [../README.md](../README.md). The per-dispatch rows are in [../ceilings.md](../ceilings.md). This file records the isolated sum of those dispatches. It is a measurement of this shader library.


This pipeline is not in the actively developed tree; it exists only in git history.

## Remainder

On an M5 Pro with 48 GB, one command buffer per kernel, then a wait, sums to 8090.6 µs (123.60 tok/s) for one decode token and 18523.0 µs (3023.27 tok/s) for a 56-token prefill. A search on 1 November 2026 did not find that sum.

## Predecessors

- Method. The Open Compute Project MX specification gives the fp8 E4M3 block and the E8M0 scale. Apple, [WWDC26 session 330](https://developer.apple.com/videos/play/wwdc2026/330/), specifies `matmul2d` on a quantized tensor. The decode projections in this sum are matrix-vector products, one 256-thread group per 32 rows.
- Model. [XHToken/Spark-X2.6-1.7B](https://huggingface.co/XHToken/Spark-X2.6-1.7B). The [DSpark note](https://dev.to/sparkllm/769-toks-on-a-single-rtx-4090-spark-x26-17b-dspark-6ga) reports 138 tok/s for an INT4 MLX copy on an M5 Pro, and 165 tok/s with that draft model. Those are end-to-end rates, not this dispatch sum.
- Measurement. [DaBench](https://dabench.ai/model/spark-x-2-6-1-7b) reports 106 tok/s for llama.cpp Q8_0 on an M5 Pro with 48 GB, 30 October 2026. [LiteRT-LM](https://huggingface.co/litert-community/Spark-X2.6-1.7B) reports 151.5 tok/s decode and 3,013 tok/s prefill for its int8 file on an M4 Max with 128 GB. None of those pages publishes this per-kernel sum.
- Mechanism. Overlapping command buffers are not added. The bandwidth column divides the projection bytes by the 2 GiB streaming read, 7093.5 µs, 302.74 GB/s, recorded in the Qwen3.6 ceilings note. Attention at one key is outside that byte count.

## Rule

Each listed kernel runs in its own command buffer. Two launches are discarded. The figure is the median of the next four `GPUEndTime` minus `GPUStartTime` intervals. The decode rows are one token. The prefill rows are one 56-token chunk: the projections, the cast, the norm, the residual, and the activation use the padded count of 64, and store and attention cover the 56 tokens. Measured on 1 November 2026.

## Command

`uv run favlo --spark-x26-1-7b kernels` checks that the shader library compiles. That command does not print this table. No committed command regenerates the medians. Recompute [../ceilings.md](../ceilings.md) when a listed kernel changes.

## Table

| Forward | Kernel sum | Bandwidth ceiling |
|---|---|---|
| Decode, one token | 8090.6 µs, 123.60 tok/s | 5834.8 µs, 171.39 tok/s |
| Prefill, 56-token chunk | 18523.0 µs, 3023.27 tok/s | |

The prefill bandwidth cell is empty. The ceilings note does not divide that chunk into a tok/s.

## Search

1 November 2026. Queries: "Spark-X2.6-1.7B tok/s decode prefill llama.cpp MLX Apple 2026", "Spark-X2.6-1.7B 123.60 OR 8090.6 OR 3023 tok/s Metal GEMV", "Spark-X2.6-1.7B M5 Pro MLX INT4 tok/s decode", and "Spark-X2.6-1.7B 89.65 OR 76.02 OR 73.40 OR 123.60 tok/s". The predecessors in this note are the overlaps those queries returned. The search did not find 8090.6 µs, 123.60 tok/s, or 18523.0 µs.

## Corrections

Later finds are appended here. The sections above stay as committed.
