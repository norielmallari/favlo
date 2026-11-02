# Kernel ceilings

Measurement note for the `qwen35-4b` pipeline. The serving contract is [../README.md](../README.md). The per-dispatch rows are in [../ceilings.md](../ceilings.md). This file records the isolated sum of those dispatches. It is a measurement of this shader library.

## Remainder

On an M5 Pro with 48 GB, one command buffer per kernel, then a wait, the decode gemv and listed elementwise sum is 53018.9 µs, 18.86 tok/s, at context 2048. The bandwidth ceiling of that token is 67.22 tok/s. The last full-chunk prefill composition under half×fp8 dense tiles is 241125.0 µs, 1061.69 tok/s, for a 256-token chunk. A search on 2 November 2026 did not find that decode sum.

## Predecessors

- Method. The Open Compute Project MX specification gives the fp8 E4M3 block and the E8M0 scale. Apple, [WWDC26 session 330](https://developer.apple.com/videos/play/wwdc2026/330/), specifies `matmul2d` on a quantized tensor. Decode projections are matrix-vector products (`gemv_fp8`, mid-width `gemv_fp8_x4` in the forward).
- Model. [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B). [oMLX](https://omlx.ai/benchmarks/performance/x0t6bie9) and [llmcheck](https://llmcheck.net/models/qwen-35-4b-on-m5-pro/) publish end-to-end tok/s for this model on Apple Silicon. They are not this per-kernel sum.
- Measurement. Favlo's [qwen36-35b-a3b kernel-ceilings](../../qwen36-35b-a3b/research/kernel-ceilings.md) and [qwen38-27b kernel-ceilings](../../qwen38-27b/research/kernel-ceilings.md) are the same method on other pipelines. None of those pages publishes this 4B dispatch sum.
- Mechanism. Overlapping command buffers are not added. The bandwidth column divides language weight bytes plus float16 K/V at 2048 keys on eight layers and Gated DeltaNet state traffic by the 2 GiB streaming read, 7093.5 µs, 302.74 GB/s, recorded in the Qwen3.6 ceilings note. The prefill chunk figure is the earlier half×fp8-only composition; int8 dense tiles are timed in the ceilings note but not yet folded into a replacement chunk sum.

## Rule

Each listed kernel runs in its own command buffer. Two launches are discarded. The figure is the median of the next four `GPUEndTime` minus `GPUStartTime` intervals. Decode is one token at context 2048. Prefill is one 256-token chunk with a 2048-key attention window on the eight full-attention layers. Measured on 2 November 2026 on the default `f16` cache.

## Command

`uv run favlo --qwen35-4b kernels` checks that the shader library compiles. That command does not print this table. No committed command regenerates the medians. Recompute [../ceilings.md](../ceilings.md) when a listed kernel changes.

## Table

| Forward | Kernel sum | Bandwidth ceiling |
|---|---|---|
| Decode, one token, context 2048 | 53018.9 µs, 18.86 tok/s | 14.88 ms, 67.22 tok/s |
| Prefill, 256-token chunk (half×fp8 composition) | 241125.0 µs, 1061.69 tok/s | 22.39 ms, 11432.78 tok/s |

## Search

2 November 2026. Queries: "Qwen3.5-4B prefill tok/s int8 Apple Silicon M5 Pro 2026", "Qwen3.5-4B 18.86 OR 53018 OR 1061.69 tok/s Metal GEMV ceiling", and "Qwen3.5-4B-fp8 decode kernel sum M5 Pro 48GB". The predecessors in this note are the overlaps those queries returned. The search did not find 53018.9 µs or 18.86 tok/s.

## Corrections

Later finds are appended here. The sections above stay as committed.
