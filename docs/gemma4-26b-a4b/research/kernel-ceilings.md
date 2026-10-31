# Kernel ceilings

Measurement note for the `gemma4-26b-a4b` pipeline. The serving contract is [../README.md](../README.md). The per-dispatch rows are in [../ceilings.md](../ceilings.md). This file records the isolated decode sum of those dispatches. It is a measurement of this shader library.

## Remainder

On an M5 Pro with 48 GB, one command buffer per kernel, then a wait, sums to 35.595 ms (28.09 tok/s) at one key, 56.658 ms (17.65 tok/s) at 1,024 keys, and 396.713 ms (2.52 tok/s) at 65,536 keys. A search on 31 October 2026 did not find that sum.

## Predecessors

- Method. Apple, WWDC26 session 330, specifies `matmul2d` on a quantized tensor. The OCP MX specification gives the fp4 E2M1 block and the E8M0 scale. [Gemma Team, arXiv:2607.02770](https://arxiv.org/abs/2607.02770) describes the 26B-A4B layer, including the sliding window and the global layers.
- Model. [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it). [oMLX n50yftgp](https://omlx.ai/benchmarks/performance/n50yftgp) reports 1,289 prefill tok/s and 72.5 decode tok/s at 1,024 tokens for a 4-bit MLX copy on an M5 Pro with 48 GB, oMLX 0.6.0rc1. [oMLX gv7qn8l9](https://omlx.ai/benchmarks/performance/gv7qn8l9) reports 2,082 and 76.9 at the same context on the same chip, oMLX 0.6.2. Those are end-to-end rates, not this dispatch sum.
- Measurement. [ComputeArena](https://computearena.ai/models/gemma-4-26b-a4b-it) lists BaseRT prompt and generation rates for this checkpoint on other GPUs, including 80.3 generation tok/s and 1,669 prompt tok/s at 512 tokens on an M4 Max. [Rapid-MLX](https://rapidmlx.com/models/gemma-4-26b-4bit) reports 78.5 decode tok/s on an M4 Pro with 48 GB. None of those pages publishes this per-kernel sum.
- Mechanism. Overlapping command buffers are not added. The bandwidth column divides bytes by the 2 GiB streaming read, 7093.5 µs, 302.74 GB/s, recorded in the Qwen3.6 ceilings note.

## Rule

Each listed kernel runs in its own command buffer. The sample is the median of four device intervals after one discarded run. `gemv_fp4_routed` and `attend_span` were measured on 31 October 2026. The other dispatches stay the earlier figures in the ceilings note. The sum replaces the previous routed-expert and attention rows with those two and leaves the rest.

## Command

`uv run favlo --gemma4-26b-a4b kernels` checks that the shader library compiles and that the listed products match. That command does not print this table. No committed command regenerates the medians. Recompute [../ceilings.md](../ceilings.md) when a listed kernel changes.

## Table

| Forward | Kernel sum | Bandwidth ceiling |
|---|---|---|
| Decode, 1 key | 35.595 ms, 28.09 tok/s | 10.61 ms, 94.21 tok/s |
| Decode, 1,024 keys | 56.658 ms, 17.65 tok/s | 14.49 ms, 69.02 tok/s |
| Decode, 65,536 keys | 396.713 ms, 2.52 tok/s | 84.32 ms, 11.86 tok/s |

Local attention at 65,536 keys stays inside the 1,024-key window. Global attention walks every key.

## Search

31 October 2026. Queries: "Gemma 4 26B-A4B oMLX prefill tok/s 1289 M5 Pro 48GB 2026", "gemma-4-26B-A4B 1161.9 OR 18.84 OR 46.64 tok/s Metal decode", and "gemma-4-26b-a4b-it-4bit PP tok/s TG tok/s M5 Pro 20c 1161 OR 18.84 OR 46.64 OR 56.658". The predecessors in this note are the overlaps those queries returned. The search did not find 35.595 ms, 56.658 ms, or 396.713 ms.

## Corrections

Later finds are appended here. The sections above stay as committed.
