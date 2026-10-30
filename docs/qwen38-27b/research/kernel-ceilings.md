# Kernel ceilings

Measurement note for the `qwen38-27b` pipeline. The serving contract is [../README.md](../README.md). The per-dispatch rows are in [../ceilings.md](../ceilings.md). This file records the isolated projection sum and the warm fused step. It is a measurement of this shader library.

## Remainder

The isolated `gemv_fp4_x4` sum is 325.9 ms, 3.07 tok/s. A warm language step, projections and mixers in one command buffer, measured 63.4 to 64.6 ms. A search on 21 October 2026 did not find that pair.

## Predecessors

- Method. Yang, Kautz, and Hatamizadeh ([arXiv:2412.06464](https://arxiv.org/abs/2412.06464)) give the Gated DeltaNet step. Apple, WWDC26 session 330, specifies `matmul2d`.
- Model. [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). [MTPLX](https://mtplx.com/models/qwen3.8-27b/) reports up to 87.6 tok/s decode for this model with a draft of three tokens.
- Measurement. That rate is another engine and includes draft acceptance. It does not publish this GEMV sum or this fused-step time.
- Mechanism. Overlapping command buffers are not added. The isolated sum is not the designed decode step. The fused buffer is.

## Rule

Each GEMV time is one command buffer, then a wait, the median of five dispatches after one discarded sample. Counts are the language stack. The forward issues the fused projection kernels instead of `gemv_fp4_x4` alone. Convolution, norms, and the embedding row are outside the isolated sum.

## Command

`favlo --qwen38-27b kernels` checks the dispatches. That command does not print this table. No committed command regenerates the medians. Recompute [../ceilings.md](../ceilings.md) when a listed kernel changes.

## Table

| Forward | Recorded result |
|---|---|
| Isolated projection sum | 325.9 ms, 3.07 tok/s |
| Warm fused step | 63.4 to 64.6 ms |
| Attention, 64 head steps, context 65536 | 6.31 ms, 158 tok/s |
| Attention bandwidth, context 65536 | 8.04 ms, 124 tok/s |

## Search

21 October 2026. Query: "Qwen3.8-27B inference tok/s Apple Silicon Qwen3.8-Flash-Next". The predecessor in this note is the overlap that query returned for this checkpoint. The search did not find the 325.9 ms sum or the 63.4 to 64.6 ms step.

## Corrections

Later finds are appended here. The sections above stay as committed.

23 October 2026. A 2048-token prefill of `Qwen3.8-27B-fp8-e20` is tabulated in [../ceilings.md](../ceilings.md). The projection bandwidth ceiling is 1020 tok/s, and 994 tok/s once the 2048-key score and value reads are added. The timed kernel set sums to 5594.6 ms, 366 tok/s. The decode rows above are unchanged.

23 October 2026. The down projection now uses `prefill_fp4_m32` and `prefill_fp8_m32`. That chunk's projection bandwidth ceiling is 834 tok/s, and 816 tok/s with the 2048-key score and value reads. The timed set sums to 5321.3 ms, 385 tok/s.

23 October 2026. A full chunk no longer clears the norm buffer or the half tile, and it submits every 4 layers. The timed set sums to 5245.3 ms, 390 tok/s.

30 October 2026. A full-accept MTP window on an M5 Pro with 48 GB sums to 146.7 ms, 20.45 tok/s. The bandwidth ceiling of that window is 135.3 ms, 22.2 tok/s, at the 254.6 GB/s copy in [../ceilings.md](../ceilings.md). The table is in that note. The isolated fp4 sum above is not this window.
