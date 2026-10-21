# Kernel ceilings

Measurement note for the `qwen38-flash-next` pipeline. The serving contract is [../architecture.md](../architecture.md). The per-dispatch rows are in [../ceilings.md](../ceilings.md). This file records the isolated decode sum and the prefill chunk without routed experts.

## Remainder

One decode token at context 2048, experts resident, sums to 75.23 ms, 13.29 tok/s. One 2048-token prefill chunk, routed experts excluded, is 3950.9 ms, 518.4 tok/s. A search on 21 October 2026 did not find that pair.

## Predecessors

- Method. Qwen Team ([arXiv:2608.30320](https://arxiv.org/abs/2608.30320)) gives Gated DeltaNet and sparse attention. Apple, WWDC26 session 330, specifies `matmul2d`.
- Model. The same report. [MTPLX](https://mtplx.com/docs/run-qwen-3-8-on-a-mac/) reports 125.8 tok/s on one cached request for this model, with a draft.
- Measurement. That rate is another engine. It does not publish this per-kernel sum.
- Mechanism. Overlapping command buffers are not added. The 90% test for decode uses the kernel column. A 2 GiB streaming read finished in 8043 µs, 267.0 GB/s.

## Rule

Each time is one command buffer, then a wait. Context 2048 uses dense attention and does not run the indexer. A longer context uses the selected window. The prefill chunk ceiling excludes routed experts and is not the 90% bar for prefill. A draft's extra MTP layer and head are outside the decode totals.

## Command

`favlo --qwen38-flash-next kernels` checks the dispatches. That command does not print this table. No committed command regenerates the medians. Recompute [../ceilings.md](../ceilings.md) when a listed kernel changes.

## Table

| Forward | Kernel sum | Bandwidth ceiling |
|---|---|---|
| Decode, context 2048, dense | 75.23 ms, 13.29 tok/s | 15.58 ms, 64.19 tok/s |
| Decode, context 32768, selected | 78.93 ms, 12.67 tok/s | 17.54 ms, 57.01 tok/s |
| Decode, context 262144, selected | 89.29 ms, 11.20 tok/s | 32.20 ms, 31.06 tok/s |
| Prefill chunk, experts excluded | 3950.9 ms, 518.4 tok/s | — |

## Search

21 October 2026. Query: "Qwen3.8-27B inference tok/s Apple Silicon Qwen3.8-Flash-Next". The predecessor in this note is the overlap that query returned for this model. The search did not find the 13.29 and 518.4 tok/s sums.

## Corrections

Later finds are appended here. The sections above stay as committed.
