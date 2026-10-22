# Kernel ceilings

Measurement note for the `qwen36-35b-a3b` pipeline. The serving contract is [../README.md](../README.md). The per-dispatch rows are in [../ceilings.md](../ceilings.md). This file records the isolated sum of those dispatches. It is a measurement of this shader library. It does not record a hidden-state error on the [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) checkout.

## Remainder

One command buffer per kernel, then a wait, summed without overlap, is 97.25 tok/s for one decode token at context 2048 and 815.71 tok/s for a 65536-token prefill. A search on 21 October 2026 did not find that sum.

## Predecessors

- Method. FlashAttention ([arXiv:2205.14135](https://arxiv.org/abs/2205.14135)) tiles the score and the value mix. Yang, Kautz, and Hatamizadeh ([arXiv:2412.06464](https://arxiv.org/abs/2412.06464)) give the Gated DeltaNet chunk. The OCP MX specification gives the fp4 and fp8 blocks. Apple, WWDC26 session 330, specifies `matmul2d` on a quantized tensor with an E8M0 plane.
- Model. [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). Perplexity's [on-device note](https://www.perplexity.ai/hub/blog/optimizing-on-device-inference-for-apple-silicon) runs this model on an Apple GPU and publishes end-to-end prefill and decode rates, including a 4K-token pair of 5749.9 and 186.6 tok/s. Those rates are a different engine and a different quantizer.
- Measurement. [omlx issue 2238](https://github.com/jundot/omlx/issues/2238) reports decode tok/s for an MLX 6-bit gather path. [mlx-swift-lm 83ff028](https://github.com/ekryski/mlx-swift-lm/commit/83ff028def2e734bf2da9b65f51a479c5c736c3d) reports GDN prefill tok/s at context 1024 for this model. Neither publishes this per-kernel sum.
- Mechanism. Overlapping command buffers are not added. The 90% test uses the kernel column. The bandwidth column divides bytes by the 2 GiB streaming read, 7093.5 µs, 302.74 GB/s.

## Rule

Each listed kernel runs in its own command buffer. The sample is the median of four runs after one discarded run. Decode weights are fp4 and the key/value cache is fp8 E4M3. Prefill experts are one 64-row tile per expert. The embed, the top-8 pick, RoPE, and the key/value store are outside the sum.

## Command

`favlo --qwen36-35b-a3b kernels` checks that the dispatches compile and match the host formulas in `qwen36_35b_a3b/checks.py`. That command does not print this table. No committed command regenerates the medians. Recompute [../ceilings.md](../ceilings.md) when a listed kernel changes.

## Table

| Forward | Kernel sum | Bandwidth ceiling |
|---|---|---|
| Decode, context 2048 | 10.28 ms, 97.25 tok/s | 5.77 ms, 173.36 tok/s |
| Decode, context 65536 | 17.17 ms, 58.23 tok/s | 8.32 ms, 120.21 tok/s |
| Prefill, one 2048-token chunk | 1346.2 ms, 1521.37 tok/s | 24,762,900,480 bytes |
| Prefill, 65536 tokens | 80342.7 ms, 815.71 tok/s | 13.65 s, 4801.58 tok/s |

The first-chunk bandwidth cell is the byte count. The ceilings note does not divide that count into a tok/s.

## Search

21 October 2026. Queries: "Qwen3.6-35B-A3B MLX Apple Silicon inference Gated DeltaNet", and "Qwen3.6-35B-A3B Metal kernel microsecond dispatch prefill decode ceiling matmul2d". The predecessors in this note are the overlaps those queries returned. The search did not find the 97.25 and 815.71 tok/s sums.

## Corrections

Later finds are appended here. The sections above stay as committed.

23 October 2026. The `Qwen3.6-35B-A3B-fp8-e20` record stores the language projections, the shared expert, and `lm_head` as fp8, and 2,048 of 10,240 routed experts as fp8. At the same 302.74 GB/s, a decode token that reads the pool mix of those experts is 9.27 ms, 107.90 tok/s at context 2048, and 11.82 ms, 84.61 tok/s at context 65536. A 65536-token prefill is 14.06 s, 4659.90 tok/s. The kernel sums above were measured on fp4 weight images.

23 October 2026. The fp8 decode dot indexes a 256-entry E4M3 table and loads both scale blocks of a 2048-wide row before that lookup. The isolated sums in the table stay the fp4-weight measurement.
