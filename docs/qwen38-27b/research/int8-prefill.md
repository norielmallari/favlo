# Int8 prefill

Measurement note for the `qwen38-27b` pipeline. The serving contract is [../README.md](../README.md). The run is in that note. The isolated kernel sum is [kernel-ceilings.md](kernel-ceilings.md).

## Remainder

On `Qwen3.8-27B-fp8-fp4`, on an M5 Pro with 20 shader cores and 48 GB, a warm run recorded 512.61 tok/s on a 2048-token prefill and 15.66 tok/s on 64 argmax tokens. A search on 25 October 2026 did not find that pair.

## Predecessors

- Method. [oMLX pull 3548](https://github.com/jundot/omlx/pull/3548) routes prefill matmuls as int8 × int8, decodes packed affine Q4 or Q5 weights in the fragment, and applies the group-64 correction there. Decode stays on the packed weights. On an M5 Pro with 15 or 16 shader cores and 48 GB it reports 518.5 tok/s at 4096 prompt tokens, up from 391.8. [Cider](https://github.com/jundot/omlx/discussions/3136) requantizes an 8-bit checkpoint to per-channel symmetric int8 and reports 1044 tok/s at 1975 tokens on an M5 Max with 40 shader cores and 128 GB.
- Model. [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). [oMLX](https://omlx.ai/benchmarks/performance/qq23wztb) reports 406.1 prefill tok/s at 1024 tokens and 451.1 at 4096 for an 8-bit checkpoint on an M5 Pro with 20 shader cores and 48 GB.
- Measurement. Those figures are another engine, another checkpoint, or another chip. They do not publish this mixed-record run.
- Mechanism. This prefill restores each listed packed matrix into one int8 stage and replaces each group of 32 with one channel scale. [A block-scale alignment](https://axiomofchoice-hjt.github.io/pages/7f2a8c/) keeps the group scales by shifting each block onto the channel peak before the int8 product. This run does not. The resulting weight error is not measured. Decode does not read the stage.

## Rule

Prefill is the summed prompt time. Decode is the summed sampled-token time. Temperature 0 is an argmax. The 2048-token prefill repeated one token id and was a cache miss. The counted prefill is the second such run. The first printed 505.26 tok/s (2048 tok in 4.05 s). The decode prompt was the carrot question, 51 tokens. The live length was 4096. The MTP head stayed resident, and all 64 language feed-forward layers stayed wired.

## Command

The lines are the ones printed in [../README.md](../README.md). No committed command reprints the row.

## Table

| Run | Prefill | Decode |
|---|---|---|
| Warm, context 4096 | 512.61 tok/s (2048 tok in 4.00 s) | 15.66 tok/s (64 tok in 4.09 s) |

## Search

25 October 2026. Queries: "Qwen3.8-27B prefill tok/s int8 Apple M5 Pro 512", "NVFP4 dequantize to int8 matmul prefill W8A8 runtime requantize E2M1", "MXFP4 group scale collapse per-channel int8 GEMM decode keeps FP4", "int8_prefill Qwen3.8-27B 512.61 tok/s fp8-fp4 M5 Pro 2048", "Cider W8A8 Qwen3.8-27B M5 Pro 48GB 2048 prefill 15.66 decode", and "oMLX int8 activation prefill Qwen3.8-27B 518.5 tok/s pp4096 group_size 64". The predecessors in this note are the overlaps those queries returned. The 518.5 tok/s figure is an affine Q4 checkpoint on an M5 Pro with 15 or 16 shader cores. The 1044 tok/s figure is an M5 Max with 128 GB. The search did not find the 512.61 tok/s prefill or the 15.66 tok/s decode on `Qwen3.8-27B-fp8-fp4`.

## Corrections

Later finds are appended here. The sections above stay as committed.
