# Recorded forward

Measurement note for the `qwen38-27b` pipeline. The serving contract is [../README.md](../README.md). The runs are in that note. The isolated projection sum is [kernel-ceilings.md](kernel-ceilings.md).

## Remainder

A warm run with the full cache allocated recorded 353.54 tok/s on a 2048-token prefill and 15.29 tok/s on 64 argmax tokens. A search on 21 October 2026 did not find that run.

## Predecessors

- Method. The forward is this pipeline's shader library. The kernel sum leaves overlap and the mixers' fusion out.
- Model. [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). [MTPLX](https://mtplx.com/models/qwen3.8-27b/) reports up to 87.6 tok/s decode with a three-token draft.
- Measurement. That rate is another engine. It does not publish these runs.
- Mechanism. The 2048-token prefill repeated one token id. The decode prompt was the carrot question. Later runs read repository files past 65536 tokens.

## Rule

Prefill is the summed prompt time. Decode is the summed sampled-token time. Temperature 0 is an argmax. The MTP-resident run keeps the MTP plane and still draws the serial argmax.

## Command

The runs are the ones printed in [../README.md](../README.md). No committed command reprints the rows below.

## Table

| Run | Prefill | Decode |
|---|---|---|
| Warm, full cache | 353.54 tok/s (2048 tok in 5.79 s) | 15.29 tok/s (64 tok in 4.19 s) |
| Files to 83461 | 230.07 tok/s (83435 tok in 362.65 s) | 13.18 tok/s (39 tok in 2.96 s) |
| MTP resident, to 86590 | 200.38 tok/s (86391 tok in 431.13 s) | 9.03 tok/s (327 tok in 36.21 s) |
| Argmax, to 86641 | 197.39 tok/s (86442 tok in 437.92 s) | 9.65 tok/s (327 tok in 33.87 s) |

## Search

21 October 2026. Query: "Qwen3.8-27B inference tok/s Apple Silicon Qwen3.8-Flash-Next". The predecessor in this note is the overlap that query returned for this checkpoint. The search did not find the 353.54 and 15.29 tok/s pair.

## Corrections

Later finds are appended here. The sections above stay as committed.
