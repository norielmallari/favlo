# MTP window

Measurement note for the `qwen38-27b` pipeline. The serving contract is [../README.md](../README.md). The per-dispatch rows are in [../ceilings.md](../ceilings.md). This file records one full-accept verify window. It is a measurement of this shader library.

## Remainder

On an M5 Pro with 48 GB, a full-accept MTP window sums to 146.7 ms, 20.45 tok/s. The bandwidth ceiling of that window is 135.3 ms, 22.2 tok/s. A search on 30 October 2026 did not find that pair.

## Predecessors

- Method. DeepSeek-V3 ([arXiv:2412.19437](https://arxiv.org/abs/2412.19437)) trains a multi-token head and says it can be reused for speculative decoding. Leviathan, Kalman, and Matias, and the TensorRT-LLM MTP note, verify drafted tokens in one target forward. [OpenJet](https://richland.au/blog/fusing-the-whole-speculative-decoding-cycle-into-one-cuda-kernel-qwen38-27b-at-140-toks-on-a-single-3090) reports that a shared-activation verify makes four tokens cost about the same memory time as one, and measures 140 tok/s on code on one RTX 3090. [oMLX pull 3958](https://github.com/jundot/omlx/pull/3958) lowers Lightning MTP verify cost on this checkpoint, including a 64k verify forward of 43.9 ms on an M3 Ultra.
- Model. [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). [MTPLX](https://mtplx.com/models/qwen3.8-27b/) reports up to 87.6 tok/s with a draft of three tokens on an M5 Max, and 65.2 tok/s on a coding task. Those rates include acceptance. They are not this dispatch sum.
- Measurement. [oMLX](https://omlx.ai/benchmarks/performance/sg4ax9hd) reports 35.5 tok/s decode at 4k on an M5 Pro with 20 GPU cores and 48 GB, MTP on. [A thinking run](https://omlx.ai/benchmarks/performance/ev8uxazm) on that machine reports 33.6 tok/s at 16k. [llama.cpp on one RTX 5090](https://huggingface.co/Qwen/Qwen3.8-27B/discussions/112) reports 133.6 tok/s at draft depth 3 for Q4_K_M. None of those runs publishes this window's kernel sum or its 22.2 tok/s bandwidth ceiling.
- Mechanism. The first two feeds share one read of each fp4 language weight. The third feed is one serial step. Twelve language feed-forward layers are fp8 and are read once per token. Three drafts read the MTP block, and the window reads `lm_head` six times.

## Rule

The window feeds three tokens and emits three output tokens in a full accept. Each listed time is one command buffer, then a wait, the median of five dispatches after one discarded sample. Overlapping command buffers are not added. Mixers, norms, and the attention window are outside the kernel sum. The streaming copy in [../ceilings.md](../ceilings.md) is 254.6 GB/s. The language projections are 14,544,732,160 bytes, the MTP projections are 437,944,320 bytes, and `lm_head` is 675,430,400 bytes. The window reads the language image twice, the MTP image three times, and the head six times.

## Command

`favlo --qwen38-27b kernels` checks the dispatches. That command does not print this table. No committed command regenerates the medians. Recompute [../ceilings.md](../ceilings.md) when a listed kernel changes.

## Table

| Forward | Recorded result |
|---|---|
| Full-accept MTP window | 146.7 ms, 20.45 tok/s |
| Bandwidth ceiling | 135.3 ms, 22.2 tok/s |
| Same bytes plus three attention passes at 4096 tokens | 136.8 ms, 21.9 tok/s |
| `ffn_gate_up_span2`, 17408×5120, two tokens | 352.2 µs |
| Replaced one-row body of that kernel | 2238.3 µs |

## Search

30 October 2026. Queries: "Qwen3.8-27B MTP speculative decoding tok/s depth 3 Apple Silicon M5", "Qwen3.8-27B 20.45 tok/s 146.7 ms MTP span GEMV M5 Pro 48GB", "Qwen3.8-27B 146.7 OR 20.45 tok/s OR ffn_gate_up_span2 flash-decoding verify kernel", and "DeepSeek-V3 multi-token prediction speculative decoding verify one forward draft acceptance". The predecessors in this note are the overlaps those queries returned. The 87.6 tok/s and 65.2 tok/s figures are an M5 Max. The 35.5 tok/s and 33.6 tok/s figures are another engine on an M5 Pro with 48 GB. The 133.6 tok/s figure is one RTX 5090. The 140 tok/s figure is one RTX 3090. The search did not find the 146.7 ms sum or the 135.3 ms bandwidth ceiling.

## Corrections

Later finds are appended here. The sections above stay as committed.

30 October 2026. The verify window now reads down, out, the Gated DeltaNet input, and attention q/k/v once for three tokens. Gate and up of the third token are a second read. The sum of that window is 136.5 ms, 21.98 tok/s, and its bandwidth ceiling is 123.5 ms, 24.3 tok/s. The table is in ../ceilings.md. A three-activation gate/up kernel measured 1512.0 µs and is not the launch.
