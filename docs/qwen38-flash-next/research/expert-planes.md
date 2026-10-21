# Expert-width dispatches

Measurement note for the `qwen38-flash-next` pipeline. The serving contract is [../architecture.md](../architecture.md). The per-dispatch rows are in [../ceilings.md](../ceilings.md). This file records the expert-layer times for fp8, fp4, and int2. It is a measurement of this shader library. It does not record a hidden-state error on the checkout.

## Remainder

One resident expert layer measures 572.0 µs at fp8, 115.6 µs at fp4, and 135.1 µs at int2. At context 2048 those paths sum to 10.20, 13.22, and 13.05 tok/s. A search on 21 October 2026 did not find that table.

## Predecessors

- Method. The OCP MX specification gives the fp8 and fp4 blocks. The int2 grid is the one in [int2-miss.md](int2-miss.md). Apple, WWDC26 session 330, specifies `matmul2d`.
- Model. Qwen Team ([arXiv:2608.30320](https://arxiv.org/abs/2608.30320)). [RockTalk/Qwen3.8-Flash-Next-MLX-4bit](https://huggingface.co/RockTalk/Qwen3.8-Flash-Next-MLX-4bit) reports 26.4 to 26.6 tok/s plain greedy on an M3 Ultra and says an 8-bit expert path is much slower end to end.
- Measurement. [Whallm](https://shadowgraph.io/yanun0323/Whallm) reports 9.7 tok/s decode for an fp8 Flash Next checkpoint at 16,384 tokens on an M5 Pro. [HaberstrohSystems/Qwen3.8-Flash-Next-int2-mixed-AutoRound-24GB-SGLang](https://huggingface.co/HaberstrohSystems/Qwen3.8-Flash-Next-int2-mixed-AutoRound-24GB-SGLang) reports 54.5 tok/s decode for a mixed int2 checkpoint at 10,001 tokens on an NVIDIA GPU. Neither publishes these layer times.
- Mechanism. The served record is fp4. A mixed layer runs fp8, then fp4, then int2. The counts in a mixed row are a synthetic split of the ten ids.

## Rule

Each layer time is one command buffer, then a wait, the median after one discarded sample, with the expert images resident. The token total keeps the non-expert rows in the decode table and replaces the gate and down time with 49 times the layer time. A draft's extra MTP layer and head are outside the totals.

## Command

`favlo --qwen38-flash-next kernels` checks the dispatches. That command does not print this table. No committed command regenerates the medians. Recompute [../ceilings.md](../ceilings.md) when a listed kernel changes.

## Table

| Assignment | µs per layer | ms in the token | Context 2048 | Context 32768 | Context 262144 |
|---|---:|---:|---|---|---|
| all fp4 | 115.6 | 5.664 | 75.65 ms, 13.22 tok/s | 79.35 ms, 12.60 tok/s | 89.71 ms, 11.15 tok/s |
| all fp8 | 572.0 | 28.028 | 98.02 ms, 10.20 tok/s | 101.72 ms, 9.83 tok/s | 112.08 ms, 8.92 tok/s |
| all int2 | 135.1 | 6.620 | 76.61 ms, 13.05 tok/s | 80.31 ms, 12.45 tok/s | 90.67 ms, 11.03 tok/s |
| 1 fp8, 8 fp4, 1 int2 | 180.2 | 8.830 | 78.82 ms, 12.69 tok/s | 82.52 ms, 12.12 tok/s | 92.88 ms, 10.77 tok/s |
| 2 fp8, 6 fp4, 2 int2 | 236.0 | 11.564 | 81.55 ms, 12.26 tok/s | 85.25 ms, 11.73 tok/s | 95.61 ms, 10.46 tok/s |

The fp4 row in [kernel-ceilings.md](kernel-ceilings.md) is an earlier settle of the same served path, 13.29 tok/s at context 2048. This table uses the later head time.

## Search

21 October 2026. Queries: "Qwen3.8-Flash-Next tokens per second Apple Silicon MLX inference", "Qwen3.8-Flash-Next fp8 int2 expert layer microseconds decode Apple Silicon moe_decode". The predecessors in this note are the overlaps those queries returned. The search did not find the 572.0, 115.6, and 135.1 µs layer times.

## Corrections

Later finds are appended here. The sections above stay as committed.
