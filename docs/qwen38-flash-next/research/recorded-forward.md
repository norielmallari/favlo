# Recorded forward

Measurement note for the `qwen38-flash-next` pipeline. The serving contract is [../architecture.md](../architecture.md). The request rows are in [../performance.md](../performance.md). The isolated dispatch sum is [kernel-ceilings.md](kernel-ceilings.md).

## Remainder

The file-reading run recorded 200.37 tok/s prefill over 92152 tokens and 4.65 tok/s decode over 318 tokens. A search on 21 October 2026 did not find that run.

## Predecessors

- Method. The forward is this pipeline's shader library. The kernel sum leaves overlap and expert loads out of the prefill chunk.
- Model. Qwen Team ([arXiv:2608.30320](https://arxiv.org/abs/2608.30320)). [MTPLX](https://mtplx.com/docs/run-qwen-3-8-on-a-mac/) reports 125.8 tok/s on one OpenCode request, with a draft and 18,364 cached prompt tokens, and 61.8 tok/s on a 109k-token turn, on an M5 Max.
- Measurement. [oMLX issue 3353](https://github.com/jundot/omlx/issues/3353) reports 29.7 tok/s greedy with MTP off, and 31.9 tok/s at 91k tokens, on an M3 Ultra. [Whallm](https://shadowgraph.io/yanun0323/Whallm) reports 113.3 tok/s prefill and 9.7 tok/s decode at 16,384 tokens for an fp8 checkpoint on an M5 Pro. None of those runs is this harness table.
- Mechanism. The file-reading series set temperature 0 and left thinking on. The instruct series and the coding series stay separate.

## Rule

Prefill is the summed suffix time. Decode is the summed sampled-token time. The coding-agent harness summary counted decode from the first request only. The server log's second decode line is not in that summary.

## Command

```text
uv run python -m harness --model qwen38-flash-next
uv run python -m harness.coding --model qwen38-flash-next
```

Those commands are what the performance note names. No committed command reprints the rows below.

## Table

| Run | Prefill | Decode |
|---|---|---|
| File-reading | 200.37 tok/s (92152 tok in 459.91 s) | 4.65 tok/s (318 tok in 68.42 s) |
| Instruct file agent | 190.34 tok/s (68231 tok in 358.46 s) | 6.61 tok/s (521 tok in 78.86 s) |
| Coding agent, harness summary | 59.58 tok/s (457 tok in 7.67 s) | 10.37 tok/s (893 tok in 86.11 s) |

## Search

21 October 2026. Queries: "Qwen3.8-Flash-Next tokens per second Apple Silicon MLX inference", "Qwen3.8-Flash-Next fp8 int2 expert layer microseconds decode Apple Silicon moe_decode". The predecessors in this note are the overlaps those queries returned. The search did not find the 200.37 and 4.65 tok/s sums.

## Corrections

Later finds are appended here. The sections above stay as committed.
