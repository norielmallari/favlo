# Recorded forward

Measurement note for the `qwen36-35b-a3b` pipeline. The serving contract is [../README.md](../README.md). The request rows are in [../performance.md](../performance.md). This file records the summed prefill and decode of those runs. The isolated dispatch sum is [kernel-ceilings.md](kernel-ceilings.md).

## Remainder

The file-agent forward, instruct card, recorded 597.98 tok/s prefill over 88004 tokens and 51.26 tok/s decode over 1298 tokens. A search on 21 October 2026 did not find that run.

## Predecessors

- Method. The forward is the same shader library as [kernel-ceilings.md](kernel-ceilings.md). Overlap, the prefix store, and the atlas are inside these times. The kernel sum leaves them out.
- Model. [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). Perplexity's [on-device note](https://www.perplexity.ai/hub/blog/optimizing-on-device-inference-for-apple-silicon) publishes end-to-end rates for this model on an Apple GPU, 5749.9 prefill tok/s and 186.6 decode tok/s at 4K tokens, on a different engine.
- Measurement. [omlx issue 2238](https://github.com/jundot/omlx/issues/2238) and [mlx-swift-lm 83ff028](https://github.com/ekryski/mlx-swift-lm/commit/83ff028def2e734bf2da9b65f51a479c5c736c3d) publish other tok/s for this model. They do not publish this harness run.
- Mechanism. The file agent and the coding agent are separate series. A 16GiB cap holds fewer than every expert image.

## Rule

Prefill is the summed suffix time. Decode is the summed sampled-token time. The file agent uses the instruct card. The coding agent uses the thinking card. The earlier file-reading series in the performance note was timed against the previous value mix and is not this table.

## Command

```text
uv run python -m harness --model qwen36-35b-a3b
uv run python -m harness.coding --model qwen36-35b-a3b
uv run python -m harness --model qwen36-35b-a3b --max-memory 16GiB
uv run python -m harness.coding --model qwen36-35b-a3b --max-memory 16GiB
```

Those commands are what the performance note names. No committed command reprints the rows below.

## Table

| Run | Prefill | Decode |
|---|---|---|
| File agent | 597.98 tok/s (88004 tok in 147.17 s) | 51.26 tok/s (1298 tok in 25.32 s) |
| Coding agent | 399.05 tok/s (419 tok in 1.05 s) | 74.06 tok/s (197 tok in 2.66 s) |
| File agent, 16GiB | 493.77 tok/s (88004 tok in 178.23 s) | 29.72 tok/s (1034 tok in 34.79 s) |
| Coding agent, 16GiB | 134.29 tok/s (419 tok in 3.12 s) | 34.19 tok/s (252 tok in 7.37 s) |

## Search

21 October 2026. Queries: "Qwen3.6-35B-A3B MLX Apple Silicon inference Gated DeltaNet", and "Qwen3.6-35B-A3B Metal kernel microsecond dispatch prefill decode ceiling matmul2d". The predecessors in this note are the overlaps those queries returned. The search did not find the 597.98 and 51.26 tok/s file-agent sums.

## Corrections

Later finds are appended here. The sections above stay as committed.
