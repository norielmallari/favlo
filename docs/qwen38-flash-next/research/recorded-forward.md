# Recorded forward

Measurement note for the `qwen38-flash-next` pipeline. The serving contract is [../architecture.md](../architecture.md). The request rows are in [../performance.md](../performance.md). The isolated dispatch sum is [kernel-ceilings.md](kernel-ceilings.md).

## Remainder

The file-reading run recorded 200.37 tok/s prefill over 92152 tokens and 4.65 tok/s decode over 318 tokens. A search on 21 October 2026 did not find that run.

## Predecessors

- Method. The forward is this pipeline's shader library. The kernel sum leaves overlap and expert loads out of the prefill chunk.
- Model. Qwen Team ([arXiv:2608.30320](https://arxiv.org/abs/2608.30320)). [MTPLX](https://mtplx.com/docs/run-qwen-3-8-on-a-mac/) reports 125.8 tok/s on one cached request, with a draft.
- Measurement. That rate is another engine. It does not publish this harness run.
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

21 October 2026. Query: "Qwen3.8-27B inference tok/s Apple Silicon Qwen3.8-Flash-Next". The predecessor in this note is the overlap that query returned for this model. The search did not find the 200.37 and 4.65 tok/s sums.

## Corrections

Later finds are appended here. The sections above stay as committed.
