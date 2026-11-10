# Coding agent, half cache

Measurement note for the `qwen38_27b` pipeline. The serving contract is [../README.md](../README.md). This file records coding-agent runs on `half` with MTP off. It is a recorded run, not a format bound. The `fp8_e4m3` table is [coding-agent.md](coding-agent.md).

## Remainder

On an M5 Pro with 48 GB, `Qwen3.8-27B-fp8-e20` with `half` keys and values and MTP off passed the vision coding agent. Thinking recorded 269.03 tok/s prefill and 11.71 tok/s decode. Instruct recorded 284.35 tok/s prefill and 12.12 tok/s decode. A search on 8 November 2026 did not find that table.

## Predecessors

- Method. The ledger writes `ledger.py` through the coding tools, restarts serve so the next turn loads the prefix from disk, then checks the chat contract. Exit 0 requires `collect` to return 6, 0, and -6. Which tools were called is not the result. With `--vision`, the still is the first chat.
- Model. [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). The card does not publish this harness rate table.
- Rate. The same harness on `fp8_e4m3` keys and values, with the still and MTP off, recorded prefill between 299.89 and 316.78 tok/s and decode between 14.16 and 14.36 tok/s. That table is in [coding-agent.md](coding-agent.md). [llm-bench](https://llm-bench.io/benchmarks/cmtdr2v3y00c201noaeefqfp5) reports 17.1 tok/s for a coding HTML game on an M5 Pro with 48 GB under LM Studio. [llm-bench](https://llm-bench.io/benchmarks/cmvac9wso009z01obo2won597) reports 249.0 tok/s prompt processing and 21.4 tok/s generation for `Qwen3.8-27B-oQ4e-fp16-mtp` on an M5 Pro with 48 GB, with the KV cache unset. [oMLX](https://omlx.ai/benchmarks/performance/uk5k6rtm) reports 491.2 tok/s prompt processing and 16.1 tok/s generation at 16k for that pack. [mlx-works/Qwen3.8-27B-oQ2e-mtp](https://huggingface.co/mlx-works/Qwen3.8-27B-oQ2e-mtp) reports 15.0 tok/s generation and 11.71 GB peak memory on a 1024-token prompt. None of those runs is this harness table on `half` with MTP off.

## Rule

MTP is off. Thinking on uses the thinking card. Thinking off uses the instruct card. `--vision` sends the horse still as the first chat with `tool_choice` `none`, then the ledger on that session. The banner prints the loaded plane. This table is the pair whose banners printed `kv half`, `mtp off`, and `vision on`. A banner that prints `fp8_e4m3` or `mtp on` is a different row. Request 2 of each run loaded the prefix from disk. `store` is off.

## Command

```text
uv run python -m harness.coding --model qwen38-27b --vision --thinking on
uv run python -m harness.coding --model qwen38-27b --vision --thinking off
```

The banner prints the loaded plane, MTP, thinking, and vision.

## Table

| Thinking | Vision | Result | Prefill | Decode |
|---|---|---|---|---|
| on | on | pass | 269.03 tok/s (2997 tok in 11.14 s) | 11.71 tok/s (750 tok in 64.06 s) |
| off | on | pass | 284.35 tok/s (3125 tok in 10.99 s) | 12.12 tok/s (1037 tok in 85.55 s) |

## Search

8 November 2026. Queries: "269.03 OR 284.35 OR 11.71 Qwen3.8-27B half coding agent tok/s M5 Pro", "Qwen3.8-27B fp16 KV cache coding agent vision M5 Pro 48GB tok/s", and "269.03 OR 284.35 OR 11.71 tok Qwen3.8 half". The predecessors in this note are the overlaps those queries returned. The search did not find 269.03 and 11.71, or 284.35 and 12.12, as this harness pair. The 11.71 figure in the MLX card is gigabytes of peak memory, not this decode rate.

## Corrections

Later finds are appended here. The sections above stay as committed.
