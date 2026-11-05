# Coding agent

Measurement note for the `qwen35-4b` pipeline. The serving contract is [../README.md](../README.md). This file records vision coding-agent runs. It is a recorded run, not a format bound.

## Remainder

On an M5 Pro with 48 GB, `Qwen3.5-4B-fp8` with `fp8_e4m3` keys and values, MTP off, passed the vision coding agent. Thinking recorded 1460.00 tok/s prefill and 27.68 tok/s decode. Instruct recorded 1414.22 tok/s prefill and 27.12 tok/s decode. A search on 3 November 2026 did not find that table.

## Predecessors

- Method. The ledger writes `ledger.py` through the coding tools, restarts serve so the next turn loads the prefix from disk, then checks the chat contract. Exit 0 requires `collect` to return 6, 0, and -6. Which tools were called is not the result.
- Model. [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B). The card reports coding and agent scores for this checkpoint. Those scores are not this harness rate.
- Rate. [oMLX Qwen3.5-4B-8bit on an M5 Pro with 48 GB](https://omlx.ai/benchmarks/performance/x0t6bie9) reports 2643 tok/s prefill and 55.4 tok/s decode at 4096 context. [oMLX Qwen3.5-4B-MLX-4bit on an M5 Pro with 64 GB](https://omlx.ai/benchmarks/performance/ngfyp6l9) reports 3159 tok/s prefill and 92.6 tok/s decode at 4096 context. An earlier Favlo coding run on the default cache without the still is in [recorded-forward.md](recorded-forward.md): 1673.36 tok/s prefill and 40.77 tok/s decode. None of them is this `fp8_e4m3` vision table.

## Rule

MTP is off. The still is the first chat, `detail` `auto`, `tool_choice` `none`. The ledger is the next chat on that session. Thinking on uses the thinking card. Thinking off uses the instruct card. The banner prints the loaded plane. This table is the pair whose banners printed `kv fp8_e4m3` and `vision on`. A banner that prints `half` or `vision off` is a different row. Request 2 of each run loaded the prefix from disk.

## Command

```text
uv run python -m harness.coding --model qwen35-4b --vision --thinking on
uv run python -m harness.coding --model qwen35-4b --vision --thinking off
```

The banner prints the loaded plane, MTP, thinking, and vision.

## Table

| Thinking | Vision | Result | Prefill | Decode |
|---|---|---|---|---|
| on | on | pass | 1460.00 tok/s (2993 tok in 2.05 s) | 27.68 tok/s (1088 tok in 39.31 s) |
| off | on | pass | 1414.22 tok/s (2984 tok in 2.11 s) | 27.12 tok/s (632 tok in 23.30 s) |

## Search

3 November 2026. Queries: "Qwen3.5-4B coding agent vision ledger tok/s M5 Pro fp8_e4m3 1460 OR 1414 OR 27.68", "Qwen3.5-4B oMLX prefill decode tok/s M5 Pro 2026", and "1460.00 OR 27.68 OR 1414.22 OR 27.12 Qwen3.5-4B coding". The predecessors in this note are the overlaps those queries returned. The search did not find 1460.00 and 27.68, or 1414.22 and 27.12, in those sources.

## Corrections

Later finds are appended here. The sections above stay as committed.

### 5 November 2026 — fused sample submit

Same plane, MTP off, and vision on. Thinking recorded 1416.98 tok/s prefill and 27.46 tok/s decode. Instruct recorded 1414.22 tok/s prefill and 27.24 tok/s decode. A search on 5 November 2026 did not find that table. Queries: "1416.98 OR 27.46 OR 27.24 Qwen3.5-4B coding agent vision tok/s", and "Qwen3.5-4B coding agent vision ledger prefill decode tok/s M5 Pro fp8 2026". Closest overlaps were [Token Atlas](https://tokatlas.github.io/models/qwen3-5-4b/) at 2579 tok/s prefill and 98.7 tok/s decode for a 4-bit oMLX build on an M5 Pro at 1024 context, and [qwen35-thor](https://github.com/thomas-hiddenpeak/qwen35-thor/) at 26.3 tok/s BF16 decode with MTP off. Neither is this `fp8_e4m3` vision table.

```text
uv run python -m harness.coding --model qwen35-4b --vision --thinking on
uv run python -m harness.coding --model qwen35-4b --vision --thinking off
```

| Thinking | Vision | Result | Prefill | Decode |
|---|---|---|---|---|
| on | on | pass | 1416.98 tok/s (3004 tok in 2.12 s) | 27.46 tok/s (1160 tok in 42.25 s) |
| off | on | pass | 1414.22 tok/s (2984 tok in 2.11 s) | 27.24 tok/s (632 tok in 23.20 s) |
