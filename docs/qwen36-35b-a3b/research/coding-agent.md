# Coding agent

Measurement note for the `qwen36-35b-a3b` pipeline. The serving contract is [../README.md](../README.md). This file records one coding-agent run. It is a recorded run, not a format bound.

## Remainder

On an M5 Pro with 48 GB, `Qwen3.6-35B-A3B-fp8-e20` with `fp8_e4m3` keys and values passed the vision coding agent. Thinking recorded 632.95 tok/s prefill and 68.72 tok/s decode. Instruct recorded 671.74 tok/s prefill and 69.33 tok/s decode. A search on 3 November 2026 did not find that table.

## Predecessors

- Method. The ledger writes `ledger.py` through the coding tools, restarts serve so the next turn loads the prefix from disk, then checks the chat contract. Exit 0 requires `collect` to return 6, 0, and -6. Which tools were called is not the result.
- Model. [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) and [Qwen/Qwen3.6-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8). The card reports SWE-bench Verified 73.4 for this checkpoint. That score is not this harness rate.
- Rate. [Baobab](https://baobabtech.ai/posts/local-moe-m5-max-qwen-gemma-speed) reports 107 tok/s on a coding prompt under `mlx_lm.server`, 4-bit, on an M5 Max with 128 GB, measured 19 October 2026. [How To Spark](https://howtospark.com/recipes/qwen3-6-35b-a3b-fp8) reports 3.6k tok/s prefill and 33.3 tok/s decode at 128K for the official FP8 checkpoint on one DGX Spark. [NInfer](https://github.com/Neroued/ninfer/blob/master/model-cards/Qwen3.6-35B-A3B-NInfer/README.md) reports 677.2 tok/s decode on its code fixtures, MTP 3, INT8 group-64 keys and values, on one RTX 5090, measured 2026-09-07. The same harness on half keys and values, without the still, recorded 597.65 tok/s prefill and 61.46 tok/s decode with thinking on, and 612.20 tok/s prefill and 59.29 tok/s decode with thinking off, on 3 November 2026. Those rows are in [../performance.md](../performance.md). None of them is this `fp8_e4m3` vision table.

## Rule

MTP is off. The still is the first chat, `detail` `auto`, `tool_choice` `none`. The ledger is the next chat on that session. Thinking on uses the thinking card. Thinking off uses the instruct card. The banner prints the loaded plane. This table is the pair whose banners printed `kv fp8_e4m3` and `vision on`. A banner that prints `half`, `e2m1`, `f32`, or `vision off` is a different row. Request 2 of each run loaded the prefix from disk.

## Command

```text
uv run python -m harness.coding --model qwen36-35b-a3b --vision --thinking on
uv run python -m harness.coding --model qwen36-35b-a3b --vision --thinking off
```

The banner prints the loaded plane, MTP, thinking, and vision.

## Table

| Thinking | Vision | Result | Prefill | Decode |
|---|---|---|---|---|
| on | on | pass | 632.95 tok/s (2747 tok in 4.34 s) | 68.72 tok/s (1358 tok in 19.76 s) |
| off | on | pass | 671.74 tok/s (2734 tok in 4.07 s) | 69.33 tok/s (495 tok in 7.14 s) |

## Search

3 November 2026. Queries: "632.95 OR 671.74 OR 67.55 Qwen3.6-35B-A3B coding agent fp8_e4m3 vision ledger tok/s", and "632.95 OR 68.72 OR 671.74 OR 69.33 Qwen3.6-35B-A3B". The predecessors in this note are the overlaps those queries returned. The search did not find 632.95 and 68.72, or 671.74 and 69.33, in those sources.

## Corrections

Later finds are appended here. The sections above stay as committed.

5 November 2026. On an M5 Pro with 48 GB, `Qwen3.6-35B-A3B-fp8-e20` with `fp8_e4m3` keys and values, MTP off, thinking on, vision off, store off: the coding agent passed at 651.69 tok/s prefill (2509 tok in 3.85 s) and 67.50 tok/s decode (806 tok in 11.94 s). Command: `uv run python -m harness.coding --model qwen36-35b-a3b --thinking on --store off`. The banner printed `vision off`. Query: "651.69 OR 67.50 Qwen3.6-35B-A3B coding agent fp8_e4m3 vision off". The card and the Metal MTP issue above do not list those rates. A search on 5 November 2026 did not find 651.69 and 67.50 in those sources.
