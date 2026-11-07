# Coding agent, half cache

Measurement note for the `qwen36-35b-a3b` pipeline. The serving contract is [../README.md](../README.md). This file records one coding-agent run. It is a recorded run, not a format bound.

## Remainder

On an M5 Pro with 48 GB, `Qwen3.6-35B-A3B-fp8-e20` with `half` keys and values passed the vision coding agent. Thinking recorded 574.21 tok/s prefill and 66.23 tok/s decode. Instruct recorded 547.92 tok/s prefill and 67.44 tok/s decode. A search on 7 November 2026 did not find that table.

## Predecessors

- Method. The ledger writes `ledger.py` through the coding tools, restarts serve so the next turn loads the prefix from disk, then checks the chat contract. Exit 0 requires `collect` to return 6, 0, and -6. Which tools were called is not the result.
- Model. [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) reports SWE-bench Verified 73.4 for this checkpoint. That score is not this harness rate.
- Rate. The same harness on `fp8_e4m3` keys and values, with the still, recorded 632.95 tok/s prefill and 68.72 tok/s decode with thinking on, and 671.74 tok/s prefill and 69.33 tok/s decode with thinking off, on 3 November 2026. That table is in [coding-agent.md](coding-agent.md). The same harness on half keys and values, without the still, recorded 597.65 tok/s prefill and 61.46 tok/s decode with thinking on, and 612.20 tok/s prefill and 59.29 tok/s decode with thinking off, on 3 November 2026. Those rows are in [../performance.md](../performance.md). [NInfer](https://github.com/Neroued/ninfer/blob/master/model-cards/Qwen3.6-35B-A3B-NInfer/README.md) reports 677.2 tok/s decode on its code fixtures, MTP 3, INT8 group-64 keys and values, on one RTX 5090, measured 2026-09-07. [A 64K llama-bench row on an M5 Max](https://www.reddit.com/r/LocalLLaMA/comments/1sz9v9z/qwen_3635ba3b_kv_cache_part_2_ppl_kl_divergence/) reports 602.0 tok/s prefill at 512 tokens and 59.8 tok/s generation of 128 tokens for symmetric f16 keys and values. None of them is this `half` vision table.

## Rule

MTP is off. The still is the first chat, `detail` `auto`, `tool_choice` `none`. The ledger is the next chat on that session. Thinking on uses the thinking card. Thinking off uses the instruct card. The banner prints the loaded plane. This table is the pair whose banners printed `kv half` and `vision on`. A banner that prints `fp8_e4m3`, `e2m1`, `f32`, or `vision off` is a different row. Request 2 of each run loaded the prefix from disk. `store` is off.

## Command

```text
uv run python -m harness.coding --model qwen36-35b-a3b --vision --thinking on
uv run python -m harness.coding --model qwen36-35b-a3b --vision --thinking off
```

The banner prints the loaded plane, MTP, thinking, and vision.

## Table

| Thinking | Vision | Result | Prefill | Decode |
|---|---|---|---|---|
| on | on | pass | 574.21 tok/s (2716 tok in 4.73 s) | 66.23 tok/s (1226 tok in 18.51 s) |
| off | on | pass | 547.92 tok/s (2767 tok in 5.05 s) | 67.44 tok/s (582 tok in 8.63 s) |

## Search

7 November 2026. Queries: "574.21 OR 547.92 OR 66.23 OR 67.44 Qwen3.6-35B-A3B coding agent half vision tok/s", "Qwen3.6-35B-A3B half kv cache coding agent vision prefill decode tok/s M5", and "574.21 66.23 Qwen3.6 OR 547.92 67.44 coding agent half". The predecessors in this note are the overlaps those queries returned. The search did not find 574.21 and 66.23, or 547.92 and 67.44, in those sources.

## Corrections

Later finds are appended here. The sections above stay as committed.
