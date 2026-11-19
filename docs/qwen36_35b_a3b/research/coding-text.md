# Coding agent, text, half cache

Measurement note for the `qwen36_35b_a3b` pipeline. The serving contract is [../ARCHITECTURE.md](../ARCHITECTURE.md). This file records one text coding-agent run on `Qwen3.6-35B-A3B-fp8-e30-24h` with `half` keys and values. It is a recorded run, not a format bound. The vision runs are in [coding-agent.md](coding-agent.md) and [coding-agent-half.md](coding-agent-half.md).

## Remainder

- On an M5 Pro with 48 GB, `Qwen3.6-35B-A3B-fp8-e30-24h` with `half` keys and values and MTP off passed the text coding agent.
- Thinking on, vision off, recorded 539.66 tok/s prefill and 65.19 tok/s decode.
- Thinking off, vision off, recorded 492.10 tok/s prefill and 66.72 tok/s decode.
- A search on 19 November 2026 did not find that table.

## Predecessors

- Method. The ledger writes `ledger.py` through the coding tools, restarts serve so the next turn loads the prefix from disk, then checks the chat contract. Exit 0 requires `collect` to return 6, 0, and -6. Which tools were called is not the result.
- Model. [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) reports SWE-bench Verified 73.4 for this checkpoint. That score is not this harness rate.
- Rate. The same harness on `fp8_e4m3` keys and values, with the still, recorded 632.95 tok/s prefill and 68.72 tok/s decode with thinking on, and 671.74 tok/s prefill and 69.33 tok/s decode with thinking off, on 3 November 2026. That table is [coding-agent.md](coding-agent.md).
- Rate. The same harness on `half` keys and values, with the still, on `Qwen3.6-35B-A3B-fp8-e20`, recorded 574.21 tok/s prefill and 66.23 tok/s decode with thinking on, and 547.92 tok/s prefill and 67.44 tok/s decode with thinking off, on 7 November 2026. That table is [coding-agent-half.md](coding-agent-half.md).
- Rate. The same harness on `half` keys and values, without the still, on `Qwen3.6-35B-A3B-fp8-e20`, recorded 597.65 tok/s prefill and 61.46 tok/s decode with thinking on, and 612.20 tok/s prefill and 59.29 tok/s decode with thinking off, on 3 November 2026. Those rows are in [../performance.md](../performance.md).
- Rate. [NInfer](https://github.com/Neroued/ninfer/blob/master/model-cards/Qwen3.6-35B-A3B-NInfer/README.md) reports 677.2 tok/s decode on its code fixtures, MTP 3, on one RTX 5090, measured 2026-09-07.
- Rate. [A community Q4_K_M run](https://llmcheck.net/models/qwen-36-35b-a3b-on-m5-pro/) reports 79.48 tok/s on an M5 Pro with 48 GB for about 85 prompt tokens and 512-token coding answers.
- Rate. None of them is this `half` text table on `Qwen3.6-35B-A3B-fp8-e30-24h`.

## Rule

- MTP is off.
- The still is omitted.
- Thinking on uses the thinking card.
- Thinking off uses the instruct card.
- The banner prints the loaded plane.
- This table is the pair whose banners printed `kv half`, `mtp off`, and `vision off`.
- A banner that prints `fp8_e4m3`, `e2m1`, `f32`, or `vision on` is a different row.
- Request 2 of each run loaded the prefix from disk.
- `store` is off.

## Command

```text
uv run python -m harness.coding --model qwen36_35b_a3b --thinking on --mtp off
uv run python -m harness.coding --model qwen36_35b_a3b --thinking off --mtp off
```

The banner prints the loaded plane, MTP, thinking, and vision.

## Table

| Thinking | Vision | Result | Prefill | Decode |
|---|---|---|---|---|
| on | off | pass | 539.66 tok/s (2504 tok in 4.64 s) | 65.19 tok/s (751 tok in 11.52 s) |
| off | off | pass | 492.10 tok/s (2554 tok in 5.19 s) | 66.72 tok/s (447 tok in 6.70 s) |

## Search

- 19 November 2026.
- Queries: "539.66 OR 65.19 OR 492.10 OR 66.72 Qwen3.6-35B-A3B coding agent half tok/s", and "Qwen3.6-35B-A3B SWE-bench Verified 73.4 NInfer 677.2 M5 Pro half cache 539.66 492.10 coding harness vision off".
- The predecessors in this note are the overlaps those queries returned.
- The search did not find 539.66 and 65.19, or 492.10 and 66.72, in those sources.

## Corrections

Later finds are appended here. The sections above stay as committed.
