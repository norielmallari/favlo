# Coding agent (MTP on)

Measurement note for the `qwen35-4b` pipeline. The serving contract is [../README.md](../README.md). This file records one vision coding-agent run with MTP on. It is a recorded run, not a format bound. The MTP-off table is [coding-agent.md](coding-agent.md).

## Remainder

On an M5 Pro with 48 GB, `Qwen3.5-4B-fp8` with `fp8_e4m3` keys and values and MTP on at depth 3 passed the vision coding agent on the instruct card. Prefill recorded 1469.46 tok/s and decode recorded 23.91 tok/s. A search on 5 November 2026 did not find that table.

## Predecessors

- Method. The ledger writes `ledger.py` through the coding tools, restarts serve so the next turn loads the prefix from disk, then checks the chat contract. Exit 0 requires `collect` to return 6, 0, and -6. Which tools were called is not the result.
- Model. [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B). The card reports coding and agent scores for this checkpoint and documents an MTP server launch. Those scores are not this harness rate.
- Rate. [oMLX Qwen3.5-4B oQ4e FP16 MTP on an M5 Pro](https://omlx.ai/benchmarks/performance/hbz79itz) reports 2638 tok/s prefill and 89.0 tok/s decode at 4096 context. [DiggTop Qwen3.5-4B-MTP Q6_K_XL on an RTX 3090](https://www.diggtop.com/models/qwen3-5-4b-mtp) reports 2167.6 tok/s prefill and 240.4 tok/s decode. [Treeish on an M4 Max](https://huggingface.co/treeish/Qwen3.5-4B-oQ4e-MTP-MLX) reports 66.4 tok/s at MTP depth 3 on a 1066-token fixture. The MTP-off vision table is in [coding-agent.md](coding-agent.md). None of them is this `fp8_e4m3` vision table.

## Rule

MTP is on at depth 3. The still is the first chat, `detail` `auto`, `tool_choice` `none`. The ledger is the next chat on that session. Thinking is off, so the request uses the instruct card. The banner prints the loaded plane. This table is the row whose banner printed `kv fp8_e4m3`, `mtp on`, `depth 3`, `thinking off`, and `vision on`. A banner that prints `half`, `mtp off`, `thinking on`, or `vision off` is a different row. Request 2 loaded the prefix from disk.

## Command

```text
uv run python -m harness.coding --model qwen35-4b --mtp on --vision --thinking off
```

The banner prints the loaded plane, MTP, thinking, and vision.

## Table

| Thinking | Vision | MTP | Result | Prefill | Decode |
|---|---|---|---|---|---|
| off | on | on, depth 3 | pass | 1469.46 tok/s (2983 tok in 2.03 s) | 23.91 tok/s (707 tok in 29.57 s) |

## Search

5 November 2026. Queries: "1469.46 OR 23.91 Qwen3.5-4B coding agent vision MTP tok/s", "Qwen3.5-4B MTP speculative decoding prefill decode tok/s M5 Pro fp8 2026", and "Qwen3.5-4B oQ4e MTP M5 Pro 2638 OR 89.0 vision coding agent fp8_e4m3". The predecessors in this note are the overlaps those queries returned. The search did not find 1469.46 and 23.91 in those sources.

## Corrections

Later finds are appended here. The sections above stay as committed.
