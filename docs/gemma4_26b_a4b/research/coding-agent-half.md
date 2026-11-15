# Coding agent, half cache

Measurement note for the `gemma4_26b_a4b` pipeline. The serving contract is [../ARCHITECTURE.md](../ARCHITECTURE.md). This file records vision coding-agent runs on `half` with MTP off. It is a recorded run, not a format bound. The `fp8_e4m3` table is in [coding-agent.md](coding-agent.md).

## Remainder

On an M5 Pro with 48 GB, `gemma-4-26B-A4B-it` with `half` keys and values and MTP off passed the vision coding agent. Thinking recorded 249.31 tok/s prefill and 31.11 tok/s decode. Instruct recorded 322.65 tok/s prefill and 31.47 tok/s decode. A search on 15 November 2026 did not find that table.

## Predecessors

- Method. The ledger writes `ledger.py` through the coding tools, restarts serve so the next turn loads the prefix from disk, then checks the chat contract. Exit 0 requires `collect` to return 6, 0, and -6. Which tools were called is not the result. With `--vision`, the still is the first chat.
- Model. [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it). The card describes thinking, tool use, and a 256K context. It does not publish this harness rate table.
- Rate. The same harness on `fp8_e4m3` keys and values, with the still, recorded 274.02 tok/s prefill and 8.49 tok/s decode with thinking on, and 276.61 tok/s prefill and 9.11 tok/s decode with thinking off, on 3 November 2026. That table is in [coding-agent.md](coding-agent.md).
- Rate. [oMLX gemma-4-26b-a4b-it-4bit on an M5 Pro with 48 GB](https://omlx.ai/benchmarks/performance/6w533fgx) reports 1,721 tok/s prefill and 74.1 tok/s decode at 4,096 tokens, and 1,855 tok/s prefill and 62.5 tok/s decode at 16,384 tokens. KV compression is off.
- Rate. [oMLX on an M5 Pro with 64 GB](https://omlx.ai/benchmarks/performance/3uxvqscd) reports 2,481 tok/s prefill and 81.9 tok/s decode at 4,096 tokens for the 4-bit checkpoint.
- Rate. [Token Atlas](https://tokatlas.github.io/models/gemma-4-26b-a4b/) lists an 8-bit oMLX row on an M5 Pro at 110.4 tok/s decode and 2,373 tok/s prefill at 4,096 tokens.
- Rate. [Simon P. Couch](https://simonpcouch.com/blog/2026-04-16-local-agents-2/) reports about 690 tok/s prefill and 53 tok/s decode for a 5,000-token coding-agent prompt on an M4 Pro with 48 GB under Ollama.
- None of those runs is this `half` vision table with MTP off.

## Rule

MTP is off. The still is the first chat, `detail` `auto`, `tool_choice` `none`. The ledger is the next chat on that session.

- Thinking on uses the thinking card.
- Thinking off uses the instruct card.
- The banner prints the loaded plane.
- This table is the pair whose banners printed `kv half`, `mtp off`, and `vision on`.
- A banner that prints `fp8_e4m3` or `vision off` is a different row.
- Request 2 of each run loaded the prefix from disk.

## Command

```text
uv run python -m harness.coding --model gemma4_26b_a4b --vision --thinking on
uv run python -m harness.coding --model gemma4_26b_a4b --vision --thinking off
```

The banner prints the loaded plane, MTP, thinking, and vision.

## Table

| Thinking | Vision | Result | Prefill | Decode |
|---|---|---|---|---|
| on | on | pass | 249.31 tok/s (3069 tok in 12.31 s) | 31.11 tok/s (1448 tok in 46.54 s) |
| off | on | pass | 322.65 tok/s (2849 tok in 8.83 s) | 31.47 tok/s (411 tok in 13.06 s) |

## Search

15 November 2026. Queries: "249.31 OR 31.11 OR 322.65 OR 31.47 Gemma 4 26B coding agent half kv vision tok/s" and "gemma-4-26B-A4B-it half KV cache coding agent prefill decode tok/s M5 Pro vision". The predecessors in this note are the overlaps those queries returned. The search did not find 249.31 and 31.11, or 322.65 and 31.47, in those sources.

## Corrections

Later finds are appended here. The sections above stay as committed.

15 November 2026. This run uses the packed folder `gemma-4-26B-A4B-it-fp8-e20`. Resident matrices are fp8. Of the 3,840 routed experts, 768 are fp8 and 3,072 are fp4. Keys and values are `half`.

A search the same day, "gemma-4-26B-A4B oMLX M5 Pro 48GB 1721 74.1 4096 site:omlx.ai", returned [oMLX 4-bit on an M5 Pro with 48 GB](https://omlx.ai/benchmarks/performance/6w533fgx) with `turboquant_kv_enabled` false and `mtp_enabled` false: 1,721 tok/s prefill and 74.1 tok/s generation at 4,096 tokens, and 1,855 tok/s prefill and 62.5 tok/s generation at 16,384 tokens. Those rows are a 4-bit pack. The search did not find 249.31 and 31.11, or 322.65 and 31.47.
