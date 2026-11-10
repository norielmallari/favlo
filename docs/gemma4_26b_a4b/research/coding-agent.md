# Coding agent

Measurement note for the `gemma4_26b_a4b` pipeline. The serving contract is [../README.md](../README.md). This file records vision coding-agent runs on `fp8_e4m3` with MTP off. It is a recorded run, not a format bound. An earlier coding-agent decode on the half cache is in [recorded-forward.md](recorded-forward.md).

## Remainder

On an M5 Pro with 48 GB, `gemma-4-26B-A4B-it` with `fp8_e4m3` keys and values and MTP off passed the vision coding agent. Thinking recorded 274.02 tok/s prefill and 8.49 tok/s decode. Instruct recorded 276.61 tok/s prefill and 9.11 tok/s decode. A search on 3 November 2026 did not find that table.

## Predecessors

- Method. The ledger writes `ledger.py` through the coding tools, restarts serve so the next turn loads the prefix from disk, then checks the chat contract. Exit 0 requires `collect` to return 6, 0, and -6. Which tools were called is not the result. With `--vision`, the still is the first chat.
- Model. [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it). The card describes thinking, tool use, and a 256K context. It does not publish this harness rate table.
- Rate. [oMLX gemma-4-26b-a4b-it-4bit on an M5 Pro with 48 GB](https://omlx.ai/benchmarks/performance/gv7qn8l9) reports 2,082 tok/s prefill and 76.9 tok/s decode at 1,024 tokens, and the same session's 4,096-token row is 2,012 and 74.0. [oMLX at 16,384 tokens](https://omlx.ai/benchmarks/performance/lnir2jdu) reports 1,855 tok/s prefill and 62.5 tok/s decode. [Token Atlas](https://tokatlas.github.io/models/gemma-4-26b-a4b/) lists an 8-bit oMLX row on an M5 Pro at 110.4 tok/s decode and 2,373 tok/s prefill at 4,096 tokens. [Simon P. Couch](https://simonpcouch.com/blog/2026-04-16-local-agents-2/) reports about 690 tok/s prefill and 53 tok/s decode for a 5,000-token coding-agent prompt on an M4 Pro with 48 GB under Ollama. An earlier Favlo coding log on the half cache is in [recorded-forward.md](recorded-forward.md): 18.84 tok/s decode over 820 tokens. None of those runs is this `fp8_e4m3` vision table with MTP off.

## Rule

MTP is off. The still is the first chat, `detail` `auto`, `tool_choice` `none`. The ledger is the next chat on that session. Thinking on uses the thinking card. Thinking off uses the instruct card. The banner prints the loaded plane. This table is the pair whose banners printed `kv fp8_e4m3`, `mtp off`, and `vision on`. A banner that prints `half` or `vision off` is a different row. Request 2 of each run loaded the prefix from disk.

## Command

```text
uv run python -m harness.coding --model gemma4-26b-a4b --vision --thinking on
uv run python -m harness.coding --model gemma4-26b-a4b --vision --thinking off
```

The banner prints the loaded plane, MTP, thinking, and vision.

## Table

| Thinking | Vision | Result | Prefill | Decode |
|---|---|---|---|---|
| on | on | pass | 274.02 tok/s (2869 tok in 10.47 s) | 8.49 tok/s (1144 tok in 134.80 s) |
| off | on | pass | 276.61 tok/s (2542 tok in 9.19 s) | 9.11 tok/s (367 tok in 40.28 s) |

## Search

3 November 2026. Queries: "Gemma 4 26B A4B coding agent tok/s M5 Pro fp8", "gemma-4-26B-A4B-it oMLX M5 Pro 48GB prefill decode tok/s coding agent", and "274.02 OR 8.49 Gemma 4 26B coding agent fp8_e4m3 vision ledger". The predecessors in this note are the overlaps those queries returned. The search did not find 274.02 and 8.49, or 276.61 and 9.11, in those sources.

## Corrections

Later finds are appended here. The sections above stay as committed.
