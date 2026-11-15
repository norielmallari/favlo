# Coding agent, text

Measurement note for the `qwen35_4b` pipeline. The serving contract is [../ARCHITECTURE.md](../ARCHITECTURE.md). This file records one text coding-agent run on `fp8_e4m3` with MTP off. It is a recorded run, not a format bound. The vision runs are in [coding-agent.md](coding-agent.md). The half-cache coding run is in [recorded-forward.md](recorded-forward.md).

## Remainder

- On an M5 Pro with 48 GB, `Qwen3.5-4B-fp8` with `fp8_e4m3` keys and values and MTP off passed the text coding agent.
- Thinking on, vision off, recorded 738.46 tok/s prefill and 28.26 tok/s decode.
- A search on 15 November 2026 did not find that pair.

## Predecessors

- Method. The ledger writes `ledger.py` through the coding tools, restarts serve so the next turn loads the prefix from disk, then checks the chat contract. Exit 0 requires `collect` to return 6, 0, and -6. Which tools were called is not the result.
- Model. [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B). The card reports coding and agent scores for this checkpoint. Those scores are not this harness rate.
- Rate. [oMLX Qwen3.5-4B-8bit on a 20-core M5 Pro with 48 GB](https://omlx.ai/benchmarks/performance/x0t6bie9) reports 2643 tok/s prefill and 55.4 tok/s decode at 4096 context. [oMLX Qwen3.5-4B-MLX-4bit on a 20-core M5 Pro with 64 GB](https://omlx.ai/benchmarks/performance/ngfyp6l9) reports 3159 tok/s prefill and 92.6 tok/s decode at 4096 context. [Token Atlas](https://tokatlas.github.io/models/qwen3-5-4b/) reports 2579 tok/s prefill and 98.7 tok/s decode for a 4-bit oMLX build on an M5 Pro at 1024 context. The vision table is in [coding-agent.md](coding-agent.md). The half-cache coding run is in [recorded-forward.md](recorded-forward.md). None of them is this `fp8_e4m3` text table.

## Rule

- MTP is off.
- Thinking on uses the thinking card.
- The banner prints the loaded plane.
- This row is the run whose banner printed `kv fp8_e4m3`, `mtp off`, `thinking on`, and `vision off`.
- A banner that prints `half` or `vision on` is a different row.
- Request 2 loaded the prefix from disk.

## Command

Historical (hyphen id as recorded):

```text
uv run python -m harness.coding --model qwen35-4b --thinking on --mtp off
```

The live CLI id is `qwen35_4b`. The banner prints the loaded plane, MTP, thinking, and vision.

## Table

| Thinking | Vision | Result | Prefill | Decode |
|---|---|---|---|---|
| on | off | pass | 738.46 tok/s (2784 tok in 3.77 s) | 28.26 tok/s (890 tok in 31.49 s) |

## Search

15 November 2026.

- Queries: "738.46 OR 28.26 Qwen3.5-4B coding agent fp8_e4m3 vision off tok/s M5 Pro", "Qwen3.5-4B fp8 coding agent text harness M5 Pro tok/s 2026 vision off thinking", and "738.46 OR 28.26 Qwen3.5-4B oMLX Token Atlas text coding agent fp8 M5 Pro".
- The predecessors in this note are the overlaps those queries returned.
- The search did not find 738.46 and 28.26 in those sources.

## Corrections

Later finds are appended here. The sections above stay as committed.