# Coding agent, text, CUDA

Measurement note for the CUDA runtime of `gemma4_26b_a4b`. The serving contract is [../ARCHITECTURE.md](../ARCHITECTURE.md). This file records text coding-agent runs on `half` keys and values with MTP off. It is a recorded run, not a format bound. The Metal text run is in [coding-text.md](coding-text.md).

## Remainder

On a GeForce RTX 3070 Laptop GPU with 8192 MiB, `gemma-4-26B-A4B-it-int8-e10` with `half` keys and values and MTP off passed the text coding agent. Thinking on recorded 146.52 tok/s prefill over 2885 tokens and 20.97 tok/s decode over 2146 tokens. Thinking off recorded 139.46 tok/s prefill over 2746 tokens and 20.59 tok/s decode over 827 tokens. A search on 23 November 2026 did not find that pair.

## Predecessors

- Method. The ledger writes `ledger.py` through the coding tools, restarts serve so a later turn loads the prefix from disk, then checks the chat contract. Exit 0 requires `collect` to return 6, 0, and -6. Which tools were called is not the result.
- Model. [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it). The card describes thinking, tool use, and a 256K context. It does not publish this harness rate.
- Rate. [Will It Run AI](https://willitrunai.com/can-run/gemma-4-26b-a4b-on-rtx-3090-24gb) estimates about 144.9 tok/s decode for Q4_K_M on an RTX 3090 24GB. [A Blackwell 24GB estimate](https://willitrunai.com/can-run/gemma-4-26b-a4b-on-rtx-pro-4000-blackwell-24gb) puts that decode near 138.7 tok/s and prefill near 3491 tok/s. [Caleb Coffie](https://calebcoffie.com/benchmarks/gemma-4-26b-a4b-it-gguf) records 1229.9 tok/s prefill and 116.3 tok/s decode for an agent-shaped prompt on one RTX 3090. [Runs on My](https://runsonmy.com/fit/gemma-4-26b-a4b-it/rtx-3070/) estimates 27–47 tok/s decode for Q4_K_M with expert offload on an RTX 3070 and labels that estimate as not a benchmark. The Metal text coding agent, in [coding-text.md](coding-text.md), recorded 836.90 tok/s prefill and 25.94 tok/s decode with thinking on and vision off on an M5 Pro with `fp8_e4m3`. None of those is this half text table.

## Rule

MTP is off. Thinking on uses the thinking card. Thinking off uses the instruct card. The banner prints the loaded plane. This table is the pair whose banners printed `kv half`, `mtp off`, and `vision off`. A banner that prints `vision on` or `fp8_e4m3` is a different row. A later request loaded the prefix from disk.

## Command

```text
uv run python -m harness.coding --model gemma4_26b_a4b --thinking on
uv run python -m harness.coding --model gemma4_26b_a4b --thinking off
```

The banner prints the loaded plane, MTP, thinking, and vision.

## Table

| Thinking | Vision | Result | Prefill | Decode |
|---|---|---|---|---|
| on | off | pass | 146.52 tok/s (2885 tok in 19.69 s) | 20.97 tok/s (2146 tok in 102.33 s) |
| off | off | pass | 139.46 tok/s (2746 tok in 19.69 s) | 20.59 tok/s (827 tok in 40.16 s) |

## Search

23 November 2026.

- Queries: "146.52 OR 20.97 OR 139.46 OR 20.59 Gemma 4 26B tok/s coding agent", and "Gemma 4 26B A4B RTX 3070 tokens per second prefill decode benchmark 2026".
- The predecessors in this note are the overlaps those queries returned.
- The search did not find 146.52 and 20.97, or 139.46 and 20.59, as this text coding agent.

## Corrections

Later finds are appended here. The sections above stay as committed.
