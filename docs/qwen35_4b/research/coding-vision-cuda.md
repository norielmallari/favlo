# Coding agent, vision, CUDA

Measurement note for the CUDA runtime of `qwen35_4b`. The serving contract is [../ARCHITECTURE.md](../ARCHITECTURE.md). This file records vision coding-agent runs on int8 keys and values with MTP off. It is a recorded run, not a format bound.

## Remainder

On a GeForce RTX 3070 Laptop GPU with 8192 MiB, `Qwen3.5-4B-int8` with int8 keys and values and MTP off passed the vision coding agent. Thinking on recorded 773.16 tok/s prefill over 5876 tokens and 51.09 tok/s decode over 2245 tokens. Thinking off recorded 709.64 tok/s prefill over 5592 tokens and 42.35 tok/s decode over 1124 tokens. A search on 23 November 2026 did not find that pair.

## Predecessors

- Method. The ledger writes `ledger.py` through the coding tools, restarts serve so a later turn loads the prefix from disk, then checks the chat contract. Exit 0 requires `collect` to return 6, 0, and -6. Which tools were called is not the result. The still is the seeded horse.
- Model. [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B). The card publishes vision and agent scores and says throughput varies by framework. It does not publish this harness rate.
- Rate. [Will It Run AI](https://willitrunai.com/can-run/qwen-3.5-4b-on-rtx-3070-8gb) estimates about 55.9 tok/s decode and 3863 tok/s prefill for Q4_K_M on an RTX 3070 8GB. [A second estimate](https://willitrunai.com/can-run/hf-unsloth--qwen3-5-4b-gguf-on-rtx-3070-8gb) puts Q4_K_M decode near 48.0 tok/s. [GPUfit](https://gpufit.tech/can-i-run/qwen3-5-4b/on/rtx-3070/) estimates 40–54 tok/s for UD-Q8_K_XL and labels that estimate as not a benchmark. [An RTX 5060 Ti llama.cpp run](https://dev.to/elcances/qwen35-4b-on-a-16-gb-rtx-5060-ti-823-tokens-per-second-for-16-users-6i) records 5050 tok/s prompt processing and 114 tok/s generation at 512 tokens. The Metal vision coding agent, in [coding-agent.md](coding-agent.md), recorded 1416.98 / 27.46 tok/s thinking and 1414.22 / 27.24 tok/s instruct on an M5 Pro with `fp8_e4m3`. None of those is this int8 vision table.

## Rule

MTP is off. Thinking on uses the thinking card. Thinking off uses the instruct card. The banner prints the loaded plane. This table is the pair whose banners printed `kv int8`, `mtp off`, and `vision on`. A banner that prints `vision off` or `fp8_e4m3` is a different row. A later request loaded the prefix from disk.

## Command

```text
uv run python -m harness.coding --model qwen35_4b --vision --thinking on
uv run python -m harness.coding --model qwen35_4b --vision --thinking off
```

The banner prints the loaded plane, MTP, thinking, and vision.

## Table

| Thinking | Vision | Result | Prefill | Decode |
|---|---|---|---|---|
| on | on | pass | 773.16 tok/s (5876 tok in 7.60 s) | 51.09 tok/s (2245 tok in 43.94 s) |
| off | on | pass | 709.64 tok/s (5592 tok in 7.88 s) | 42.35 tok/s (1124 tok in 26.54 s) |

## Search

23 November 2026.

- Queries: "773.16 OR 51.09 OR 709.64 OR 42.35 Qwen3.5-4B vision coding agent tok/s", "Qwen3.5-4B RTX 3070 tokens per second int8 vision benchmark 2026", and "773.16 OR 709.64 OR 51.09 Qwen3.5-4B prompt processing GeForce RTX 3070".
- The predecessors in this note are the overlaps those queries returned.
- The search did not find 773.16 and 51.09, or 709.64 and 42.35, as this vision coding agent.

## Corrections

Later finds are appended here. The sections above stay as committed.
