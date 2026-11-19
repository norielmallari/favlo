# Coding agent, vision, CUDA

Measurement note for the CUDA runtime of `qwen35_4b`. The serving contract is [../ARCHITECTURE.md](../ARCHITECTURE.md). This file records vision coding-agent runs on int8 keys and values with MTP off. It is a recorded run, not a format bound. The text runs are in [coding-text-cuda.md](coding-text-cuda.md) and [coding-instruct-cuda.md](coding-instruct-cuda.md).

## Remainder

On a GeForce RTX 3070 Laptop GPU with 8192 MiB, `Qwen3.5-4B` with int8 keys and values and MTP off passed the vision coding agent. Thinking on recorded 424.58 tok/s prefill over 4543 tokens and 48.76 tok/s decode over 1434 tokens. Thinking off recorded 423.92 tok/s prefill over 4129 tokens and 41.83 tok/s decode over 545 tokens. A search on 19 November 2026 did not find that pair.

## Predecessors

- Method. The ledger writes `ledger.py` through the coding tools, restarts serve so a later turn loads the prefix from disk, then checks the chat contract. Exit 0 requires `collect` to return 6, 0, and -6. Which tools were called is not the result. The still is the seeded horse.
- Model. [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B). The card publishes vision scores and says throughput varies by framework. It does not publish this harness rate.
- Rate. [Will It Run AI](https://willitrunai.com/can-run/qwen-3.5-4b-on-rtx-3070-8gb) estimates about 55.9 tok/s decode and 3863 tok/s prefill for Q4_K_M on an RTX 3070 8GB. [A second estimate](https://willitrunai.com/can-run/hf-unsloth--qwen3-5-4b-gguf-on-rtx-3070-8gb) puts Q4_K_M decode near 48.0 tok/s. [GPUfit](https://gpufit.tech/can-i-run/qwen3-5-4b/on/rtx-3070/) estimates 40–54 tok/s for UD-Q8_K_XL and labels that estimate as not a benchmark. [An RTX 5060 Ti llama.cpp run](https://dev.to/elcances/qwen35-4b-on-a-16-gb-rtx-5060-ti-823-tokens-per-second-for-16-users-6i) records 5050 tok/s prompt processing and 114 tok/s generation at 512 tokens. [oMLX Qwen3.5-4B-4bit on an M4](https://omlx.ai/benchmarks/performance/q4k741pp) records 433.7 tok/s prefill and 36.0 tok/s decode at 4096 tokens. The Metal vision coding agent, in [coding-agent.md](coding-agent.md), recorded 1416.98 / 27.46 tok/s thinking and 1414.22 / 27.24 tok/s instruct on an M5 Pro with `fp8_e4m3`. None of those is this int8 vision table.

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
| on | on | pass | 424.58 tok/s (4543 tok in 10.70 s) | 48.76 tok/s (1434 tok in 29.41 s) |
| off | on | pass | 423.92 tok/s (4129 tok in 9.74 s) | 41.83 tok/s (545 tok in 13.03 s) |

## Search

19 November 2026. Queries: "424.58 OR 48.76 OR 423.92 OR 41.83 Qwen3.5-4B vision coding agent tok/s", "Qwen3.5-4B RTX 3070 tokens per second int8 vision benchmark 2026", and "Qwen3.5-4B int8 vision coding agent prefill decode tok/s GeForce RTX 3070 424.58 48.76". The predecessors in this note are the overlaps those queries returned. The search did not find 424.58 and 48.76, or 423.92 and 41.83, as this vision coding agent.

## Corrections

Later finds are appended here. The sections above stay as committed.
