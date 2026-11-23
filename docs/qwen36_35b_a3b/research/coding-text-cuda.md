# Coding agent, text, CUDA

Measurement note for the CUDA runtime of `qwen36_35b_a3b`. The serving contract is [../ARCHITECTURE.md](../ARCHITECTURE.md). This file records text coding-agent runs on the int4 pack with int8 keys and values and MTP off. It is a recorded run, not a format bound.

## Remainder

On a GeForce RTX 3070 Laptop GPU with 8192 MiB, `Qwen3.6-35B-A3B-int4` with int8 keys and values and MTP off passed the text coding agent. Thinking on recorded 184.06 tok/s prefill over 2932 tokens and 12.42 tok/s decode over 1428 tokens. Thinking off recorded 187.49 tok/s prefill over 3043 tokens and 11.75 tok/s decode over 1505 tokens. A search on 23 November 2026 did not find that pair.

## Predecessors

- Method. The ledger writes `ledger.py` through the coding tools, restarts serve so a later turn loads the prefix from disk, then checks the chat contract. Exit 0 requires `collect` to return 6, 0, and -6. Which tools were called is not the result.
- Model. [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). The card reports SWE-bench Verified 73.4 for this checkpoint. That score is not this harness rate.
- Rate. [llm-bench](https://llm-bench.io/benchmarks/cmvjm53mf01ql01mw91cy60wr) records 30.5 tok/s and 285.7 prompt tok/s for UD-IQ4_XS under llama.cpp on a GeForce RTX 3070. [An RTX 3060 llama.cpp sweep](https://smeltcore.com/recipes/qwen3-6-35b-a3b-on-rtx-3060-38-9-tok-s-from-a-12gb-card-via-moe-expert-cpu-offload/) records 413 tok/s prompt processing and 38.9 tok/s generation for UD-Q4_K_M with expert offload. [One MTP writeup](https://johnpaulwile.substack.com/p/running-qwen36-35b-a3b-mtp-at-55) reports about 55 tok/s for Q4_K_XL with MTP on an RTX 3070 8GB. [LLM Benchmark](https://llm-benchmark.de/modell/qwen3-6-35b-a3b) records 184.04 combined tok/s for UD-Q4_K_M on five Radeon PRO W7900 cards. The Metal text coding agent, in [coding-text.md](coding-text.md), recorded 539.66 / 65.19 tok/s thinking and 492.10 / 66.72 tok/s instruct on an M5 Pro with `half`. None of those is this int4 text table.

## Rule

MTP is off. Thinking on uses the thinking card. Thinking off uses the instruct card. The banner prints the loaded plane. This table is the pair whose banners printed `kv int8`, `mtp off`, and `vision off`. A banner that prints `vision on` or `half` is a different row. A later request loaded the prefix from disk.

## Command

```text
uv run python -m harness.coding --model qwen36-35b-a3b --thinking on
uv run python -m harness.coding --model qwen36-35b-a3b --thinking off
```

The banner prints the loaded plane, MTP, thinking, and vision.

## Table

| Thinking | Vision | Result | Prefill | Decode |
|---|---|---|---|---|
| on | off | pass | 184.06 tok/s (2932 tok in 15.93 s) | 12.42 tok/s (1428 tok in 114.97 s) |
| off | off | pass | 187.49 tok/s (3043 tok in 16.23 s) | 11.75 tok/s (1505 tok in 128.05 s) |

## Search

23 November 2026.

- Queries: "184.06 OR 12.42 OR 187.49 OR 11.75 Qwen3.6-35B-A3B coding agent tok/s RTX 3070", and "184.06 OR 187.49 OR 12.42 Qwen3.6-35B-A3B prompt processing int8 coding agent".
- The predecessors in this note are the overlaps those queries returned.
- The search did not find 184.06 and 12.42, or 187.49 and 11.75, as this text coding agent.

## Corrections

Later finds are appended here. The sections above stay as committed.
