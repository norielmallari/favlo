# Coding agent, instruct

Measurement note for the CUDA runtime of `qwen36-35b-a3b`. The serving contract is [../README.md](../ARCHITECTURE.md). This file records one text coding-agent run on int8 keys and values with MTP off and thinking off. It is a recorded run, not a format bound. The thinking-on text run is in [coding-text.md](coding-text.md). The Metal vision runs are in [../../research/coding-agent.md](../../research/coding-agent.md).

## Remainder

On a GeForce RTX 3070 Laptop GPU with 8192 MiB, `Qwen3.6-35B-A3B` with int8 keys and values and MTP off passed the text coding agent. Thinking off, vision off, recorded 9.68 tok/s prefill over 3134 tokens and 5.22 tok/s decode over 684 tokens. A search on 7 November 2026 did not find that pair.

## Predecessors

- Method. The ledger writes `ledger.py` through the coding tools, restarts serve so the next turn loads the prefix from disk, then checks the chat contract. Exit 0 requires `collect` to return 6, 0, and -6. Which tools were called is not the result.
- Model. [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). The card reports SWE-bench Verified 73.4 for this checkpoint. That score is not this harness rate.
- Rate. [llm-bench](https://llm-bench.io/benchmarks/cmvjm53mf01ql01mw91cy60wr) records 30.5 tok/s and 285.7 prompt tok/s for UD-IQ4_XS under llama.cpp on a GeForce RTX 3070. A [llama.cpp sweep](https://note.com/ripe_hosta6920/n/nb81b72d54ad7) on an RTX 3070 8GB records about 517 tok/s prompt processing at 512 tokens and about 27 tok/s generation at 128 tokens for the same UD-IQ4_XS file. [One MTP writeup](https://johnpaulwile.substack.com/p/running-qwen36-35b-a3b-mtp-at-55) reports about 55 tok/s for Q4_K_XL with MTP on an RTX 3070 8GB. [Colibri](https://huggingface.co/Kreuzzelg/qwen36-35b-a3b-colibri-i4) records 9.9 tok/s decode for an int4 container on one 8GB GPU. The Metal vision coding agent, in [coding-agent.md](../../research/coding-agent.md), recorded 671.74 tok/s prefill and 69.33 tok/s decode on an M5 Pro with `fp8_e4m3` and thinking off. The thinking-on text row is in [coding-text.md](coding-text.md). None of those is this int8 instruct row.

## Rule

MTP is off. Thinking off uses the instruct card. The banner prints the loaded plane. This row is the run whose banner printed `kv int8`, `mtp off`, `thinking off`, and `vision off`. A banner that prints `thinking on`, `fp8_e4m3`, or `vision on` is a different row. Request 2 loaded the prefix from disk.

## Command

```text
uv run python -m harness.coding --model qwen36-35b-a3b --thinking off
```

The banner prints the loaded plane, MTP, thinking, and vision.

## Table

| Thinking | Vision | Result | Prefill | Decode |
|---|---|---|---|---|
| off | off | pass | 9.68 tok/s (3134 tok in 323.76 s) | 5.22 tok/s (684 tok in 131.13 s) |

## Search

7 November 2026. Queries: "Qwen3.6-35B-A3B tokens per second CUDA RTX 3070 int8 benchmark tok/s", "Qwen3.6-35B-A3B tok/s OR tokens per second prefill decode", and "9.68 OR 5.22 Qwen3.6-35B-A3B tok/s int8 coding instruct 3134". The predecessors in this note are the overlaps those queries returned. The search did not find 9.68 prefill and 5.22 decode as this int8 instruct coding agent.

## Corrections

Later finds are appended here. The sections above stay as committed.

8 November 2026. The same banner passed again after leftover device memory began holding the expert images last used, each at its own size. Prefill summed to 21.76 tok/s over 3052 tokens and decode to 6.26 tok/s over 812 tokens. Query: "21.76 OR 6.26 Qwen3.6-35B-A3B tok/s int8 coding instruct 3052". The search did not find that pair as this int8 instruct coding agent. The predecessors above still stand.
