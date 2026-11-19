# Coding agent, text

Measurement note for the CUDA runtime of `qwen36-35b-a3b`. The serving contract is [../README.md](../ARCHITECTURE.md). This file records one text coding-agent run on int8 keys and values with MTP off. It is a recorded run, not a format bound. The Metal vision runs are in [../../research/coding-agent.md](../../research/coding-agent.md).

## Remainder

On a GeForce RTX 3070 Laptop GPU with 8192 MiB, `Qwen3.6-35B-A3B` with int8 keys and values and MTP off passed the text coding agent. Thinking on, vision off, recorded 9.95 tok/s prefill over 2908 tokens and 5.17 tok/s decode over 980 tokens. A search on 7 November 2026 did not find that pair.

## Predecessors

- Method. The ledger writes `ledger.py` through the coding tools, restarts serve so the next turn loads the prefix from disk, then checks the chat contract. Exit 0 requires `collect` to return 6, 0, and -6. Which tools were called is not the result.
- Model. [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). The card reports SWE-bench Verified 73.4 for this checkpoint. That score is not this harness rate.
- Rate. [llm-bench](https://llm-bench.io/benchmarks/cmvjm53mf01ql01mw91cy60wr) records 30.5 tok/s and 285.7 prompt tok/s for UD-IQ4_XS under llama.cpp on a GeForce RTX 3070. A [llama.cpp sweep](https://note.com/ripe_hosta6920/n/nb81b72d54ad7) on an RTX 3070 8GB records about 517 tok/s prompt processing at 512 tokens and about 27 tok/s generation at 128 tokens for the same UD-IQ4_XS file. [One MTP writeup](https://johnpaulwile.substack.com/p/running-qwen36-35b-a3b-mtp-at-55) reports about 55 tok/s for Q4_K_XL with MTP on an RTX 3070 8GB. [Colibri](https://huggingface.co/Kreuzzelg/qwen36-35b-a3b-colibri-i4) records 9.9 tok/s decode for an int4 container on one 8GB GPU. The Metal vision coding agent, in [coding-agent.md](../../research/coding-agent.md), recorded 632.95 tok/s prefill and 68.72 tok/s decode on an M5 Pro with `fp8_e4m3`. None of those is this int8 text row.

## Rule

MTP is off. Thinking on uses the thinking card. The banner prints the loaded plane. This row is the run whose banner printed `kv int8`, `mtp off`, `thinking on`, and `vision off`. A banner that prints `fp8_e4m3` or `vision on` is a different row. Request 2 loaded the prefix from disk.

## Command

```text
uv run python -m harness.coding --model qwen36-35b-a3b --thinking on
```

The banner prints the loaded plane, MTP, thinking, and vision.

## Table

| Thinking | Vision | Result | Prefill | Decode |
|---|---|---|---|---|
| on | off | pass | 9.95 tok/s (2908 tok in 292.23 s) | 5.17 tok/s (980 tok in 189.58 s) |

## Search

7 November 2026. Queries: "Qwen3.6-35B-A3B tokens per second CUDA RTX 3070 int8 benchmark tok/s", "Qwen3.6-35B-A3B tok/s OR tokens per second prefill decode", and "9.95 OR 5.17 Qwen3.6-35B-A3B tok/s coding agent 2908". The predecessors in this note are the overlaps those queries returned. The search did not find 9.95 prefill and 5.17 decode as this int8 text coding agent.

## Corrections

Later finds are appended here. The sections above stay as committed.

8 November 2026. The same banner passed again after leftover device memory began holding the expert images last used. Prefill summed to 18.14 tok/s over 3043 tokens and decode to 5.42 tok/s over 769 tokens. Query: "18.14 OR 5.42 Qwen3.6-35B-A3B tok/s int8 coding 3043". The search did not find that pair as this int8 text coding agent. A CPU-only run reports 5.40 tok/s generation for a GGUF of this model. That is not this prefill and decode pair. The predecessors above still stand.

19 November 2026. The same banner passed again after each device copy finished before its read buffer was filled again, and after one warp updated each Gated DeltaNet state row as a contiguous vector. Prefill summed to 200.12 tok/s over 3288 tokens and decode to 7.56 tok/s over 1058 tokens. Query: "200.12 OR 7.56 Qwen3.6-35B-A3B tok/s int8 coding 3288". The search did not find that pair as this int8 text coding agent. Published rates for other quantizations and other GPUs are not this pair. The predecessors above still stand.
