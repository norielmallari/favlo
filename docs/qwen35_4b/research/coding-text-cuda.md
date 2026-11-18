# Coding agent, text

Measurement note for the CUDA runtime of `qwen35-4b`. The serving contract is [../README.md](../ARCHITECTURE.md). This file records one text coding-agent run on int8 keys and values with MTP off. It is a recorded run, not a format bound. The Metal vision runs are in [../../metal/research/coding-agent.md](../../metal/research/coding-agent.md).

## Remainder

On a GeForce RTX 3070 Laptop GPU with 8192 MiB, `Qwen3.5-4B` with int8 keys and values and MTP off passed the text coding agent. Thinking on, vision off, recorded 18.23 tok/s prefill over 3880 tokens and 9.97 tok/s decode over 3323 tokens. A search on 6 November 2026 did not find that pair.

## Predecessors

- Method. The ledger writes `ledger.py` through the coding tools, restarts serve so the next turn loads the prefix from disk, then checks the chat contract. Exit 0 requires `collect` to return 6, 0, and -6. Which tools were called is not the result.
- Model. [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B). The card publishes task scores and says throughput varies by framework. It does not publish this harness rate.
- Rate. [ModelFit](https://modelfit.io/models/qwen3.5-4b) reports 93.0 tok/s for Qwen3.5 4B Q4_K_M under Ollama on a GeForce RTX 3070. [Will It Run AI](https://willitrunai.com/can-run/qwen-3.5-4b-on-rtx-3070-8gb) estimates about 55.9 tok/s decode for Q4_K_M on an RTX 3070 8GB. [GPUfit](https://gpufit.tech/can-i-run/qwen3-5-4b/on/rtx-3070/) estimates 40–54 tok/s for UD-Q8_K_XL and labels that estimate as not a benchmark. The Metal vision coding agent, in [coding-agent.md](../../metal/research/coding-agent.md), recorded 1416.98 tok/s prefill and 27.46 tok/s decode on an M5 Pro with `fp8_e4m3`. None of those is this int8 text row.

## Rule

MTP is off. Thinking on uses the thinking card. The banner prints the loaded plane. This row is the run whose banner printed `kv int8`, `mtp off`, `thinking on`, and `vision off`. A banner that prints `fp8_e4m3` or `vision on` is a different row. Request 2 loaded the prefix from disk.

## Command

```text
uv run python -m harness.coding --model qwen35-4b --thinking on
```

The banner prints the loaded plane, MTP, thinking, and vision.

## Table

| Thinking | Vision | Result | Prefill | Decode |
|---|---|---|---|---|
| on | off | pass | 18.23 tok/s (3880 tok in 212.84 s) | 9.97 tok/s (3323 tok in 333.21 s) |

## Search

6 November 2026. Queries: "Qwen3.5-4B tokens per second CUDA RTX 3070 int8 benchmark tok/s", "18.23 OR 9.97 Qwen3.5-4B tok/s coding", and "Qwen3.5-4B 93.0 OR 18.23 tok/s OR 3880 tok RTX 3070 coding agent". The predecessors in this note are the overlaps those queries returned. The search did not find 18.23 and 9.97 in those sources.

## Corrections

Later finds are appended here. The sections above stay as committed.

6 November 2026. The same banner, after prefill became a 16-token `__dp4a` tile, passed again. Prefill summed to 57.42 tok/s over 2946 tokens and decode to 11.91 tok/s over 1167 tokens. Query: "57.42 OR 11.91 Qwen3.5-4B tok/s int8 coding". The search did not find that pair as this int8 text coding agent. The predecessors above still stand.

7 November 2026. The same banner passed again. Prefill summed to 1357.62 tok/s over 2851 tokens and decode to 50.61 tok/s over 1486 tokens. Queries: "1357.62 OR 50.61 Qwen3.5-4B tok/s int8 coding" and "Qwen3.5-4B text coding agent prefill decode tok/s GeForce RTX 3070 int8 thinking on 2851 tok". The search did not find that pair. [LiteRT](https://huggingface.co/litert-community/Qwen3.5-4B) records 672 tok/s prefill and 62.0 tok/s decode for an int8 graph on an M4 Max at 256 tokens. That is not this text coding agent. The predecessors above still stand.
