# Coding agent, instruct

Measurement note for the CUDA runtime of `qwen35-4b`. The serving contract is [../README.md](../ARCHITECTURE.md). This file records one text coding-agent run on int8 keys and values with MTP off and thinking off. It is a recorded run, not a format bound. The thinking-on text run is in [coding-text.md](coding-text.md). The Metal vision runs are in [../../metal/research/coding-agent.md](../../metal/research/coding-agent.md).

## Remainder

On a GeForce RTX 3070 Laptop GPU with 8192 MiB, `Qwen3.5-4B` with int8 keys and values and MTP off passed the text coding agent. Thinking off, vision off, recorded 1402.56 tok/s prefill over 2735 tokens and 37.31 tok/s decode over 444 tokens. A search on 7 November 2026 did not find that pair.

## Predecessors

- Method. The ledger writes `ledger.py` through the coding tools, restarts serve so the next turn loads the prefix from disk, then checks the chat contract. Exit 0 requires `collect` to return 6, 0, and -6. Which tools were called is not the result.
- Model. [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B). The card publishes task scores and says throughput varies by framework. It does not publish this harness rate.
- Rate. [ModelFit](https://modelfit.io/models/qwen3.5-4b) reports 93.0 tok/s for Qwen3.5 4B Q4_K_M under Ollama on a GeForce RTX 3070. [Will It Run AI](https://willitrunai.com/can-run/qwen-3.5-4b-on-rtx-3070-8gb) estimates about 55.9 tok/s decode and 3863 tok/s prefill for Q4_K_M on an RTX 3070 8GB. [GPUfit](https://gpufit.tech/can-i-run/qwen3-5-4b/on/rtx-3070/) estimates 40–54 tok/s for UD-Q8_K_XL and labels that estimate as not a benchmark. [DiggTop](https://www.diggtop.com/models/qwen3-5-4b) records 2111.9 tok/s prompt processing and 78.6 tok/s generation for Q4_K_M under llama.cpp on an RTX 3060 at a 32768-token context. [An RTX 4060 Laptop llama-bench](https://www.insidentally.com/articles/000051/) records 2998 t/s for pp512 and 69.0 t/s for tg128. The Metal vision coding agent, in [coding-agent.md](../../metal/research/coding-agent.md), recorded 1414.22 tok/s prefill and 27.24 tok/s decode on an M5 Pro with `fp8_e4m3` and thinking off. None of those is this int8 text row.

## Rule

MTP is off. Thinking off uses the instruct card. The banner prints the loaded plane. This row is the run whose banner printed `kv int8`, `mtp off`, `thinking off`, and `vision off`. A banner that prints `thinking on`, `fp8_e4m3`, or `vision on` is a different row. Request 2 loaded the prefix from disk.

## Command

```text
uv run python -m harness.coding --model qwen35-4b --thinking off
```

The banner prints the loaded plane, MTP, thinking, and vision.

## Table

| Thinking | Vision | Result | Prefill | Decode |
|---|---|---|---|---|
| off | off | pass | 1402.56 tok/s (2735 tok in 1.95 s) | 37.31 tok/s (444 tok in 11.90 s) |

## Search

7 November 2026. Queries: "Qwen3.5-4B tokens per second CUDA RTX 3070 int8 instruct coding agent tok/s 2026", "1402.56 OR 37.31 Qwen3.5-4B tok/s coding", "Qwen3.5-4B model card throughput tokens per second framework", and "Qwen3.5-4B prompt processing pp tok/s GeForce RTX 3070 agentic coding instruct llama.cpp". The predecessors in this note are the overlaps those queries returned. The search did not find 1402.56 and 37.31 in those sources.

## Corrections

Later finds are appended here. The sections above stay as committed.
