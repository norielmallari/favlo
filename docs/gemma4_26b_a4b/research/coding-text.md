# Coding agent, text

Measurement note for the `gemma4_26b_a4b` pipeline. The serving contract is [../README.md](../README.md). This file records one text coding-agent run on `fp8_e4m3` with MTP off. It is a recorded run, not a format bound. The vision runs are in [coding-agent.md](coding-agent.md).

## Remainder

On an M5 Pro with 48 GB, `gemma-4-26B-A4B-it` with `fp8_e4m3` keys and values and MTP off passed the text coding agent. Thinking on, vision off, recorded 836.90 tok/s prefill and 25.94 tok/s decode. A search on 3 November 2026 did not find that pair.

## Predecessors

- Method. The ledger writes `ledger.py` through the coding tools, restarts serve so the next turn loads the prefix from disk, then checks the chat contract. Exit 0 requires `collect` to return 6, 0, and -6. Which tools were called is not the result.
- Model. [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it). The card describes thinking, tool use, and a 256K context. It does not publish this harness rate.
- Rate. [oMLX gemma-4-26b-a4b-it-4bit on an M5 Pro with 48 GB](https://omlx.ai/benchmarks/performance/gv7qn8l9) reports 2,082 tok/s prefill and 76.9 tok/s decode at 1,024 tokens. [Gemma 4 26B-A4B vLLM FP8 on DGX Spark](https://gauravmm.github.io/autobench/configs/gemma-4-26b-a4b-it-vllm-fp8/) reports 362.64 tok/s prefill and 316.93 tok/s decode at concurrency 32. [Caleb Coffie](https://calebcoffie.com/benchmarks/gemma-4-26b-a4b-it-gguf) reports 1,229.9 tok/s prefill and 116.3 tok/s decode for an agent-shaped prompt on one RTX 3090. The vision coding agent on the earlier attention, in [coding-agent.md](coding-agent.md), recorded 274.02 tok/s prefill and 8.49 tok/s decode. None of those is this text row.

## Rule

MTP is off. Thinking on uses the thinking card. The banner prints the loaded plane. This row is the run whose banner printed `kv fp8_e4m3`, `mtp off`, `thinking on`, and `vision off`. A banner that prints `half` or `vision on` is a different row. Request 2 loaded the prefix from disk.

## Command

```text
uv run python -m harness.coding --model gemma4-26b-a4b --thinking on
```

The banner prints the loaded plane, MTP, thinking, and vision.

## Table

| Thinking | Vision | Result | Prefill | Decode |
|---|---|---|---|---|
| on | off | pass | 836.90 tok/s (2268 tok in 2.71 s) | 25.94 tok/s (429 tok in 16.54 s) |

## Search

3 November 2026. Queries: "836.90 OR 25.94 Gemma 4 26B coding agent fp8_e4m3 vision ledger tok/s", and "836.90 OR 25.94 OR 46.954 OR 1522.48 gemma-4-26B-A4B tok/s". The predecessors in this note are the overlaps those queries returned. The search did not find 836.90 and 25.94 in those sources.

## Corrections

Later finds are appended here. The sections above stay as committed.
