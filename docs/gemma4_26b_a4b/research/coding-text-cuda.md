# Coding agent, text

Measurement note for the CUDA runtime of `gemma4-26b-a4b`. The serving contract is [../ARCHITECTURE.md](../ARCHITECTURE.md). This file records one text coding-agent run on half keys and values with MTP off. It is a recorded run, not a format bound. The Metal text run is in [coding-text.md](coding-text.md).

## Remainder

On a GeForce RTX 3070 Laptop GPU with 8192 MiB, `Gemma 4 26B-A4B` with int8 dense weights, int8 for 10% of the experts and int4 for the rest, half keys and values, and MTP off passed the text coding agent. Thinking on and vision off recorded 15.49 tok/s prefill over 2436 tokens and 5.47 tok/s decode over 485 tokens. A search on 19 November 2026 did not find that pair.

## Predecessors

- Method. The ledger writes `ledger.py` through the coding tools, restarts serve so the next turn loads the prefix from disk, then checks the chat contract. Exit 0 requires `collect` to return 6, 0, and -6. Which tools were called is not the result.
- Model. [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it). The card publishes task scores. It does not publish this harness rate.
- Rate. [GPUfit](https://gpufit.tech/can-i-run/gemma-4-26b-a4b-it/on/rtx-3070/) estimates 18–25 tok/s for UD-Q4_K_M with expert offload on an RTX 3070 and says that figure is not a benchmark. [llm-bench.io](https://llm-bench.io/benchmarks/cmusmf3af00p601oxfmjf6ihe) records 57.7 tok/s decode and 570.7 tok/s prefill for UD-Q4_K_XL on an RTX 3070 Ti. [Caleb Coffie](https://calebcoffie.com/benchmarks/gemma-4-26b-a4b-it-gguf) records 116.3 tok/s decode and 1229.9 tok/s prefill for an agent shape of UD-Q4_K_M on an RTX 3090. The Metal text coding agent, in [coding-text.md](coding-text.md), recorded 836.90 tok/s prefill and 25.94 tok/s decode on an M5 Pro with `fp8_e4m3`. None of those is this int8-e10 half row.

## Rule

MTP is off. Thinking on uses the thinking card. The banner prints the loaded plane. This row is the run whose banner printed `kv half`, `mtp off`, `thinking on`, and `vision off`. A banner that prints `fp8_e4m3` or `vision on` is a different row. Request 2 loaded the prefix from disk.

## Command

```text
uv run python -m harness.coding --model gemma4_26b_a4b --thinking on
```

The banner prints the loaded plane, MTP, thinking, and vision.

## Table

| Thinking | Vision | Result | Prefill | Decode |
|---|---|---|---|---|
| on | off | pass | 15.49 tok/s (2436 tok in 157.25 s) | 5.47 tok/s (485 tok in 88.64 s) |

## Search

19 November 2026. Queries: "Gemma 4 26B-A4B tokens per second CUDA RTX 3070 benchmark tok/s" and "15.49 OR 5.47 Gemma 4 26B tok/s coding agent int8". The predecessors in this note are the overlaps those queries returned. The search did not find 15.49 and 5.47 as this text coding agent.

## Corrections

Later finds are appended here. The sections above stay as committed.

20 November 2026. The same command passed again, and the instruct card passed in the same session. Both banners printed `kv half`, `mtp off`, and `vision off`. Request 2 loaded the prefix from disk on each run. The sums:

| Thinking | Vision | Result | Prefill | Decode |
|---|---|---|---|---|
| on | off | pass | 178.46 tok/s (2436 tok in 13.65 s) | 9.42 tok/s (504 tok in 53.52 s) |
| off | off | pass | 167.31 tok/s (2411 tok in 14.41 s) | 9.87 tok/s (477 tok in 48.32 s) |

The instruct command is `uv run python -m harness.coding --model gemma4_26b_a4b --thinking off`.

Queries that day: "178.46 OR 9.42 OR 167.31 Gemma 4 26B coding agent tok/s int8". The search did not find 178.46 / 9.42 or 167.31 / 9.87 as this int8-e10 half text coding agent. [LocalAgentBoard](https://localagentboard.com/models/gemma-4-26b-a4b/) records a Hermes agent on Q8_0 at 2625.5 tok/s prefill and 46.2 tok/s decode on a DGX Spark. [conorseabrook/gemma4-turboquant-bench](https://github.com/conorseabrook/gemma4-turboquant-bench) records about 129 tok/s generation and about 5600 tok/s prefill for a TurboQuant agent on an RTX 4090. Neither is this 8192 MiB row.

20 November 2026. The same two cards passed after every routed expert was requantized to int4 from the BF16 snapshot. Dense matrices stayed int8. The packed folder name stayed `gemma-4-26B-A4B-it-int8-e10`. Both banners printed `kv half`, `mtp off`, and `vision off`. Request 2 loaded the prefix from disk: 1652 tokens with thinking on, 1650 tokens with thinking off.

| Thinking | Vision | Result | Prefill | Decode |
|---|---|---|---|---|
| on | off | pass | 191.36 tok/s (2436 tok in 12.73 s) | 12.38 tok/s (518 tok in 41.84 s) |
| off | off | pass | 172.34 tok/s (2411 tok in 13.99 s) | 11.63 tok/s (485 tok in 41.70 s) |

Queries that day: "191.36 OR 12.38 OR 172.34 Gemma 4 26B coding agent tok/s int4" and "Gemma 4 26B-A4B tokens per second CUDA RTX 3070 benchmark tok/s int4". The search did not find 191.36 / 12.38 or 172.34 / 11.63 as this half text coding agent. [Towards AI](https://pub.towardsai.net/deploying-gemma-4-26b-a4b-on-an-rtx-5090-69dacb0c116d) records about 195.8 tok/s decode for an AWQ int4 coding endpoint on an RTX 5090. [Intelinside.ai](https://www.intelinside.ai/results/61) records 95.3 tok/s decode for an OpenVINO int4 run on an Arc Pro B70. Neither is this 8192 MiB row.

20 November 2026. The same two cards passed again after decode attention split the key span across the multiprocessors. Dense matrices stayed int8 and every routed expert stayed int4. Both banners printed `kv half`, `mtp off`, and `vision off`. Request 2 loaded the prefix from disk: 1652 tokens with thinking on, 1650 tokens with thinking off.

| Thinking | Vision | Result | Prefill | Decode |
|---|---|---|---|---|
| on | off | pass | 188.69 tok/s (2436 tok in 12.91 s) | 24.94 tok/s (441 tok in 17.68 s) |
| off | off | pass | 187.77 tok/s (2411 tok in 12.84 s) | 25.84 tok/s (478 tok in 18.50 s) |

Queries that day: "188.69 OR 24.94 OR 187.77 OR 25.84 Gemma 4 26B coding agent tok/s" and "Gemma 4 26B-A4B coding agent tok/s RTX 3070 int4 2026". The search did not find 188.69 / 24.94 or 187.77 / 25.84 as this half text coding agent. [DEV Community](https://dev.to/gde/can-a-26b-moe-run-on-an-8-gb-gpu-gemma-4-26b-a4b-on-an-rtx-3050-2efn) records 33.7 tok/s generation with an empty cache and 22.8 tok/s with 65,536 tokens held for Gemma 4 26B-A4B on an RTX 3050. That is not this 8192 MiB row.
