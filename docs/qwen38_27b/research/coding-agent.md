# Coding agent

Measurement note for the `qwen38_27b` pipeline. The serving contract is [../README.md](../README.md). This file records coding-agent runs on `fp8_e4m3` with MTP off. It is a recorded run, not a format bound. Earlier mixed-record agent sums are [mixed-agent-forward.md](mixed-agent-forward.md).

## Remainder

On an M5 Pro with 48 GB, `Qwen3.8-27B-fp8-e20` with `fp8_e4m3` keys and values and MTP off passed the coding agent on both thinking cards, with and without the horse still. Prefill summed between 295.35 and 322.56 tok/s. Decode summed between 14.01 and 14.35 tok/s. A search on 3 November 2026 did not find that table.

## Predecessors

- Method. The ledger writes `ledger.py` through the coding tools, restarts serve so the next turn loads the prefix from disk, then checks the chat contract. Exit 0 requires `collect` to return 6, 0, and -6. Which tools were called is not the result. With `--vision`, the still is the first chat.
- Model. [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). The card does not publish this harness rate table.
- Rate. [llm-bench](https://llm-bench.io/benchmarks/cmtdr2v3y00c201noaeefqfp5) reports 17.1 tok/s for a coding HTML game on an M5 Pro with 48 GB under LM Studio. [oMLX](https://omlx.ai/benchmarks/performance/e78ir7ct) reports 251.5 tok/s prefill and 17.7 tok/s decode at 64k with Lightning MTP on an M5 Pro with 20 GPU cores and 48 GB. [Billy Xu](https://billyxu.dev/notes/local-model-lab-3/) reports 32.9 tok/s on a short prompt and 30.0 tok/s at 25k tokens with Lightning MTP on an M5 Pro with 48 GB. [kutaelee/qwen38-5090-128k-runtime-recipe](https://github.com/kutaelee/qwen38-5090-128k-runtime-recipe) reports about 171 tok/s aggregate decode on a Hermes coding agent on one RTX 5090. None of those runs is this harness table on `fp8_e4m3` with MTP off.

## Rule

MTP is off. Thinking on uses the thinking card. Thinking off uses the instruct card. `--vision` sends the horse still as the first chat with `tool_choice` `none`, then the ledger on that session. The banner prints the loaded plane. This table is the runs whose banners printed `kv fp8_e4m3` and `mtp off`. A banner that prints `half` or `mtp on` is a different row. Request 2 of each vision run loaded the prefix from disk.

## Command

```text
uv run python -m harness.coding --model qwen38-27b --thinking on
uv run python -m harness.coding --model qwen38-27b --thinking off
uv run python -m harness.coding --model qwen38-27b --vision --thinking on
uv run python -m harness.coding --model qwen38-27b --vision --thinking off
```

The banner prints the loaded plane, MTP, thinking, and vision.

## Table

| Thinking | Vision | Result | Prefill | Decode |
|---|---|---|---|---|
| on | off | pass | 307.18 tok/s (2654 tok in 8.64 s) | 14.35 tok/s (729 tok in 50.80 s) |
| off | off | pass | 322.56 tok/s (2574 tok in 7.98 s) | 14.25 tok/s (357 tok in 25.06 s) |
| on | on | pass | 295.35 tok/s (2794 tok in 9.46 s) | 14.09 tok/s (926 tok in 65.71 s) |
| off | on | pass | 298.47 tok/s (2928 tok in 9.81 s) | 14.01 tok/s (853 tok in 60.87 s) |

## Search

3 November 2026. Queries: "295.35 OR 307.18 OR 322.56 OR 298.47 Qwen3.8-27B coding agent tok/s fp8_e4m3 vision", and "Qwen3.8-27B coding agent ledger vision horse M5 Pro 48GB MTP off tok/s". The predecessors in this note are the overlaps those queries returned. The search did not find 307.18 and 14.35, or 322.56 and 14.25, or 295.35 and 14.09, or 298.47 and 14.01, in those sources.

## Corrections

Later finds are appended here. The sections above stay as committed.

### 4 November 2026 — vision × thinking × store

`--store off` is the harness default. `--store on` sends `store` true for the ledger and the contract. Vision stayed on. MTP stayed off. Prefill summed between 299.89 and 316.78 tok/s. Decode summed between 14.16 and 14.36 tok/s. A search the same day did not find that table. Queries: "299.89 OR 315.96 OR 316.78 OR 301.51 Qwen3.8-27B coding agent vision store tok/s", and "Qwen3.8-27B coding agent ledger vision horse store on M5 Pro 48GB". Closest overlaps were [an RTX PRO 6000 coding-agent decode](https://safaia.net/hanxiao/Qwen3.8-27B-abliterated-AWQ-SGLang-DFlash2-RTX-PRO-6000) at about 220 tok/s and [an NVFP4 vision build](https://huggingface.co/dbrasdasilva/Qwen3.8-27B-Text-NVFP4-MTP/raw/main/README.md) at about 77 tok/s decode with MTP off. Neither is this `fp8_e4m3` table.

```text
uv run python -m harness.coding --model qwen38-27b --vision --thinking on --store off
uv run python -m harness.coding --model qwen38-27b --vision --thinking off --store off
uv run python -m harness.coding --model qwen38-27b --vision --thinking on --store on
uv run python -m harness.coding --model qwen38-27b --vision --thinking off --store on
```

| Thinking | Store | Result | Prefill | Decode |
|---|---|---|---|---|
| on | off | pass | 299.89 tok/s (2813 tok in 9.38 s) | 14.24 tok/s (781 tok in 54.83 s) |
| off | off | pass | 315.96 tok/s (2831 tok in 8.96 s) | 14.33 tok/s (811 tok in 56.61 s) |
| on | on | pass | 316.78 tok/s (2775 tok in 8.76 s) | 14.36 tok/s (698 tok in 48.61 s) |
| off | on | pass | 301.51 tok/s (2795 tok in 9.27 s) | 14.16 tok/s (468 tok in 33.05 s) |
