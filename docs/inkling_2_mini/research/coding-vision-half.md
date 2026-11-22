# Coding agent, vision, half cache

Measurement note for the `inkling_2_mini` pipeline. The serving contract is [../ARCHITECTURE.md](../ARCHITECTURE.md). This file records one vision coding-agent run on `Inkling-2-Mini` with the GPTQ dense fp4 pack and `half` keys and values. It is a recorded run, not a format bound.

## Remainder

- On an M5 Pro with 48 GB, `Inkling-2-Mini` with the GPTQ dense fp4 pack, `half` keys and values, and MTP off passed the vision coding agent.
- Thinking on, vision on, recorded 769.97 tok/s prefill and 65.07 tok/s decode.
- A search on 22 November 2026 did not find that table.

## Predecessors

- Method. The ledger writes `ledger.py` through the coding tools, unloads the model so a later turn loads the prefix from disk, then checks the chat contract. Exit 0 requires `collect` to return 6, 0, and -6. Which tools were called is not the result. The still is described twice. The second copy reuses the first encode.
- Model. [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini), Apache-2.0. The [model card](https://thinkingmachines.ai/model-card/inkling-2-mini/) reports SWE-bench Verified (bash only) 79.2% and MMMU Pro Standard 74.9%. Those scores are not this harness rate.
- Rate. The same harness without the still, thinking on, `half`, and MTP off, recorded 816.27 tok/s prefill and 64.60 tok/s decode on 22 November 2026. That row is in [coding-text-half.md](coding-text-half.md).
- Rate. [llama.cpp on a 64 GB M5 Pro Mac mini](https://huggingface.co/datasets/pratikgx/inkling-2-mini-m5-pro-mac-mini-bench) with the official IQ4_NL GGUF reports 357.5 tok/s prompt and 28.42 tok/s generation at 4096 tokens.
- Rate. [oMLX on an M5 Pro with 64 GB](https://omlx.ai/benchmarks/performance/robl2c08), pack `Inkling-2-Mini-oQ4e-mtp`, Lightning MTP, reports 708.9 tok/s prefill and 30.5 tok/s generation at 64k tokens.
- Rate. None of them is this `half` vision table.

## Rule

- MTP is off.
- The still is the first chat and the chat after the ledger, `detail` `auto`, `tool_choice` `none`.
- Thinking on uses the thinking card.
- The banner prints the loaded plane.
- This table is the run whose banner printed `kv half`, `mtp off`, `thinking on`, and `vision on`.
- The first request's prefill clock includes the image encode.
- Request 2 loaded the prefix from disk.
- `store` is off.
- The dense fp4 planes came from `dense-fp4.safetensors`. The vision linears came from `vision-fp8.safetensors`.

## Command

```text
uv run python -m harness.coding --model inkling_2_mini --thinking on --vision
```

The banner prints the loaded plane, MTP, thinking, and vision.

## Table

| Thinking | Vision | Result | Prefill | Decode |
|---|---|---|---|---|
| on | on | pass | 769.97 tok/s (2872 tok in 3.73 s) | 65.07 tok/s (965 tok in 14.83 s) |

## Search

- 22 November 2026.
- Queries: "Inkling-2-Mini 769.97 65.07 tok/s vision coding agent M5", "Inkling-2-Mini vision image tokens per second MLX llama.cpp M5 Pro benchmark", and "thinkingmachines Inkling-2-Mini model card SWE-bench vision benchmark".
- The predecessors in this note are the overlaps those queries returned, plus the text row from this harness.
- The search did not find 769.97 or 65.07 in those sources.

## Corrections

Later finds are appended here. The sections above stay as committed.
