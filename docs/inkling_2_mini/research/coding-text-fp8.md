# Coding agent, text, fp8 attention

Measurement note for the `inkling_2_mini` pipeline. The serving contract is [../ARCHITECTURE.md](../ARCHITECTURE.md). This file records one text coding-agent run on the FOEM fp8 attention pack with `half` keys and values. It is a recorded run, not a format bound.

## Remainder

- On an M5 Pro with 48 GB, `Inkling-2-Mini` with FOEM E4M3 attention and output projections, a float16 head, `half` keys and values, and MTP off passed the text coding agent.
- Thinking on, vision off, recorded 613.12 tok/s prefill and 48.48 tok/s decode.
- A search on 23 November 2026 did not find that table.

## Predecessors

- Method. The ledger writes `ledger.py` through the coding tools, unloads the model so a later turn loads the prefix from disk, then checks the chat contract. Exit 0 requires `collect` to return 6, 0, and -6. Which tools were called is not the result.
- Model. [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini), Apache-2.0. The [model card](https://thinkingmachines.ai/model-card/inkling-2-mini/) reports SWE-bench Verified (bash only) 79.2% and DeepSWE v1.1 48.2%. Those scores are not this harness rate.
- Rate. [llama.cpp on a 64 GB M5 Pro Mac mini](https://huggingface.co/datasets/pratikgx/inkling-2-mini-m5-pro-mac-mini-bench) with the official IQ4_NL GGUF reports 357.5 tok/s prompt and 28.42 tok/s generation at 4096 tokens.
- Rate. [A Metal engine on an M5 Pro with 48 GB](https://github.com/Eriskii/erisinky) reading a Q4_K_M GGUF reports 63.0 tok/s short-context decode.
- Rate. The same harness on the GPTQ dense fp4 pack, thinking on, recorded 816.27 tok/s prefill and 64.60 tok/s decode on 22 November 2026. That row is in [coding-text-half.md](coding-text-half.md).
- Rate. None of them is this fp8 text table.

## Rule

- MTP is off.
- The still is omitted.
- Thinking on uses the thinking card.
- The banner prints the loaded plane.
- This table is the run whose banner printed `kv half`, `mtp off`, `thinking on`, and `vision off`.
- A later request loaded the prefix from disk.
- `store` is off.
- Attention and the output projection came from `dense-fp8.safetensors`. The head stayed float16.

## Command

```text
uv run python -m harness.coding --model inkling_2_mini --thinking on
```

The banner prints the loaded plane, MTP, thinking, and vision.

## Table

| Thinking | Vision | Result | Prefill | Decode |
|---|---|---|---|---|
| on | off | pass | 613.12 tok/s (1729 tok in 2.82 s) | 48.48 tok/s (1312 tok in 27.06 s) |

## Search

- 23 November 2026.
- Queries: "Inkling-2-Mini 1197.33 OR 48.48 OR 787.76 tok/s prefill M5 Pro" and "Inkling-2-Mini fp8 E4M3 FOEM prefill decode tok/s Apple M5 Pro 48GB 2026".
- The predecessors in this note are the overlaps those queries returned.
- The search did not find 613.12 and 48.48.

## Corrections

Later finds are appended here. The sections above stay as committed.
