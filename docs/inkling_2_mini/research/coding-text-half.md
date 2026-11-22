# Coding agent, text, half cache

Measurement note for the `inkling_2_mini` pipeline. The serving contract is [../ARCHITECTURE.md](../ARCHITECTURE.md). This file records one text coding-agent run on `Inkling-2-Mini` with the GPTQ dense fp4 pack and `half` keys and values. It is a recorded run, not a format bound.

## Remainder

- On an M5 Pro with 48 GB, `Inkling-2-Mini` with the GPTQ dense fp4 pack, `half` keys and values, and MTP off passed the text coding agent.
- Thinking on, vision off, recorded 816.27 tok/s prefill and 64.60 tok/s decode.
- Thinking off, vision off, recorded 859.82 tok/s prefill and 64.48 tok/s decode.
- A search on 22 November 2026 did not find that table.

## Predecessors

- Method. The ledger writes `ledger.py` through the coding tools, unloads the model so a later turn loads the prefix from disk, then checks the chat contract. Exit 0 requires `collect` to return 6, 0, and -6. Which tools were called is not the result.
- Model. [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini), Apache-2.0. The [model card](https://thinkingmachines.ai/model-card/inkling-2-mini/) reports SWE-bench Verified (bash only) 79.2% and DeepSWE v1.1 48.2%. Those scores are not this harness rate.
- Rate. [llama.cpp on a 64 GB M5 Pro Mac mini](https://huggingface.co/datasets/pratikgx/inkling-2-mini-m5-pro-mac-mini-bench) with the official IQ4_NL GGUF reports 357.5 tok/s prompt and 28.42 tok/s generation at 4096 tokens, and 10 of 10 on its local code suite.
- Rate. [mlx-lm PR #2086 on an M5 Pro with 48 GB](https://github.com/ml-explore/mlx-lm/pull/2086), 4-bit MLX, reports about 830 tok/s on a cold 31,187-token prompt, 0.5 to 0.6 s to first token on appended agent turns, and 28.7 tok/s generation.
- Rate. [An MLX decode path on an M5 Max with 128 GB](https://github.com/zuiho-kai/inkling2-mlx-fastpath) reports 66.6 tok/s decode on code-generation prompts under 1k tokens, with 4-bit weights.
- Rate. None of them is this `half` text table.

## Rule

- MTP is off.
- The still is omitted.
- Thinking on uses the thinking card.
- Thinking off uses the instruct card.
- The banner prints the loaded plane.
- This table is the pair whose banners printed `kv half`, `mtp off`, and `vision off`.
- Request 2 of each run loaded the prefix from disk.
- `store` is off.
- The dense fp4 planes came from `dense-fp4.safetensors`. A run on the nearest-code planes is a different row.

## Command

```text
uv run python -m harness.coding --model inkling_2_mini --thinking on
uv run python -m harness.coding --model inkling_2_mini --thinking off
```

The banner prints the loaded plane, MTP, thinking, and vision.

## Table

| Thinking | Vision | Result | Prefill | Decode |
|---|---|---|---|---|
| on | off | pass | 816.27 tok/s (2759 tok in 3.38 s) | 64.60 tok/s (752 tok in 11.64 s) |
| off | off | pass | 859.82 tok/s (2932 tok in 3.41 s) | 64.48 tok/s (550 tok in 8.53 s) |

## Search

- 22 November 2026.
- Queries: "Inkling-2-Mini tok/s prefill decode Apple M5 OR M4 llama.cpp OR MLX benchmark pp512 tg128", "Inkling-2-Mini MLX nvfp4 mxfp8 M5 Pro 48GB prefill decode tok/s", and "thinkingmachines Inkling-2-Mini model card benchmark SWE-bench active parameters".
- The predecessors in this note are the overlaps those queries returned.
- The search did not find 816.27 and 64.60, or 859.82 and 64.48, in those sources.

## Corrections

Later finds are appended here. The sections above stay as committed.
