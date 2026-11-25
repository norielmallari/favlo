# Coding agent, text, half experts in MXFP8

Measurement note for the `inkling_2_mini` pipeline. The serving contract is [../ARCHITECTURE.md](../ARCHITECTURE.md). This file records one text coding-agent run on `Inkling-2-mini-fp8-e50` with `half` keys and values. It is a recorded run, not a format bound.

## Remainder

- On an M5 Pro with 48 GB, `Inkling-2-mini-fp8-e50` with nearest E4M3 attention and output projections, a float16 head, 1536 MXFP8 experts and 1536 Metal fp4 experts, `half` keys and values, and MTP off passed the text coding agent.
- The 1536 MXFP8 experts are the higher fp4 restoration scores. Ties keep the lower index.
- Thinking on, vision off, recorded 538.40 tok/s prefill and 42.30 tok/s decode.
- A search on 25 November 2026 did not find that table.

## Predecessors

- Method. The ledger writes `ledger.py` through the coding tools, unloads the model so a later turn loads the prefix from disk, then checks the chat contract. Exit 0 requires `collect` to return 6, 0, and -6. Which tools were called is not the result.
- Model. [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini), Apache-2.0. The [model card](https://thinkingmachines.ai/model-card/inkling-2-mini/) reports SWE-bench Verified (bash only) 79.2% and DeepSWE v1.1 48.2%. Those scores are for the official checkpoint, not this expert mix.
- Rate. [llama.cpp on a 64 GB M5 Pro Mac mini](https://huggingface.co/datasets/pratikgx/inkling-2-mini-m5-pro-mac-mini-bench) with the official IQ4_NL GGUF reports 357.5 tok/s prompt and 28.42 tok/s generation at 4096 tokens.
- Rate. [The same machine on 24 November 2026](https://github.com/pratikgx/local-ai-field-notes/blob/main/notes/2026-11-24-thinking-machines-inkling-2-mini-mlx-vllm.md) reports oMLX prefill about 930–1160 tok/s and, with MTP off, 44.47 tok/s generation at 512 tokens.
- Rate. [An MLX pack on an M5 Pro with 48 GB](https://huggingface.co/baytizzel/Inkling-2-Mini-MLX-M5Pro-48gb) using the release NVFP4 and MXFP8 weights reports 557–611 tok/s prefill at 3.3K tokens on the stock path, and 34.2 tok/s decode on the tuned path.
- Rate. [oMLX on an M5 Pro with 64 GB](https://omlx.ai/benchmarks/performance/vwp4avlb) reports 514.0 tok/s prefill and 19.8 tok/s generation at 195k tokens with MTP on.
- Rate. The same harness on the nearest-fp8 official pack, thinking on, recorded 593.44 tok/s prefill and 49.21 tok/s decode on 23 November 2026. That row is in [coding-text-fp8-nearest.md](coding-text-fp8-nearest.md).
- Rate. None of them is this 1536-expert MXFP8 table.

## Rule

- MTP is off.
- The still is omitted.
- Thinking on uses the thinking card.
- The banner prints the loaded plane.
- This table is the run whose banner printed `kv half`, `mtp off`, `thinking on`, and `vision off`.
- A later request loaded the prefix from disk.
- `store` is off.
- Attention and the output projection are nearest E4M3. The head stayed float16.
- 1536 of 3072 routed experts are MXFP8. The rest are Metal fp4. The rank is the fp4 restoration score.

## Command

```text
uv run python -m harness.coding --model Inkling-2-mini-fp8-e50
```

The banner prints the loaded plane, MTP, thinking, and vision.

## Table

| Thinking | Vision | Result | Prefill | Decode |
|---|---|---|---|---|
| on | off | pass | 538.40 tok/s (2019 tok in 3.75 s) | 42.30 tok/s (2359 tok in 55.77 s) |

## Search

- 25 November 2026.
- Queries: "Inkling-2-Mini 538.40 OR 42.30 tok/s prefill decode M5 Pro coding agent 2026" and "Inkling-2-Mini MXFP8 Metal fp4 expert mix tok/s Apple M5 Pro llama.cpp MLX 2026".
- The predecessors in this note are the overlaps those queries returned, plus the earlier harness row on the official pack.
- The search did not find 538.40 and 42.30.

## Corrections

Later finds are appended here. The sections above stay as committed.
