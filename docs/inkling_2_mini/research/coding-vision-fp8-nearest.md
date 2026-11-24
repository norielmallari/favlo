# Coding agent, vision, nearest fp8 attention

Measurement note for the `inkling_2_mini` pipeline. The serving contract is [../ARCHITECTURE.md](../ARCHITECTURE.md). This file records one vision coding-agent run on nearest E4M3 attention with GPTQ-FOEM Metal fp4 experts and `half` keys and values. It is a recorded run, not a format bound.

## Remainder

- On an M5 Pro with 48 GB, `Inkling-2-Mini` with nearest E4M3 attention and output projections, a float16 head, GPTQ-FOEM Metal fp4 experts in layers 0–19, MXFP8 experts in layers 20–23, `half` keys and values, and MTP off passed the vision coding agent.
- Thinking on, vision on, recorded 634.98 tok/s prefill and 44.45 tok/s decode.
- A search on 24 November 2026 did not find that table.

## Predecessors

- Method. The ledger writes `ledger.py` through the coding tools, unloads the model so a later turn loads the prefix from disk, then checks the chat contract. Exit 0 requires `collect` to return 6, 0, and -6. Which tools were called is not the result. The still is described twice. The second copy reuses the first encode.
- Model. [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini), Apache-2.0. The [model card](https://thinkingmachines.ai/model-card/inkling-2-mini/) reports MMMU Pro Standard 74.9%. That score is not this harness rate.
- Rate. The same harness without the still, thinking on, on this weight mix, recorded 593.44 tok/s prefill and 49.21 tok/s decode on 23 November 2026. That row is in [coding-text-fp8-nearest.md](coding-text-fp8-nearest.md).
- Rate. The same harness with the still, thinking on, on the GPTQ dense fp4 pack, recorded 769.97 tok/s prefill and 65.07 tok/s decode on 22 November 2026. That row is in [coding-vision-half.md](coding-vision-half.md).
- Rate. [llama.cpp on a 64 GB M5 Pro Mac mini](https://huggingface.co/datasets/pratikgx/inkling-2-mini-m5-pro-mac-mini-bench) with the official IQ4_NL GGUF reports 357.5 tok/s prompt and 28.42 tok/s generation at 4096 tokens. That run did not send an image.
- Rate. [oMLX on an M5 Pro with 64 GB](https://omlx.ai/benchmarks/performance?chip=M5) with `Inkling-2-Mini-oQ4e-mtp` and Lightning MTP reports 1236 tok/s prefill and 46.9 tok/s generation at 32k tokens. That row is text.
- Rate. None of them is this nearest-fp8 vision table.

## Rule

- MTP is off.
- The still is the first chat and the chat after the ledger, `detail` `auto`, `tool_choice` `none`.
- Thinking on uses the thinking card.
- The banner prints the loaded plane.
- This table is the run whose banner printed `kv half`, `mtp off`, `thinking on`, and `vision on`.
- The first request's prefill clock includes the image encode.
- A later request loaded the prefix from disk.
- `store` is off.
- Attention and the output projection are nearest E4M3 from `dense-fp8.safetensors`. The head stayed float16.
- Experts in layers 0–19 are Metal fp4 from GPTQ-FOEM on the dequantized NVFP4 weights. Experts in layers 20–23 stay MXFP8.
- The vision linears came from `vision-fp8.safetensors`.

## Command

```text
uv run python -m harness.coding --model inkling_2_mini --thinking on --vision
```

The banner prints the loaded plane, MTP, thinking, and vision.

## Table

| Thinking | Vision | Result | Prefill | Decode |
|---|---|---|---|---|
| on | on | pass | 634.98 tok/s (2051 tok in 3.23 s) | 44.45 tok/s (1114 tok in 25.06 s) |

## Search

- 24 November 2026.
- Queries: "Inkling-2-Mini 634.98 OR 44.45 tok/s vision coding agent M5 Pro", "Inkling-2-Mini vision image tokens per second MLX llama.cpp M5 Pro benchmark 2026", and "634.98 OR 44.45 Inkling-2-Mini tok/s vision MMMU oQ4e".
- The predecessors in this note are the overlaps those queries returned, plus the text and GPTQ vision rows from this harness.
- The search did not find 634.98 or 44.45.

## Corrections

Later finds are appended here. The sections above stay as committed.
