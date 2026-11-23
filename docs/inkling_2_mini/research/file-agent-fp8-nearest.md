# File agent, nearest fp8 attention

Measurement note for the `inkling_2_mini` pipeline. The serving contract is [../ARCHITECTURE.md](../ARCHITECTURE.md). This file records one file-agent run on nearest E4M3 attention with GPTQ-FOEM Metal fp4 experts and `half` keys and values. It is a recorded run, not a format bound.

## Remainder

- On an M5 Pro with 48 GB, `Inkling-2-Mini` with nearest E4M3 attention and output projections, a float16 head, GPTQ-FOEM Metal fp4 experts in layers 0–19, MXFP8 experts in layers 20–23, `half` keys and values, and MTP off passed the file agent.
- Thinking off, vision off, recorded 792.21 tok/s prefill and 26.01 tok/s decode.
- The last prompt was 70555 tokens. The prefill sum is 70174 tokens in 88.58 s. The decode sum is 452 tokens in 17.38 s.
- A search on 23 November 2026 did not find that table.

## Predecessors

- Method. The file agent reads shared package sources until the prompt reaches 65536 tokens, unloads the model after the first turn, and requires a later request to load the prefix from disk. Exit 0 is that target and that reload. Which files were read is not the result.
- Model. [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini), Apache-2.0. The [model card](https://thinkingmachines.ai/model-card/inkling-2-mini/) reports SWE-bench Verified (bash only) 79.2%. That score is not this harness rate.
- Rate. [An MLX nvfp4 and mxfp8 repack on an M5 Pro with 48 GB](https://huggingface.co/baytizzel/Inkling-2-Mini-MLX-M5Pro-48gb) reports 786 tok/s prefill on a 12.9K-token prompt and 26.6 tok/s decode.
- Rate. [llama.cpp on a 64 GB M5 Pro Mac mini](https://huggingface.co/datasets/pratikgx/inkling-2-mini-m5-pro-mac-mini-bench) with the official IQ4_NL GGUF reports 201.6 tok/s prompt and 24.97 tok/s generation at 32768 tokens.
- Rate. The same harness on the FOEM fp8 attention pack recorded 787.76 tok/s prefill and 26.72 tok/s decode on 23 November 2026. That row is in [file-agent-fp8.md](file-agent-fp8.md).
- Rate. The same harness on the GPTQ dense fp4 pack recorded 766.49 tok/s prefill and 39.93 tok/s decode on 23 November 2026. That row is in [file-agent-half.md](file-agent-half.md).
- Rate. None of them is this nearest-fp8 file-agent table.

## Rule

- MTP is off.
- The still is omitted.
- Thinking off uses the instruct card and `reasoning.effort` `none`.
- The harness line printed `thinking off`. Keys and values are `half`, the plane this listener loaded for the pack.
- Exit 0 means the prompt reached 65536 tokens and a later request loaded the prefix from disk.
- The prefill sum adds every request's prefill tokens and those clocks. A cache hit counts only its suffix.
- The decode sum adds every request's kept tokens and those clocks.
- `store` is off.
- Attention and the output projection are nearest E4M3 from `dense-fp8.safetensors`. The head stayed float16.
- Experts in layers 0–19 are Metal fp4 from GPTQ-FOEM on the dequantized NVFP4 weights. Experts in layers 20–23 stay MXFP8.

## Command

```text
uv run python -m harness --model inkling_2_mini
```

The harness prints each request, then the summed prefill and decode lines.

## Table

| Thinking | Vision | Result | Prefill | Decode |
|---|---|---|---|---|
| off | off | pass | 792.21 tok/s (70174 tok in 88.58 s) | 26.01 tok/s (452 tok in 17.38 s) |

## Search

- 23 November 2026.
- Queries: "Inkling-2-Mini 792.21 OR 26.01 tok/s file agent prefill M5 Pro nearest fp8 2026".
- The predecessors in this note are the overlaps that query returned.
- The search did not find 792.21 and 26.01.

## Corrections

Later finds are appended here. The sections above stay as committed.
