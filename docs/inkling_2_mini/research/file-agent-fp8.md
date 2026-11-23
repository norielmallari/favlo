# File agent, fp8 attention

Measurement note for the `inkling_2_mini` pipeline. The serving contract is [../ARCHITECTURE.md](../ARCHITECTURE.md). This file records one file-agent run on the FOEM fp8 attention pack with `half` keys and values. It is a recorded run, not a format bound.

## Remainder

- On an M5 Pro with 48 GB, `Inkling-2-Mini` with FOEM E4M3 attention and output projections, a float16 head, `half` keys and values, and MTP off passed the file agent.
- Thinking off, vision off, recorded 787.76 tok/s prefill and 26.72 tok/s decode.
- The last prompt was 70584 tokens. The prefill sum is 70174 tokens in 89.08 s. The decode sum is 502 tokens in 18.79 s.
- A search on 23 November 2026 did not find that table.

## Predecessors

- Method. The file agent reads shared package sources until the prompt reaches 65536 tokens, unloads the model after the first turn, and requires a later request to load the prefix from disk. Exit 0 is that target and that reload. Which files were read is not the result.
- Model. [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini), Apache-2.0. The [model card](https://thinkingmachines.ai/model-card/inkling-2-mini/) reports SWE-bench Verified (bash only) 79.2%. That score is not this harness rate.
- Rate. [An MLX nvfp4 and mxfp8 repack on an M5 Pro with 48 GB](https://huggingface.co/baytizzel/Inkling-2-Mini-MLX-M5Pro-48gb) reports 750 tok/s prefill on a 3.3K-token prompt and 18.2 tok/s decode with 125K tokens in context.
- Rate. [llama.cpp on a 64 GB M5 Pro Mac mini](https://huggingface.co/datasets/pratikgx/inkling-2-mini-m5-pro-mac-mini-bench) with the official IQ4_NL GGUF reports 201.6 tok/s prompt and 24.97 tok/s generation at 32768 tokens.
- Rate. The same harness on the GPTQ dense fp4 pack recorded 766.49 tok/s prefill and 39.93 tok/s decode on 23 November 2026. That row is in [file-agent-half.md](file-agent-half.md).
- Rate. None of them is this fp8 file-agent table.

## Rule

- MTP is off.
- The still is omitted.
- Thinking off uses the instruct card and `reasoning.effort` `none`.
- The load line prints `kv half`.
- Exit 0 means the prompt reached 65536 tokens and a later request loaded the prefix from disk.
- The prefill sum adds every request's prefill tokens and those clocks. A cache hit counts only its suffix.
- The decode sum adds every request's kept tokens and those clocks.
- `store` is off.
- Attention and the output projection came from `dense-fp8.safetensors`.

## Command

```text
uv run python -m harness --model inkling_2_mini
```

The harness prints each request, then the summed prefill and decode lines.

## Table

| Thinking | Vision | Result | Prefill | Decode |
|---|---|---|---|---|
| off | off | pass | 787.76 tok/s (70174 tok in 89.08 s) | 26.72 tok/s (502 tok in 18.79 s) |

## Search

- 23 November 2026.
- Queries: "Inkling-2-Mini 1197.33 OR 48.48 OR 787.76 tok/s prefill M5 Pro" and "Inkling-2-Mini fp8 E4M3 FOEM prefill decode tok/s Apple M5 Pro 48GB 2026".
- The predecessors in this note are the overlaps those queries returned.
- The search did not find 787.76 and 26.72.

## Corrections

Later finds are appended here. The sections above stay as committed.
