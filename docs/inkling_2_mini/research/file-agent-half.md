# File agent, half cache

Measurement note for the `inkling_2_mini` pipeline. The serving contract is [../ARCHITECTURE.md](../ARCHITECTURE.md). This file records one file-agent run on `Inkling-2-Mini` with the GPTQ dense fp4 pack and `half` keys and values. It is a recorded run, not a format bound.

## Remainder

- On an M5 Pro with 48 GB, `Inkling-2-Mini` with the GPTQ dense fp4 pack, `half` keys and values, and MTP off passed the file agent.
- Thinking off, vision off, recorded 766.49 tok/s prefill and 39.93 tok/s decode.
- The last prompt was 70653 tokens. The prefill sum is 70195 tokens in 91.58 s. The decode sum is 543 tokens in 13.60 s.
- A search on 23 November 2026 did not find that table.

## Predecessors

- Method. The file agent reads shared package sources until the prompt reaches 65536 tokens, unloads the model after the first turn, and requires a later request to load the prefix from disk. Exit 0 is that target and that reload. Which files were read is not the result.
- Model. [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini), Apache-2.0. The [model card](https://thinkingmachines.ai/model-card/inkling-2-mini/) reports SWE-bench Verified (bash only) 79.2%. That score is not this harness rate.
- Rate. [An MLX nvfp4 and mxfp8 repack on an M5 Pro with 48 GB](https://huggingface.co/baytizzel/Inkling-2-Mini-MLX-M5Pro-48gb) reports 750 tok/s prefill on a 3.3K-token prompt, 786 tok/s on a 12.9K-token prompt, 26.6 tok/s decode, and 18.2 tok/s decode with 125K tokens in context.
- Rate. [oMLX on an M5 Pro with 64 GB](https://omlx.ai/benchmarks/performance/rdexuhjk), pack `Inkling-2-Mini-oQ8e`, reports 713.8 tok/s prefill and 32.0 tok/s generation at 64k tokens.
- Rate. [llama.cpp on a 64 GB M5 Pro Mac mini](https://huggingface.co/datasets/pratikgx/inkling-2-mini-m5-pro-mac-mini-bench) with the official IQ4_NL GGUF reports 201.6 tok/s prompt and 24.97 tok/s generation at 32768 tokens.
- Rate. [LexiEngine on an AMD RX 7900 XTX](https://llamaperf.com/) reports 767.0 tok/s prompt and 33.7 tok/s generation on one 523264-token prompt. That row is not this summed table.
- Rate. None of them is this `half` file-agent table.

## Rule

- MTP is off.
- The still is omitted.
- Thinking off uses the instruct card and `reasoning.effort` `none`.
- The load line prints `kv half`.
- Exit 0 means the prompt reached 65536 tokens and a later request loaded the prefix from disk.
- The prefill sum adds every request's prefill tokens and those clocks. A cache hit counts only its suffix.
- The decode sum adds every request's kept tokens and those clocks.
- `store` is off.
- The dense fp4 planes came from `dense-fp4.safetensors`. This run loaded the pack written from 237766 calibration tokens.

## Command

```text
uv run python -m harness --model inkling_2_mini
```

The harness prints each request, then the summed prefill and decode lines.

## Table

| Thinking | Vision | Result | Prefill | Decode |
|---|---|---|---|---|
| off | off | pass | 766.49 tok/s (70195 tok in 91.58 s) | 39.93 tok/s (543 tok in 13.60 s) |

## Search

- 23 November 2026.
- Queries: "Inkling-2-Mini 766.49 OR 39.93 tok/s prefill M5 Pro file agent", "Inkling-2-Mini MLX OR llama.cpp OR oMLX M5 Pro 48GB long context prefill tok/s 65536", and "766.49 OR 39.93 tok Inkling-2-Mini prefill decode".
- The predecessors in this note are the overlaps those queries returned.
- The search did not find 766.49 and 39.93 in those sources.

## Corrections

Later finds are appended here. The sections above stay as committed.
