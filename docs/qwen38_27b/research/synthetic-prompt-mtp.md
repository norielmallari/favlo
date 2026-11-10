# Synthetic prompt (MTP on)

Measurement note for the `qwen38_27b` pipeline. The serving contract is [../README.md](../README.md). This file records one synthetic-prompt run with MTP on. It is a recorded run, not a format bound. The MTP-off table is [synthetic-prompt.md](synthetic-prompt.md).

## Remainder

On an M5 Pro with 48 GB, `Qwen3.8-27B-fp8-e20` with `fp8_e4m3` keys and values and MTP on at depth 3 passed a cold instruct prompt. A 512-token prompt prefilled at 359.96 tok/s and decoded 128 tokens at 14.97 tok/s. A 10000-token prompt prefilled at 468.03 tok/s and decoded 92 tokens at 15.17 tok/s. The end-to-end rates, prefill plus decode, were 64.19 tok/s and 367.92 tok/s. A search on 4 November 2026 did not find that table.

## Predecessors

- Method. [llama-bench](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md) defaults to a 512-token prompt and 128 generated tokens. [`mlx_lm.benchmark`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/benchmark.py) uses the same prompt default. [Artificial Analysis](https://artificialanalysis.ai/methodology/performance-benchmarking) times a prompt of about 10000 tokens and reports time to first token, not prefill tok/s.
- Model. [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). The card does not publish this harness rate table.
- Rate. [oMLX oQ4e with MTP on an M5 Pro](https://omlx.ai/benchmarks/performance/k68ktgtl) reports 503.2 tok/s prefill and 31.7 tok/s generation at 4k. [oMLX MTPLX on an M5 Pro with 48 GB](https://omlx.ai/benchmarks/performance/7kjna4i0) reports 433.3 tok/s prefill and 30.1 tok/s generation at 1k with MTP depth 3. [llm-speed on an RTX 4090](https://llm-speed.com/blog/mtp-speculative-decoding-rtx-4090) reports chat-short prefill falling from 428.6 to 299.3 tok/s when MTP turns on, with decode rising from 43.06 to 84.51 tok/s. [Weschera/Qwen3.8-27B-oMLX-MTP-Mac](https://github.com/Weschera/Qwen3.8-27B-oMLX-MTP-Mac) reports 273.7 tok/s prefill at 4k with MTP k=3 on an M4 Max. None of them is this `fp8_e4m3` table.

## Rule

MTP is on at depth 3. Thinking is off, so the request uses the instruct card and `reasoning.effort` `none`. The banner prints the loaded plane. Each measured prompt is one user message and encodes to that length under the chat template, including the template tokens. A 64-token warmup runs first. The measured prompts start with different labels, so each prefill is a cache miss. Decode is capped at 128 tokens. A stop before that cap is the kept length. The end-to-end rate adds the prefill tokens to the decode tokens and adds those two clocks. A banner that prints `half`, `mtp off`, or `thinking on` is a different row.

## Command

```text
uv run python -m harness.synthetic --model qwen38-27b --mtp on
```

The banner prints the loaded plane, MTP, thinking, and the decode cap.

## Table

| Prompt tokens | Prefill | Decode | End to end |
|---:|---|---|---|
| 512 | 359.96 tok/s (512 tok in 1.42 s) | 14.97 tok/s (128 tok in 8.55 s) | 64.19 tok/s (640 tok in 9.97 s) |
| 10000 | 468.03 tok/s (10000 tok in 21.37 s) | 15.17 tok/s (92 tok in 6.06 s) | 367.92 tok/s (10092 tok in 27.43 s) |

## Search

4 November 2026. Queries: "359.96 OR 468.03 OR 64.19 OR 367.92 Qwen3.8-27B MTP prefill tok/s", and "Qwen3.8-27B MTP depth 3 pp512 OR 512 tokens prefill decode M5 Pro oMLX Metal". The predecessors in this note are the overlaps those queries returned. The search did not find 359.96 and 14.97, or 468.03 and 15.17, in those sources.

## Corrections

Later finds are appended here. The sections above stay as committed.
