# Synthetic prompt

Measurement note for the `qwen38_27b` pipeline. The serving contract is [../README.md](../README.md). This file records one synthetic-prompt run with MTP off. It is a recorded run, not a format bound. The MTP-on table is [synthetic-prompt-mtp.md](synthetic-prompt-mtp.md).

## Remainder

On an M5 Pro with 48 GB, `Qwen3.8-27B-fp8-e20` with `fp8_e4m3` keys and values and MTP off passed a cold instruct prompt. A 512-token prompt prefilled at 427.59 tok/s and decoded 128 tokens at 14.61 tok/s. A 10000-token prompt prefilled at 483.79 tok/s and decoded 20 tokens at 13.19 tok/s. The end-to-end rates, prefill plus decode, were 64.26 tok/s and 451.55 tok/s. A search on 4 November 2026 did not find that table.

## Predecessors

- Method. [llama-bench](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md) defaults to a 512-token prompt and 128 generated tokens. [`mlx_lm.benchmark`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/benchmark.py) uses the same prompt default. [Artificial Analysis](https://artificialanalysis.ai/methodology/performance-benchmarking) times a prompt of about 10000 tokens and reports time to first token, not prefill tok/s.
- Model. [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). The card does not publish this harness rate table.
- Rate. [oMLX on an M5 Pro](https://omlx.ai/benchmarks/performance/yjjrwjqi) reports 432.4 tok/s prefill and 10.0 tok/s generation at 1k for an 8-bit build. [oMLX oQ4e with MTP](https://omlx.ai/benchmarks/performance/8cs39b09) reports 319.1 tok/s prefill and 19.3 tok/s generation at 16k on an M5 Pro with 64 GB. [Just Jason](https://junxinzhang.com/qwen38-27b-m5pro-local/) reports 272 tok/s prefill at about 3.6k under Ollama Q4_K_M on an M5 Pro with 64 GB. [An M1 Max MLX and llama.cpp comparison](https://www.reddit.com/r/oMLX/comments/1waczb5/qwen3827b_on_m1_max_32gb_mlx_158_toks_vs_llamacpp/) reports 81.76 and 99.61 tok/s prefill on a 512-token prompt. None of them is this `fp8_e4m3` table.

## Rule

MTP is off. Thinking is off, so the request uses the instruct card and `reasoning.effort` `none`. The banner prints the loaded plane. Each measured prompt is one user message and encodes to that length under the chat template, including the template tokens. A 64-token warmup runs first. The measured prompts start with different labels, so each prefill is a cache miss. Decode is capped at 128 tokens. A stop before that cap is the kept length. The end-to-end rate adds the prefill tokens to the decode tokens and adds those two clocks. A banner that prints `half`, `mtp on`, or `thinking on` is a different row.

## Command

```text
uv run python -m harness.synthetic --model qwen38-27b
```

The banner prints the loaded plane, MTP, thinking, and the decode cap.

## Table

| Prompt tokens | Prefill | Decode | End to end |
|---:|---|---|---|
| 512 | 427.59 tok/s (512 tok in 1.20 s) | 14.61 tok/s (128 tok in 8.76 s) | 64.26 tok/s (640 tok in 9.96 s) |
| 10000 | 483.79 tok/s (10000 tok in 20.67 s) | 13.19 tok/s (20 tok in 1.52 s) | 451.55 tok/s (10020 tok in 22.19 s) |

## Search

4 November 2026. Queries: "427.59 OR 483.79 OR 64.26 OR 451.55 Qwen3.8-27B prefill tok/s", and "Qwen3.8-27B pp512 OR 512 tokens prefill tok/s Metal OR MLX OR llama.cpp 10000 prompt M5 Pro". The predecessors in this note are the overlaps those queries returned. The search did not find 427.59 and 14.61, or 483.79 and 13.19, in those sources.

## Corrections

Later finds are appended here. The sections above stay as committed.
