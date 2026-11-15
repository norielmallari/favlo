# Synthetic prompt

Measurement note for the `qwen35_4b` pipeline. The serving contract is [../ARCHITECTURE.md](../ARCHITECTURE.md). This file records one synthetic-prompt run. It is a recorded run, not a format bound.

## Remainder

On an M5 Pro with 48 GB, `Qwen3.5-4B-fp8` with `fp8_e4m3` keys and values and MTP off passed a cold instruct prompt.

- A 512-token prompt prefilled at 2239.26 tok/s and decoded 61 tokens at 40.79 tok/s.
- A 10000-token prompt prefilled at 991.90 tok/s and decoded 128 tokens at 14.28 tok/s.
- The end-to-end rates, prefill plus decode, were 331.21 tok/s and 531.93 tok/s.

A search on 4 November 2026 did not find that table.

## Predecessors

- Method. [llama-bench](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md) defaults to a 512-token prompt and 128 generated tokens. [`mlx_lm.benchmark`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/benchmark.py) uses the same prompt default. [Artificial Analysis](https://artificialanalysis.ai/methodology/performance-benchmarking) times a prompt of about 10000 tokens and reports time to first token, not prefill tok/s.
- Model. [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B). The card does not publish this harness rate table.
- Rate. [mac-llm-bench on base M5](https://github.com/enescingoz/mac-llm-bench/blob/main/results/m5/base/speed/README.md) reports 1074.2 tok/s at pp512 and 48.7 tok/s at tg128 for Qwen3.5-4B-4bit under MLX, and 628.0 tok/s at pp512 and 29.4 tok/s at tg128 for Q4_K_M under llama.cpp. [oMLX Qwen3.5-4B-8bit on an M5 Pro with 48 GB](https://omlx.ai/benchmarks/performance/x0t6bie9) reports 2643 tok/s prefill and 55.4 tok/s decode at 4096 context. [llmcheck](https://llmcheck.net/models/qwen-35-4b-on-m5-pro/) reports about 66 tok/s decode for llama.cpp Q4_K_M on an M5 Pro. None of them is this `fp8_e4m3` table.

## Rule

- MTP is off.
- Thinking is off, so the request uses the instruct card and `reasoning.effort` `none`.
- The banner prints the loaded plane.
- Each measured prompt is one user message and encodes to that length under the chat template, including the template tokens.
- A 64-token warmup runs before the measured prompts.
- The measured prompts start with different labels, so each prefill is a cache miss.
- Decode is capped at 128 tokens.
- A stop before that cap is the kept length.
- The end-to-end rate adds the prefill tokens to the decode tokens and adds those two clocks.
- A banner that prints `half`, `mtp on`, or `thinking on` is a different row.

## Command

```text
uv run python -m harness.synthetic --model qwen35-4b
```

The banner prints the loaded plane, MTP, thinking, and the decode cap.

## Table

| Prompt tokens | Prefill | Decode | End to end |
|---:|---|---|---|
| 512 | 2239.26 tok/s (512 tok in 0.23 s) | 40.79 tok/s (61 tok in 1.50 s) | 331.21 tok/s (573 tok in 1.73 s) |
| 10000 | 991.90 tok/s (10000 tok in 10.08 s) | 14.28 tok/s (128 tok in 8.96 s) | 531.93 tok/s (10128 tok in 19.04 s) |

## Search

4 November 2026.

- Queries: "Qwen3.5-4B pp512 tg128 prefill tok/s Metal OR MLX OR llama.cpp M5 Pro 2026", and "2239.26 OR 991.90 OR 331.21 OR 531.93 Qwen3.5-4B synthetic prompt fp8_e4m3".
- The predecessors in this note are the overlaps those queries returned.
- The search did not find 2239.26 and 40.79, or 991.90 and 14.28, in those sources.

## Corrections

Later finds are appended here. The sections above stay as committed.

### 15 November 2026 — 2048-token prompt

Same plane, MTP off, and the instruct card. A cold 2048-token prompt prefilled at 1471.55 tok/s and decoded 128 tokens at 26.63 tok/s. End to end was 350.97 tok/s. A search on 15 November 2026 did not find 1471.55 and 26.63 in these sources.

- Queries: "1471.55 OR 26.63 OR 350.97 Qwen3.5-4B prefill decode tok/s", and "Qwen3.5-4B pp2048 tg128 MLX M5 Pro fp8 prefill tok/s 2026".
- Closest overlaps were [oMLX Qwen3.5-4B-8bit on a 20-core M5 Pro](https://omlx.ai/benchmarks/performance/x0t6bie9) at 2643 tok/s prefill and 55.4 tok/s decode at 4096 context, [oMLX Qwen3.5-4B-MLX-8bit on a 20-core M5 Pro](https://omlx.ai/benchmarks/performance/wia8vwen) at 2564 tok/s prefill and 52.5 tok/s decode at 4096 context, and [FastFlowLM](https://fastflowlm.com/docs/benchmarks/qwen3.5_results/) at 445.63 tok/s prefill and 15.6 tok/s decode for Qwen3.5-4B at 2048 tokens on XDNA2.
- None of them is this `fp8_e4m3` table.
- The 512 and 10000 rows above stay the 4 November measurement.

```text
uv run python -m harness.synthetic --model qwen35-4b --tokens 2048 --decode 128 --thinking off --mtp off
```

| Prompt tokens | Prefill | Decode | End to end |
|---:|---|---|---|
| 2048 | 1471.55 tok/s (2048 tok in 1.39 s) | 26.63 tok/s (128 tok in 4.81 s) | 350.97 tok/s (2176 tok in 6.20 s) |

### 15 November 2026 — protocol lengths

Same pack, MTP off, and the instruct card. Temperature is 0. Each prefill is a cache miss. Decode is 128 tokens at every length. The packed folder is `Qwen3.5-4B-fp8`. Language matrices are fp8. Keys and values are `fp8_e4m3`. The banner printed depth 3; that depth is unused while MTP is off.

| Prompt tokens | Prefill | Decode | End to end |
|---:|---|---|---|
| 512 | 1322.13 tok/s (512 tok in 0.39 s) | 40.37 tok/s (128 tok in 3.17 s) | 179.78 tok/s (640 tok in 3.56 s) |
| 1024 | 1457.07 tok/s (1024 tok in 0.70 s) | 36.19 tok/s (128 tok in 3.54 s) | 271.70 tok/s (1152 tok in 4.24 s) |
| 4096 | 1278.14 tok/s (4096 tok in 3.20 s) | 23.41 tok/s (128 tok in 5.47 s) | 487.20 tok/s (4224 tok in 8.67 s) |
| 10000 | 874.93 tok/s (10000 tok in 11.43 s) | 13.61 tok/s (128 tok in 9.40 s) | 486.22 tok/s (10128 tok in 20.83 s) |

```text
uv run python -m harness.synthetic --model qwen35_4b --thinking off --mtp off
```

15 November 2026. Queries: "1322.13 OR 1457.07 OR 1278.14 OR 874.93 Qwen3.5-4B prefill tok/s", and "Qwen3.5-4B M5 Pro pp512 tg128 OR pp1024 OR pp4096 fp8 OR 8bit prefill tok/s 2026". The search did not find 1322.13, 1457.07, 1278.14, or 874.93.

- [oMLX Qwen3.5-4B-8bit on a 20-core M5 Pro with 48 GB](https://omlx.ai/benchmarks/performance/zocwcytw) reports 2105 tok/s prefill and 57.2 tok/s decode at 1024 tokens. The same session at 4096 tokens is [x0t6bie9](https://omlx.ai/benchmarks/performance/x0t6bie9), 2643 tok/s prefill and 55.4 tok/s decode.
- [oMLX Qwen3.5-4B-MLX-8bit on a 20-core M5 Pro with 48 GB](https://omlx.ai/benchmarks/performance/xozhrdwg) reports 2481 tok/s prefill and 54.3 tok/s decode at 1024 tokens, and 2630 tok/s prefill and 51.5 tok/s decode at 4096 tokens.
- [Token Atlas](https://tokatlas.github.io/models/qwen3-5-4b/) lists an oMLX 4-bit row on an M5 Pro at 2579 tok/s prefill and 98.7 tok/s decode at 1024 tokens, and 2919 tok/s prefill and 84.0 tok/s decode at 16384 tokens. That page has no 10000-token row.
- [mac-llm-bench on base M5](https://github.com/enescingoz/mac-llm-bench/blob/main/results/m5/base/speed/README.md) reports 1074.2 tok/s at pp512 and 48.7 tok/s at tg128 for Qwen3.5-4B-4bit under MLX. That machine is not an M5 Pro.
- None of them is this `fp8_e4m3` table. The 512, 2048, and 10000 rows above stay the earlier measurements.
