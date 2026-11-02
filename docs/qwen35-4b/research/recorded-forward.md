# Recorded forward

Measurement note for the `qwen35-4b` pipeline. The serving contract is [../README.md](../README.md). The isolated dispatch sum is [kernel-ceilings.md](kernel-ceilings.md). This file records the carrot smoke and the coding agent printed on 2 November 2026 after int8 prefill, the resident int8 atlas, and mid-width `gemv_fp8_x4` decode.

## Remainder

On an M5 Pro with 48 GB, on `Qwen3.5-4B-fp8` with the default `f16` cache, the coding agent recorded 1673.36 tok/s prefill over 25569 tokens and 40.77 tok/s decode over 2172 tokens. A search on 2 November 2026 did not find that run.

## Predecessors

- Method. The forward is the shader library summed in [kernel-ceilings.md](kernel-ceilings.md). Prefill uses the int8 path in [int8-prefill.md](int8-prefill.md). The prefix store, the sampler, and the wait between tokens are inside these times. The kernel sum leaves them out.
- Model. [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B). [oMLX Qwen3.5-4B-8bit](https://omlx.ai/benchmarks/performance/x0t6bie9) reports 2643 prefill tok/s and 55.4 generation tok/s at 4096 context on an M5 Pro with 20 shader cores and 48 GB. [oMLX Optimized-MTPLX](https://omlx.ai/benchmarks/performance/7bdbt1iy) reports 46.8 generation tok/s at 4k. [llmcheck](https://llmcheck.net/models/qwen-35-4b-on-m5-pro/) reports about 66 tok/s decode for llama.cpp Q4_K_M on an M5 Pro.
- Measurement. Those figures are another engine or another quant. They do not publish this harness run on the fp8 record.
- Mechanism. The carrot cases are greedy, then the instruct and thinking cards. The coding agent uses the thinking card. Prefill restores selected planes into a resident int8 atlas when the budget fits. Decode uses `gemv_fp8_x4` for mid-width projections.

## Rule

Prefill is the summed prompt time. Decode is the summed sampled-token time. The carrot rows are the lines `tests/test_qwen35_4b_smoke.py` printed under the smoke lock on 2 November 2026 after the mid-width GEMV change. The coding-agent row is the harness summary of the same day's thinking-card run: 25569 prompt tokens and 2172 sampled tokens.

## Command

```text
uv run python -m pytest tests/test_qwen35_4b_smoke.py -s
uv run python -m harness.coding --model qwen35-4b --thinking on
```

The pytest command prints the carrot rows. The coding command prints `passed` when `ledger.py` `collect` returns 6, 0, and -6. No committed command reprints the 1673.36 and 40.77 tok/s rows.

## Table

| Run | Prefill | Decode |
|---|---|---|
| Carrot, greedy | 613.54 tok/s (51 tok in 0.08 s) | 47.63 tok/s (107 tok in 2.25 s) |
| Carrot, instruct | 1132.46 tok/s (51 tok in 0.05 s) | 46.90 tok/s (105 tok in 2.24 s) |
| Carrot, thinking | 1123.10 tok/s (49 tok in 0.04 s) | 45.66 tok/s (509 tok in 11.15 s) |
| Coding agent, thinking | 1673.36 tok/s (25569 tok in 15.28 s) | 40.77 tok/s (2172 tok in 53.28 s) |

## Search

2 November 2026. Queries: "Qwen3.5-4B prefill tok/s int8 Apple Silicon M5 Pro 2026", "Qwen3.5-4B oMLX OR mlx OR MTPLX decode tok/s M5 Pro", "Qwen3.5-4B-fp8 1673.36 OR 40.77 OR 47.63 tok/s Favlo coding", and "Qwen 3.5 4B on M5 Pro: 66 tok/s". The predecessors in this note are the overlaps those queries returned. The search did not find 1673.36 and 40.77, or 47.63, 46.90, and 45.66, on this harness.

## Corrections

Later finds are appended here. The sections above stay as committed.
