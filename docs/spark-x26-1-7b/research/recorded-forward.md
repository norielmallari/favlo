# Recorded forward

Measurement note for the `spark-x26-1-7b` pipeline. The serving contract is [../README.md](../README.md). The isolated dispatch sum is [kernel-ceilings.md](kernel-ceilings.md). This file records the carrot smoke and the coding agent printed on 1 November 2026, after the packed record was one Metal buffer.

## Remainder

On an M5 Pro with 48 GB the carrot smoke decoded the greedy case at 89.65 tok/s, and the coding agent decoded 1,534 tokens at 76.02 tok/s. A search on 1 November 2026 did not find that run.

## Predecessors

- Method. The forward is the shader library summed in [kernel-ceilings.md](kernel-ceilings.md). The prefix store, the sampler, and the wait between tokens are inside these times. The kernel sum leaves them out.
- Model. [XHToken/Spark-X2.6-1.7B](https://huggingface.co/XHToken/Spark-X2.6-1.7B). The [DSpark note](https://dev.to/sparkllm/769-toks-on-a-single-rtx-4090-spark-x26-17b-dspark-6ga) reports 138 tok/s for an INT4 MLX copy on an M5 Pro, and 68 tok/s for a BF16 MLX copy on that chip. This record is fp8 E4M3 and has no draft model.
- Measurement. [DaBench](https://dabench.ai/model/spark-x-2-6-1-7b) reports 106 tok/s for llama.cpp Q8_0 on an M5 Pro with 48 GB, 30 October 2026, with reasoning on. [LiteRT-LM](https://huggingface.co/litert-community/Spark-X2.6-1.7B) reports 151.5 tok/s decode for its int8 file on an M4 Max with 128 GB. They do not publish this harness run.
- Mechanism. The carrot cases are greedy, then the `none`, `high`, and `low` cards. The coding agent uses the `high` template and the card sampler. The long thinking request in that agent decoded 731 tokens at 73.40 tok/s.

## Rule

Prefill is the forward of the named prompt. Decode is the sampled-token time after that forward. The carrot rows are the lines `tests/test_spark_x26_1_7b_smoke.py` printed on 1 November 2026. The coding-agent row is the harness summary of the same day's run: 2,663 prompt tokens and 1,534 sampled tokens. The weights were the single resident buffer of `weights.bin`.

## Command

```text
uv run python -m pytest tests/test_spark_x26_1_7b_smoke.py -s
uv run python -m harness.coding --model spark-x26-1-7b
```

The pytest command prints the carrot rows. The coding command prints `passed` when `ledger.py` `collect` returns 6, 0, and -6. No committed command reprints the 89.65 and 76.02 tok/s rows.

## Table

| Run | Prefill | Decode |
|---|---|---|
| Carrot, greedy | 1797.22 tok/s (56 tok in 0.03 s) | 89.65 tok/s (66 tok in 0.74 s) |
| Carrot, none | 2269.79 tok/s (56 tok in 0.02 s) | 82.89 tok/s (63 tok in 0.76 s) |
| Carrot, high | 2156.72 tok/s (107 tok in 0.05 s) | 81.64 tok/s (180 tok in 2.20 s) |
| Carrot, low | 2140.70 tok/s (107 tok in 0.05 s) | 81.19 tok/s (100 tok in 1.23 s) |
| Coding agent | 1256.13 tok/s (2663 tok in 2.12 s) | 76.02 tok/s (1534 tok in 20.18 s) |

## Search

1 November 2026. Queries: "Spark-X2.6-1.7B tok/s decode prefill llama.cpp MLX Apple 2026", "Spark-X2.6-1.7B M5 Pro MLX INT4 tok/s decode", and "Spark-X2.6-1.7B 89.65 OR 76.02 OR 73.40 OR 123.60 tok/s". The predecessors in this note are the overlaps those queries returned. The search did not find 89.65, 76.02, or 73.40 tok/s.

## Corrections

Later finds are appended here. The sections above stay as committed.
