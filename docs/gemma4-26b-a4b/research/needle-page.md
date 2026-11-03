# Page needle

Measurement note for the `gemma4-26b-a4b` pipeline. The serving contract is [../README.md](../README.md). This file records a page-mode needle run on `fp8_e4m3` with MTP off. It is a recorded run, not a format bound. Page mode is a long-transcript rate. It is not a retrieval claim.

## Remainder

On an M5 Pro with 48 GB, `gemma-4-26B-A4B-it` with `fp8_e4m3` keys and values, MTP off, and thinking off passed a 65536-token page needle at depth 0.5. Prefill summed to 54.25 tok/s and decode to 1.30 tok/s. A search on 3 November 2026 did not find that table.

## Predecessors

- Method. [Hsieh et al., RULER](https://arxiv.org/abs/2404.06654) and [Kamradt's needle](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) plant a fact in one prompt. This harness pages `haystack.txt` with `read_file` until a window shows the magic-number line, then replies. A pass needs the number and a read of that line. The number may sit in the newest window.
- Model. [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it). The card reports 44.1% on MRCR v2, 8 needles at 128k. That score is not this page-needle rate.
- Rate. [oMLX gemma-4-26B-A4B-it-8bit at 65,536 tokens](https://omlx.ai/benchmarks/performance/nwc7hy4n) reports 1,463 tok/s prefill and 22.5 tok/s decode on an M5 Pro with 64 GB. [A second 65,536-token row](https://omlx.ai/benchmarks/performance/kqesi9le) reports 1,462 and 22.9. [DGX Spark Autobench](https://gauravmm.github.io/autobench/configs/gemma-4-26b-a4b-it-vllm-fp8/) reports 362.64 tok/s prefill and 316.93 tok/s decode for FP8 weights at concurrency 32 and a 65,536 context, aggregate over ShareGPT. None of those runs is this page-needle sum on the fp8 record with `fp8_e4m3` keys and values.

## Rule

`--mode page` is the default. Thinking is off. MTP is off. Seed is 0. Depth is 0.5. The loaded plane is the registry `kv_cache` after aliases. An omitted registry cache uses `fp8_e4m3`. This table is the row whose banner printed `kv fp8_e4m3`, `mtp off`, `thinking off`, and `mode page`. A banner that prints `half`, `mtp on`, thinking on, or another depth is a different row.

The file is replayed inside tool results, so the finished transcript has to fit the served window. This run closed at 45000 prompt tokens over five requests. Four reads reached the needle line.

## Command

```text
uv run python -m harness.needle --model gemma4-26b-a4b --tokens 65536 --depth 0.5 --mtp off --thinking off --mode page
```

The banner prints the loaded plane, MTP, thinking, and mode.

## Table

| Plane | Tokens | Depth | MTP | Thinking | Result | Prefill | Decode |
|---|---|---|---|---|---|---|---|
| fp8_e4m3 | 65536 | 0.5 | off | off | pass | 54.25 tok/s (44894 tok in 827.55 s) | 1.30 tok/s (117 tok in 90.32 s) |

## Search

3 November 2026. Queries: "Gemma 4 26B A4B needle in a haystack 65536 tok/s M5 Pro fp8" and "54.25 OR 1.30 Gemma 4 26B page needle fp8_e4m3 65536 tok/s". The predecessors in this note are the overlaps those queries returned. The search did not find 54.25 and 1.30 in those sources.

## Corrections

Later finds are appended here. The sections above stay as committed.
