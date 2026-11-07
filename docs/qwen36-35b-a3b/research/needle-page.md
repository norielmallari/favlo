# Page needle

Measurement note for the `qwen36-35b-a3b` pipeline. The serving contract is [../README.md](../README.md). This file records one page-mode needle run on `half` with MTP off. It is a recorded run, not a format bound. Page mode is a long-transcript rate. It is not a retrieval claim.

## Remainder

On an M5 Pro with 48 GB, `Qwen3.6-35B-A3B-fp8-e20` with `half` keys and values, MTP off, and thinking off passed a 65536-token page needle at depth 0.5. Prefill summed to 374.79 tok/s and decode to 53.07 tok/s. A search on 7 November 2026 did not find that table.

## Predecessors

- Method. [Hsieh et al., RULER](https://arxiv.org/abs/2404.06654) and [Kamradt's needle](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) plant a fact in one prompt. This harness pages `haystack.txt` with `read_file` until a window shows the magic-number line, then replies. A pass needs the number and a read of that line. The number may sit in the newest window.
- Model. [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) and [Qwen/Qwen3.6-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8). The FP8 card states a 262144-token context. It does not publish this page-needle rate.
- Task score. [ZoliBen Csupra](https://zoliben.com/en/posts/2026-04-23-qwen-36-35b-vs-27b-benchmark-results/) reports one-shot needle-in-a-haystack at 100% (74/74) for Qwen3.6-35B-A3B IQ4_XS with turbo3 keys and values, including 64k at 50% depth. That score is not `half` and not this paging protocol.
- Rate. The same harness in retrieve mode on `fp8_e4m3` recorded 655.76 tok/s prefill and 56.73 tok/s decode at 65536 tokens and depth 0.5 on 3 November 2026. That row is in [needle-retrieve.md](needle-retrieve.md). [oMLX Qwen3.6-35B-A3B-oQ4-fp16-mtp at 65,536 tokens](https://omlx.ai/benchmarks/performance/z6lqzgn2) reports 1,263 tok/s prefill and 51.0 tok/s decode on an M5 Pro with 64 GB. [oMLX Qwen3.6-35B-A3B-4bit at 65,536 tokens](https://omlx.ai/benchmarks/performance/ztqi9ews) reports 1,079 tok/s prefill and 45.9 tok/s decode with 4-bit TurboQuant keys and values on an M5 Pro with 64 GB. [NInfer](https://github.com/Neroued/ninfer/blob/master/model-cards/Qwen3.6-35B-A3B-NInfer/README.md) reports 11,758.0 tok/s prefill and 298.1 tok/s decode at 64,512 prompt tokens, INT8 group-64 keys and values, on one RTX 5090, measured 2026-09-07. [How To Spark](https://howtospark.com/recipes/qwen3-6-35b-a3b-fp8) reports 3.6k tok/s prefill and 33.3 tok/s decode at 128K for the official FP8 checkpoint on one DGX Spark. None of those runs is this page-needle sum on `half`.

## Rule

`--mode page` is the default. Thinking is off. MTP is off. Seed is 0. Depth is 0.5. The loaded plane is the registry `kv_cache` after aliases. This table is the row whose banner printed `kv half`, `mtp off`, `thinking off`, and `mode page`. A banner that prints `fp8_e4m3`, `e2m1`, `f32`, `mtp on`, thinking on, or another depth is a different row.

The file is replayed inside tool results, so the finished transcript has to fit the 131072-token served window. This run closed at 46021 prompt tokens over five requests. Four reads reached the needle line.

## Command

```text
uv run python -m harness.needle --model qwen36-35b-a3b --tokens 65536 --depth 0.5 --mtp off --thinking off --mode page
```

The banner prints the loaded plane, MTP, thinking, and mode.

## Table

| Plane | Tokens | Depth | MTP | Thinking | Result | Prefill | Decode |
|---|---|---|---|---|---|---|---|
| half | 65536 | 0.5 | off | off | pass | 374.79 tok/s (45871 tok in 122.39 s) | 53.07 tok/s (164 tok in 3.09 s) |

## Search

7 November 2026. Queries: "374.79 OR 53.07 Qwen3.6-35B-A3B needle page half tok/s", "Qwen3.6-35B-A3B half kv cache needle in a haystack 65536 page tok/s M5 Pro", "374.79 OR 53.07 Qwen3.6 needle half OR f16 OR page", and "Qwen3.6-35B-A3B f16 KV needle haystack read_file page 65536 tok/s M5 Pro 48". The predecessors in this note are the overlaps those queries returned. The search did not find 374.79 and 53.07 in those sources.

## Corrections

Later finds are appended here. The sections above stay as committed.
