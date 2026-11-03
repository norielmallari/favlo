# Retrieve needle

Measurement note for the `qwen36-35b-a3b` pipeline. The serving contract is [../README.md](../README.md). This file records one retrieve-mode needle run. It is a recorded run, not a format bound.

## Remainder

On an M5 Pro with 48 GB, `Qwen3.6-35B-A3B-fp8-e20` with `fp8_e4m3` keys and values passed a 65536-token retrieve at depth 0 and depth 0.5. The sums are 650.84 and 655.76 tok/s prefill, and 52.82 and 56.73 tok/s decode. A search on 3 November 2026 did not find that table.

## Predecessors

- Method. [Hsieh et al., RULER](https://arxiv.org/abs/2404.06654) and [Kamradt's needle](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) plant a fact in one prompt. This harness pages `haystack.txt` with `read_file` until a result has no `next_offset`, then requires the last window to start after the needle line. A RULER 128K row found the same day is [Qwen3.5-35B-A3B-Base at 88.3](https://www.sota2.com/research/sota/long-context-understanding-on-ruler-128k), not this checkpoint.
- Model. [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) and [Qwen/Qwen3.6-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8). The FP8 card states a 262144-token context and block-size 128 fp8 weights. It does not publish this retrieve table.
- Task score. [ZoliBen Csupra](https://zoliben.com/en/posts/2026-04-23-qwen-36-35b-vs-27b-benchmark-results/) reports one-shot needle-in-a-haystack at 100% (74/74) for Qwen3.6-35B-A3B IQ4_XS with turbo3 keys and values, including 64k at 0%, 25%, 50%, and 100% depth and 128k at 0%, 25%, 50%, 75%, and 100%. That score is not `fp8_e4m3` and not this paging protocol.
- Rate. [How To Spark](https://howtospark.com/recipes/qwen3-6-35b-a3b-fp8) reports 3.6k tok/s prefill and 33.3 tok/s decode at 128K for the official FP8 checkpoint on one DGX Spark. [NInfer](https://github.com/Neroued/ninfer/blob/master/model-cards/Qwen3.6-35B-A3B-NInfer/README.md) reports 11,758.0 tok/s prefill and 298.1 tok/s decode at 64,512 prompt tokens, INT8 group-64 keys and values, on one RTX 5090, measured 2026-09-07. [Hugging Face discussion 17](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8/discussions/17) reports 64.3 tok/s decode at 121604 prompt tokens for the official FP8 checkpoint with fp8 keys and values and MTP 3, on 4× RTX 5060 Ti, measured 2026-08-14. None of those runs is this retrieve sum.

## Rule

`--mode retrieve` follows `next_offset` until a read has none. A pass needs the number, a read of its line, a last read with no `next_offset`, and a last window whose first line is after the needle line. Depth 1 has no later line, so it was not run. Thinking is off. MTP is off. Seed is 0. The loaded plane is the registry `kv_cache` after aliases. This table is the run whose banner printed `kv fp8_e4m3`. A banner that prints `half`, `e2m1`, or `f32` is a different row.

The file is replayed inside tool results, so the finished transcript has to fit the 131072-token served window. A 131072-token file on this plane stopped with `input does not fit` at request 12 and is not a row. The 65536-token transcripts closed at 84582 and 84555 prompt tokens. An empty page at the end of the file is not a window.

## Command

```text
uv run python -m harness.needle --model qwen36-35b-a3b --tokens 65536 --mode retrieve --depth 0
uv run python -m harness.needle --model qwen36-35b-a3b --tokens 65536 --mode retrieve --depth 0.5
```

The banner prints the loaded plane. These rows are the banner that printed `kv fp8_e4m3`.

## Table

| Tokens | Depth | Result | Prefill | Decode |
|---|---|---|---|---|
| 65536 | 0 | pass | 650.84 tok/s (84167 tok in 129.32 s) | 52.82 tok/s (506 tok in 9.58 s) |
| 65536 | 0.5 | pass | 655.76 tok/s (84239 tok in 128.46 s) | 56.73 tok/s (333 tok in 5.87 s) |

## Search

3 November 2026. Queries: "Qwen3.6-35B-A3B needle-in-a-haystack RULER passkey 65536 fp8 agent read_file next_offset", "Qwen3.6-35B-A3B needle in a haystack 128k IQ4_XS 100% zoliben RULER", "Qwen3.6-35B-A3B-FP8 DGX Spark 33.3 tok/s 128K needle How To Spark", "647.64 OR 648.99 OR 54.07 tok OR 55.59 tok Qwen3.6 needle", and "647.64 OR 648.99 OR 54.07 OR 55.59 Qwen3.6-35B-A3B needle retrieve tok/s". The predecessors in this note are the overlaps those queries returned. The search did not find 650.84 and 52.82, or 655.76 and 56.73, in those sources. A later query the same day, "650.84 OR 655.76 OR 56.73 Qwen3.6-35B-A3B needle retrieve tok/s fp8", returned Spark and vLLM fp8 rates, including 64.3 tok/s decode at 121604 prompt tokens, and did not return this table.

## Corrections

Later finds are appended here. The sections above stay as committed.

3 November 2026. An earlier pass the same day, before an empty trailing page was ignored, printed 648.99 tok/s prefill and 55.59 tok/s decode at depth 0, and 647.64 tok/s prefill and 54.07 tok/s decode at depth 0.5. The table above is the rerun. Queries that named 648.99, 647.64, 55.59, and 54.07 did not find those sums either.
