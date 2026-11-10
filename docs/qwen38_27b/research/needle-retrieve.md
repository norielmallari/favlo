# Retrieve needle

Measurement note for the `qwen38_27b` pipeline. The serving contract is [../README.md](../README.md). This file records retrieve-mode needle runs. It is a recorded run, not a format bound. Page mode is [needle-page.md](needle-page.md).

## Remainder

On an M5 Pro with 48 GB, `Qwen3.8-27B-fp8-e20` with `fp8_e4m3` keys and values and MTP off passed a 65536-token retrieve at depth 0 at 236.59 tok/s prefill and 9.33 tok/s decode. The same plane at depth 0.5 found the number but failed the end-of-file rule at 340.89 tok/s prefill and 12.39 tok/s decode. A search on 3 November 2026 did not find that table.

## Predecessors

- Method. [Hsieh et al., RULER](https://arxiv.org/abs/2404.06654) and [Kamradt's needle](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) plant a fact in one prompt. This harness pages `haystack.txt` with `read_file` until a result has no `next_offset`, then requires the last window to start after the needle line.
- Model. [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). The card states a 262144-token context. It does not publish this retrieve table.
- Task score. [RAMGPT on NInfer](https://ramgpt.org/articles/qwen38-27b-ninfer-rtx4090-5000-prefill-260k/) reports PASS at 64512 prompt tokens with 3.84k tok/s prefill and 146.4 tok/s decode on one RTX 4090. [How To Spark / DGX playbook](https://github.com/tsarihan/qwen3.8-27b-dgx-spark-gb10-playbook) reports 5/5 YaRN needles at 128K with 625 tok/s prefill and 38.6 tok/s decode. [0xWhiteMage/Qwen3.8-27B-Kearuga](https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga) reports 3/3 NIAH at 64K at depths 25%, 50%, and 90%. Those scores are not this paging protocol and not this Metal plane.
- Rate. [NInfer](https://github.com/Neroued/ninfer/blob/master/docs/performance/qwen3.8-27b.md) reports 2964.4 tok/s prefill and 76.8 tok/s decode at 64512 prompt tokens on one RTX 5090. [oMLX on an M5 Pro with 20 GPU cores and 64 GB](https://omlx.ai/benchmarks/pjsukgv8) reports 399.9 tok/s prefill and 33.4 tok/s decode at 8k with Lightning MTP. None of those runs is this retrieve sum on an M5 Pro with 48 GB.

## Rule

`--mode retrieve` follows `next_offset` until a read has none. A pass needs the number, a read of its line, a last read with no `next_offset`, and a last window whose first line is after the needle line. Depth 1 has no later line, so it was not run. Thinking is off. MTP is off. Seed is 0. The loaded plane is the registry `kv_cache` after aliases. This table is the run whose banner printed `kv fp8_e4m3` and `mtp off`. A banner that prints `half`, `mtp on`, or thinking on is a different row.

The file is replayed inside tool results, so the finished transcript has to fit the 131072-token served window. The depth-0 transcript closed at 97309 prompt tokens over twelve requests. The depth-0.5 transcript closed at 46034 prompt tokens over five requests after the agent replied with the number before paging to the end.

## Command

```text
uv run python -m harness.needle --model qwen38-27b --tokens 65536 --mode retrieve --depth 0 --mtp off
uv run python -m harness.needle --model qwen38-27b --tokens 65536 --mode retrieve --depth 0.5 --mtp off
```

The banner prints the loaded plane. These rows are the banners that printed `kv fp8_e4m3` with `mtp off`.

## Table

| Tokens | Depth | Result | Prefill | Decode |
|---|---|---|---|---|
| 65536 | 0 | pass | 236.59 tok/s (96749 tok in 408.93 s) | 9.33 tok/s (644 tok in 69.03 s) |
| 65536 | 0.5 | fail | 340.89 tok/s (45829 tok in 134.44 s) | 12.39 tok/s (246 tok in 19.85 s) |

Depth 0.5 found `7463343` on line 707. The harness rejected the run because the file was not read to the end and the last window still held the needle line.

## Search

3 November 2026. Queries: "Qwen3.8-27B needle-in-a-haystack retrieve read_file next_offset 65536 tok/s", "236.59 OR 9.33 OR 340.89 OR 12.39 Qwen3.8-27B tok/s needle retrieve M5 Pro", and "Qwen3.8-27B NIAH RULER 64k depth 0 passkey Apple Silicon fp8". The predecessors in this note are the overlaps those queries returned. The search did not find 236.59 and 9.33, or 340.89 and 12.39, in those sources.

## Corrections

Later finds are appended here. The sections above stay as committed.
