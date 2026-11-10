# Page needle

Measurement note for the `qwen38_27b` pipeline. The serving contract is [../README.md](../README.md). This file records page-mode needle runs. It is a recorded run, not a format bound. Page mode is a long-transcript rate. It is not a retrieval claim.

## Remainder

On an M5 Pro with 48 GB, `Qwen3.8-27B-fp8-e20` with MTP on at depth 3 passed page needles at depth 0.5. At 65536 tokens, `half` keys and values summed to 127.11 tok/s prefill and 4.96 tok/s decode, and `fp8_e4m3` summed to 320.80 tok/s prefill and 13.32 tok/s decode. At 131072 tokens on `fp8_e4m3` the sums are 232.28 tok/s prefill and 10.42 tok/s decode. A search on 3 November 2026 did not find that table.

## Predecessors

- Method. [Hsieh et al., RULER](https://arxiv.org/abs/2404.06654) and [Kamradt's needle](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) plant a fact in one prompt. This harness pages `haystack.txt` with `read_file` until a window shows the magic-number line, then replies. A pass needs the number and a read of that line. The number may sit in the newest window.
- Model. [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). The card states a 262144-token context. It does not publish this page-needle rate table.
- Task score. [YoozLabs/Qwen3.8-27B-lean-4bit-mlx](https://huggingface.co/YoozLabs/Qwen3.8-27B-lean-4bit-mlx) reports NIAH single 1.000 at 65K on an M4 Pro with q8_0 keys and values. [leoncca/Qwen3.8-27B-AEON-Mixed-FP8](https://huggingface.co/leoncca/Qwen3.8-27B-AEON-Mixed-FP8) reports 4/4 needle retrieval at 8K through 246K. [RAMGPT on NInfer](https://ramgpt.org/articles/qwen38-27b-ninfer-rtx4090-5000-prefill-260k/) reports PASS at 64512 prompt tokens with 3.84k tok/s prefill and 146.4 tok/s decode on one RTX 4090. [Anvil Serving](https://fakoli.github.io/anvil-serving/findings/2026-08-17-qwen38-27b-radixark-nvfp4-rtx5090-128k/) reports a 119675-token retrieval pass in 29.811 s on one RTX 5090. Those scores are not this paging protocol and not this Metal plane pair.
- Rate. [oMLX on an M5 Pro with 20 GPU cores and 64 GB](https://omlx.ai/benchmarks/performance/w2yt01xn) reports 312.1 tok/s prefill and 9.8 tok/s decode at 131072 context. [oMLX issue 2795](https://github.com/jundot/omlx/issues/2795) reports 231.7 tok/s prefill and 27.0 tok/s decode at 16K on an M3 Max with 128 GB. [How To Spark / DGX playbook](https://github.com/tsarihan/qwen3.8-27b-dgx-spark-gb10-playbook) reports 625 tok/s prefill and 38.6 tok/s decode at 128K for a YaRN needle. [NInfer](https://github.com/Neroued/ninfer/blob/master/docs/performance/qwen3.8-27b.md) reports 2964.4 tok/s prefill and 76.8 tok/s decode at 64512 prompt tokens on one RTX 5090. None of those runs is this page-needle sum on an M5 Pro with 48 GB.

## Rule

`--mode page` is the default. Thinking is off. MTP is on at depth 3. Seed is 0. Depth is 0.5. The loaded plane is the registry `kv_cache` after aliases. An omitted registry cache uses `fp8_e4m3`. This table is the rows whose banners printed `mtp on` with the named plane. A banner that prints `mtp off`, thinking on, or another depth is a different row.

The file is replayed inside tool results, so the finished transcript has to fit the 131072-token served window. The 65536-token runs closed near 46000 prompt tokens over five requests. The 131072-token run closed at 92053 prompt tokens over nine requests.

## Command

```text
uv run python -m harness.needle --model qwen38-27b --tokens 65536 --mtp on --depth 0.5
uv run python -m harness.needle --model qwen38-27b --tokens 131072 --mtp on --depth 0.5
```

The banner prints the loaded plane. Set the registry `kv_cache` to `half` or `fp8_e4m3` before each row when the plane is not already the registry default. These rows are the banners that printed those planes with `mtp on`.

## Table

| Plane | Tokens | Depth | MTP | Result | Prefill | Decode |
|---|---|---|---|---|---|---|
| half | 65536 | 0.5 | on | pass | 127.11 tok/s (45871 tok in 360.88 s) | 4.96 tok/s (157 tok in 31.66 s) |
| fp8_e4m3 | 65536 | 0.5 | on | pass | 320.80 tok/s (45871 tok in 142.99 s) | 13.32 tok/s (218 tok in 16.37 s) |
| fp8_e4m3 | 131072 | 0.5 | on | pass | 232.28 tok/s (91737 tok in 394.94 s) | 10.42 tok/s (323 tok in 31.01 s) |

## Search

3 November 2026. Queries: "Qwen3.8-27B needle-in-a-haystack RULER passkey 65536 tok/s", "Qwen3.8-27B Metal Apple Silicon tok/s prefill decode MTP fp8", "320.80 OR 13.32 OR 127.11 OR 4.96 Qwen3.8-27B needle tok/s", "Qwen3.8-27B needle-in-a-haystack 64k NIAH RULER", "232.28 OR 10.42 OR 92053 Qwen3.8-27B needle page tok/s 131072", and "232.28 tok/s 10.42 tok/s Qwen3.8-27B page needle fp8_e4m3 131072 M5 Pro". The predecessors in this note are the overlaps those queries returned, including the oMLX 312.1 / 9.8 pair at 131072 on an M5 Pro with 64 GB. The search did not find 127.11 and 4.96, or 320.80 and 13.32, or 232.28 and 10.42, in those sources.

## Corrections

Later finds are appended here. The sections above stay as committed.

### 5 November 2026 — MTP off

Same page mode, `fp8_e4m3`, thinking off, 65536 tokens, depth 0.5, seed 0. MTP is off. The run passed. Prefill summed to 336.42 tok/s and decode to 11.57 tok/s over five requests that closed at 46021 prompt tokens. A search on 5 November 2026 did not find that pair. Queries: "336.42 OR 11.57 Qwen3.8-27B needle page tok/s fp8_e4m3 MTP off 65536", and "\"336.42\" \"11.57\" Qwen3.8-27B needle tok/s". The overlaps were unrelated numbers and the rate reports already named above, including NInfer at 64512 and the DGX Spark FP8+MTP pair at 264.55 / 203.01. None is this page-needle sum with MTP off on an M5 Pro with 48 GB.

```text
uv run python -m harness.needle --model qwen38-27b --tokens 65536 --depth 0.5 --mtp off --thinking off --mode page
```

| Plane | Tokens | Depth | MTP | Thinking | Result | Prefill | Decode |
|---|---|---|---|---|---|---|---|
| fp8_e4m3 | 65536 | 0.5 | off | off | pass | 336.42 tok/s (45871 tok in 136.35 s) | 11.57 tok/s (157 tok in 13.57 s) |

### 5 November 2026 — MTP on rerun

Same page mode, `fp8_e4m3`, thinking off, 65536 tokens, depth 0.5, seed 0. MTP is on at depth 3. The run passed. Prefill summed to 319.21 tok/s and decode to 13.16 tok/s over five requests that closed at 46021 prompt tokens. A search on 5 November 2026 did not find that pair. Queries: "319.21 OR 13.16 Qwen3.8-27B needle page tok/s fp8_e4m3 MTP 65536", and "\"319.21\" \"13.16\" Qwen3.8-27B needle tok/s". The overlaps were unrelated numbers and the rate reports already named above, including NInfer at 64512 and the DGX Spark FP8+MTP pair at 264.55 / 203.01. None is this page-needle sum with MTP on on an M5 Pro with 48 GB.

```text
uv run python -m harness.needle --model qwen38-27b --tokens 65536 --depth 0.5 --mtp on --thinking off --mode page
```

| Plane | Tokens | Depth | MTP | Thinking | Result | Prefill | Decode |
|---|---|---|---|---|---|---|---|
| fp8_e4m3 | 65536 | 0.5 | on | off | pass | 319.21 tok/s (45871 tok in 143.70 s) | 13.16 tok/s (157 tok in 11.93 s) |
