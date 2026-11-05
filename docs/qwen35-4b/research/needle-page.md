# Page needle

Measurement note for the `qwen35-4b` pipeline. The serving contract is [../README.md](../README.md). This file records a page-mode needle run. It is a recorded run, not a format bound. Page mode is a long-transcript rate. It is not a retrieval claim.

## Remainder

On an M5 Pro with 48 GB, `Qwen3.5-4B-fp8` with `fp8_e4m3` keys and values, MTP off, and thinking off passed a 65536-token page needle at depth 0.5. Prefill summed to 863.02 tok/s and decode to 25.58 tok/s. A search on 3 November 2026 did not find that table.

## Predecessors

- Method. [Hsieh et al., RULER](https://arxiv.org/abs/2404.06654) and [Kamradt's needle](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) plant a fact in one prompt. This harness pages `haystack.txt` with `read_file` until a window shows the magic-number line, then replies. A pass needs the number and a read of that line. The number may sit in the newest window.
- Model. [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B). The card states a long context. It does not publish this page-needle rate table.
- Rate. [oMLX Qwen3.5-4B on M4 at 64k](https://omlx.ai/benchmarks/35bhiqlc) reports 237.6 tok/s prompt processing and 10.9 tok/s generation for a 4-bit MLX build. [oMLX Qwen3.5-4B-8bit on an M5 Pro with 48 GB](https://omlx.ai/benchmarks/performance/x0t6bie9) reports 2643 tok/s prefill and 55.4 tok/s decode at 4096 context. [Token Atlas](https://tokatlas.github.io/models/qwen3-5-4b/) aggregates community Apple Silicon rates for other quants and engines. None of those runs is this page-needle sum on the fp8 record with `fp8_e4m3` keys and values.

## Rule

`--mode page` is the default. Thinking is off. MTP is off. Seed is 0. Depth is 0.5. The loaded plane is the registry `kv_cache` after aliases. An omitted registry cache uses `fp8_e4m3`. This table is the row whose banner printed `kv fp8_e4m3`, `mtp off`, and `thinking off`. A banner that prints `half`, `mtp on`, thinking on, or another depth is a different row.

The file is replayed inside tool results, so the finished transcript has to fit the served window. This run closed at 11989 prompt tokens over six requests.

## Command

```text
uv run python -m harness.needle --model qwen35-4b --tokens 65536 --depth 0.5 --mtp off --thinking off --mode page
```

The banner prints the loaded plane, MTP, thinking, and mode.

## Table

| Plane | Tokens | Depth | MTP | Thinking | Result | Prefill | Decode |
|---|---|---|---|---|---|---|---|
| fp8_e4m3 | 65536 | 0.5 | off | off | pass | 863.02 tok/s (46465 tok in 53.84 s) | 25.58 tok/s (88 tok in 3.44 s) |

## Search

3 November 2026. Queries: "Qwen3.5-4B needle-in-a-haystack 65536 tok/s Apple Silicon Metal fp8", "Qwen3.5-4B oMLX prefill decode tok/s M5 Pro 2026", and "863.02 OR 25.58 Qwen3.5-4B needle page fp8_e4m3". The predecessors in this note are the overlaps those queries returned. The search did not find 863.02 and 25.58 in those sources.

## Corrections

Later finds are appended here. The sections above stay as committed.

### 5 November 2026 — same page needle

Same plane, MTP off, thinking off, 65536 tokens, depth 0.5, seed 0. The run passed. Prefill summed to 864.79 tok/s and decode to 25.81 tok/s over six requests that closed at 11989 prompt tokens. A search on 5 November 2026 did not find that table. Queries: "864.79 OR 25.81 Qwen3.5-4B needle page 65536 tok/s", and "Qwen3.5-4B 65536 needle-in-a-haystack prefill decode tok/s M5 Pro fp8 2026". The closest 64k rate is [oMLX on an M5 with 32 GB](https://omlx.ai/benchmarks/performance/oxv7o7p8) at 681.9 tok/s prompt processing and 24.0 tok/s generation for a 4-bit build. [Andrew Rose](https://andrewroseit.com/posts/e21j-multikey-needle-4b-context-curve/) records a multikey needle pass or fail from 128k to 512k. Neither is this page-needle sum on the fp8 record with `fp8_e4m3` keys and values.

```text
uv run python -m harness.needle --model qwen35-4b --tokens 65536 --depth 0.5 --mtp off --thinking off --mode page
```

| Plane | Tokens | Depth | MTP | Thinking | Result | Prefill | Decode |
|---|---|---|---|---|---|---|---|
| fp8_e4m3 | 65536 | 0.5 | off | off | pass | 864.79 tok/s (46465 tok in 53.73 s) | 25.81 tok/s (88 tok in 3.41 s) |
