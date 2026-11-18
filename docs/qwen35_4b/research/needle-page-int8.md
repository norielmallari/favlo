# Page needle, int8

Measurement note for the CUDA runtime of `qwen35-4b`. The serving contract is [../README.md](../ARCHITECTURE.md). This file records one page-mode needle run on int8 keys and values with MTP off. It is a recorded run, not a format bound. Page mode is a long-transcript rate. It is not a retrieval claim. The Metal page needle is in [../../metal/research/needle-page.md](../../metal/research/needle-page.md).

## Remainder

On a GeForce RTX 3070 Laptop GPU with 8192 MiB, `Qwen3.5-4B` with int8 keys and values, MTP off, and thinking off passed a 65536-token page needle at depth 0.5. Prefill summed to 483.64 tok/s and decode to 15.50 tok/s. A search on 7 November 2026 did not find that table.

## Predecessors

- Method. [Hsieh et al., RULER](https://arxiv.org/abs/2404.06654) and [Kamradt's needle](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) plant a fact in one prompt. This harness pages `haystack.txt` with `read_file` until a window shows the magic-number line, then replies. A pass needs the number and a read of that line. The number may sit in the newest window.
- Model. [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B). The card states a native context of 262,144 tokens and says throughput varies by framework. It does not publish this page-needle rate table.
- Rate. [Andrew Rose](https://andrewroseit.com/posts/e21j-multikey-needle-4b-context-curve/) records a Q4_K_M multikey needle on an RTX 3080 10 GB with a q4_0 cache: the 128k depth-50% row prefilled at 1,969 tok/s and passed. [whisper-bye](https://gist.github.com/whisper-bye/5bcf1ce721b55700451a474932d3a9bd) records Q4_K_M generation on an RTX 4090 with a q8_0 cache; the 65536 section lists 135.081 tok/s and a 7.717 s time to first token. [ModelFit](https://modelfit.io/models/qwen3.5-4b) reports 93.0 tok/s for Q4_K_M under Ollama on a GeForce RTX 3070. [Will It Run AI](https://willitrunai.com/can-run/qwen-3.5-4b-on-rtx-3070-8gb) estimates about 55.9 tok/s decode for Q4_K_M, and [GPUfit](https://gpufit.tech/can-i-run/qwen3-5-4b/on/rtx-3070/) estimates 40–54 tok/s for UD-Q8_K_XL. The Metal page needle, in [needle-page.md](../../metal/research/needle-page.md), recorded 864.79 tok/s prefill and 25.81 tok/s decode on an M5 Pro with `fp8_e4m3`. None of those is this int8 page-needle sum.

## Rule

`--mode page` is the default. Thinking is off. MTP is off. Seed is 0. Depth is 0.5. The loaded plane is the one the banner prints. This table is the row whose banner printed `kv int8`, `mtp off`, `thinking off`, and `mode page`. A banner that prints `fp8_e4m3`, `mtp on`, or thinking on is a different row.

The file is replayed inside tool results, so the finished transcript has to fit the served window. This run closed at 34833 prompt tokens over six requests.

## Command

```text
uv run python -m harness.needle --model qwen35-4b --tokens 65536 --depth 0.5 --mtp off --thinking off --mode page
```

The banner prints the loaded plane, MTP, thinking, and mode.

## Table

| Plane | Tokens | Depth | MTP | Thinking | Result | Prefill | Decode |
|---|---|---|---|---|---|---|---|
| int8 | 65536 | 0.5 | off | off | pass | 483.64 tok/s (46449 tok in 96.04 s) | 15.50 tok/s (242 tok in 15.61 s) |

## Search

7 November 2026. Queries: "Qwen3.5-4B needle-in-a-haystack CUDA RTX 3070 tok/s int8 2026", "483.64 OR 15.50 Qwen3.5-4B needle page tok/s", "Qwen3.5-4B 65536 needle prefill decode tokens per second CUDA GeForce 3070", "Qwen3.5-4B RULER multikey needle llama.cpp q8_0 prompt eval RTX 3070 int8 page", and "Qwen3.5-4B model card needle-in-a-haystack long context throughput 262144". The predecessors in this note are the overlaps those queries returned. The search did not find 483.64 and 15.50 in those sources.

## Corrections

Later finds are appended here. The sections above stay as committed.
