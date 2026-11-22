# Synthetic prompt

Measurement note for the CUDA runtime of `qwen36_35b_a3b`. The serving contract is [../ARCHITECTURE.md](../ARCHITECTURE.md). This file records one synthetic-prompt run. It is a recorded run, not a format bound. The thinking-on int8 run is in [synthetic-prompt-thinking-cuda.md](synthetic-prompt-thinking-cuda.md). The `half` instruct run is in [synthetic-prompt-half.md](synthetic-prompt-half.md).

## Remainder

- On a GeForce RTX 3070 Laptop GPU with 8192 MiB, `Qwen3.6-35B-A3B-int8-e10` with int8 keys and values and MTP off passed a cold instruct prompt. 1024 of 10240 routed experts stay int8 and the rest stay int4.
- A 512-token prompt prefilled at 369.75 tok/s and decoded 128 tokens at 20.50 tok/s.
- A 1024-token prompt prefilled at 324.82 tok/s and decoded 128 tokens at 17.87 tok/s.
- A 4096-token prompt prefilled at 304.49 tok/s and decoded 128 tokens at 9.61 tok/s.
- A 10000-token prompt prefilled at 260.71 tok/s and decoded 128 tokens at 5.16 tok/s.
- The end-to-end rates, prefill plus decode, were 83.99, 111.74, 157.79, and 160.38 tok/s.
- A search on 19 November 2026 did not find that table.

## Predecessors

- Method. [llama-bench](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md) defaults to a 512-token prompt and 128 generated tokens. [`mlx_lm.benchmark`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/benchmark.py) uses the same prompt default. [oMLX](https://omlx.ai/benchmarks/performance/x0t6bie9) reports 1024-token and 4096-token sessions. [Artificial Analysis](https://artificialanalysis.ai/methodology/performance-benchmarking) times a prompt of about 10000 tokens and reports time to first token, not prefill tok/s.
- Model. [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). The card publishes task scores. It does not publish this harness rate table.
- Rate. The same harness with thinking on recorded 328.12 / 19.02 tok/s at 512 tokens and 241.90 / 4.98 tok/s at 10000 tokens. That table is [synthetic-prompt-thinking-cuda.md](synthetic-prompt-thinking-cuda.md). The `half` table in [synthetic-prompt-half.md](synthetic-prompt-half.md) recorded 691.10 / 71.36 tok/s at 512 tokens and 1317.11 / 67.50 tok/s at 10000 tokens on an M5 Pro with 48 GB.
- Rate. [llama.cpp UD-IQ4_XS on an RTX 3070](https://llm-bench.io/benchmarks/cmvjm53mf01ql01mw91cy60wr) reports 30.5 tok/s and a prompt-processing rate of 285.7. [The same quant on an RTX 3070 with 27 expert layers on the CPU](https://note.com/ripe_hosta6920/n/nb81b72d54ad7) reports about 517 tok/s at pp512 and about 27 tok/s at tg128. [Token Atlas](https://tokatlas.github.io/models/qwen3-6-35b-a3b/) lists an RTX 5090 Q4_K_M llama.cpp row at 281 tok/s decode and 8316 tok/s prompt processing. [ToshLLM](https://toshllm.com/benchmarks/model/qwen3-6-35b-a3b) reports a median prompt rate of 284.7 tok/s and a median generation rate of 20.22 tok/s across its recorded GPUs. None of them is this int8-e10 instruct table.

## Rule

- MTP is off.
- Thinking is off, so the request uses the instruct card and `reasoning.effort` `none`.
- Temperature is 0.
- The banner prints the loaded plane.
- Each measured prompt is one user message and encodes to that length under the chat template, including the template tokens.
- A 64-token warmup runs before the measured prompts.
- The measured prompts start with different labels, so each prefill is a cache miss.
- Decode is 128 tokens at every length. `favlo:fill_decode` keeps sampling through stop ids until that cap.
- The end-to-end rate adds the prefill tokens to the decode tokens and adds those two clocks.
- The banner printed depth 3. That depth is unused while MTP is off.
- A banner that prints `thinking on`, `half`, `fp8_e4m3`, or `mtp on` is a different row.

## Command

```text
uv run python -m harness.synthetic --model qwen36-35b-a3b --thinking off --mtp off
```

The banner prints the loaded plane, MTP, thinking, and the decode cap.

## Table

| Prompt tokens | Prefill | Decode | End to end |
|---:|---|---|---|
| 512 | 369.75 tok/s (512 tok in 1.38 s) | 20.50 tok/s (128 tok in 6.24 s) | 83.99 tok/s (640 tok in 7.62 s) |
| 1024 | 324.82 tok/s (1024 tok in 3.15 s) | 17.87 tok/s (128 tok in 7.16 s) | 111.74 tok/s (1152 tok in 10.31 s) |
| 4096 | 304.49 tok/s (4096 tok in 13.45 s) | 9.61 tok/s (128 tok in 13.32 s) | 157.79 tok/s (4224 tok in 26.77 s) |
| 10000 | 260.71 tok/s (10000 tok in 38.36 s) | 5.16 tok/s (128 tok in 24.79 s) | 160.38 tok/s (10128 tok in 63.15 s) |

## Search

19 November 2026.

- Queries: "369.75 OR 324.82 OR 304.49 OR 260.71 Qwen3.6-35B-A3B tok/s", and "369.75 20.50 OR 260.71 5.16 Qwen3.6-35B-A3B int8 prefill RTX 3070".
- The predecessors in this note are the overlaps those queries returned.
- The [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) card publishes task scores. It does not publish 369.75, 324.82, 304.49, or 260.71.
- [Token Atlas](https://tokatlas.github.io/models/qwen3-6-35b-a3b/) lists other Qwen3.6-35B-A3B rates. It does not list 369.75 and 20.50, 324.82 and 17.87, 304.49 and 9.61, or 260.71 and 5.16.
- The search did not find 369.75 and 20.50, 324.82 and 17.87, 304.49 and 9.61, or 260.71 and 5.16 in those sources.

## Corrections

Later finds are appended here. The sections above stay as committed.

19 November 2026. The same instruct banner passed again after an empty expert slot is taken from a list, a full pool takes the smallest stamp, and the expert launch returns while its copy is still queued. A 512-token prompt prefilled at 409.86 tok/s and decoded 128 tokens at 21.50 tok/s. A 1024-token prompt prefilled at 422.93 tok/s and decoded at 19.25 tok/s. A 4096-token prompt prefilled at 386.11 tok/s and decoded at 10.81 tok/s. A 10000-token prompt prefilled at 327.16 tok/s and decoded at 5.51 tok/s. The end-to-end rates were 88.89, 127.01, 188.15, and 188.29 tok/s.

Queries: "409.86 OR 422.93 OR 386.11 OR 327.16 Qwen3.6-35B-A3B tok/s".

- [llm-bench.io](https://llm-bench.io/models/qwen3-6-35b-a3b) lists a peak of 437.3 tok/s on an RTX 5090 with vLLM. That is not 409.86 with 21.50, and it is not this int8-e10 table.
- [Token Atlas](https://tokatlas.github.io/models/qwen3-6-35b-a3b/) lists other Qwen3.6-35B-A3B rates. It does not list 409.86, 422.93, 386.11, or 327.16.
- The search did not find 409.86 and 21.50, 422.93 and 19.25, 386.11 and 10.81, or 327.16 and 5.51 in those sources.
