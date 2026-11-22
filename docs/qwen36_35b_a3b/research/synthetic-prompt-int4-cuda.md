# Synthetic prompt, int4

Measurement note for the CUDA runtime of `qwen36_35b_a3b`. The serving contract is [../ARCHITECTURE.md](../ARCHITECTURE.md). This file records one synthetic-prompt run. It is a recorded run, not a format bound. The int8-e10 instruct run is in [synthetic-prompt-cuda.md](synthetic-prompt-cuda.md). The thinking-on run is in [synthetic-prompt-thinking-cuda.md](synthetic-prompt-thinking-cuda.md).

## Remainder

- On a GeForce RTX 3070 Laptop GPU with 8192 MiB, `Qwen3.6-35B-A3B-int4` with int8 keys and values and MTP off passed a cold instruct prompt. Every routed expert is int4. Dense language matrices stay int8.
- A 512-token prompt prefilled at 338.60 tok/s and decoded 128 tokens at 20.36 tok/s.
- A 1024-token prompt prefilled at 328.13 tok/s and decoded 128 tokens at 17.59 tok/s.
- A 4096-token prompt prefilled at 284.92 tok/s and decoded 128 tokens at 9.79 tok/s.
- A 10000-token prompt prefilled at 263.25 tok/s and decoded 128 tokens at 5.32 tok/s.
- The end-to-end rates, prefill plus decode, were 82.05, 110.77, 153.82, and 163.22 tok/s.
- A search on 19 November 2026 did not find that table.

## Predecessors

- Method. [llama-bench](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md) defaults to a 512-token prompt and 128 generated tokens. [`mlx_lm.benchmark`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/benchmark.py) uses the same prompt default. [oMLX](https://omlx.ai/benchmarks/performance/x0t6bie9) reports 1024-token and 4096-token sessions. [Artificial Analysis](https://artificialanalysis.ai/methodology/performance-benchmarking) times a prompt of about 10000 tokens and reports time to first token, not prefill tok/s.
- Model. [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). The card publishes task scores. It does not publish this harness rate table.
- Rate. The int8-e10 instruct table in [synthetic-prompt-cuda.md](synthetic-prompt-cuda.md) recorded 369.75 / 20.50 tok/s at 512 tokens, and a later correction on that mix recorded 409.86 / 21.50 tok/s. The thinking-on int4 correction in [synthetic-prompt-thinking-cuda.md](synthetic-prompt-thinking-cuda.md) recorded 389.41 / 21.84 tok/s at 512 tokens and 272.87 / 5.32 tok/s at 10000 tokens.
- Rate. [llama.cpp UD-IQ4_XS on an RTX 3070](https://llm-bench.io/benchmarks/cmvjm53mf01ql01mw91cy60wr) reports 30.5 tok/s and a prompt-processing rate of 285.7. [The same quant on an RTX 3070 with 27 expert layers on the CPU](https://note.com/ripe_hosta6920/n/nb81b72d54ad7) reports about 517 tok/s at pp512 and about 27 tok/s at tg128. [An RTX 3070 Mobile Q4_K_M session](https://specpicks.com/reviews/qwen-3-6-35b-a3b-krasis-8gb-laptop-local-llm-2026) reports 220 tok/s prefill and 5.6 tok/s generation for a 1024-token prompt. [Token Atlas](https://tokatlas.github.io/models/qwen3-6-35b-a3b/) lists an RTX 5090 Q4_K_M llama.cpp row at 281 tok/s decode and 8316 tok/s prompt processing. [ToshLLM](https://toshllm.com/benchmarks/model/qwen3-6-35b-a3b) reports a median prompt rate of 284.7 tok/s and a median generation rate of 20.22 tok/s across its recorded GPUs. None of them is this int4 instruct table.

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
- The banner printed nonce `6145167a`.
- A banner that prints `thinking on`, `half`, `fp8_e4m3`, or `mtp on` is a different row.

## Command

```text
uv run python -m harness.synthetic --model qwen36-35b-a3b --thinking off --mtp off
```

The banner prints the loaded plane, MTP, thinking, and the decode cap.

## Table

| Prompt tokens | Prefill | Decode | End to end |
|---:|---|---|---|
| 512 | 338.60 tok/s (512 tok in 1.51 s) | 20.36 tok/s (128 tok in 6.29 s) | 82.05 tok/s (640 tok in 7.80 s) |
| 1024 | 328.13 tok/s (1024 tok in 3.12 s) | 17.59 tok/s (128 tok in 7.28 s) | 110.77 tok/s (1152 tok in 10.40 s) |
| 4096 | 284.92 tok/s (4096 tok in 14.38 s) | 9.79 tok/s (128 tok in 13.08 s) | 153.82 tok/s (4224 tok in 27.46 s) |
| 10000 | 263.25 tok/s (10000 tok in 37.99 s) | 5.32 tok/s (128 tok in 24.06 s) | 163.22 tok/s (10128 tok in 62.05 s) |

## Search

19 November 2026.

- Queries: "338.60 OR 328.13 OR 284.92 OR 263.25 Qwen3.6-35B-A3B tok/s".
- The predecessors in this note are the overlaps that query returned.
- The [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) card publishes task scores. It does not publish 338.60, 328.13, 284.92, or 263.25.
- [Token Atlas](https://tokatlas.github.io/models/qwen3-6-35b-a3b/) lists other Qwen3.6-35B-A3B rates. It does not list 338.60 and 20.36, 328.13 and 17.59, 284.92 and 9.79, or 263.25 and 5.32.
- The search did not find 338.60 and 20.36, 328.13 and 17.59, 284.92 and 9.79, or 263.25 and 5.32 in those sources.

## Corrections

Later finds are appended here. The sections above stay as committed.

20 November 2026. The same command passed again on that int4 record after decode reused one activation from shared memory, split the key span across the streaming multiprocessors, and copied expert images on a side stream while the shared expert ran. The banner printed `kv int8`, `mtp off`, `thinking off`, and nonce `3bd36702`.

| Prompt tokens | Prefill | Decode | End to end |
|---:|---|---|---|
| 512 | 367.43 tok/s (512 tok in 1.39 s) | 25.36 tok/s (128 tok in 5.05 s) | 99.38 tok/s (640 tok in 6.44 s) |
| 1024 | 341.75 tok/s (1024 tok in 3.00 s) | 24.87 tok/s (128 tok in 5.15 s) | 141.35 tok/s (1152 tok in 8.15 s) |
| 4096 | 301.15 tok/s (4096 tok in 13.60 s) | 20.32 tok/s (128 tok in 6.30 s) | 212.26 tok/s (4224 tok in 19.90 s) |
| 10000 | 275.76 tok/s (10000 tok in 36.26 s) | 16.53 tok/s (128 tok in 7.74 s) | 230.18 tok/s (10128 tok in 44.00 s) |

Query: "367.43 OR 341.75 OR 301.15 OR 275.76 Qwen3.6-35B-A3B tok/s". The search did not find 367.43 and 25.36, 341.75 and 24.87, 301.15 and 20.32, or 275.76 and 16.53. [Token Atlas](https://tokatlas.github.io/models/qwen3-6-35b-a3b/) does not list those pairs. An M3 Ultra Q4_K_M row at 365.5 tok/s and an RTX 5090 row at 301.7 tok/s are other machines. [ToshLLM](https://toshllm.com/benchmarks/model/qwen3-6-35b-a3b) reports a median prompt rate of 284.7 tok/s and a median generation rate of 20.22 tok/s. That is not this table. The predecessors above still stand.
