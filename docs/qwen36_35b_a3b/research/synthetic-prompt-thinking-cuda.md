# Synthetic prompt, thinking

Measurement note for the CUDA runtime of `qwen36_35b_a3b`. The serving contract is [../ARCHITECTURE.md](../ARCHITECTURE.md). This file records one synthetic-prompt run. It is a recorded run, not a format bound. The `fp8_e4m3` instruct run is in [synthetic-prompt.md](synthetic-prompt.md). The `half` instruct run is in [synthetic-prompt-half.md](synthetic-prompt-half.md).

## Remainder

- On a GeForce RTX 3070 Laptop GPU with 8192 MiB, `Qwen3.6-35B-A3B-int8-e10` with int8 keys and values and MTP off passed a cold thinking prompt. 1024 of 10240 routed experts stay int8 and the rest stay int4.
- A 512-token prompt prefilled at 328.12 tok/s and decoded 128 tokens at 19.02 tok/s.
- A 1024-token prompt prefilled at 311.10 tok/s and decoded 128 tokens at 16.05 tok/s.
- A 4096-token prompt prefilled at 282.56 tok/s and decoded 128 tokens at 9.36 tok/s.
- A 10000-token prompt prefilled at 241.90 tok/s and decoded 128 tokens at 4.98 tok/s.
- The end-to-end rates, prefill plus decode, were 77.20, 102.22, 149.89, and 151.05 tok/s.
- A search on 19 November 2026 did not find that table.

## Predecessors

- Method. [llama-bench](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md) defaults to a 512-token prompt and 128 generated tokens. [`mlx_lm.benchmark`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/benchmark.py) uses the same prompt default. [oMLX](https://omlx.ai/benchmarks/performance/x0t6bie9) reports 1024-token and 4096-token sessions. [Artificial Analysis](https://artificialanalysis.ai/methodology/performance-benchmarking) times a prompt of about 10000 tokens and reports time to first token, not prefill tok/s.
- Model. [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). The card publishes task scores. It does not publish this harness rate table.
- Rate. The same harness on `half` keys and values recorded 691.10 / 71.36 tok/s at 512 tokens and 1317.11 / 67.50 tok/s at 10000 tokens on an M5 Pro with 48 GB. That table is [synthetic-prompt-half.md](synthetic-prompt-half.md). The `fp8_e4m3` table in [synthetic-prompt.md](synthetic-prompt.md) recorded 825.26 / 67.97 tok/s at 512 tokens and 1272.73 / 67.79 tok/s at 10000 tokens.
- Rate. [llama.cpp UD-IQ4_XS on an RTX 3070](https://llm-bench.io/benchmarks/cmvjm53mf01ql01mw91cy60wr) reports 30.5 tok/s and a prompt-processing rate of 285.7 with thinking used. [The same quant on an RTX 3070 with 27 expert layers on the CPU](https://note.com/ripe_hosta6920/n/nb81b72d54ad7) reports about 517 tok/s at pp512 and about 27 tok/s at tg128. [An RTX 3070 Mobile Q4_K_M session](https://specpicks.com/reviews/qwen-3-6-35b-a3b-krasis-8gb-laptop-local-llm-2026) reports 220 tok/s prefill and 5.6 tok/s generation for a 1024-token prompt. [Token Atlas](https://tokatlas.github.io/models/qwen3-6-35b-a3b/) lists an RTX 5090 Q4_K_M llama.cpp row at 281 tok/s decode and 8316 tok/s prompt processing. None of them is this int8-e10 thinking table.

## Rule

- MTP is off.
- Thinking is on, so the request uses the thinking card and `reasoning.effort` `high`.
- Temperature is 0.
- The banner prints the loaded plane.
- Each measured prompt is one user message and encodes to that length under the chat template, including the template tokens.
- A 64-token warmup runs before the measured prompts.
- The measured prompts start with different labels, so each prefill is a cache miss.
- Decode is 128 tokens at every length. `favlo:fill_decode` keeps sampling through stop ids until that cap.
- The end-to-end rate adds the prefill tokens to the decode tokens and adds those two clocks.
- The banner printed depth 3. That depth is unused while MTP is off.
- A banner that prints `thinking off`, `half`, `fp8_e4m3`, or `mtp on` is a different row.

## Command

```text
uv run python -m harness.synthetic --model qwen36-35b-a3b --thinking on --mtp off
```

The banner prints the loaded plane, MTP, thinking, and the decode cap.

## Table

| Prompt tokens | Prefill | Decode | End to end |
|---:|---|---|---|
| 512 | 328.12 tok/s (512 tok in 1.56 s) | 19.02 tok/s (128 tok in 6.73 s) | 77.20 tok/s (640 tok in 8.29 s) |
| 1024 | 311.10 tok/s (1024 tok in 3.29 s) | 16.05 tok/s (128 tok in 7.98 s) | 102.22 tok/s (1152 tok in 11.27 s) |
| 4096 | 282.56 tok/s (4096 tok in 14.50 s) | 9.36 tok/s (128 tok in 13.68 s) | 149.89 tok/s (4224 tok in 28.18 s) |
| 10000 | 241.90 tok/s (10000 tok in 41.34 s) | 4.98 tok/s (128 tok in 25.71 s) | 151.05 tok/s (10128 tok in 67.05 s) |

## Search

19 November 2026.

- Queries: "328.12 OR 311.10 OR 282.56 OR 241.90 Qwen3.6-35B-A3B tok/s", "Qwen3.6-35B-A3B RTX 3070 prefill tokens per second CUDA int8 benchmark 2026", and "328.12 19.02 OR 311.10 16.05 OR 282.56 9.36 Qwen3.6-35B-A3B pp512 tg128".
- The predecessors in this note are the overlaps those queries returned.
- The [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) card publishes task scores. It does not publish 328.12, 311.10, 282.56, or 241.90.
- [Token Atlas](https://tokatlas.github.io/models/qwen3-6-35b-a3b/) lists other Qwen3.6-35B-A3B rates. It does not list 328.12 and 19.02, 311.10 and 16.05, 282.56 and 9.36, or 241.90 and 4.98.
- The search did not find 328.12 and 19.02, 311.10 and 16.05, 282.56 and 9.36, or 241.90 and 4.98 in those sources.

## Corrections

Later finds are appended here. The sections above stay as committed.

19 November 2026. The same command passed on `Qwen3.6-35B-A3B-int4`, where every routed expert is int4. The banner printed `kv int8`, `mtp off`, `thinking on`, and nonce `1c5b9714`.

| Prompt tokens | Prefill | Decode | End to end |
|---:|---|---|---|
| 512 | 389.41 tok/s (512 tok in 1.31 s) | 21.84 tok/s (128 tok in 5.86 s) | 89.26 tok/s (640 tok in 7.17 s) |
| 1024 | 369.61 tok/s (1024 tok in 2.77 s) | 18.94 tok/s (128 tok in 6.76 s) | 120.88 tok/s (1152 tok in 9.53 s) |
| 4096 | 302.71 tok/s (4096 tok in 13.53 s) | 10.18 tok/s (128 tok in 12.57 s) | 161.84 tok/s (4224 tok in 26.10 s) |
| 10000 | 272.87 tok/s (10000 tok in 36.65 s) | 5.32 tok/s (128 tok in 24.04 s) | 166.88 tok/s (10128 tok in 60.69 s) |

Query: "389.41 OR 369.61 OR 302.71 OR 272.87 Qwen3.6-35B-A3B tok/s". The search did not find 389.41 and 21.84, 369.61 and 18.94, 302.71 and 10.18, or 272.87 and 5.32. [Token Atlas](https://tokatlas.github.io/models/qwen3-6-35b-a3b/) does not list those pairs. A DGX Spark total of 301.74 tok/s is not 302.71 with 10.18. The predecessors above still stand.

20 November 2026. The same command passed again on that int4 record after decode reused one activation from shared memory, split the key span across the streaming multiprocessors, and copied expert images on a side stream while the shared expert ran. The banner printed `kv int8`, `mtp off`, `thinking on`, and nonce `81badcde`.

| Prompt tokens | Prefill | Decode | End to end |
|---:|---|---|---|
| 512 | 393.69 tok/s (512 tok in 1.30 s) | 24.75 tok/s (128 tok in 5.17 s) | 98.92 tok/s (640 tok in 6.47 s) |
| 1024 | 379.38 tok/s (1024 tok in 2.70 s) | 24.13 tok/s (128 tok in 5.30 s) | 144.00 tok/s (1152 tok in 8.00 s) |
| 4096 | 313.29 tok/s (4096 tok in 13.07 s) | 20.48 tok/s (128 tok in 6.25 s) | 218.63 tok/s (4224 tok in 19.32 s) |
| 10000 | 283.46 tok/s (10000 tok in 35.28 s) | 15.99 tok/s (128 tok in 8.00 s) | 234.01 tok/s (10128 tok in 43.28 s) |

Query: "393.69 OR 379.38 OR 313.29 OR 283.46 Qwen3.6-35B-A3B tok/s". The search did not find 393.69 and 24.75, 379.38 and 24.13, 313.29 and 20.48, or 283.46 and 15.99. [Token Atlas](https://tokatlas.github.io/models/qwen3-6-35b-a3b/) does not list those pairs. An M5 Ultra oQ4e row at 313.7 tok/s is not 313.29 with 20.48. The predecessors above still stand.
