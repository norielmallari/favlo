# Recorded forward

Measurement note for the `qwen36-35b-a3b` pipeline. The serving contract is [../README.md](../README.md). The request rows are in [../performance.md](../performance.md). This file records the summed prefill and decode of those runs. The isolated dispatch sum is [kernel-ceilings.md](kernel-ceilings.md).

## Remainder

The file-agent forward, instruct card, recorded 597.98 tok/s prefill over 88004 tokens and 51.26 tok/s decode over 1298 tokens. A search on 21 October 2026 did not find that run.

## Predecessors

- Method. The forward is the same shader library as [kernel-ceilings.md](kernel-ceilings.md). Overlap, the prefix store, and the atlas are inside these times. The kernel sum leaves them out.
- Model. [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). Perplexity's [on-device note](https://www.perplexity.ai/hub/blog/optimizing-on-device-inference-for-apple-silicon) publishes end-to-end rates for this model on an Apple GPU, 5749.9 prefill tok/s and 186.6 decode tok/s at 4K tokens, on a different engine.
- Measurement. [omlx issue 2238](https://github.com/jundot/omlx/issues/2238) and [mlx-swift-lm 83ff028](https://github.com/ekryski/mlx-swift-lm/commit/83ff028def2e734bf2da9b65f51a479c5c736c3d) publish other tok/s for this model. They do not publish this harness run.
- Mechanism. The file agent and the coding agent are separate series. A 16GiB cap holds fewer than every expert image.

## Rule

Prefill is the summed suffix time. Decode is the summed sampled-token time. The file agent uses the instruct card. The coding agent uses the thinking card. The earlier file-reading series in the performance note was timed against the previous value mix and is not this table.

## Command

```text
uv run python -m harness --model qwen36-35b-a3b
uv run python -m harness.coding --model qwen36-35b-a3b
uv run python -m harness --model qwen36-35b-a3b --max-memory 16GiB
uv run python -m harness.coding --model qwen36-35b-a3b --max-memory 16GiB
```

Those commands are what the performance note names. No committed command reprints the rows below.

## Table

| Run | Prefill | Decode |
|---|---|---|
| File agent | 597.98 tok/s (88004 tok in 147.17 s) | 51.26 tok/s (1298 tok in 25.32 s) |
| Coding agent | 399.05 tok/s (419 tok in 1.05 s) | 74.06 tok/s (197 tok in 2.66 s) |
| File agent, 16GiB | 493.77 tok/s (88004 tok in 178.23 s) | 29.72 tok/s (1034 tok in 34.79 s) |
| Coding agent, 16GiB | 134.29 tok/s (419 tok in 3.12 s) | 34.19 tok/s (252 tok in 7.37 s) |

## Search

21 October 2026. Queries: "Qwen3.6-35B-A3B MLX Apple Silicon inference Gated DeltaNet", and "Qwen3.6-35B-A3B Metal kernel microsecond dispatch prefill decode ceiling matmul2d". The predecessors in this note are the overlaps those queries returned. The search did not find the 597.98 and 51.26 tok/s file-agent sums.

## Corrections

Later finds are appended here. The sections above stay as committed.

23 October 2026. On `Qwen3.6-35B-A3B-fp8-e20`, with `moe_group` splitting assignments into 16 slices, the file agent recorded 545.24 tok/s prefill over 101868 tokens and 43.18 tok/s decode over 1460 tokens. The coding agent recorded 195.71 tok/s prefill over 638 tokens and 62.93 tok/s decode over 314 tokens. The request rows are in [../performance.md](../performance.md). Queries that day: "Qwen3.6-35B-A3B file agent prefill 545 tok/s Metal Gated DeltaNet 2026". [q36](https://git.sr.ht/~gotlou/q36) publishes 448.75 tok/s prefill at 2048 tokens for a resident q2 GGUF on Metal. [SparkInfer](https://github.com/gittensor-ai-lab/sparkinfer/pull/403) reports 7412.20 tok/s at 4096 tokens for a grouped int8 prefill on an RTX 5090. An [R9700 llama.cpp note](https://deepindeepsea.github.io/r9700-vllm-blogs/part13-r9700-the-35b-only-one-runtime-could-load.html) reports 2752.05 tok/s at 512 tokens. Perplexity's on-device note remains the predecessor named above. A search on 23 October 2026 did not find 545.24 and 43.18, or 195.71 and 62.93, in those sources.

25 October 2026. On `Qwen3.6-35B-A3B-fp8-e20`, after the resident projections restored into an int8 stage, the file agent recorded 563.85 tok/s prefill over 104301 tokens and 42.56 tok/s decode over 1349 tokens. The coding agent recorded 676.53 tok/s prefill over 663 tokens and 63.48 tok/s decode over 405 tokens. The request rows are in [../performance.md](../performance.md). Queries that day: "Qwen3.6-35B-A3B file agent prefill tok/s Metal Gated DeltaNet 2026". [q36](https://git.sr.ht/~gotlou/q36) publishes 448.75 tok/s prefill at 2048 tokens for a resident q2 GGUF on Metal. [llamaperf](https://llamaperf.com/model/qwen3-6) lists 464 tok/s prefill and 62 tok/s decode at 128000 tokens on an M1 Max. [SparkInfer 530](https://github.com/gittensor-ai-lab/sparkinfer/pull/530) reports 1490 tok/s prefill at 32768 tokens on an RTX 5090. The R9700 note above reports 2752.05 tok/s at 512 tokens. A search on 25 October 2026 did not find 563.85 and 42.56, or 676.53 and 63.48, in those sources.

31 October 2026. On an M5 Pro with 48 GB, `Qwen3.6-35B-A3B-fp8-e20`, the coding agent with MTP on recorded 556.99 tok/s prefill over 3069 tokens and 43.03 tok/s decode over 1204 tokens. The harness printed `mtp on depth 3`. The language forward verifies one draft. The request rows are in [../performance.md](../performance.md). The file agent was not rerun. [Leviathan, Kalman, and Matias](https://arxiv.org/abs/2211.17192) publish the acceptance rule this run uses: keep a draft when a uniform draw falls under the target probability divided by the draft probability. [oMLX](https://omlx.ai/benchmarks/performance/77y1hqh1) reports 224.0 generation tok/s at 1k context and 183.4 at 4k for `Qwen3.6-35B-A3B-oQ4-mtp` on an M5 Max with 128 GB, Lightning MTP on, 7 October 2026. [benching-local-llms-on-apple-silicon](https://github.com/stared/benching-local-llms-on-apple-silicon) reports 105 tok/s at 128 tokens and 97 at 8k for Qwen3.6-35B-A3B Q8 with llama.cpp MTP on an M5 Max with 128 GB, 14 June 2026. [MTPLX](https://mtplx.com/models/) reports 145 tok/s at draft depth 2 and 132 at depth 3 for this model on an M5 Max. [lightning-mlx](https://github.com/samuelfaj/lightning-mlx) reports 226.01 tok/s raw decode, and 182.87 and 215.43 tok/s on short and long agentic turns, for Qwen3.6-35B. Queries that day: "Qwen3.6-35B-A3B MTP speculative decode tok/s Apple Silicon Metal 2026", then "Qwen3.6-35B-A3B 556.99 tok/s 43.03 decode Lightning MTP coding agent harness". A search on 31 October 2026 did not find 556.99 and 43.03 in those sources.

31 October 2026. Later the same day, on the same machine and record, the coding agent with MTP on recorded 561.30 tok/s prefill over 3104 tokens and 42.59 tok/s decode over 1098 tokens. The harness printed `mtp on depth 3`. The language forward verifies one draft. The request rows are in [../performance.md](../performance.md). The greedy carrot in `tests/test_qwen36_35b_a3b_smoke.py` decoded at 68.94 tok/s with MTP off, 108 tokens in 1.57 s, and at 57.54 tok/s with MTP on, 149 tokens in 2.59 s, keeping 67 of 82 drafts. The file agent was not rerun. [llama.cpp 23011](https://github.com/ggml-org/llama.cpp/issues/23011) records 26.23 tok/s without MTP and 1.93 tok/s with self-MTP for this model on an M1 Pro with 32 GB, at 95.6% acceptance. [LM Studio 1969](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1969) records 25.0 tok/s without MTP and 5.5 tok/s with MTP for this model on an M1 Max with 32 GB. [MTPLX 174](https://github.com/youssofal/MTPLX/pull/174) records 119.5 tok/s autoregressive and 199 tok/s at one draft after a compiled verify on an M5 Max, and 124.5 tok/s at two drafts against 209.9 at one draft on a coding prompt. [Youssofal's speed card](https://huggingface.co/Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed) records 94.46 tok/s autoregressive, 138.39 at depth 1, and 107.67 at depth 3. [Caleb Coffie](https://calebcoffie.com/benchmarks/qwen3-6-35b-a3b-mtp-gguf-ud-q4-k-m) records 148.1 tok/s without MTP and 162.7 tok/s at draft n=2 for an agent prompt on an RTX 3090. [AshyIsMe](https://gist.github.com/AshyIsMe/ef9dc0e57d7af5626f3935f55c38543a) records 79.6 tok/s without MTP and 135.2 tok/s with MTP at about 16k tokens on an RX 7900 XTX. Queries that day: "Qwen3.6-35B-A3B 42.59 tok/s 561.30 decode MTP coding agent harness", "Qwen3.6-35B-A3B MTP speculative decode slower than baseline Apple Silicon Metal 2026 depth 1", and "57.54 OR 68.94 OR 42.59 OR 561.30 Qwen3.6-35B-A3B MTP tok/s". A search on 31 October 2026 did not find 561.30 and 42.59, or 68.94 and 57.54, in those sources.
