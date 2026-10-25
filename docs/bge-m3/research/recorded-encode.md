# Recorded encode

Measurement note for the `bge-m3` pipeline. The serving contract is [../README.md](../README.md). This file records one warm run of the packed int8 record. It is a recorded run, not a kernel sum.

## Remainder

A warm call to `Runtime.embed` recorded 9.02 ms at 12 tokens, 11.39 ms at 128, 58.56 ms at 512, and 3973.08 ms at 4096. Opening the record took 371.3 ms. A search on 25 October 2026 did not find that table.

## Predecessors

- Method. Chen et al. ([arXiv:2402.03216](https://arxiv.org/abs/2402.03216)) define the dense vector as the normalized CLS state. The int8 product is the packed GPU record in the serving note.
- Task. [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3). [FlagEmbedding issue 434](https://github.com/FlagOpen/FlagEmbedding/issues/434) reports about 28 ms to score four short pairs on one A800 after the padding fix, and 504 ms for the first call at a cap of 8192. [manicule-mlx](https://pypi.org/project/manicule-mlx/) reports 136.7 chunks/s at 128 tokens and 25.6 chunks/s at 512 tokens for `BAAI/bge-m3` at batch 32 on an M4 Max. [bge-m3-onnx](https://github.com/yuniko-software/bge-m3-onnx/blob/main/performance_comparison.ipynb) reports 6.980 ms average and 143.3 texts/s for an ONNX CUDA run over 1000 mixed texts.
- Measurement. Those figures are other engines, other batch sizes, and other sequence lengths. They do not publish this single-sequence table.
- Mechanism. The clock is the host interval around `embed`, including tokenization, the pad, the GPU wait, the readback, and the L2 norm. The rate divides the real token count by that interval. The 12-token sentence is padded to 64.

## Rule

One call is discarded, then the sample is taken. The 12-token sentence is repeated 5 times and the median is kept. The 128-token and 512-token inputs are repeated 3 times. The 4096-token input is one timed call. The longer inputs repeat the word `word` until the tokenizer count matches. The short sentence is `A stone bridge under a clear sky.`

## Command

`favlo --bge-m3 embed` prints one vector. `favlo --bge-m3 kernels` checks a synthetic tile. Neither prints this table. No committed command regenerates the rows.

## Table

| Tokens | Pad | Samples | Time | Rate |
|---:|---:|---|---:|---:|
| 12 | 64 | median of 5 | 9.02 ms | 1330.1 tok/s |
| 128 | 128 | median of 3 | 11.39 ms | 11237.4 tok/s |
| 512 | 512 | median of 3 | 58.56 ms | 8742.9 tok/s |
| 4096 | 4096 | one call | 3973.08 ms | 1030.9 tok/s |
| open |  | one call | 371.3 ms |  |

## Search

25 October 2026. Queries: "BGE-M3 embedding inference latency milliseconds tokens per second FlagEmbedding", then "bge-m3 ONNX TensorRT throughput sentences per second int8", then "FlagEmbedding bge-m3 max_passage_length 8192 cost_time A800 encode milliseconds", then "bge-m3 MLX Metal encode latency milliseconds 512 tokens Apple Silicon". The predecessors in this note are the overlaps those queries returned. The search did not find the 9.02 ms, 11.39 ms, 58.56 ms, and 3973.08 ms table.

## Corrections

Later finds are appended here. The sections above stay as committed.
