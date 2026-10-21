# Float16 peak restore

Measurement note for the `qwen36-35b-a3b` pipeline. The serving contract is [../README.md](../README.md). The forward already divides each value-mix window by its peak before the float16 product and multiplies that peak back onto the partial. This file records the float16 code-point bound for that cast. It does not record the `matmul2d` product error, the fp8 value error, or a hidden-state error on the [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) checkout.

## Remainder

Softmax stores `exp(score / 16 − m)`, where `m` is the running max of the scaled scores and a score at that max has weight 1. A direct cast of that weight to float16 flushes to zero at `2^-25`. Dividing by the window peak first maps the largest weight in the window to 1, which float16 holds, and the partial is multiplied by the peak afterward. A window whose own peak is at or below `2^-25` is dropped by the direct cast and kept by the restore. A search on 21 October 2026 did not find this bound for a per-window peak ahead of a float16 value mix.

## Predecessors

- Method. FlashAttention ([arXiv:2205.14135](https://arxiv.org/abs/2205.14135)) rescales a float32 accumulator by the running max. The float16 softmax was left unused because a small weight rounds to zero ([Dao-AILab/flash-attention#613](https://github.com/Dao-AILab/flash-attention/issues/613)).
- Method. Modular's FP8 attention multiplies every probability by 256 before the value product so a tail below the fp8 floor is not stored as zero, and the same factor cancels in the row sum ([modular/modular@c3e9320](https://github.com/modular/modular/commit/c3e93203f8dd9236b5d5627ee2118235ba6c8988)). The factor is a compile-time constant, not the measured peak of one window.
- Method. BAPS ([arXiv:2602.02071](https://arxiv.org/abs/2602.02071)) rescales a block when its local max is far from the running max, and it may recompute the block.
- Measurement. IEEE binary16 rounds ties to even. The midpoint between 0 and the smallest positive subnormal `2^-24` is `2^-25`, and that midpoint rounds to 0.
- Mechanism. The row sum stays the float32 sum of the unnormalized exponentials. The peak multiplies the value partial only. Prefill takes one peak over the keys of that score window. Decode takes one peak over each 512-key block.

## Rule

`kv_cast_probs` and `kv_cast_blocks` set `peak` to the max absolute weight in the span, store `half(weight / peak)` when `peak > 0`, and store 0 when `peak` is 0. `kv_scale_partial` and `kv_restore_sum` multiply the partial by that peak. A positive peak entry divides to 1.

## Command

```text
uv run python -c "import numpy as np; x=np.float32(2.0**-25); print(float(np.float16(x)), float(np.float16(np.nextafter(x, np.float32(1)))), float(np.float16(np.float32(1))), 25*np.log(2))"
```

The three printed magnitudes are the cast of `2^-25`, the cast of the next float32 above it, and the cast of 1. The last number is the scaled-score gap `25 ln 2`.

## Table

A weight flushes when its absolute value is at most `2^-25` after the scale this cast applies. `2^-25` is `1/33554432`. The scaled-score gap is `25 ln 2`, about 17.329. The raw dot gap is 16 times that, because the exponential subtracts `score / 16`.

| Cast | Flushes | Largest weight in a positive window |
|---|---|---|
| `half(weight)` | `abs(weight) ≤ 2^-25` | Dropped when the window peak is at most `2^-25`, a scaled score at least `25 ln 2` below the running max |
| `half(weight / peak)`, then multiply the partial by `peak` | `abs(weight) ≤ peak · 2^-25` | Kept. That entry divides to 1, and float16 holds 1 |

## Search

21 October 2026. Query: "softmax attention rescale probabilities by block maximum float16 GEMM underflow". The predecessors in this note are the overlaps that query returned. The search did not find the `2^-25` bound for a measured per-window peak stored beside a float16 value mix.

## Corrections

Later finds are appended here. The sections above stay as committed.
