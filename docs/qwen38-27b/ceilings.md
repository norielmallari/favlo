# Ceilings

Kernel and bandwidth figures for the decode projections. Reuse them while `gemv_fp4_x4` and the fused projection kernels are unchanged. Recompute when one of those kernels changes.

Each GEMV time is one command buffer, then a wait. The median of five dispatches after one discarded sample. Overlapping command buffers are not added. A 2 GiB buffer copy finished in 15705 µs, which is 254.6 GB/s counting the read and the write.

## Decode projections

One token. fp4 weights, one activation. Counts are the language stack: 48 Gated DeltaNet layers, 16 full-attention layers, a feed-forward on all 64 layers, and one head. `in_proj_a` and `in_proj_b` are the stored 64-row matrices. These times are `gemv_fp4_x4` alone. The forward issues the fused `gdn_in_proj`, `attn_qkv`, and `ffn_gate_up` kernels instead, in one command buffer with the rest of the step.

| Kernel | Shape | Count | µs | ms |
|---|---|---:|---:|---:|
| `gemv_fp4_x4` qkv | 10240×5120 | 48 | 251 | 12.048 |
| `gemv_fp4_x4` z | 6144×5120 | 48 | 232 | 11.136 |
| `gemv_fp4_x4` a | 64×5120 | 48 | 168 | 8.064 |
| `gemv_fp4_x4` b | 64×5120 | 48 | 168 | 8.064 |
| `gemv_fp4_x4` out | 5120×6144 | 48 | 223 | 10.704 |
| `gemv_fp4_x4` q | 12288×5120 | 16 | 316 | 5.056 |
| `gemv_fp4_x4` k | 1024×5120 | 16 | 152 | 2.432 |
| `gemv_fp4_x4` v | 1024×5120 | 16 | 152 | 2.432 |
| `gemv_fp4_x4` o | 5120×6144 | 16 | 223 | 3.568 |
| `gemv_fp4_x4` gate | 17408×5120 | 64 | 1351 | 86.464 |
| `gemv_fp4_x4` up | 17408×5120 | 64 | 1351 | 86.464 |
| `gemv_fp4_x4` down | 5120×17408 | 64 | 1358 | 86.912 |
| `gemv_fp4_x4` head | 248320×5120 | 1 | 2552 | 2.552 |

The isolated sum is 325.9 ms, 3.07 tok/s. The small projections are mostly dispatch overhead at that size: the 64×5120 GEMV is 168 µs. The weight image of these shapes is 12,814,909,440 bytes. The convolution, the norms, and the embedding row are not in that image.

A warm language step, projections and mixers in one command buffer, measured 63.4 to 64.6 ms. That buffer is the designed decode step. It is not the sum above.

`gemv_fp4_span2` on a 17408×5120 matrix was 0.50 ms beside 0.85 ms for `gemv_fp4_x4` on the same bytes. Decode does not use it. A kept two-token state drifted from the serial DeltaNet state, and the completion changed by the tenth token.
