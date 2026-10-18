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

## Attention weights

Softmax writes float32 weights. `matmul2d` does not accept a float activation against fp8 values, so each value-mix window divides by that row's peak, multiplies in float16, and restores the peak. These times are one command buffer each, the median of five dispatches after one discarded sample. The value stride is 262144. `kv_pv_split8` is one 512-key block and eight query rows. The streaming bandwidth above is 254.6 GB/s.

| Context | `kv_softmax_max` | `kv_softmax_prob` | `kv_softmax_close` | `kv_cast_probs` | `kv_pv_split8` | `kv_scale_partial` |
|---|---:|---:|---:|---:|---:|---:|
| 2048 | 8.4 | 15.6 | 15.1 | 21.8 | 30.2 | 7.1 |
| 8192 | 8.8 | 15.8 | 15.1 | 21.9 | 30.3 | 7.5 |
| 32768 | 12.0 | 22.2 | 17.2 | 22.0 | 30.3 | 7.0 |
| 65536 | 5.2 | 13.3 | 7.6 | 7.6 | 10.4 | 2.9 |
| 262144 | 10.9 | 59.6 | 13.2 | 7.6 | 11.0 | 2.7 |

Times are microseconds. A figure under about 30 µs is dispatch and clock noise: the 512-key block does not get slower as the cache stride stays 262144.

One decode token uses one softmax span for the live context, then `context / 512` blocks of cast, value mix, and scale. That is one head. The language stack has 4 key/value heads and 16 full-attention layers, so 64 of those head steps. Each kernel is timed alone, so the sum counts one dispatch per block. At 65536 tokens the softmax is 26.1 µs and one block is 20.9 µs, and 128 blocks make 2.70 ms per head. Across 64 heads that isolated sum is 173 ms, 5.8 tok/s. A decode step encodes the blocks in one command buffer. This sum is not that step.

The bytes of one head at context C are the float scores read twice (64 C), the float weight write (32 C), the cast reread (32 C), the float16 window write and read (32 C), the fp8 values and their scales (264 C), and the partial write plus the scale restore (48 C). That is 472 C bytes per head, 30208 C bytes for 64 heads. At 65536 tokens that is 1.844 GiB. Divided by 254.6 GB/s the bandwidth ceiling of this traffic is 7.78 ms, 129 tok/s. At 262144 tokens it is 31.1 ms, 32.2 tok/s.

One prefill value-mix tile, `prefill_fp8_n64_stride`, is 12288 rows by 2048 keys by 256 channels and measured 614.6 µs. That is one head of one 2048-token chunk. Sixty-four such tiles, each in its own command buffer, are 39.3 ms.
