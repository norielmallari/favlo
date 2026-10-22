# Ceilings

Kernel and bandwidth figures for the decode projections. The fp4 rows predate the E8M0 scale bit insert in `gemv_fp4_x4` and the fused fp4 projection kernels. Recompute the fp8 rows when `gemv_fp8_x4` changes.

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

Softmax writes float32 weights. `matmul2d` does not accept a float activation against fp8 values, so each 512-key block divides by that row's peak, multiplies in float16, and restores the peak in the same dispatch. One decode head runs `kv_softmax_max`, `kv_softmax_prob`, `kv_softmax_close`, then `kv_cast_probs` over every block, `kv_pv_split8` over every block, and `kv_part_sum`. These times are one command buffer each, the median of five dispatches after one discarded sample. The value stride is 262144. The streaming bandwidth above is 254.6 GB/s.

| Context | `kv_softmax_max` | `kv_softmax_prob` | `kv_softmax_close` | `kv_cast_probs` | `kv_pv_split8` | `kv_part_sum` |
|---|---:|---:|---:|---:|---:|---:|
| 2048 | 9.7 | 15.6 | 15.4 | 22.1 | 32.7 | 10.7 |
| 8192 | 8.8 | 15.9 | 5.8 | 7.8 | 12.7 | 3.6 |
| 32768 | 4.1 | 7.9 | 6.1 | 8.9 | 30.2 | 5.5 |
| 65536 | 4.7 | 11.9 | 7.0 | 12.1 | 54.9 | 8.0 |
| 262144 | 8.8 | 39.0 | 9.7 | 27.5 | 344.8 | 16.8 |

Times are microseconds. A figure under about 30 µs is dispatch and clock noise.

One head at 65536 tokens is 98.6 µs. The language stack has 4 key/value heads and 16 full-attention layers, so 64 of those head steps, 6.31 ms, 158 tok/s. One head at 262144 tokens is 446.6 µs, and 64 heads are 28.6 ms, 35.0 tok/s. `kv_scale_partial` is the prefill restore. Decode applies the peak inside `kv_pv_split8`.

The bytes of one head at context C are the float scores read twice (64 C), the float weight write (32 C), the cast reread (32 C), the float16 window write and read (32 C), the fp8 values and their scales (264 C), the partial write and the in-place scale (48 C), and the block reduction's read of those partials (16 C). That is 488 C bytes per head, 31232 C bytes for 64 heads. At 65536 tokens that is 1.907 GiB. Divided by 254.6 GB/s the bandwidth ceiling of this traffic is 8.04 ms, 124 tok/s. At 262144 tokens it is 32.2 ms, 31.1 tok/s. The 65536 kernel sum sits under that line because most of its dispatches are under 30 µs. The 262144 sum is the one past that noise: 28.6 ms beside the 32.2 ms bandwidth ceiling.

One prefill value-mix tile, `prefill_fp8_n64_stride`, is 12288 rows by 2048 keys by 256 channels and measured 614.6 µs. That is one head of one 2048-token chunk. Sixty-four such tiles, each in its own command buffer, are 39.3 ms.

## Feed-forward fp8

`gemv_fp8_x4`. Decode launches it once per matrix. The threadgroup is 160 when the K dimension has a multiple of 160 groups of 16 bytes, and 256 otherwise. Gate and up use 160. Down uses 256. Same timing rule as the fp4 rows: one command buffer, then a wait, the median of five dispatches after one discarded sample. The copy above counts a read and a write, so one direction of that copy is 127.3 GB/s.

| Kernel | Shape | Threads | µs | GB/s |
|---|---|---:|---:|---:|
| `gemv_fp8_x4` gate or up | 17408×5120 | 160 | 354.6 | 241.6 |
| `gemv_fp8_x4` down | 5120×17408 | 256 | 336.8 | 254.4 |

The gate dispatch reads the 91,914,240-byte image and the 20,480-byte activation, and writes 69,632 bytes. A promoted layer is those two dispatches plus the down GEMV, 1.047 ms. `ffn_gate_up_fp8` at 128 threads measured 830.9 µs for both images and is not the decode launch. The fp4 feed-forward rows above stay the comparison for a layer that was not promoted. Prefill of a promoted layer is `prefill_fp8_n64`. A cold 2048×17408×5120 tile of that kernel measured 14.9 ms, beside 15.1 ms for `prefill_fp4_n64` on the fp4 image of the same shape.

## Prefill, 2048 tokens

The mixed record `Qwen3.8-27B-fp8-fp4`. Twelve language feed-forward layers are fp8 and the other fifty-two are fp4. MTP, the embedding, `lm_head`, the vision tower, and the float16 vectors are not in the projection image. The data plane plus the scale plane of the language projections is 14,544,732,160 bytes.

Gate, up, and the attention projections use a 64-row tile, so those 10,979,573,760 bytes are read 32 times in a 2048-token chunk. The down projection uses a 32-row tile, so its 3,565,158,400 bytes are read 64 times. Weight traffic is 579,516,497,920 bytes. The chunk also reads 13,891,534,848 bytes of half activations and writes 31,994,150,912 bytes of float outputs. That is 625,402,183,680 bytes. Divided by 254.6 GB/s the projection bandwidth ceiling is 2.456 s, 834 tok/s.

One full-attention head in that chunk has 12,288 query rows and one 2048-key window. The key plane and the value plane are 264 bytes per token, and each is read once per 64 query rows. Sixty-four heads move 13,287,555,072 bytes. Adding that traffic makes 638,689,738,752 bytes, 2.509 s, 816 tok/s. A longer window multiplies only this attention term.

Decode reads the projection image once and adds `lm_head`, 675,430,400 bytes. 15,220,162,560 bytes at 254.6 GB/s is 59.8 ms, 16.7 tok/s.

Each prefill time below is one command buffer, then a wait, the median of five dispatches after one discarded sample. Counts are one 2048-token chunk of the mixed record. `gdn_state_rows` keeps the state row on the thread for the chunk.

| Kernel | Shape | Count | µs | ms |
|---|---|---:|---:|---:|
| `prefill_fp4_n64` qkv | 2048×10240×5120 | 48 | 8803 | 422.56 |
| `prefill_fp4_n64` z | 2048×6144×5120 | 48 | 5351 | 256.85 |
| `prefill_fp4_n64` a | 2048×64×5120 | 48 | 133 | 6.37 |
| `prefill_fp4_n64` b | 2048×64×5120 | 48 | 134 | 6.41 |
| `prefill_fp4_n64` out | 2048×5120×6144 | 48 | 5423 | 260.28 |
| `prefill_fp4_n64` q | 2048×12288×5120 | 16 | 10636 | 170.17 |
| `prefill_fp4_n64` k | 2048×1024×5120 | 16 | 934 | 14.95 |
| `prefill_fp4_n64` v | 2048×1024×5120 | 16 | 924 | 14.79 |
| `prefill_fp4_n64` o | 2048×5120×6144 | 16 | 5478 | 87.65 |
| `prefill_fp4_n64` gate | 2048×17408×5120 | 52 | 15012 | 780.63 |
| `prefill_fp4_n64` up | 2048×17408×5120 | 52 | 14953 | 777.56 |
| `prefill_fp4_m32` down | 2048×5120×17408 | 52 | 15830 | 823.16 |
| `prefill_fp8_n64` gate | 2048×17408×5120 | 12 | 14760 | 177.12 |
| `prefill_fp8_n64` up | 2048×17408×5120 | 12 | 14715 | 176.58 |
| `prefill_fp8_m32` down | 2048×5120×17408 | 12 | 15618 | 187.42 |
| `prefill_fp8_n64` score | 12288×2048×256 | 64 | 492 | 31.48 |
| `prefill_fp8_n64_stride` value | 12288×256×2048 | 64 | 609 | 38.99 |
| `gdn_state_rows` | 2048 | 48 | 12234 | 587.23 |
| `gdn_conv_rows` | 2048 | 48 | 869 | 41.69 |
| `cast_f32_f16` hidden | 2048×5120 | 64 | 244 | 15.64 |
| `cast_f32_f16` intermediate | 2048×17408 | 64 | 798 | 51.04 |
| `cast_f32_f16` z or o | 2048×6144 | 64 | 964 | 61.66 |
| `zero_f32` | 2048×5120 | 1 | 138 | 0.14 |
| `swiglu_f32` | 2048×17408 | 64 | 1687 | 107.94 |
| `residual_rows` | 2048×5120 | 64 | 481 | 30.78 |
| `rmsnorm_rows` | 2048×5120 | 128 | 331 | 42.34 |
| `split_q_gate_rows` | 2048 | 16 | 726 | 11.62 |
| `qk_norm_rope_rows` | 2048 | 16 | 423 | 6.76 |
| `kv_softmax_max_rows` | 12288×2048 | 64 | 190 | 12.18 |
| `kv_softmax_prob_rows` | 12288×2048 | 64 | 592 | 37.89 |
| `kv_softmax_close` | 12288 | 64 | 45 | 2.86 |
| `kv_acc_add` | 12288×256 | 64 | 41 | 2.59 |

The sum is 5245.3 ms, 390 tok/s. That is the kernel ceiling for this set. A full chunk does not clear the norm buffer or the half tile, because the norm and the cast overwrite them. Key and value stores, the query cast, the row init, the attend write, and the embedding row are outside it.
