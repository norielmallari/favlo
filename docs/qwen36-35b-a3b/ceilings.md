# Ceilings

Kernel and bandwidth ceilings for the kernels this pipeline runs. Reuse these figures while those kernels are unchanged. Recompute this note when one of them changes.

Each time is one command buffer, then a wait. Overlapping command buffers are not added. The times are the median of four samples after one discarded sample.

A 2 GiB streaming read finished in 7093.5 µs, 302.74 GB/s. The bandwidth ceiling divides bytes by 302.74 GB/s. The value-mix timing uses token stride 262144, the position limit.

## Decode

One token. Weights are fp4. K and V are fp8 E4M3. The forward is 30 Gated DeltaNet layers, 10 attention layers, and the head. Top 8 routed experts. `gdn_in_proj` is the fused qkv, z, a, and b projection. `gemv_fp4_vec` is the output projection, the attention output, the shared expert, and the head. `router_coeff` is the float16 router and the shared-expert coefficient. `moe_gate_silu` and `moe_down_mix` read eight expert images. `rmsnorm_f32` is the layer norm, including the final norm. Attention is two key/value heads: `prefill_fp8_m8` scores the span, the three softmax kernels reduce it, and `kv_pv_split8` with `kv_part_sum` mixes values in blocks of 512.

| Kernel | Shape | Count | µs |
|---|---|---:|---:|
| `gdn_in_proj` | qkv 8192×2048, z 4096×2048, a/b 32×2048 | 30 | 48.0 |
| `gemv_fp4_vec` out | 2048×4096 | 30 | 19.9 |
| `gdn_conv_shift` | 8192×4 | 30 | 2.2 |
| `gdn_state` | 32×128×128 | 30 | 11.4 |
| `attn_qkv` | q 8192×2048, k/v 512×2048 | 10 | 37.0 |
| `gemv_fp4_vec` o | 2048×4096 | 10 | 19.9 |
| `gemv_fp4_vec` shared up | 1024×2048 | 40 | 7.7 |
| `gemv_fp4_vec` shared down | 2048×512 | 40 | 5.1 |
| `router_coeff` | 256×2048 and 1×2048 | 40 | 13.4 |
| `moe_gate_silu` | 8 images | 40 | 38.7 |
| `moe_down_mix` | 8 images | 40 | 24.8 |
| `rmsnorm_f32` | 2048 | 81 | 5.3 |
| `gemv_fp4_vec` head | 248320×2048 | 1 | 929.3 |
| `prefill_fp8_m8` | 8×2048, inner 256 | 20 | 28.4 |
| `kv_softmax_max` | 2048 keys, 8 rows | 20 | 4.1 |
| `kv_softmax_prob` | 2048 keys, 8 rows | 20 | 4.4 |
| `kv_softmax_close` | 8 rows | 20 | 4.1 |
| `kv_pv_split8` | 512-key blocks, span 2048 | 20 | 7.7 |
| `kv_part_sum` | 4 blocks | 20 | 2.8 |

That sum is 8.99 ms, 111.24 tok/s. A context of 65536 tokens uses the same projections and replaces the attention rows:

| Kernel | Shape | Count | µs |
|---|---|---:|---:|
| `prefill_fp8_m8` | 8×65536, inner 256 | 20 | 110.3 |
| `kv_softmax_max` | 65536 keys, 8 rows | 20 | 7.4 |
| `kv_softmax_prob` | 65536 keys, 8 rows | 20 | 9.0 |
| `kv_softmax_close` | 8 rows | 20 | 5.2 |
| `kv_pv_split8` | 512-key blocks, span 65536 | 20 | 47.3 |
| `kv_part_sum` | 128 blocks | 20 | 25.4 |

That token is 12.05 ms, 82.97 tok/s. The byte count is the resident projections, eight expert images per layer, the float16 routers and coefficients, the head, the GDN state read and write, K and V across the visible tokens, and the score and probability writes. Context 2048 is 1,745,018,880 bytes, 5.76 ms, 173.49 tok/s. Context 65536 is 2,476,400,640 bytes, 8.18 ms, 122.25 tok/s. The 90% test for decode uses the kernel ceiling.

The embed, the query/gate split, RoPE, the top-8 pick, and the key/value store are not in this total.

## Prefill

One chunk of 2048 tokens, then a prompt of 65536 tokens as 32 of those chunks. Projections are `prefill_fp4_n64` where the output width is a multiple of 64, and `prefill_fp4_e8m0` for the 32-wide a and b projections. Routed experts are one 64-row tile for each of the 256 experts. `moe_group` is one thread over the 16384 assignments. The Gated DeltaNet recurrence is 32 sub-chunks of 64 tokens (Yang et al., §3.3). A score window is 16384 query rows by 2048 keys. The value product uses token stride 65536. Chunk c of the long prompt, counting from 1, walks c windows per key/value head. The 32 chunks walk 528 windows per head.

| Kernel | Count in one chunk | µs per call |
|---|---:|---:|
| `prefill_fp4_n64` qkv | 30 | 2705.7 |
| `prefill_fp4_n64` z | 30 | 1369.9 |
| `prefill_fp4_e8m0` a | 30 | 52.6 |
| `prefill_fp4_e8m0` b | 30 | 52.4 |
| `prefill_fp4_n64` out | 30 | 1410.8 |
| `gdn_conv_rows` | 30 | 484.6 |
| `gdn_l2_rows` | 30 | 220.5 |
| `gdn_chunk_solve` | 960 | 316.4 |
| `gdn_chunk_out` | 960 | 102.4 |
| `gdn_chunk_state` | 960 | 151.0 |
| `gdn_y_norm` | 30 | 363.5 |
| `prefill_fp4_n64` q | 10 | 2705.7 |
| `prefill_fp4_n64` k | 10 | 195.2 |
| `prefill_fp4_n64` v | 10 | 195.2 |
| `prefill_fp4_n64` o | 10 | 1411.7 |
| `prefill_fp8_n64` score | 20 | 651.8 |
| `kv_softmax_max_rows` | 20 | 245.5 |
| `kv_softmax_prob_rows` | 20 | 512.3 |
| `kv_softmax_close` | 20 | 61.9 |
| `prefill_fp8_n64_stride` | 20 | 775.9 |
| `kv_acc_add` | 20 | 161.0 |
| `prefill_fp4_n64` shared up | 40 | 362.3 |
| `prefill_fp4_n64` shared down | 40 | 180.8 |
| `prefill_fp4_expert_tile` gate | 40 | 4199.8 |
| `prefill_fp4_expert_tile` down | 40 | 2004.3 |
| `prefill_f16` router | 40 | 139.8 |
| `rmsnorm_rows` | 80 | 61.5 |
| `cast_f32_f16` hidden | 120 | 49.7 |
| `cast_f32_f16` wide | 40 | 176.1 |
| `cast_f32_f16` gated | 40 | 10.9 |
| `moe_group` | 40 | 3366.9 |
| `gather_half_grouped` | 40 | 479.0 |
| `silu_mul_rows` | 40 | 22.5 |
| `scatter_add_grouped` | 40 | 609.1 |
| `residual_rows` | 40 | 55.9 |

The first chunk, one window per head, is 1315.2 ms, 1557.15 tok/s. The 65536-token prompt is 65978.4 ms, 993.29 tok/s. That is the kernel ceiling for prefill. The byte count repeats the projections, all 256 expert images, and the float16 routers on every chunk, and adds the GDN state read and write, the K and V bytes of each window, and the score and probability writes. One first chunk is 22,077,235,200 bytes. The 65536-token prompt is 2,714,358,251,520 bytes, 8.97 s at 302.74 GB/s, 7309.45 tok/s. The 90% test for prefill uses the kernel ceiling.

The embed, the coefficient row, the top-8 pick, the query/gate split, RoPE, the key/value store, and the grouped SiLU after the expert tile are not in this total.
