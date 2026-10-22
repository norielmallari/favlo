# Ceilings

Kernel and bandwidth ceilings for the kernels this pipeline runs. Reuse these figures while those kernels are unchanged. Recompute this note when one of them changes. Projection times are the earlier measurement. The attention rows were measured again after softmax began writing float32 weights. The kernel times below were measured when every weight image was fp4. The byte counts are the `Qwen3.6-35B-A3B-fp8-e20` record. Language projections, the shared expert, and `lm_head` are fp8. Of the 10,240 routed experts, 2,048 are fp8 and 8,192 are fp4. A decode token reads eight images per layer, counted at that pool mix: 64 fp8 and 256 fp4 of the 320 images.

Each time is one command buffer, then a wait. Overlapping command buffers are not added. The times are the median of four samples after one discarded sample.

A 2 GiB streaming read finished in 7093.5 µs, 302.74 GB/s. The bandwidth ceiling divides bytes by 302.74 GB/s. The value-mix timing uses token stride 262144, the position limit.

## Decode

One token. The times are the fp4 weight images. K and V are fp8 E4M3. The forward is 30 Gated DeltaNet layers, 10 attention layers, and the head. Top 8 routed experts. `gdn_in_proj` is the fused qkv, z, a, and b projection. `gemv_fp4_vec` is the output projection, the attention output, the shared expert, and the head. `router_coeff` is the float16 router and the shared-expert coefficient. `moe_gate_silu` and `moe_down_mix` read eight expert images. `rmsnorm_f32` is the layer norm, including the final norm. Attention is two key/value heads: `prefill_fp8_m8` scores the span and the three softmax kernels reduce it. The value mix is `kv_cast_blocks`, `kv_pv_split8`, and `kv_restore_sum`, each in its own command buffer. `kv_cast_blocks` writes one float16 plane in span order and one peak per row per block. `kv_pv_split8` mixes every block. `kv_restore_sum` multiplies each partial by that peak.

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
| `kv_softmax_prob` | 2048 keys, 8 rows | 20 | 15.7 |
| `kv_softmax_close` | 8 rows | 20 | 4.1 |
| `kv_cast_blocks` | 4 blocks of 512 keys, 8 rows | 20 | 21.6 |
| `kv_pv_split8` | 4 blocks of 512 keys | 20 | 30.5 |
| `kv_restore_sum` | 4 blocks, 8×256 | 20 | 11.7 |

That sum is 10.28 ms, 97.25 tok/s. A context of 65536 tokens uses the same projections and replaces the attention rows:

| Kernel | Shape | Count | µs |
|---|---|---:|---:|
| `prefill_fp8_m8` | 8×65536, inner 256 | 20 | 110.3 |
| `kv_softmax_max` | 65536 keys, 8 rows | 20 | 7.4 |
| `kv_softmax_prob` | 65536 keys, 8 rows | 20 | 36.1 |
| `kv_softmax_close` | 8 rows | 20 | 5.2 |
| `kv_cast_blocks` | 128 blocks of 512 keys, 8 rows | 20 | 33.3 |
| `kv_pv_split8` | 128 blocks of 512 keys | 20 | 166.3 |
| `kv_restore_sum` | 128 blocks, 8×256 | 20 | 102.1 |

That token is 17.17 ms, 58.23 tok/s. Those times are the fp4-weight measurement. The byte count is the resident projections, eight expert images per layer, the float16 routers and coefficients, the head, the GDN state read and write, K and V across the visible tokens, the float32 score and probability writes, the float16 probability scratch, and one float scale per row per block. On this record the projections, the shared expert, and the head are fp8 images. Context 2048 is 2,805,787,136 bytes, 9.27 ms, 107.90 tok/s. Context 65536 is 3,577,880,576 bytes, 11.82 ms, 84.61 tok/s. The 90% test for decode uses a kernel ceiling measured on the kernels the forward launches. The rows above are not that measurement.

The embed, the query/gate split, RoPE, the top-8 pick, and the key/value store are not in this total.

## Prefill

One chunk of 2048 tokens, then a prompt of 65536 tokens as 32 of those chunks. Projections are `prefill_fp4_n64` where the output width is a multiple of 64, and `prefill_fp4_e8m0` for the 32-wide a and b projections. Routed experts are one 64-row tile for each of the 256 experts. `moe_group` is one thread over the 16384 assignments. The Gated DeltaNet recurrence is 32 sub-chunks of 64 tokens (Yang et al., §3.3). A score window is 16384 query rows by up to 4096 keys. The first chunk's limit is 2048 keys, so that chunk is the one 2048-key window per head in the table. A later chunk walks 4096-key windows and, when the limit is not a multiple of 4096, one 2048-key tail. The value product uses token stride 262144.

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
| `kv_softmax_prob_rows` | 20 | 1074.5 |
| `kv_softmax_close` | 20 | 61.9 |
| `kv_cast_probs` | 20 | 923.4 |
| `prefill_fp8_n64_stride` | 20 | 775.9 |
| `kv_scale_partial` | 20 | 60.6 |
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

The attention window is the only part of a chunk that changes with the keys already stored. Without that window, one chunk is 1267.066 ms. A 2048-key window is `prefill_fp8_n64` 651.8 µs, `kv_softmax_max_rows` 245.5 µs, `kv_softmax_prob_rows` 1074.5 µs, `kv_softmax_close` 61.9 µs, `kv_cast_probs` 923.4 µs, `prefill_fp8_n64_stride` 775.9 µs, `kv_scale_partial` 60.6 µs, and `kv_acc_add` 161.0 µs, 3954.6 µs together. A 4096-key window is `prefill_fp8_n64` 1295.8 µs, `kv_softmax_max_rows` 370.5 µs, `kv_softmax_prob_rows` 1564.6 µs, `kv_softmax_close` 63.9 µs, `kv_cast_probs` 2422.4 µs, `prefill_fp8_n64_stride` 1570.1 µs, `kv_scale_partial` 60.6 µs, and `kv_acc_add` 177.7 µs, 7525.6 µs together. The first chunk, one 2048-key window per head, is 1346.2 ms, 1521.37 tok/s.

A prompt of 65536 tokens is 32 chunks. Chunk n sees n×2048 keys: the floor of n/2 windows of 4096 keys, and one 2048-key window when n is odd. One head walks 256 windows of 4096 keys and 16 windows of 2048 keys. Ten layers and two heads make 20 of each. That prompt is 80342.7 ms, 815.71 tok/s. That is the kernel ceiling for prefill.

The byte count repeats the projections, all 256 expert images, and the float16 routers on every chunk, and adds the GDN state read and write, the K and V bytes of each window, the float32 score and probability writes, the float16 probability scratch, and one float scale per row per window. On this record the projections and the shared expert are fp8, and 2,048 of the 10,240 expert images are fp8. One first chunk is 28,688,637,952 bytes. The 65536-token prompt is 4,257,677,574,144 bytes, 14.06 s at 302.74 GB/s, 4659.90 tok/s. The 90% test for prefill uses a kernel ceiling measured on the kernels the forward launches. The rows above are not that measurement.

The embed, the coefficient row, the top-8 pick, the query/gate split, RoPE, the key/value store, and the grouped SiLU after the expert tile are not in this total.
