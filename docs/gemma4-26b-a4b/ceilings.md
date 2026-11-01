# Ceilings

Bandwidth and kernel ceilings for the kernels `gemma4-26b-a4b` launches. Recompute when `gemv_q4_0`, `gemv_fp4_routed`, `gemv_q4_0_routed`, `mix_experts`, `attend_span`, `attend_decode`, `gemm_q4_0_n`, `gemm_q4_0`, `dequant_q4_0`, `embed_q4_0`, `matmul_f16`, `matmul_f16_64`, `write_row`, `write_rows`, `rmsnorm`, `rmsnorm_rows`, `rmsnorm_heads`, `router_topk`, `rope_pairs`, `rope_rows`, `store_kv`, `store_kv_rows`, `attend`, `geglu`, `geglu_rows`, `geglu_pair`, `gather_rows`, `scatter_axpy`, `scatter_slot_rows`, `group_expert_slots`, `residual_scale`, `scale_copy`, `zero_buffer`, `softcap`, `topk_groups`, or `sample_topk` changes. The CPU reads one sampled token. Prefill expert ids stay on device.

The divisor is the 2 GiB streaming read on an M5 Pro with 48 GB: 7093.5 µs, 302.74 GB/s. That measurement is recorded in [../qwen36-35b-a3b/ceilings.md](../qwen36-35b-a3b/ceilings.md). Bytes below are divided by 302.74 GB/s.

One token runs 30 layers. Twenty-five are local, with a sliding window of 1024. Five are global and attend every key. Eight of 128 experts run. The expert files are read once when the runtime opens and stay q4_0. A one-token q4_0 projection is `gemv_q4_0`. Two simdgroups own eight output rows. Each thread dots half of a 32-value block. The scale is bfloat16. The code bytes are read as ushorts, and the nibbles stay in place at bits 0, 4, 8, and 12. The sixteen activations are scaled once, by 1, 1/16, 1/256, and 1/4096, and reused for the four rows. The zero point is one subtraction per half-block. The routed experts are one `gemv_q4_0_routed` per gate-up and one per down projection: eight experts, the ids left on device, eight output rows in a threadgroup of 64. The token does not expand the matrix into float16. A q4_0 block is 18 bytes. The logits matrix reaches the copy rate. The shorter projections do not. Decode attention is `attend_decode`: the key range splits across four simdgroups at 256 keys and eight at 1024, and those partials merge into one online softmax. The `attend_span` rows below are the previous one-simd walk. This table does not time `attend_decode`. A value is an RMSNorm with no scale. A global value is that norm of the key projection and is stored apart from the keys. The token embedding is one q4_0 row. The logits are the full tied matrix. The router is one float16 `matmul_f16` of 8 by 128 by 2816. Decode is this one-token launch. A prefill batch of eight or fewer tokens uses `gemm_q4_0_n`, the same dot applied to every token after one load of the weight half. A longer batch dequantizes one matrix to float16 and multiplies it with `matmul2d`. The one-row table is not the prefill ceiling.

## Bandwidth

| Traffic | Bytes | ms |
|---|---:|---:|
| q4_0 read | 2,143,696,896 | 7.08 |
| Activation read by `gemv_q4_0` | 1,000,558,592 | 3.31 |
| `gemv_q4_0` output | 7,289,856 | 0.02 |
| Attention at 1 key | 1,720,320 | 0.01 |
| Attention at 65,536 keys | 22,314,270,720 | 73.71 |
| Everything else | 60,040,752 | 0.20 |

| Context | Bytes | ms | tok/s |
|---:|---:|---:|---:|
| 1 | 3,213,306,416 | 10.61 | 94.21 |
| 1,024 | 4,386,564,656 | 14.49 | 69.02 |
| 65,536 | 25,525,856,816 | 84.32 | 11.86 |
| 131,072 | 47,000,693,296 | 155.25 | 6.44 |

Walking 65,536 tokens one row at a time, with global attention growing over the keys already stored, sums to 3,199.96 s, 20.48 tok/s. That is the one-row launch. Prefill does not use it.

## Prefill

Prefill dequantizes one q4_0 matrix into a reused float16 buffer, then calls `matmul2d`. q4_0 is not a `matmul2d` right-hand type. A chunk of 32 or more tokens uses `matmul_f16_64`, the 64×64 descriptor with four simdgroups from [../synthetic-matmul2d.md](../synthetic-matmul2d.md), when the output row count is a multiple of 64. A batch of nine to 31 tokens uses `matmul_f16`, the 8×32 descriptor. A batch of eight or fewer uses `gemm_q4_0_n` and does not write the float16 matrix. Measured on 31 October 2026, same method as the decode table. A 128×4096×2816 product took 112.3 µs, 26.29 TFLOP/s. Dequant of that matrix took 89.3 µs, of 8192×2816 took 237.8 µs, and of 1408×2816 took 25.3 µs. The products:

| Launch | Shape | µs | TFLOP/s |
|---|---|---:|---:|
| `matmul_f16_64` | 128×4096×2816 | 112.3 | 26.29 |
| `matmul_f16_64` | 128×8192×2816 | 282.1 | 20.93 |
| `matmul_f16_64` | 128×2112×2816 | 60.4 | 25.22 |
| `matmul_f16_64` | 128×2816×8192 | 275.1 | 21.47 |
| `matmul_f16_64` | 128×1408×2816 | 48.1 | 21.09 |
| `matmul_f16` | 8×1408×2816 | 25.6 | 2.48 |

The previous `gemm_q4_0` of 128×4096×2816 was 1072.7 µs, 2.75 TFLOP/s. A chunk is 1024 tokens. Local attention is `attend_span4`: four queries share each key load. Global attention is `attend_span`. On 31 October 2026 the fp4 expert prefill sorted routed rows into slots of 64 and called `matmul_fp4_64_batch`. That launch of 128 experts at 64×1408×2816 took 4.097 ms, 15.85 TFLOP/s, and 64×2816×1024 took 2.632 ms, 17.95 TFLOP/s. The live expert prefill repacks all 128 experts of a layer to fp4 once when the runtime opens, for as many layers as the budget still holds. `group_expert_slots` writes the slot tables and the indirect matmul grid, so the host does not read the ids. Occupied slots are padded to 64 tokens, the down K is padded from 704 to 768, and the product is `matmul_fp4_64_batch`. A layer that does not fit is repacked again on the next chunk. These rows do not time `group_expert_slots` or that launch. A varied prompt the same day prefilled 512 tokens at 1134.1 tok/s and 1024 tokens at 1161.9 tok/s. The 1024-token run spent 0.429 s in the expert path and 0.419 s in the rest of the chunk. The carrot smoke on this path prefilled 50 tokens at 283.07 tok/s and 276.53 tok/s, and 53 tokens at 279.27 tok/s. After decode attention moved to `attend_span`, the same smoke decoded 54 tokens at 46.64 tok/s, 77 tokens at 44.03 tok/s, and 200 tokens at 41.97 tok/s. The coding-agent runtime log the same day decoded 820 tokens at 18.84 tok/s across eight requests, from 20.36 tok/s on the first to 16.84 tok/s on the last. That prompt rate is not a kernel ceiling. The composed prefill ceiling is not the one-row sum above.

## Decode kernels

Measured on 31 October 2026. Each kernel is one command buffer, then a wait. The time is the device interval, `GPUEndTime` minus `GPUStartTime`. The first sample is discarded. The figure is the median of the next four. The sum is every dispatch in the token. The 90% test uses this kernel ceiling.

Context 1 is 35.595 ms, 28.09 tok/s. Context 1024 is 56.658 ms, 17.65 tok/s. Context 65536 is 396.713 ms, 2.52 tok/s. The projection rows do not change with context. The attention rows do. `gemv_fp4_routed` and `attend_span` were measured on 31 October 2026 after a warmup, median of four device intervals. The other rows are the earlier measurement the same day. The sums replace the earlier `gemv_q4_0_routed` and `attend` rows with the fp4 expert kernel and `attend_span` and leave every other dispatch as it was. The live decode expert kernel is `gemv_q4_0_routed` again. This table does not time that launch. The sums count `scale_copy` at 2.0 µs. A reference `gemv_q4_0` of 4096×2816 in the routed session was 85.1 µs, beside the 87.9 µs row below.

| Kernel | Shape or launch | Count | µs |
|---|---|---:|---:|
| `gemv_q4_0` | 4096×2816, 8 rows, 64 threads | 25 | 87.9 |
| `gemv_q4_0` | 2048×2816, 8 rows, 64 threads | 50 | 49.5 |
| `gemv_q4_0` | 2816×4096, 8 rows, 64 threads | 25 | 89.4 |
| `gemv_q4_0` | 8192×2816, 8 rows, 64 threads | 5 | 150.7 |
| `gemv_q4_0` | 1024×2816, 8 rows, 64 threads | 5 | 42.6 |
| `gemv_q4_0` | 2816×8192, 8 rows, 64 threads | 5 | 166.8 |
| `gemv_q4_0` | 2112×2816, 8 rows, 64 threads | 60 | 51.2 |
| `gemv_q4_0` | 2816×2112, 8 rows, 64 threads | 30 | 56.7 |
| `gemv_fp4_routed` | 8×1408×2816, 64 threads | 30 | 295.0 |
| `gemv_fp4_routed` | 8×2816×1024, active 704, 64 threads | 30 | 213.0 |
| `geglu_rows` | 8×704 | 30 | 8.9 |
| `mix_experts` | 8×2816 | 30 | 14.3 |
| `gemv_q4_0` | 262144×2816, 8 rows, 64 threads | 1 | 1746.2 |
| `embed_q4_0` | one row | 1 | 4.1 |
| `rmsnorm` | 2816 | 241 | 6.2 |
| `rmsnorm_heads` | 16×256 | 25 | 2.8 |
| `rmsnorm_heads` | 8×256 | 25 | 2.7 |
| `rmsnorm_heads` | 16×512 | 5 | 3.1 |
| `rmsnorm_heads` | 2×512 | 5 | 3.1 |
| `rope_pairs` | local | 25 | 2.7 |
| `rope_pairs` | global | 5 | 3.0 |
| `store_kv` | local key or value | 50 | 2.3 |
| `store_kv` | global key | 5 | 2.3 |
| `attend_span` | local, 1 key | 25 | 16.5 |
| `attend_span` | local, 1024 keys | 25 | 683.1 |
| `attend_span` | global, 1 key | 5 | 5.4 |
| `attend_span` | global, 1024 keys | 5 | 885.1 |
| `attend_span` | global, 65536 keys | 5 | 68896.2 |
| `geglu_pair` | 2112 | 30 | 2.1 |
| `residual_scale` | 2816 | 120 | 1.9 |
| `scale_copy` | hidden, 1024, 2048, 2112, 4096, or 8192 | 205 | 1.9 to 2.0 |
| `zero_buffer` | 11264 bytes | 60 | 1.8 |
| `softcap` | 262144 | 1 | 4.0 |
| `write_row` | router activation | 30 | 1.9 |
| `router_topk` | 128 to 8 | 30 | 11.8 |
| `matmul_f16` | 8×128×2816 | 30 | 24.6 |
| `topk_groups` | 1024, 256, 64, 16, 4, then 1 group | 6 | 67.7, 19.2, 8.3, 6.1, 6.1, 6.1 |
| `sample_topk` | 64 | 1 | 40.1 |

Context 1 uses the 1-key attention rows. Context 1024 uses the 1024-key rows. Context 65536 keeps the local window at 1024 and uses the 65536-key global row. `scale_copy` is one launch per copy; the measured launches sit between 1.9 and 2.0 µs.
