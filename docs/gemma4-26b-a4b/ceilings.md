# Ceilings

Bandwidth and kernel ceilings for the kernels `gemma4-26b-a4b` launches. Recompute when `gemv_fp8`, `gemv_fp4_at`, `gemv_fp8_at`, `embed_fp8`, `matmul_fp8`, `matmul_half`, `matmul_fp4_64_at`, `matmul_fp8_64_at`, `restore_fp8_i8`, `quantize_half_i8`, `quantize_float_i8`, `prefill_i8_n64`, `scale_i8_f32`, `write_row`, `write_rows_tight`, `rmsnorm`, `rmsnorm_rows`, `rmsnorm_heads`, `router_topk`, `rope_pairs`, `rope_rows`, `store_kv`, `store_kv_rows`, `attend_span`, `attend_span4`, `attend_decode`, `geglu_rows`, `geglu_pair`, `group_expert_slots`, `gather_half_slots`, `pad_half_rows`, `scatter_slot_rows`, `mix_experts`, `residual_scale`, `scale_copy`, `zero_buffer`, `softcap`, `topk_groups`, or `sample_topk` changes. The CPU reads one sampled token. Expert ids stay on device.

The divisor is the 2 GiB streaming read on an M5 Pro with 48 GB: 7093.5 µs, 302.74 GB/s. That measurement is recorded in [../qwen36-35b-a3b/ceilings.md](../qwen36-35b-a3b/ceilings.md). Bytes below are divided by 302.74 GB/s.

One token runs 30 layers. Twenty-five are local, with a sliding window of 1024. Five are global and attend every key. Eight of 128 experts run. The timed token uses six fp4 experts and two fp8 experts. A 20% promotion expects 1.6 fp8 experts in those eight. Both `gemv_fp4_at` and `gemv_fp8_at` launch, and a threadgroup returns when the width differs. Dense projections and the tied embedding are fp8. Decode reads them in `gemv_fp8` and `embed_fp8`. The router is one Metal half `matmul_half` of 8 by 128 by 2816. Decode attention at one key is `attend_span`. At 1024 keys and above it is `attend_decode`, with the key range split across eight simdgroups. A value is an RMSNorm with no scale. A global value is that norm of the key projection and is stored apart from the keys.

## Bandwidth

Weight bytes are one read of the codes and scales the launch visits. The fp8 dense total includes the logits matrix and one embedding row. A down projection stops at K 2112, so the pad to 2176 is outside that read. Each layer reads six fp4 expert images and two fp8 expert images. Gate-up is read in full. The down plane stops at K 704. The router row is the Metal half 128 by 2816 matrix, once per layer. The activation column is the source vector once per launch. The output column is the written rows. Attention counts each query head's Metal half key and value, the float32 query, and the float32 output. The remaining row is the norm, residual, copy, rotary, store, GeGLU, mix, router scratch, soft cap, and sampler buffers, once each.

| Traffic | Bytes | ms |
|---|---:|---:|
| fp8 dense read | 2,458,134,360 | 8.12 |
| Routed experts, 6 fp4 and 2 fp8 | 936,714,240 | 3.09 |
| Router half | 21,626,880 | 0.07 |
| Activation read | 3,485,184 | 0.01 |
| Projection output | 7,289,856 | 0.02 |
| Attention at 1 key | 1,720,320 | 0.01 |
| Attention at 1,024 keys | 588,349,440 | 1.94 |
| Attention at 65,536 keys | 11,157,995,520 | 36.86 |
| Attention at 131,072 keys | 21,895,413,760 | 72.32 |
| Everything else | 33,880,000 | 0.11 |

| Context | Bytes | ms | tok/s |
|---:|---:|---:|---:|
| 1 | 3,462,850,840 | 11.44 | 87.43 |
| 1,024 | 4,049,479,960 | 13.38 | 74.76 |
| 65,536 | 14,619,126,040 | 48.29 | 20.71 |
| 131,072 | 25,356,544,280 | 83.76 | 11.94 |

## Prefill

One chunk of 1024 tokens at position 0. The chunk finishes, so the sum includes one tied-logits `gemv_fp8`. Large attention projections restore the packed fp8 matrix into a per-row int8 stage, quantize the float activation with `quantize_float_i8`, multiply with `prefill_i8_n64`, and scale the int32 product. That path does not cast through half. Local key and value, the shared gate and up, and the shared down stay on `matmul_fp8` with `write_rows_tight`. The router is `matmul_half`. Expert prefill uses 128 slots of 64, one slot for each expert: 102 fp4 and 26 fp8. `matmul_fp4_64_at` and `matmul_fp8_64_at` both launch on that grid. The TFLOP/s column counts the slots whose width matches. Local attention is `attend_span4`. Global attention is `attend_span` over the keys this first chunk has stored. Int8 rows measured on 2 November 2026; other rows reuse the same-day fp8 measurement. The sum is 867.785 ms, 1180.02 tok/s.

| Launch | Shape | Count | µs | TFLOP/s |
|---|---|---:|---:|---:|
| `restore_fp8_i8` | 4096×2816 | 25 | 813.1 | |
| `quantize_float_i8` | 1024×2816 | 25 | 141.6 | |
| `prefill_i8_n64` | 1024×4096×2816 | 25 | 1051.8 | 22.46 |
| `scale_i8_f32` | 1024×4096 | 25 | 92.1 | |
| `matmul_fp8` | 1024×2048×2816 | 50 | 1279.8 | 9.23 |
| `restore_fp8_i8` | 2816×4096 | 25 | 794.9 | |
| `quantize_float_i8` | 1024×4096 | 25 | 170.7 | |
| `prefill_i8_n64` | 1024×2816×4096 | 25 | 1090.6 | 21.66 |
| `scale_i8_f32` | 1024×2816 | 25 | 63.6 | |
| `restore_fp8_i8` | 8192×2816 | 5 | 1620.4 | |
| `quantize_float_i8` | 1024×2816 | 5 | 62.0 | |
| `prefill_i8_n64` | 1024×8192×2816 | 5 | 869.9 | 54.34 |
| `scale_i8_f32` | 1024×8192 | 5 | 205.6 | |
| `restore_fp8_i8` | 1024×2816 | 5 | 114.1 | |
| `quantize_float_i8` | 1024×2816 | 5 | 70.4 | |
| `prefill_i8_n64` | 1024×1024×2816 | 5 | 266.4 | 22.18 |
| `scale_i8_f32` | 1024×1024 | 5 | 30.7 | |
| `restore_fp8_i8` | 2816×8192 | 5 | 1547.7 | |
| `quantize_float_i8` | 1024×8192 | 5 | 259.2 | |
| `prefill_i8_n64` | 1024×2816×8192 | 5 | 857.8 | 55.11 |
| `scale_i8_f32` | 1024×2816 | 5 | 39.9 | |
| `matmul_fp8` | 1024×2112×2816 | 60 | 1304.9 | 9.33 |
| `matmul_fp8` | 1024×2816×2176 | 30 | 1339.4 | 9.37 |
| `matmul_half` | 1024×128×2816 | 30 | 60.2 | 12.27 |
| `matmul_fp4_64_at` | 102×64×1408×2816 | 30 | 3430.6 | 15.09 |
| `matmul_fp8_64_at` | 26×64×1408×2816 | 30 | 871.5 | 15.14 |
| `matmul_fp4_64_at` | 102×64×2816×768 | 30 | 1380.6 | 20.45 |
| `matmul_fp8_64_at` | 26×64×2816×768 | 30 | 389.9 | 18.46 |
| `write_rows_tight` | K 2816, 1024 tokens | 110 | 32.8 | |
| `write_rows_tight` | K 2176, 1024 tokens | 30 | 25.1 | |
| `rmsnorm_rows` | 1024×2816 | 240 | 52.8 | |
| `rmsnorm_heads` | 16384×256 | 25 | 77.4 | |
| `rmsnorm_heads` | 8192×256 | 50 | 37.8 | |
| `rmsnorm_heads` | 16384×512 | 5 | 231.5 | |
| `rmsnorm_heads` | 2048×512 | 10 | 15.6 | |
| `rope_rows` | local, 1024 tokens | 25 | 148.3 | |
| `rope_rows` | global, 1024 tokens | 5 | 270.6 | |
| `store_kv_rows` | local | 50 | 22.8 | |
| `store_kv_rows` | global | 10 | 11.1 | |
| `attend_span4` | local, 1024 queries at position 0 | 25 | 6253.4 | |
| `attend_span` | global, 1024 queries at position 0 | 5 | 21984.0 | |
| `residual_scale` | 1024×2816 | 120 | 91.0 | |
| `scale_copy` | 1024×2816 | 30 | 44.1 | |
| `scale_copy` | global value, 1024×1024 | 5 | 13.6 | |
| `geglu_pair` | 1024×2112 | 30 | 46.2 | |
| `zero_buffer` | 1024×2816 floats | 60 | 52.9 | |
| `router_topk` | 1024 tokens, 128 to 8 | 30 | 49.4 | |
| `group_expert_slots` | 1024 tokens | 30 | 199.6 | |
| `gather_half_slots` | 8192×2816 | 30 | 293.1 | |
| `geglu_rows` | 8192×704 | 30 | 265.3 | |
| `pad_half_rows` | 8192×768 | 30 | 110.0 | |
| `scatter_slot_rows` | 8192×2816 | 30 | 711.9 | |
| `embed_fp8` | one row | 1024 | 2.5 | |
| `gemv_fp8` | 262144×2816 | 1 | 2970.1 | |
| `rmsnorm` | 2816 | 1 | 6.2 | |
| `softcap` | 262144 | 1 | 3.9 | |
| `topk_groups` | 1024, 256, 64, 16, 4, then 1 group | 6 | 67.8, 19.9, 8.5, 6.1, 6.1, 6.2 | |
| `sample_topk` | 64 | 1 | 40.2 | |

## Decode kernels

Measured on 2 November 2026. Each kernel is one command buffer, then a wait. The time is the device interval, `GPUEndTime` minus `GPUStartTime`. The first sample is discarded. The figure is the median of the next four. The sum adds those medians before the rows are rounded to 0.1 µs. The 90% test uses this kernel ceiling.

Context 1 is 37.424 ms, 26.72 tok/s. Context 1024 is 40.005 ms, 25.00 tok/s. Context 65536 is 81.759 ms, 12.23 tok/s. Context 131072 is 124.216 ms, 8.05 tok/s. The projection rows do not change with context. The attention rows do.

| Kernel | Shape or launch | Count | µs |
|---|---|---:|---:|
| `gemv_fp8` | 4096×2816, 8 rows, 64 threads | 25 | 208.2 |
| `gemv_fp8` | 2048×2816, 8 rows, 64 threads | 50 | 123.4 |
| `gemv_fp8` | 2816×4096, 8 rows, 64 threads | 25 | 201.9 |
| `gemv_fp8` | 8192×2816, 8 rows, 64 threads | 5 | 323.4 |
| `gemv_fp8` | 1024×2816, 8 rows, 64 threads | 5 | 44.2 |
| `gemv_fp8` | 2816×8192, 8 rows, 64 threads | 5 | 211.2 |
| `gemv_fp8` | 2112×2816, 8 rows, 64 threads | 60 | 69.1 |
| `gemv_fp8` | 2816×2176, active 2112, 64 threads | 30 | 69.5 |
| `gemv_fp4_at` | gate, 6 of 8 experts, 1408×2816 | 30 | 50.4 |
| `gemv_fp8_at` | gate, 2 of 8 experts, 1408×2816 | 30 | 38.1 |
| `gemv_fp4_at` | down, 6 of 8 experts, 2816×768, active 704 | 30 | 35.3 |
| `gemv_fp8_at` | down, 2 of 8 experts, 2816×768, active 704 | 30 | 25.4 |
| `geglu_rows` | 8×704 | 30 | 2.4 |
| `mix_experts` | 8×2816 | 30 | 3.7 |
| `gemv_fp8` | 262144×2816, 8 rows, 64 threads | 1 | 2970.1 |
| `embed_fp8` | one row | 1 | 2.5 |
| `rmsnorm` | 2816 | 241 | 6.2 |
| `rmsnorm_heads` | 16×256 | 25 | 2.8 |
| `rmsnorm_heads` | 8×256 | 50 | 2.8 |
| `rmsnorm_heads` | 16×512 | 5 | 3.2 |
| `rmsnorm_heads` | 2×512 | 10 | 3.0 |
| `rope_pairs` | local | 25 | 2.9 |
| `rope_pairs` | global | 5 | 2.8 |
| `store_kv` | local key or value | 50 | 2.3 |
| `store_kv` | global key or value | 10 | 2.4 |
| `attend_span` | local, 1 key | 25 | 4.5 |
| `attend_span` | global, 1 key | 5 | 4.7 |
| `attend_decode` | local, 1024 keys | 25 | 83.6 |
| `attend_decode` | global, 1024 keys | 5 | 125.1 |
| `attend_decode` | global, 65536 keys | 5 | 8475.9 |
| `attend_decode` | global, 131072 keys | 5 | 16967.4 |
| `geglu_pair` | 2112 | 30 | 2.1 |
| `residual_scale` | 2816 | 120 | 1.9 |
| `scale_copy` | 4096 | 25 | 2.0 |
| `scale_copy` | 2048 | 50 | 2.0 |
| `scale_copy` | 8192 | 5 | 2.1 |
| `scale_copy` | 1024 | 10 | 2.0 |
| `scale_copy` | 2816 | 90 | 2.0 |
| `scale_copy` | 2112 | 30 | 2.0 |
| `zero_buffer` | 11264 bytes | 60 | 1.8 |
| `softcap` | 262144 | 1 | 3.9 |
| `write_row` | router activation | 30 | 2.0 |
| `router_topk` | 128 to 8 | 30 | 11.7 |
| `matmul_half` | 8×128×2816 | 30 | 24.6 |
| `topk_groups` | 1024, 256, 64, 16, 4, then 1 group | 6 | 67.8, 19.9, 8.5, 6.1, 6.1, 6.2 |
| `sample_topk` | 64 | 1 | 40.2 |

Context 1 uses the 1-key attention rows. Context 1024 uses the 1024-key rows. Context 65536 and context 131072 keep the local window at 1024 and use the matching global row.
