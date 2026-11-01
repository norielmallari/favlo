# Ceilings

Kernel and bandwidth ceilings for the kernels `spark-x26-1-7b` launches. Recompute when `gemv_fp4_vec_rms`, `gemv_fp4_vec_add`, `gemv_group_rms`, `gemv_f16`, `matmul_fp4_32`, `rope_qk`, `geglu`, `store_full_keys`, `store_full_span`, `store_slide_keys`, `store_slide_span`, `attend_span`, `attend_tile`, or `attend_merge` changes.

Each time is one command buffer, then a wait. The time is `GPUEndTime` minus `GPUStartTime`. The first sample is discarded. The figure is the median of the next four. Measured on 1 November 2026.

The divisor is the 2 GiB streaming read on an M5 Pro with 48 GB: 7093.5 µs, 302.74 GB/s, recorded in [../qwen36-35b-a3b/ceilings.md](../qwen36-35b-a3b/ceilings.md). The fp4 record is 908,038,144 bytes. That read is 2.999 ms, 333.40 tok/s. The 90% test uses the kernel ceilings below.

## Decode

One token. Attention is one key. Seven full layers and 21 sliding layers. The up projection is the same kernel and shape as the gate projection; the gate row is that measurement, counted twice. The sum is 19500.7 µs, 51.28 tok/s.

| Kernel | Shape or launch | Count | µs |
|---|---|---:|---:|
| `gemv_fp4_vec_rms` | qkv 3072×2048 | 28 | 105.5 |
| `gemv_fp4_vec_add` | out 2048×2048 | 28 | 43.9 |
| `gemv_fp4_vec_rms` | gate or up 6656×2048 | 56 | 156.7 |
| `gemv_fp4_vec_add` | down 2048×6656 | 28 | 117.8 |
| `gemv_group_rms` | head 131072×2048 | 1 | 883.5 |
| `rope_qk` | one token | 28 | 3.1 |
| `gemv_f16` | gproj 8×2048 | 28 | 18.5 |
| `geglu` | 6656 | 28 | 2.8 |
| `store_full_span` | one token | 7 | 29.4 |
| `store_slide_span` | one token | 21 | 57.7 |
| `attend_span` | full, 1 key | 7 | 9.2 |
| `attend_span` | slide, 1 key | 21 | 9.3 |

## Prefill

One 56-token chunk. The 32×128 matmul projects it. Store and attention cover those 56 tokens. The sum is 25270.7 µs, 2216.01 tok/s.

| Kernel | Shape or launch | Count | µs |
|---|---|---:|---:|
| `matmul_fp4_32` | qkv 64×3072×2048 | 28 | 76.9 |
| `matmul_fp4_32` | out 64×2048×2048 | 28 | 60.7 |
| `matmul_fp4_32` | gate 64×6656×2048 | 28 | 96.7 |
| `matmul_fp4_32` | up 64×6656×2048 | 28 | 96.2 |
| `matmul_fp4_32` | down 64×2048×6656 | 28 | 138.0 |
| `store_full_keys` | 56 tokens | 7 | 7.4 |
| `store_full_span` | 56 tokens | 7 | 295.9 |
| `store_slide_keys` | 56 tokens | 21 | 7.4 |
| `store_slide_span` | 56 tokens | 21 | 340.0 |
| `attend_span` | full, 56 tokens | 7 | 102.9 |
| `attend_span` | slide, 56 tokens | 21 | 95.9 |
