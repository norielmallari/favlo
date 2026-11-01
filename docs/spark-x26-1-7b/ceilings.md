# Ceilings

Kernel and bandwidth ceilings for the kernels `spark-x26-1-7b` launches. Recompute when `gemv_fp8_vec_rms`, `gemv_fp8_vec_add_cached`, `gemv_fp8_vec_add_wide`, `gemv_group`, `gemv_f16`, `matmul_fp8_32`, `rmsnorm`, `cast_f16`, `rope_qk`, `residual`, `geglu`, `store_full_keys`, `store_full_span`, `store_slide_keys`, `store_slide_span`, `attend_span`, or `attend_merge` changes.

Each time is one command buffer, then a wait. The time is `GPUEndTime` minus `GPUStartTime`. Two launches are discarded. The figure is the median of the next four. Measured on 1 November 2026.

The divisor is the 2 GiB streaming read on an M5 Pro with 48 GB: 7093.5 µs, 302.74 GB/s, recorded in [../qwen36-35b-a3b/ceilings.md](../qwen36-35b-a3b/ceilings.md).

## Decode

One token. Seven full layers and 21 sliding layers. The projection byte count is 1,766,420,480: the activation, the fp8 weights, the E8M0 scales, the RMSNorm weight, the normed row, and the destination, including the destination read on an add. Attention at one key is outside that count. Dividing those bytes by 302.74 GB/s is 5834.8 µs, 171.39 tok/s.

The kernel sum is 8090.6 µs, 123.60 tok/s.

| Kernel | Shape or launch | Count | µs |
|---|---|---:|---:|
| `gemv_fp8_vec_rms` | qkv 3072×2048 | 28 | 29.5 |
| `gemv_fp8_vec_add_cached` | out 2048×2048 | 28 | 20.1 |
| `gemv_fp8_vec_rms` | gate 6656×2048 | 28 | 58.2 |
| `gemv_fp8_vec_rms` | up 6656×2048 | 28 | 54.0 |
| `gemv_fp8_vec_add_wide` | down 2048×6656 | 28 | 64.2 |
| `gemv_fp8_vec_rms` | vocab 131072×2048 | 1 | 1004.5 |
| `rope_qk` | full, one token | 7 | 2.4 |
| `rope_qk` | sliding, one token | 21 | 2.3 |
| `gemv_f16` | gproj 8×2048 | 28 | 13.3 |
| `geglu` | 6656 | 28 | 2.1 |
| `store_full_span` | one token | 7 | 3.7 |
| `store_slide_span` | one token | 21 | 6.4 |
| `attend_span` | full, 1 key | 7 | 3.7 |
| `attend_span` | slide, 1 key | 21 | 3.6 |

## Prefill

One 56-token chunk. The projections, the cast, the norm, the residual, and the activation use the padded count of 64. Store and attention cover the 56 tokens. The vocabulary projection is `gemv_group`. The sum is 18523.0 µs, 3023.27 tok/s.

| Kernel | Shape or launch | Count | µs |
|---|---|---:|---:|
| `matmul_fp8_32` | qkv 64×3072×2048 | 28 | 47.1 |
| `matmul_fp8_32` | out 64×2048×2048 | 28 | 41.5 |
| `matmul_fp8_32` | gate 64×6656×2048 | 28 | 92.0 |
| `matmul_fp8_32` | up 64×6656×2048 | 28 | 93.1 |
| `matmul_fp8_32` | down 64×2048×6656 | 28 | 140.8 |
| `cast_f16` | 64×2048 | 112 | 3.0 |
| `cast_f16` | 64×6656 | 28 | 5.2 |
| `rmsnorm` | 64×2048 | 56 | 24.7 |
| `rmsnorm` | final row | 1 | 32.7 |
| `rope_qk` | full, 64 tokens | 7 | 3.1 |
| `rope_qk` | sliding, 64 tokens | 21 | 3.5 |
| `residual` | 64×2048 | 56 | 14.5 |
| `geglu` | 64×6656 | 28 | 8.6 |
| `gemv_f16` | gproj, 64 tokens | 28 | 13.8 |
| `store_full_keys` | 56 tokens | 7 | 7.5 |
| `store_full_span` | 56 tokens | 7 | 20.2 |
| `store_slide_keys` | 56 tokens | 21 | 7.6 |
| `store_slide_span` | 56 tokens | 21 | 33.9 |
| `attend_span` | full, 56 tokens | 7 | 51.1 |
| `attend_span` | slide, 56 tokens | 21 | 44.4 |
| `gemv_group` | vocab 131072×2048 | 1 | 1129.6 |
