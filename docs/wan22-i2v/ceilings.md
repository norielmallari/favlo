# Ceilings

Kernel and bandwidth ceilings for one 576×320 clip of 9 frames and 4 Euler steps. The latent is 3×40×72, which is 2160 tokens. `matmul2d` rows are padded to 2176. Eight of the forty blocks are fp8. Text, time, and the head stay fp4. Reuse these figures while the kernels named here are unchanged. Recompute this note when one of them changes.

Each time is one command buffer, then a wait. The time is `GPUEndTime − GPUStartTime`. Each kernel is timed alone. The figure is the median of five dispatches after one discarded sample.

A 2 GiB copy finished in 15.503 ms, which is 277.04 GB/s counting the read and the write. The bandwidth ceiling divides bytes by 277.04 GB/s.

## DiT

Four steps and forty layers. A layer projects six token rows of 5120×5120 (self q, k, v, o, and cross q, o), two text rows of 5120×5120 (cross k, v), one feed-forward up of 13824×5120, and one feed-forward down of 5120×13824. Each projection is `scale_gather`, `matmul_fp4` or `matmul_fp8`, `rescale_rows`, and `add_bias`. The layer also runs self-attention over 2160 tokens and cross-attention over the 512 text positions, 40 heads of 128. Thirty-two layers in a step are fp4 and eight are fp8, so the four steps are 128 fp4 layers and 32 fp8 layers.

| Kernel | Shape | Count | ms |
|---|---|---:|---:|
| `matmul_fp4` | 2176×5120, K 5120 | 768 | 10.360 |
| `matmul_fp8` | 2176×5120, K 5120 | 192 | 9.186 |
| `matmul_fp4` | 2176×13824, K 5120 | 128 | 26.951 |
| `matmul_fp8` | 2176×13824, K 5120 | 32 | 24.906 |
| `matmul_fp4` | 2176×5120, K 13824 | 128 | 30.634 |
| `matmul_fp8` | 2176×5120, K 13824 | 32 | 29.216 |
| `matmul_fp4` | 512×5120, K 5120 | 260 | 2.286 |
| `matmul_fp8` | 512×5120, K 5120 | 64 | 2.143 |
| `matmul_fp4` | 512×5120, K 4096 | 4 | 1.820 |
| `matmul_fp4` | 64×5120, K 256 | 4 | 0.021 |
| `matmul_fp4` | 64×5120, K 5120 | 4 | 0.305 |
| `matmul_fp4` | 64×30720, K 5120 | 4 | 1.742 |
| `matmul_fp4` | 2176×64, K 5120 | 4 | 0.170 |
| `attend` | 2160×2160, 40×128 | 160 | 255.220 |
| `attend` | 2160×512, 40×128 | 160 | 59.989 |

The 512-row `matmul_fp4` with K 5120 counts the cross k and v projections and the second text embedding. The K 4096 row is the first text embedding. The 64-row rows are the time embedding and the time projection. The 2176×64 row is the head. Layer norms, gates, and the feed-forward activation run once per layer. `patch_embed` and the Euler update run once per step.

| Kernel | Where | Count | ms |
|---|---|---:|---:|
| `scale_gather` | K 5120, 2176 rows | 960 | 0.277 |
| `rescale_rows` | 5120, 2176 rows | 960 | 0.322 |
| `add_bias` | 5120, 2176 rows | 960 | 0.324 |
| `scale_gather` | K 5120, 512 rows | 324 | 0.053 |
| `rescale_rows` | 5120, 512 rows | 324 | 0.035 |
| `add_bias` | 5120, 512 rows | 324 | 0.036 |
| `scale_gather` | K 5120, N 13824 | 160 | 0.271 |
| `rescale_rows` | N 13824 | 160 | 0.894 |
| `add_bias` | N 13824 | 160 | 0.887 |
| `scale_gather` | K 13824, N 5120 | 160 | 0.930 |
| `rescale_rows` | N 5120, K 13824 | 160 | 0.333 |
| `add_bias` | N 5120, K 13824 | 160 | 0.322 |
| `layernorm` | 5120 | 324 | 0.340 |
| `layernorm_affine` | 5120 | 160 | 0.338 |
| `modulate` | 5120 | 324 | 0.320 |
| `rmsnorm` | 5120, token rows | 480 | 0.344 |
| `rmsnorm` | 5120, text rows | 160 | 0.054 |
| `rope3d` | 2160×40×128 | 320 | 0.326 |
| `gate_residual` | 5120 | 480 | 0.481 |
| `gelu_tanh` | 13824 | 160 | 0.864 |
| `add_half` | 6×5120 | 160 | 0.002 |
| `copy_f32` | 5120 | 1124 | 0.002 |
| `patch_embed` | 36×3×40×72 | 4 | 24.447 |
| `zero_bytes` | token rows | 4 | 0.199 |

The gather, rescale, and bias for the text, time, and head projections, plus `gelu_tanh` on the text rows, `silu`, `add_embed_half`, `sinusoid`, `unpatchify`, `condition36`, and `axpy`, add 0.002 s. The unrounded median sum for the DiT is 72.532 s.

`attend` reads the query on both walks, writes the destination once, and walks every key twice and every value once. One self-attention call is 286,787,174,400 bytes. One hundred sixty calls are 165.632 s at 277.04 GB/s. The measured self-attention is 40.835 s. One cross-attention call is 68,080,435,200 bytes. One hundred sixty calls are 39.319 s at the same bandwidth. The measured cross-attention is 9.598 s.

## umT5

The smoke prompt is 9 tokens. `matmul_f16` pads those rows to 64. Each of the 24 layers runs four 4096×4096 linears, two 10240×4096 linears, one 4096×10240 linear, `attend_bias` over 9×9 with 64 heads of 64, two RMSNorms, a gated GELU, and two residual adds. One final RMSNorm follows the stack. A linear is `cast_half` and half `matmul2d`. The unrounded sum is 0.049 s.

| Kernel | Shape | Count | ms |
|---|---|---:|---:|
| `matmul_f16` | 64×4096, K 4096 | 96 | 0.177 |
| `matmul_f16` | 64×10240, K 4096 | 48 | 0.421 |
| `matmul_f16` | 64×4096, K 10240 | 24 | 0.435 |
| `attend_bias` | 9×9, 64×64 | 24 | 0.010 |
| `rmsnorm` | 4096 | 49 | 0.015 |

The token-row read from the checkpoint is outside this sum.

## VAE

Encode walks the first pixel frame, then two groups of four. Decode walks the three latent frames. A convolution of at least 32 output channels, 1024 outputs, and 64 taps is `conv_pack` once, then `conv_im2col`, half `matmul_f16`, and `conv_scatter` for each 8192-row tile. That dispatch is one command buffer. Smaller convolutions stay on float `conv3d`. The shapes at or above 0.25 s in aggregate:

| Input | Weight | Count | ms |
|---|---|---:|---:|
| 96×5×320×576 | 96×96×3×3×3 | 10 | 118.624 |
| 96×6×320×576 | 96×96×3×3×3 | 10 | 118.490 |
| 192×6×160×288 | 192×192×3×3×3 | 9 | 73.077 |
| 192×5×160×288 | 192×192×3×3×3 | 9 | 72.541 |
| 96×1×320×576 | 96×96×3×3×3 | 10 | 29.946 |
| 384×3×80×144 | 384×384×3×3×3 | 8 | 32.772 |
| 384×4×80×144 | 384×384×3×3×3 | 8 | 32.128 |

The other 49 shapes are 143 calls and add 1.074 s. All 207 calls sum to 5.575 s. `spatial_attend` runs six times at 384 channels over 40×72, median 612.848 ms, 3.677 s. The channel RMS, clipped SiLU, causal cache, resample, and the latent affine are 859 calls and add 0.524 s. The VAE sum is 9.776 s.

## Clip

The DiT, the 9-token encoder, and the VAE sum to 82.357 s. The 90% test uses that kernel ceiling. The attention bandwidth ceiling is 204.951 s. The expert pack read and the AVI pack are outside the kernel sum.

A 9-frame 576×320 smoke printed umT5 2.49 s, VAE encode 4.65 s, four steps 18.88 s, 18.95 s, 18.83 s, and 19.02 s, and VAE decode 6.67 s. Those stages sum to 89.49 s, within 82.357 s / 0.9.
