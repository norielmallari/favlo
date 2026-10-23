# Ceilings

Kernel and bandwidth ceilings for one 1280×1280 generation. Reuse these figures while `matmul_fp4`, `scale_to_half`, `bias_cast`, `rmsnorm`, `rope_axes`, `attend`, `attend_q4`, `channel_scale`, `channel_gate`, and `silu_mul` are unchanged. Recompute this note when one of them changes.

Each time is one command buffer, then a wait. The time is `GPUEndTime − GPUStartTime`. Overlapping command buffers are not added. The figure is the median of five dispatches after one discarded sample.

A 2 GiB copy finished in 16.674 ms, which is 257.59 GB/s counting the read and the write. The bandwidth ceiling divides bytes by 257.59 GB/s.

## Generation

Eight velocity steps and the decoder. The sequence is 6528 tokens: 6400 image tokens from a 1280 edge, then 128 caption tokens. Each step in this sum is 2 noise-refiner blocks and 30 main blocks, 256 blocks in all. A noise refiner in the forward sees only the image tokens; this sum uses 6528 for every block. The caption prefill, the context refiner, the time embed, the image embed, and the final readout are not in the sum.

One modulated block is 4 projections of 3840×3840, 2 of 10240×3840, and 1 of 3840×10240. Each projection is `scale_to_half`, `matmul_fp4`, and `bias_cast`. The block also runs 4 hidden RMSNorms, 2 head RMSNorms, 2 `channel_scale`, 2 `channel_gate`, one `silu_mul`, 2 `rope_axes`, and one `attend` of 30 heads of 128.

| Kernel | Shape | Count | ms |
|---|---|---:|---:|
| `matmul_fp4` | 3840×3840 | 1024 | 18.516 |
| `matmul_fp4` | 10240×3840 | 512 | 48.189 |
| `matmul_fp4` | 3840×10240 | 256 | 51.311 |
| `scale_to_half` | inner 3840 | 1536 | 0.578 |
| `scale_to_half` | inner 10240 | 256 | 1.734 |
| `bias_cast` | 3840 | 1280 | 0.738 |
| `bias_cast` | 10240 | 512 | 2.027 |
| `rmsnorm` | 3840 | 1024 | 0.754 |
| `rmsnorm` | head 128 | 512 | 0.730 |
| `channel_scale` | 3840 | 512 | 0.760 |
| `channel_gate` | 3840 | 512 | 1.115 |
| `silu_mul` | 10240 | 256 | 2.901 |
| `rope_axes` | 30×128 | 512 | 0.854 |
| `attend_q4` | 6528×30×128 | 256 | 242.959 |

One block is 490.5 ms. The 256 blocks are 125.57 s.

`attend_q4` reads the query once, writes the destination once, and walks every key and every value for every query. That is 1,309,326,704,640 bytes per call. Two hundred fifty-six calls are 1301 s at 257.59 GB/s. The measured `attend_q4` is 62.20 s, under that line. Eight queries share each key, and four keys are scored before the softmax rescales. The decoder head is 512 and stays on `attend`.

The decoder adds one `attend` over the 160×160 latent, one head of 512, median 5597 ms. The 3×3 convolutions below are `matmul_fp4` on a 4096-row chunk. The last partial chunk is counted as a full chunk. `conv_in`, `conv_out`, the 1×1 shortcuts, and group norm are not in this sum.

| Kernel | Where | Chunks | ms |
|---|---|---:|---:|
| `attend` | mid-block, 25600 pixels | 1 | 5597.062 |
| `matmul_fp4` | 512→512, inner 4608 | 345 | 1.865 |
| `matmul_fp4` | 256→256, inner 2304 | 900 | 0.467 |
| `matmul_fp4` | 128→128, inner 1280 | 2000 | 0.136 |
| `matmul_fp4` | 512→256, inner 4608 | 100 | 0.959 |
| `matmul_fp4` | 256→128, inner 2304 | 400 | 0.256 |

The decoder sum is 7.13 s. One generation is 132.70 s. The 90% test uses that kernel ceiling. The attention bandwidth ceiling is 1301 s.

The latent path adds `patchify`, `unpatchify`, `euler`, `sinusoid`, `copy_rows`, `zero_bytes`, `text_positions`, `image_positions`, `joined_positions`, `affine`, and `clip_u8_hwc`. Those dispatches are outside the block sum. At 1280 the medians are 0.046 ms, 0.099 ms, 0.028 ms, 0.007 ms, 0.739 ms for a 6400×3840 copy, 0.014 ms for a 128×3840 copy, 0.446 ms to clear the joined sequence, 0.031 ms to clear the token rows, 0.002 ms for a position table, 0.005 ms for the VAE scale, and 0.054 ms for the uint8 clip. Eight steps of that glue plus the decoder clip are 11.6 ms. That glue sits outside the 132.70 s sum.

A 1280 generation of the smoke prompt printed `generation_seconds.total` 134.36 s, caption 0.17 s, and decoder 9.90 s. The eight steps were 15.48–15.72 s. That wall clock is within 132.70 s / 0.9.
