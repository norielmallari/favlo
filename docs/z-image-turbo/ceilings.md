# Ceilings

Kernel and bandwidth ceilings for one 1280×1280 generation. Reuse these figures while `matmul_fp4`, `scale_to_half`, `bias_cast`, `rmsnorm`, `rope_axes`, `attend`, `channel_scale`, `channel_gate`, and `silu_mul` are unchanged. Recompute this note when one of them changes.

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
| `attend` | 6528×30×128 | 256 | 2792.188 |

One block is 3039.7 ms. The 256 blocks are 778.17 s.

`attend` reads the query once, writes the destination once, and walks every key and every value for every query. That is 1,309,326,704,640 bytes per call. Two hundred fifty-six calls are 1301 s at 257.59 GB/s. The measured `attend` is 714.80 s, under that line, because the key and value rows are reused.

The decoder adds one `attend` over the 160×160 latent, one head of 512, median 5597 ms. The 3×3 convolutions below are `matmul_fp4` on a 4096-row chunk. The last partial chunk is counted as a full chunk. `conv_in`, `conv_out`, the 1×1 shortcuts, and group norm are not in this sum.

| Kernel | Where | Chunks | ms |
|---|---|---:|---:|
| `attend` | mid-block, 25600 pixels | 1 | 5597.062 |
| `matmul_fp4` | 512→512, inner 4608 | 345 | 1.865 |
| `matmul_fp4` | 256→256, inner 2304 | 900 | 0.467 |
| `matmul_fp4` | 128→128, inner 1280 | 2000 | 0.136 |
| `matmul_fp4` | 512→256, inner 4608 | 100 | 0.959 |
| `matmul_fp4` | 256→128, inner 2304 | 400 | 0.256 |

The decoder sum is 7.13 s. One generation is 785.30 s. The 90% test uses that kernel ceiling. The attention bandwidth ceiling is 1301 s.

A 1280 generation printed `generation_seconds.total` 771.44 s, caption 0.42 s, and decoder 10.74 s. The eight steps were 94.66–95.72 s. That wall clock is past 90% of the 785.30 s kernel ceiling.
