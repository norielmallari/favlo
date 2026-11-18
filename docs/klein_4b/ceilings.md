# Ceilings

Kernel and bandwidth ceilings for one 576×320 edit, four steps, one reference of the same size. Reuse these figures while the kernels in `shaders/kernels.metal` are unchanged. Recompute this note when one of them changes.

Each time is one command buffer, then a wait. The time is `GPUEndTime − GPUStartTime`. Overlapping command buffers are not added. The figure is the median of five dispatches after one discarded sample.

A 2 GiB copy finished in 17.461 ms, which is 245.98 GB/s counting the read and the write. The bandwidth ceiling divides bytes by 245.98 GB/s.

## Generation

The joint sequence is 1952 tokens: 512 text tokens plus 1440 image tokens (720 reference and 720 target). A double-block attention and a single-block attention both use that sequence, 24 heads of 128. There are five double blocks and twenty single blocks, so one step has 25 of those attentions. Four steps have 100. The text tower has 27 causal attentions of 512 tokens, 32 heads of 128. The encoder and the decoder each have one mid-block attention of 2880 spatial positions, one head of 512.

| Kernel | Shape | Count | ms |
|---|---|---:|---:|
| `attend_flash` | 1952×24×128 | 100 | 741.2 |
| `attend_flash` | 512×32×128 | 27 | 19.5 |
| `attend_f32` | 2880×1×512 | 2 | 177.9 |
| `quantize_f16_i8` + `matmul_i8` + `store_f16` | 1440×3072×3072 | 80 | 102.6 |
| `quantize_f16_i8` + `matmul_i8` + `store_f16` | 512×3072×3072 | 80 | 36.8 |
| `quantize_f16_i8` + `matmul_i8` + `store_f16` | 1440×18432×3072 | 20 | 147.8 |
| `quantize_f16_i8` + `matmul_i8` + `store_f16` | 1440×3072×9216 | 20 | 72.8 |
| `quantize_f16_i8` + `matmul_i8` + `store_f16` | 512×18432×3072 | 20 | 51.1 |
| `quantize_f16_i8` + `matmul_i8` + `store_f16` | 512×3072×9216 | 20 | 26.7 |
| `quantize_f16_i8` + `matmul_i8` + `store_f16` | 1952×27648×3072 | 80 | 1208.8 |
| `quantize_f16_i8` + `matmul_i8` + `store_f16` | 1952×3072×12288 | 80 | 515.5 |
| `quantize_f16_i8` + `matmul_i8` + `store_f16` | text tower, per layer | 27 | 124.3 |
| `modulate_row` | 1952×3072 | 164 | 51.3 |
| `gate_f16` | 1952×3072 | 160 | 50.1 |
| `swiglu_f16` | 1952×18432 | 120 | 125.0 |
| `rope_dit` | 1952×24×128 | 240 | 59.8 |
| `im2col_f16` | 128×320×576, 3×3 | 10 | 80.7 |
| `groupnorm_f16` | 128×320×576 | 2 | 5.1 |

The projection rows are `rows × N × K`. Counts are for the whole edit. Image-only calls are counted at the 1952-row measurement. The text-tower line is 27 layers of Q, K, V, the output projection, the two up projections, and the down projection. The ten gathers are the four 3×3 convolutions in the first encoder down block and the six in the last decoder up block.

Timestep projections, modulation projections, smaller VAE convolutions, and the head-sized RMSNorms are not in this sum.

Head dim 128 uses `attend_flash`: a 32×32 query tile and a 32×32 key tile, with the score tile kept in the threadgroup. The query-key product and the value mix are `matmul2d`. The measured medians are 7.412 ms and 0.721 ms. Head dim 512 stays on `attend_f32`.

The kernel ceiling is 3.60 s. The bandwidth ceiling is the byte bound for a later kernel pass; this note does not use it as the 90% test.
