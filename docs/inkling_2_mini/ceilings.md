# Ceilings

- Bandwidth ceilings and measured kernel costs for the kernels this pipeline runs.
- Measured on 22 November 2026 on an M5 Pro with 48 GB.
- These tables were measured on the GPTQ dense pack (`dense-fp4.safetensors`), NVFP4 experts requantized to Metal fp4 in layers 0–19, MXFP8 experts in layers 20–23, and `half` keys and values.
- The served fused attention plane and output projection are now fp8. The head is half. The decode byte totals below are that GPTQ pack, not the current layout. Recompute this note before using them as the current ceiling.
- No committed command regenerates these tables. They came from a one-off script that times the runtime's own dispatches.

## Bandwidth

- One `copy_words` dispatch copied 1 GiB to another 1 GiB buffer in 8.02 ms, the median of five runs after one discarded run.
- That is 2 GiB of traffic at 267.90 GB/s.
- The figure is wall time around one command buffer, so it is a lower bound on the device bandwidth.
- The decode ceiling below divides the bytes one token reads by 267.90 GB/s.

## Decode

- One token is one command buffer: 48 layer visits, then the head and the argmax.
- The byte count is the fp4 attention and output planes of every visit, the router and the other half planes each visit reads, the message gates of the second and third core passes, eight expert images per visit, the visible keys and values (all positions on a full-attention visit, at most 1024 on a local one), the convolution states read and written, one embedding row, and the head.
- A local visit keeps a 1024-token ring. These byte totals already counted that window. The device times below were measured while the local buffer was still context-sized; attention did not read past the window.
- The device time is the GPU start-to-end time of one step's command buffer, the median of the last 24 of 33 steps, with decode filled.
- The wall time per step was within 2% of the device time.
- At 16 GB the open keeps only the eight routed expert images per visit, so these figures describe the 48 GB layout.

| Context | Bytes per token | Byte ceiling | Device time | Device rate | Ceiling share |
|---:|---:|---|---|---|---:|
| 512 | 3,014,647,296 | 11.25 ms, 88.86 tok/s | 14.49 ms | 69.03 tok/s | 77.7% |
| 1024 | 3,115,310,592 | 11.63 ms, 85.99 tok/s | 14.78 ms | 67.64 tok/s | 78.7% |
| 4096 | 3,215,973,888 | 12.00 ms, 83.30 tok/s | 15.15 ms | 66.00 tok/s | 79.2% |
| 10000 | 3,409,436,160 | 12.73 ms, 78.57 tok/s | 15.97 ms | 62.64 tok/s | 79.7% |

### Decode cost by kernel group

- Context 4096. The step took 15.24 ms in this series.
- A group's cost is the step time with every kernel of that group, minus the step time with that group skipped.
- Skipping a group also removes the waits it causes, so the costs overlap and do not add up to the step.

| Group | Kernels | Cost |
|---|---|---:|
| Experts | `moe_gate_fp4`, `moe_gate_fp8`, `moe_down_fp4`, `moe_down_fp8` | 7.26 ms |
| Attention projection | `gemv_attn` | 2.11 ms |
| Decode attention | `attend_group` | 1.65 ms |
| Block mix and norm | `mix_blocks`, `rmsnorm` | 1.44 ms |
| Output projection | `gemv_fp4_conv_add` | 0.78 ms |
| Head | `gemv_fp4_logits`, `argmax_span`, `argmax_index` | 0.66 ms |
| Message gates | `ctl_down`, `ctl_blend` | 0.65 ms |
| Router | `route_gemv` | 0.56 ms |
| Attention prep | `attn_prep` | 0.21 ms |

- The eight expert images of 48 visits are 1,816,657,920 bytes, 6.78 ms at 267.90 GB/s. The expert kernels took 7.26 ms.

## Prefill

- Prefill walks the prompt in rows of 2048. Projections and experts multiply with `matmul2d`.
- Each figure is the faster of two prefills from a reset cache.
- A group's cost is measured the same way as for decode, and the costs overlap.

| Group | Kernels | 4096 tokens | 10000 tokens |
|---|---|---:|---:|
| Whole prefill | all | 2.876 s, 1424 tok/s | 7.469 s, 1339 tok/s |
| Expert tiles | `expert_fp4_tile`, `expert_fp8_tile` | 1.278 s | 3.068 s |
| Prefill attention | `attend_flash` | 0.532 s | 1.685 s |
| Dense fp4 projections | `matmul_fp4_m32` | 0.534 s | 1.099 s |
| Expert grouping | `moe_group`, `gather_grouped`, `swiglu_grouped`, `scatter_grouped` | 0.334 s | 1.066 s |
| Mix, norm, and cast | `mix_blocks`, `rmsnorm`, `cast_f16` | 0.311 s | 0.588 s |
| Message gates | `ctl_pair_rows`, `ctl_blend_rows` | 0.254 s | 0.479 s |
| Half projections | `matmul_f16_m64` | 0.193 s | 0.351 s |
| Attention prep and convolutions | `attn_prep`, `conv_tail`, `conv_add_rows` | 0.182 s | 0.408 s |
| Router rows | `route_rows` | 0.012 s | within noise |

- Prefill attention grows with the keys a full-attention visit reads. It is 18% of the 4096-token prefill and 23% of the 10000-token prefill.
