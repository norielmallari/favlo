# Ceilings

- Kernel and bandwidth ceilings for the kernels this pipeline runs.
- Measured on 24 November 2026 on an M5 Pro with 48 GB.
- Nearest E4M3 attention and output projection, a float16 head, Metal fp4 experts in layers 0–19, MXFP8 experts in layers 20–23, and `half` keys and values.
- Reuse these figures while those kernels are unchanged. Recompute this note when one of them changes.
- No committed command regenerates these tables. They came from a one-off script that times the runtime's own dispatches.

## Bandwidth

- One `copy_words` dispatch copied 1 GiB into another 1 GiB buffer.
- One run was discarded. The median of the next nine was 8.438 ms.
- That is 2 GiB of traffic at 254.51 GB/s.
- The figure is wall time around one command buffer, so it is a lower bound on device bandwidth.
- A decode ceiling divides the bytes one token reads and writes by 254.51 GB/s.

## Decode bytes

- One token is 48 layer visits, then the head.
- A full-attention visit reads every visible key and value. A local visit reads at most 1024.
- Eight full-attention visits and forty local visits.
- Each local or full visit reads an fp8 attention plane (`8576×2304`), an fp8 output plane (`2304×3072`), the sink, the router, the norms, the convolution taps, and the query tables.
- Layers 0–19 are Metal fp4. One expert is `2304×2304` and `2304×1280` after the K pad, 4,386,816 bytes. Forty-four visits.
- Layers 20–23 are MXFP8. One expert is `2304×2304` and `2304×1152`, 8,211,456 bytes. Four visits.
- Eight expert images per visit are 1,806,925,824 bytes.
- The head is `200064×2304` float16, 921,894,912 bytes, read once.
- Message-gate planes are read on the 24 visits of the second and third core passes.
- Convolution state is read and written on every visit.
- Fixed bytes below are those weights and that state. Keys and values are added for the context.

| Piece | Bytes |
|---|---:|
| Experts | 1,806,925,824 |
| Attention plane | 978,075,648 |
| Head | 921,894,912 |
| Output plane | 350,355,456 |
| Message gates | 113,246,208 |
| Router | 28,311,552 |
| Sink | 14,155,776 |
| Convolution state | 7,667,712 |
| Key and value projections | 4,718,592 |
| Other fixed planes | 11,917,824 |
| Fixed total | 4,237,269,504 |

| Context | Key and value bytes | Bytes per token | Bandwidth ceiling |
|---:|---:|---:|---|
| 512 | 100,663,296 | 4,337,932,800 | 17.045 ms, 58.67 tok/s |
| 1024 | 201,326,592 | 4,438,596,096 | 17.440 ms, 57.34 tok/s |
| 4096 | 301,989,888 | 4,539,259,392 | 17.836 ms, 56.07 tok/s |
| 10000 | 495,452,160 | 4,732,721,664 | 18.596 ms, 53.78 tok/s |

## Decode kernel ceiling

- Each kernel ran in its own command buffer, then waited.
- The time is that buffer's GPU start-to-end. Overlapping buffers are not added.
- Three runs of five completed steps. The first step of each run was discarded. The median of the other twelve steps is the ceiling.
- The 90% test uses this ceiling.

| Context | Kernel time | Kernel ceiling | Bandwidth share |
|---:|---|---|---:|
| 512 | 21.072 ms | 47.46 tok/s | 80.9% |
| 1024 | 21.225 ms | 47.11 tok/s | 82.2% |
| 4096 | 22.072 ms | 45.31 tok/s | 80.8% |
| 10000 | 22.698 ms | 44.06 tok/s | 81.9% |

- Context 4096, one 22.041 ms step in that series. A group's time is the sum of its kernels in that step. Smaller kernels make up the rest of the step.

| Group | Kernels | Time |
|---|---|---:|
| Attention projection | `gemv_attn` | 4.123 ms |
| fp4 expert gate | `moe_gate_fp4` | 4.098 ms |
| Head | `gemv_f16_logits` | 3.250 ms |
| fp4 expert down | `moe_down_fp4` | 2.539 ms |
| Decode attention | `attend_group` | 1.828 ms |
| Output projection | `gemv_fp8_conv_add` | 1.591 ms |
| Block mix | `mix_blocks` | 1.356 ms |
| fp8 expert gate | `moe_gate_fp8` | 0.710 ms |
| Router | `route_gemv` | 0.626 ms |
| Message gates | `ctl_down`, `ctl_blend` | 0.936 ms |
| fp8 expert down | `moe_down_fp8` | 0.366 ms |

- The expert kernels in that step are 7.713 ms. The eight expert images are 7.100 ms at 254.51 GB/s.
- The attention plane is 3.843 ms at that rate. `gemv_attn` took 4.123 ms.
- The head is 3.622 ms at that rate. `gemv_f16_logits` took 3.250 ms.

## Prefill kernel ceiling

- Prefill walks the prompt in rows of 2048 and flushes the command buffer after each visit.
- The same isolation as decode: one kernel, one waited command buffer, GPU start-to-end, summed.
- Two prefills from a reset cache. The first is discarded. The second is the ceiling.
- The prompt is one repeated token. Those expert tiles stay in cache. A chat prompt routes more experts, so its expert tiles are slower. That is a separate row below.
- A one-turn prompt does not end the piece at the first user marker. Ending there ran a second forward.
- A prefill tile reads each weight image once per piece, so the decode byte ceiling is not the prefill bandwidth ceiling.

| Tokens | Kernel time | Kernel ceiling | Kernels |
|---:|---|---|---:|
| 512 | 0.341148 s | 1500.81 tok/s | 1838 |
| 1024 | 0.655279 s | 1562.69 tok/s | 1838 |
| 4096 | 2.775520 s | 1475.76 tok/s | 3713 |
| 10000 | 7.124522 s | 1403.60 tok/s | 9338 |

- The group's time is the sum of its kernels in that second prefill.

| Group | Kernels | 4096 | 10000 |
|---|---|---:|---:|
| fp4 expert tiles | `expert_fp4_tile` | 1.0508 s | 2.4859 s |
| Prefill attention | `attend_flash` | 0.5339 s | 1.7922 s |
| fp8 projections | `matmul_fp8_m32` | 0.4222 s | 1.0253 s |
| Block mix | `mix_blocks` | 0.1015 s | 0.2434 s |
| fp8 expert tiles | `expert_fp8_tile` | 0.0902 s | 0.2153 s |
| Expert activation | `swiglu_grouped` | 0.0758 s | 0.1779 s |
| Message-gate blend | `ctl_blend_rows` | 0.0733 s | 0.1754 s |
| Half projections | `matmul_f16_m64` | 0.0712 s | 0.1711 s |
| Expert scatter | `scatter_grouped` | 0.0642 s | 0.1509 s |
| Expert gather | `gather_grouped` | 0.0588 s | 0.1342 s |
| Cast | `cast_f16` | 0.0472 s | 0.1100 s |
| Message-gate pair | `ctl_pair_rows` | 0.0413 s | 0.0988 s |

## Chat prefill at 512

- One chat prompt of 512 tokens, isolated the same way.
- With a piece break at the first user marker: 3713 kernels, 0.5145 s, 995.14 tok/s. `expert_fp4_tile` was 0.2680 s.
- Without that break: 1838 kernels, 0.4205 s, 1217.59 tok/s. `expert_fp4_tile` was 0.2249 s.
- The repeated token, same isolation: 1838 kernels, 0.3358 s, 1524.92 tok/s. `expert_fp4_tile` was 0.1564 s.
- `matmul_fp8_m32` was 0.0538 s on the chat prompt and 0.0539 s on the repeated token.
- The one-turn forward keeps the first marker in the piece, so its kernel ceiling is 0.4205 s, 1217.59 tok/s.
