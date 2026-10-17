# Ceilings

Kernel and bandwidth ceilings for the kernels this pipeline runs. Reuse these figures while those kernels are unchanged. Recompute this note when one of them changes.

Each time is one command buffer, then a wait. Overlapping command buffers are not added. The head settled at 1219 µs after a run that started at 4753 µs. `topk_batch` is one simdgroup per token; a one-token dispatch is 66.1 µs. The other times are the median after one discarded sample.

A 2 GiB streaming read finished in 8043 µs, 267.0 GB/s. A copy of that buffer finished in 16906 µs, 254.1 GB/s. The bandwidth ceiling divides bytes by 267.0 GB/s.

## Decode

One token, experts resident, fp4 weights, fp8 E4M3 K and V. A span of two to four tokens uses `gemv_int4_span` for the int4 projections. The totals below stay the one-token GEMV. The forward is 36 Gated DeltaNet layers, 12 sparse-attention layers, the MTP layer, the final mixer, one PLE injection, and one head. Top 10. The MTP layer is the cycle that runs when a draft is not taken. A draft repeats that layer and the head; those two dispatches are not in the totals below.

`gemv_int4_f32` is the one-token projection. The router is `gemv_f16` with the half cast in the same buffer. Routed experts are `moe_decode_gate` and `moe_decode_down` at top 10. `rms_rows` is the branch norm, including the final norm.

| Kernel | Shape | Count | µs | ms |
|---|---|---:|---:|---:|
| `gemv_int4_f32` qkv | 10240×2560 | 36 | 221.0 | 7.956 |
| `gemv_int4_f32` z | 6144×2560 | 36 | 147.9 | 5.324 |
| `gemv_int4_f32` a | 48×2560 | 36 | 46.2 | 1.662 |
| `gemv_int4_f32` b | 48×2560 | 36 | 43.1 | 1.551 |
| `gemv_int4_f32` out | 2560×6144 | 36 | 130.4 | 4.695 |
| `gemv_int4_f32` q | 12288×2560 | 13 | 263.1 | 3.421 |
| `gemv_int4_f32` k | 512×2560 | 13 | 42.5 | 0.552 |
| `gemv_int4_f32` v | 512×2560 | 13 | 42.9 | 0.558 |
| `gemv_int4_f32` o | 2560×6144 | 13 | 130.7 | 1.700 |
| `gemv_int4_f32` index_qk | 640×2560 | 13 | 43.3 | 0.562 |
| `gemv_int4_f32` shared gate | 640×2560 | 49 | 14.2 | 0.696 |
| `gemv_int4_f32` shared up | 640×2560 | 49 | 11.7 | 0.576 |
| `gemv_int4_f32` shared down | 2560×640 | 49 | 8.0 | 0.392 |
| `gemv_int4_f32` coeff | 1×2560 | 49 | 6.2 | 0.304 |
| `gemv_int4_f32` hyper down | 320×10240 | 99 | 24.1 | 2.384 |
| `gemv_int4_f32` hyper up | 10240×320 | 99 | 18.3 | 1.815 |
| `gemv_int4_f32` inject | 4×10240 | 98 | 25.6 | 2.511 |
| `gemv_int4_f32` PLE key | 10240×2560 | 1 | 51.5 | 0.051 |
| `gemv_int4_f32` PLE value | 2560×2560 | 1 | 16.4 | 0.016 |
| `gemv_int4_f32` head | 248320×2560 | 1 | 1219.0 | 1.219 |
| `gemv_f16` router | 512×2560 | 49 | 17.5 | 0.855 |
| `moe_decode_gate` | 1280×2560, top 10 | 49 | 62.5 | 3.062 |
| `moe_decode_down` | 2560×768, top 10 | 49 | 44.5 | 2.181 |
| `gdn_conv_scan` | 10240×4 | 36 | 3.3 | 0.117 |
| `gdn_state_scan` | 48×128×128 | 36 | 16.8 | 0.606 |
| `topk_batch` | 512, k=10 | 49 | 66.1 | 3.239 |
| `rms_rows` | 4×2560 | 99 | 36.5 | 3.614 |

That sum is 51.62 ms before attention. A context of 2048 tokens takes the dense `qsa_attend` (1816.3 µs, 13 layers, 23.61 ms) and does not run the indexer. A longer context takes `qsa_attend_selected` over a 2048-token window (1868.0 µs, 24.28 ms) plus `indexer_block_score` and `indexer_select`.

| Context | Indexer score | Indexer select | Kernel total | Kernel ceiling | Bytes | Bandwidth ceiling |
|---|---:|---:|---:|---:|---:|---:|
| 2048 dense | — | — | 75.23 ms | 13.29 tok/s | 4.160 GB | 15.58 ms, 64.19 tok/s |
| 32768 selected | 43.6 µs | 189.4 µs | 78.93 ms | 12.67 tok/s | 4.683 GB | 17.54 ms, 57.01 tok/s |
| 262144 selected | 330.7 µs | 699.3 µs | 89.29 ms | 11.20 tok/s | 8.596 GB | 32.20 ms, 31.06 tok/s |

Indexer times are one layer. The totals multiply them by 13. The byte count is the resident projections, the ten expert images per expert layer, the float16 routers, the head, the PLE matrices, the GDN state read and write, K and V across the visible tokens, and the indexer keys. The 90% test for decode uses the kernel ceiling.

### Expert planes

The served record is fp4. A later run settled the head at 1173 µs, then measured `moe_decode_gate` at 61.0 µs and `moe_decode_down` at 43.3 µs. The fp4 rows above stay the served ceiling.

The fp8 and int2 expert shaders were timed the same way: resident zero images, one command buffer, the median after one discarded sample. The fp4 layer is the decode path around the two kernels above: gather the ten weights, `moe_decode_gate`, `moe_activate`, `moe_decode_down`. The fp8 layer walks the ten experts with `gemv_int8`, `moe_activate`, `gemv_int8_half`, and an axpy into the shared row. The int2 layer is `moe_decode_gate_int2`, `moe_activate`, and `moe_decode_down_int2`. A mixed layer runs fp8, then fp4, then int2, on the experts of each width. The counts in a mixed row are a synthetic split of the ten ids.

Live image bytes are 5,068,800 for fp8, 2,785,280 for fp4, and 1,658,880 for int2. A token total keeps the non-expert rows in the decode table and replaces the 5.243 ms of gate and down with 49 times the layer time. The byte count replaces the ten fp4 images with the live images of that assignment.

| Assignment | µs per layer | ms in the token |
|---|---:|---:|
| all fp4 | 115.6 | 5.664 |
| all fp8 | 572.0 | 28.028 |
| all int2 | 135.1 | 6.620 |
| 1 fp8, 8 fp4, 1 int2 | 180.2 | 8.830 |
| 2 fp8, 6 fp4, 2 int2 | 236.0 | 11.564 |

| Assignment | 2048 dense | 32768 selected | 262144 selected |
|---|---|---|---|
| all fp4 | 75.65 ms, 13.22 tok/s | 79.35 ms, 12.60 tok/s | 89.71 ms, 11.15 tok/s |
| all fp8 | 98.02 ms, 10.20 tok/s | 101.72 ms, 9.83 tok/s | 112.08 ms, 8.92 tok/s |
| all int2 | 76.61 ms, 13.05 tok/s | 80.31 ms, 12.45 tok/s | 90.67 ms, 11.03 tok/s |
| 1 fp8, 8 fp4, 1 int2 | 78.82 ms, 12.69 tok/s | 82.52 ms, 12.12 tok/s | 92.88 ms, 10.77 tok/s |
| 2 fp8, 6 fp4, 2 int2 | 81.55 ms, 12.26 tok/s | 85.25 ms, 11.73 tok/s | 95.61 ms, 10.46 tok/s |

| Assignment | 2048 bytes | 32768 bytes | 262144 bytes |
|---|---|---|---|
| all fp4 | 4.160 GB, 15.58 ms, 64.19 tok/s | 4.683 GB, 17.54 ms, 57.01 tok/s | 8.596 GB, 32.20 ms, 31.06 tok/s |
| all fp8 | 5.279 GB, 19.77 ms, 50.58 tok/s | 5.802 GB, 21.73 ms, 46.02 tok/s | 9.715 GB, 36.39 ms, 27.48 tok/s |
| all int2 | 3.608 GB, 13.51 ms, 74.00 tok/s | 4.131 GB, 15.47 ms, 64.63 tok/s | 8.044 GB, 30.13 ms, 33.19 tok/s |
| 1 fp8, 8 fp4, 1 int2 | 4.217 GB, 15.79 ms, 63.32 tok/s | 4.740 GB, 17.75 ms, 56.33 tok/s | 8.653 GB, 32.41 ms, 30.86 tok/s |
| 2 fp8, 6 fp4, 2 int2 | 4.273 GB, 16.01 ms, 62.48 tok/s | 4.796 GB, 17.96 ms, 55.67 tok/s | 8.709 GB, 32.62 ms, 30.66 tok/s |

The fp8 layer reads 50.688 MB. At 267.0 GB/s that read is 190 µs, and the measured layer is 572.0 µs. The int2 layer reads 16.589 MB, 62 µs at that rate, and the measured layer is 135.1 µs. The 90% test for the served record uses the fp4 table above.

## Prefill

One chunk of 2048 tokens. Projections are `matmul2d` through `linear` at that count. `gdn_conv_scan` and `gdn_state_scan` run at count 2048. `qsa_attend` is count 2048, base 0, e4m3, complete 2048. `topk_batch` is count 2048. The same layer counts as decode.

Routed experts are not in this total. `rms_rows` for the chunk was not timed. A later chunk, past the dense window, was not timed.

| Kernel | Count | µs per call | ms in the chunk |
|---|---:|---:|---:|
| `matmul2d` qkv | 36 | 9956.8 | 358.4 |
| `matmul2d` z | 36 | 8490.4 | 305.7 |
| `matmul2d` a | 36 | 1763.0 | 63.5 |
| `matmul2d` b | 36 | 1675.5 | 60.3 |
| `matmul2d` out | 36 | 6275.0 | 225.9 |
| `gdn_conv_scan` | 36 | 1125.9 | 40.5 |
| `gdn_state_scan` | 36 | 24257.3 | 873.3 |
| `matmul2d` q | 13 | 11829.3 | 153.8 |
| `matmul2d` k | 13 | 650.9 | 8.5 |
| `matmul2d` v | 13 | 655.8 | 8.5 |
| `matmul2d` o | 13 | 6235.0 | 81.1 |
| `matmul2d` index_qk | 13 | 769.5 | 10.0 |
| `qsa_attend` | 13 | 29284.1 | 380.7 |
| `matmul2d` hyper down | 98 | 1868.0 | 183.1 |
| `matmul2d` hyper up | 98 | 2474.0 | 242.5 |
| `matmul2d` inject | 98 | 1024.4 | 100.4 |
| `matmul2d` final hyper down | 1 | 1868.0 | 1.9 |
| `matmul2d` final hyper up | 1 | 2474.0 | 2.5 |
| `matmul2d` shared gate | 49 | 768.7 | 37.7 |
| `matmul2d` shared up | 49 | 770.3 | 37.7 |
| `matmul2d` shared down | 49 | 900.7 | 44.1 |
| `matmul2d` coeff | 49 | 149.6 | 7.3 |
| `gemv_f16` router | 49 | 9432.1 | 462.2 |
| `topk_batch` | 49 | 264.0 | 12.9 |
| `matmul2d` PLE key | 1 | 9933.8 | 9.9 |
| `matmul2d` PLE value | 1 | 2712.0 | 2.7 |
| `matmul2d` head | 1 | 235953.8 | 236.0 |

The chunk is 3950.9 ms, 1.929 ms per token, 518.4 tok/s. That is the kernel ceiling for this chunk without routed experts. It is not the 90% bar for prefill.
