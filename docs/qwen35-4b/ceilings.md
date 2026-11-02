# Ceilings

Kernel and bandwidth ceilings for the kernels this pipeline runs. Reuse these figures while those kernels are unchanged. Recompute this note when one of them changes.

Each time is one command buffer, then a wait. Overlapping command buffers are not added. The times are the median of four samples after two warmups. Measured on 2 November 2026 on the default `f16` cache.

The divisor for bandwidth is the 2 GiB streaming read on an M5 Pro with 48 GB: 7093.5 µs, 302.74 GB/s, recorded in [../qwen36-35b-a3b/ceilings.md](../qwen36-35b-a3b/ceilings.md). An earlier write-based figure of 234.92 GB/s remains only in the prefill byte note below until that section is recomputed.

## Prefill

One chunk of 256 tokens (the matmul row multiple used for the isolated times below; the forward's `PREFILL_TILE` is 1024), with a 2048-key attention window on the eight full-attention layers. Prefill attention is `attend_prefill_f16` on the float16 cache. Gated DeltaNet is `gdn_chunk`. The head is `gemv_fp8` on the last prompt row only.

Designed dense path for FFN gate/up/down and GDN qkv/z/out: `restore_fp8_i8`, `quantize_half_i8`, `prefill_i8_n64` (or `prefill_i8_m32` for down), `scale_i8_f32`. GDN a/b and attention q/k/v/o stay on `matmul_fp8` / `matmul_fp8_n32`.

Isolated int8 gate path at 256 rows (2 November 2026): `restore_fp8_i8` 1837.9 µs, `quantize_half_i8` 3.7 µs, `prefill_i8_n64` 268.9 µs, `scale_i8_f32` 37.7 µs. The same shape on `matmul_fp8` is 491.9 µs. The restore dominates an isolated command buffer; the live forward keeps one command buffer per chunk, so the matmul win shows up in serve rates.

The earlier half×fp8-only composition for a 256-token chunk summed to 241125.0 µs, 1061.69 tok/s. That figure is kept as the last full-chunk kernel ceiling until the int8 path is summed the same way. The 90% test for prefill uses 1061.69 tok/s until that recompute.

The byte count for that first chunk (dense projections, tied embed and head, Gated DeltaNet state, float16 key/value store and attend for 2048 keys on eight layers, float32 score scratch) is 5,260,189,696 bytes, 22.39 ms at 234.92 GB/s, 11432.78 tok/s.

## Decode

One token at context 2048. Dense projections use `gemv_fp8` (eight rows per threadgroup). Mid-width outs (256–16384 rows, inner ≤4096) use `gemv_fp8_x4` in the forward; the table below is still the eight-row kernel. Attention is `attend_decode_f16` on the float16 cache.

| Kernel | Shape | Count | µs |
|---|---|---:|---:|
| `gemv_fp8` gate | 9216×2560 | 32 | 327.5 |
| `gemv_fp8` up | 9216×2560 | 32 | 327.5 |
| `gemv_fp8` down | 2560×9216 | 32 | 332.7 |
| `gemv_fp8` qkv | 8192×2560 | 24 | 291.6 |
| `gemv_fp8` z | 4096×2560 | 24 | 154.3 |
| `gemv_fp8` a | 32×2560 | 24 | 24.7 |
| `gemv_fp8` b | 32×2560 | 24 | 24.7 |
| `gemv_fp8` out | 2560×4096 | 24 | 153.4 |
| `gemv_fp8` q | 8192×2560 | 8 | 139.4 |
| `gemv_fp8` k | 1024×2560 | 8 | 23.3 |
| `gemv_fp8` v | 1024×2560 | 8 | 23.3 |
| `gemv_fp8` o | 2560×4096 | 8 | 60.9 |
| `gemv_fp8` head | 248320×2560 | 1 | 2671.0 |
| `rmsnorm` | 2560 | 65 | 6.3 |
| `residual_add` | 2560 | 64 | 2.0 |
| `silu_mul` | 9216 | 32 | 2.2 |
| `gdn_step` | 32×128×128 | 24 | 12.5 |
| `conv_silu` | 8192×4 | 24 | 2.3 |
| `gated_rmsnorm` | 32×128 | 24 | 2.4 |
| `head_rmsnorm` q/k | — | 16 | 2.5 |
| `rope_heads` q/k | — | 16 | 2.7 |
| `store_kv_f16` | — | 8 | 2.5 |
| `attend_decode_f16` | 16×2048 | 8 | 7.8 |
| `embed_row` | 2560 | 1 | 2.6 |
| `argmax` | vocab | 1 | 12.5 |

The gemv and listed elementwise sum is 53018.9 µs, 18.86 tok/s. That is the kernel ceiling for decode (isolated command buffers). One token in the forward shares a command buffer across those dispatches, so serve decode is higher.

Language weight bytes plus float16 K/V read at 2048 keys on eight layers and Gated DeltaNet state read/write are 4,503,961,600 bytes, 14.88 ms at 302.74 GB/s, 67.22 tok/s. The 90% test for decode uses the kernel ceiling.
