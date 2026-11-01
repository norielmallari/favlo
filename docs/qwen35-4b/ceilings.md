# Ceilings

Kernel and bandwidth ceilings for the kernels this pipeline runs. Reuse these figures while those kernels are unchanged. Recompute this note when one of them changes.

Each time is one command buffer, then a wait. Overlapping command buffers are not added. The times are the median of four samples after one discarded sample. Measured on 2 November 2026.

A 2 GiB streaming write finished in 8954.5 µs, 239.82 GB/s. The bandwidth ceiling divides bytes by 239.82 GB/s.

## Prefill

One chunk of 256 tokens (the matmul row multiple used for the isolated times below; the forward's `PREFILL_TILE` is 1024), with a 2048-key attention window on the eight full-attention layers. Dense projections use `matmul_fp8` (64×64) and `matmul_fp8_n32` for GDN a/b. Prefill attention in this table is `attend_prefill` on the e4m3 cache. The default cache is now `f16` (`attend_prefill_f16`); those kernels are not in this table. Gated DeltaNet is `gdn_chunk` over the tile. The head is `gemv_fp8` on the last prompt row only.

| Kernel | Shape | Count in one chunk | µs per call |
|---|---|---:|---:|
| `embed_rows` | 256×2560 | 1 | 169.2 |
| `rmsnorm_rows` | 256×2560 | 64 | 168.1 |
| `cast_f32_f16` hidden | 256×2560 | 56 | 169.9 |
| `matmul_fp8` qkv | 8192×2560 | 24 | 604.2 |
| `matmul_fp8` z | 4096×2560 | 24 | 386.7 |
| `matmul_fp8_n32` a | 32×2560 | 24 | 207.7 |
| `matmul_fp8_n32` b | 32×2560 | 24 | 200.1 |
| `conv_silu_chunk` | 8192×4 | 24 | 205.9 |
| `gdn_chunk` | 32×128×128×256 | 24 | 2018.2 |
| `gated_rmsnorm_rows` | 256×32×128 | 24 | 182.2 |
| `cast_f32_f16` gdn | 256×4096 | 24 | 133.0 |
| `matmul_fp8` out | 2560×4096 | 24 | 363.4 |
| `matmul_fp8` q | 8192×2560 | 8 | 603.0 |
| `matmul_fp8` k | 1024×2560 | 8 | 226.6 |
| `matmul_fp8` v | 1024×2560 | 8 | 220.5 |
| `head_rmsnorm_rows` | q/k | 16 | 179.7 |
| `rope_heads_rows` | q/k | 16 | 137.4 |
| `store_kv_rows` | k/v | 16 | 156.8 |
| `attend_prefill` | 16×256×2048 | 8 | 2446.1 |
| `cast_f32_f16` attn | 256×4096 | 8 | 133.6 |
| `matmul_fp8` o | 2560×4096 | 8 | 368.5 |
| `matmul_fp8` gate | 9216×2560 | 32 | 643.6 |
| `matmul_fp8` up | 9216×2560 | 32 | 648.9 |
| `silu_mul_rows` | 256×9216 | 32 | 226.7 |
| `cast_f32_f16` inter | 256×9216 | 32 | 148.9 |
| `matmul_fp8` down | 2560×9216 | 32 | 712.1 |
| `residual_add_rows` | 256×2560 | 64 | 167.6 |
| `gemv_fp8` head | 248320×2560 | 1 | 2411.1 |

One GDN layer is 7692.6 µs. One full-attention layer (2048-key window) is 8337.3 µs after the value-mix unroll in `attend_prefill` (remeasured 2 November 2026). One 256-token chunk is 254070.0 µs, 1007.60 tok/s. That is the kernel ceiling for prefill.

The byte count for that first chunk is the dense projections, the tied embed and head, Gated DeltaNet state read and write, key/value store and attend traffic for 2048 keys on eight layers, and the float32 score scratch. That is 5,195,177,984 bytes, 21.66 ms at 239.82 GB/s, 11817.62 tok/s. The 90% test for prefill uses the kernel ceiling.

## Decode

Not measured in this note. Prefill is the phase under optimization.
