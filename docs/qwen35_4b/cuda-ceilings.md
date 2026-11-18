# Ceilings

Kernel and bandwidth ceilings for the kernels this pipeline runs. Reuse these figures while those kernels are unchanged. Recompute this note when one of them changes.

Each figure is the mean of the two middle samples of four, after two warmups. A sample is 32 back-to-back launches on the engine stream, divided by 32. NVIDIA CUDA events, read from Python. The streaming read and the device-to-device copy are one launch per sample. Measured 7 November 2026 on the opened device: NVIDIA compute capability 8.6, 40 SMs, 4 MiB L2, 256-bit device memory, 8589410304 bytes. The driver reports that device as GeForce RTX 3070 Laptop GPU.

The streaming read is a `float4` load of 1 GiB, eight passes per sample. That time is 20.9745 ms, 409.54 GB/s (decimal GB). A 1 GiB device-to-device copy is 5.3365 ms, 201.21 GB/s of payload.

## Decode

One token at context 2048. A projection is `quant_blocks` then `gemv_i8`. The head is the tied embedding. Attention at this length is `attend_split` over 2048 keys on the eight full layers, and the row includes `attend_merge` of the 32 partials. Gated DeltaNet is `gdn_kernel`.

The byte count is the int8 codes and fp32 row scales read once, the int8 key/value codes and E8M0 scales for that window, and the fp32 Gated DeltaNet state read and written once per token. That is 4345577472 bytes. At 409.54 GB/s the bandwidth ceiling is 10.61 ms, 94.24 tok/s.

| Kernel | Shape | Count | µs |
|---|---|---:|---:|
| `gemv` gate | 9216×2560 | 32 | 68.6 |
| `gemv` up | 9216×2560 | 32 | 68.6 |
| `gemv` down | 2560×9216 | 32 | 68.2 |
| `gemv` qkv | 8192×2560 | 24 | 61.2 |
| `gemv` z | 4096×2560 | 24 | 39.3 |
| `gemv` a | 32×2560 | 24 | 15.9 |
| `gemv` b | 32×2560 | 24 | 15.1 |
| `gemv` out | 2560×4096 | 24 | 37.9 |
| `gemv` q | 8192×2560 | 8 | 64.3 |
| `gemv` k | 1024×2560 | 8 | 20.6 |
| `gemv` v | 1024×2560 | 8 | 22.1 |
| `gemv` o | 2560×4096 | 8 | 37.5 |
| `gemv` head | 248320×2560 | 1 | 1548.4 |
| `attend_split` | 2048 keys | 8 | 45.3 |
| `gdn_kernel` | 32×128×128 | 24 | 11.2 |
| `conv_silu` | 8192 | 24 | 7.2 |
| `rmsnorm` | 2560 | 65 | 10.5 |
| `embed_row` | 2560 | 1 | 8.4 |
| `residual` | 2560 | 64 | 7.7 |
| `silu_mul` | 9216 | 32 | 7.9 |
| `rope` query | 16×64 | 8 | 8.0 |
| `rope` key | 4×64 | 8 | 8.0 |
| `head_norm` query | 16×256 | 8 | 7.7 |
| `head_norm` key | 4×256 | 8 | 12.0 |
| `gated_norm` | 32×128 | 24 | 7.6 |
| `store_kv` | 4×256 | 16 | 8.1 |

The isolated sum is 16180.9 µs, 61.80 tok/s. That is the kernel ceiling. The 90% test uses 61.80 tok/s.

The gate product moves 23629824 weight bytes in 68.6 µs, 344.5 GB/s, and does 0.688 TOPS. The head moves 636692480 bytes in 1548.4 µs, 411.2 GB/s. Attention moves 4325376 bytes in 45.3 µs, 95.48 GB/s.

## Prefill

One tile of 128 tokens. The tile starts at cache position 1921, so the last query sees 2048 keys. `gemm_i8` reads each weight row once per group of 64, so the tile reads the projection codes twice. The head runs once, on the last token. A projection is `quant_batch` then `gemm_i8`. Gate and up, the Gated DeltaNet pair, and query and key are `gemm_pair`: one quantize and two products, timed as one interval because the products overlap. `attend_prefill` stages sixteen keys for sixteen query rows. The score is `mma.m16n8k32` and the value mix is `mma.m16n8k16`.

The vocabulary table is read once. Every other projection is read twice. The eight 16-query tiles stage 15944 keys per KV head, including one masked key on the last tile. The Gated DeltaNet state is read and written once per token. That is 20938221568 bytes. At 409.54 GB/s the bandwidth ceiling is 51.13 ms, 2503.61 tok/s.

| Kernel | Shape | Count | µs |
|---|---|---:|---:|
| `gemm_pair` gate+up | 128×9216×2560 | 32 | 395.0 |
| `gemm` down | 128×2560×9216 | 32 | 314.3 |
| `gemm_pair` qkv+z | 128×8192×2560 | 24 | 286.3 |
| `gemm_pair` a+b | 128×32×2560 | 24 | 125.7 |
| `gemm` out | 128×2560×4096 | 24 | 128.4 |
| `gemm_pair` q+k | 128×8192×2560 | 8 | 250.9 |
| `gemm` v | 128×1024×2560 | 8 | 67.2 |
| `gemm` o | 128×2560×4096 | 8 | 132.0 |
| `gemv` head | 248320×2560 | 1 | 1540.3 |
| `attend_prefill` | 128 queries, 15944 keys | 8 | 939.1 |
| `gdn_kernel` | 128×32×128×128 | 24 | 488.5 |
| `conv_silu` | 128×8192 | 24 | 59.5 |
| `rmsnorm` | 128×2560 | 64 | 11.2 |
| `rmsnorm` final | 2560 | 1 | 9.8 |
| `embed_row` | 2560 | 128 | 7.1 |
| `residual` | 128×2560 | 64 | 8.2 |
| `silu_mul` | 128×9216 | 32 | 39.1 |
| `rope` query | 128×16×64 | 8 | 14.7 |
| `rope` key | 128×4×64 | 8 | 8.4 |
| `head_norm` query | 128×16×256 | 8 | 14.7 |
| `head_norm` key | 128×4×256 | 8 | 9.0 |
| `gated_norm` | 128×32×128 | 24 | 16.4 |
| `store_kv` | 128×4×256 | 16 | 8.2 |

The isolated sum is 65783.7 µs, 1945.77 tok/s. That is the kernel ceiling. Prefill attention moves 33673728 bytes in 939.1 µs, 35.86 GB/s.
