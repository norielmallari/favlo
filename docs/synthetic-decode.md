# Synthetic decode

Measurement of one-token dispatches on an M5 Pro with 48 GB. The shape is the Qwen3.8-27B gate, N 17408 by K 5120. It is a synthetic dispatch, not a pipeline forward. The 2048-row tile of the same gate is [synthetic-matmul2d.md](synthetic-matmul2d.md). `half` in the tables is binary16.

## Remainder

On an M5 Pro with 48 GB, the scaled fp4 shader GEMV on this gate measures 0.241 ms (183.64 GiB/s) and the scaled fp8 E4M3 shader GEMV at 160 threads measures 0.403 ms (212.62 GiB/s). The smallest `matmul2d` descriptor M that compiles on the same scaled half × fp4 operands is 8, tile 8×256, and that dispatch measures 1.079 ms (41.41 GiB/s). A search on 25 October 2026 did not find that table.

## Predecessors

- Method. Apple, [WWDC26 session 330](https://developer.apple.com/videos/play/wwdc2026/330/), describes the neural accelerator in each shader core for dense prefill and treats token generation as bandwidth-bound. TensorOps dequantizes int4 and int8 on macOS 26, and fp4, fp8, and int2 on macOS 27, with an E8M0 scale plane on a block of 32 along K. The `matmul2d` header on this machine rejects a descriptor M that is not a multiple of 8 or 16.
- Measurement. [synthetic-matmul2d.md](synthetic-matmul2d.md) reports the 2048-row tile: scaled half × fp4 at 15.084 ms and scaled half × fp8 E4M3 at 14.928 ms. [Tungsten](https://tungsten-lang.org/articles/metal4-matmul2d-on-m5-max) reports `matmul2d` below `simdgroup_matrix` for half at M of 64 on an M5 Max, with K and N of 2048. [Modular `fp4_gemv`](https://docs.modular.com/api/mojo/linalg/matmul/gpu/apple/fp4_gemv/) is an M of 1 kernel whose scales are FP8 E4M3 every 16 elements. [Cider](https://github.com/Mininglamp-AI/cider/blob/main/tutorial/how_to_write_efficient_int_gemm_m5_en.md) reports int8 × int8 at 0.239 ms for M of 1 and 0.241 ms for M of 16, both at K and N of 4096, on an M5 Pro. Those figures are other shapes, and Modular's scale plane is a different layout.
- Mechanism. A GEMV row is a shader that owns four output rows in one simdgroup. A `matmul2d` row is the TensorOps op. An M of 1 in that table is an extent of 1 on the 64×64 descriptor. A descriptor of M equal to 1 does not compile.

## Rule

One command buffer, then a wait. The sample is `GPUEndTime` minus `GPUStartTime`, the median of five dispatches after one discarded sample. Payloads are constant finite codes, so the dot of one row is 5120. Every product row matched that value within 0.5. An E8M0 plane, when present, is scale byte 127 on every block of 32 along K. Codes are half 1, bfloat bits `0x3F80`, fp4 byte `0x22`, fp8 E4M3 byte `0x38`, fp8 E5M2 byte `0x3C`, int8 byte 1, int4 byte `0x11`, and int2 byte `0x55`.

The rate divides the bytes moved by the median seconds and by 2^30. A `matmul2d` row moves `rows × K` of the left type, `N × K` of the right type, `rows × N` of the destination, `N × (K / 32)` scale bytes when the plane is on the right, and another `rows × (K / 32)` when both sides carry it. A GEMV row moves one left row, the full right-hand image, one destination value per column, and the right-hand scale plane when the table says yes. A copy row moves the source and the destination. The 64×64 and 32×128 descriptors use four simdgroups. The 8×256 descriptor uses two. GEMV groups are `N / 4`. The fp4, int4, and int2 GEMVs use 32 threads. The fp8 GEMV is timed at 160 and at 32. The half, bfloat, and int8 GEMVs use 160. The GEMV left side is float, except the int8 × int8 row. The `matmul2d` left side is the type in the table. `transpose_right` is set, and K is dynamic.

## Command

`uv run python bench/synthetic_decode.py`

The command takes the smoke lock for pipeline `shared` and prints the machine line and these tables.

## Table

| Process | Kernel | M | Tile | Left | Right | Destination | E8M0 | ms | GiB/s |
|---|---|---:|---|---|---|---|---|---:|---:|
| matmul2d | mm64_h_fp4 | 64 | 64×64 | half | fp4 E2M1 | float | yes | 2.376 | 20.56 |
| matmul2d | mm64_h_fp4 | 1 | 64×64 | half | fp4 E2M1 | float | yes | 1.083 | 40.77 |
| matmul2d | mm64_h_fp4_raw | 64 | 64×64 | half | fp4 E2M1 | float | no | 0.759 | 60.94 |
| matmul2d | mm64_h_fp4_h | 64 | 64×64 | half | fp4 E2M1 | half | yes | 0.861 | 54.35 |
| matmul2d | mm64_h_e4 | 64 | 64×64 | half | fp8 E4M3 | float | yes | 0.790 | 114.42 |
| matmul2d | mm64_h_e4 | 1 | 64×64 | half | fp8 E4M3 | float | yes | 0.682 | 125.62 |
| matmul2d | mm64_h_e4_raw | 64 | 64×64 | half | fp8 E4M3 | float | no | 0.628 | 139.81 |
| matmul2d | mm64_h_e5 | 64 | 64×64 | half | fp8 E5M2 | float | yes | 0.696 | 129.74 |
| matmul2d | mm64_h_h | 64 | 64×64 | half | half | float | no | 0.948 | 180.21 |
| matmul2d | mm64_h_h | 1 | 64×64 | half | half | float | no | 1.204 | 137.93 |
| matmul2d | mm64_bf_bf | 64 | 64×64 | bfloat | bfloat | float | no | 1.277 | 133.71 |
| matmul2d | mm64_i8 | 64 | 64×64 | int8 | int8 | int32 | no | 0.882 | 99.14 |
| matmul2d | mm64_i8 | 1 | 64×64 | int8 | int8 | int32 | no | 0.741 | 112.05 |
| matmul2d | mm64_fp4_fp4 | 64 | 64×64 | fp4 E2M1 | fp4 E2M1 | float | both | 0.748 | 64.72 |
| matmul2d | mm64_e4_e4 | 64 | 64×64 | fp8 E4M3 | fp8 E4M3 | float | both | 0.677 | 133.05 |
| matmul2d | mm64_h_i4 | 64 | 64×64 | half | int4 | float | yes | 0.585 | 83.58 |
| matmul2d | mm64_h_i2 | 64 | 64×64 | half | int2 | float | yes | 0.511 | 55.02 |
| matmul2d | mm32_h_fp4 | 32 | 32×128 | half | fp4 E2M1 | float | yes | 0.441 | 105.28 |
| matmul2d | mm32_h_e4 | 32 | 32×128 | half | fp8 E4M3 | float | yes | 0.584 | 150.69 |
| matmul2d | mm32_i8 | 32 | 32×128 | int8 | int8 | int32 | no | 0.500 | 170.46 |
| matmul2d | mm8_h_fp4 | 8 | 8×256 | half | fp4 E2M1 | float | yes | 1.079 | 41.41 |
| matmul2d | mm8_h_e4 | 8 | 8×256 | half | fp8 E4M3 | float | yes | 0.718 | 120.09 |
| matmul2d | mm8_i8 | 8 | 8×256 | int8 | int8 | int32 | no | 0.497 | 168.10 |
| gemv | gemv_fp4_x4 | 1 | 32 thr | float | fp4 E2M1 | float | yes | 0.241 | 183.64 |
| gemv | gemv_fp4_x4 | 1 | 32 thr | float | fp4 E2M1 | float | no | 0.229 | 181.87 |
| gemv | gemv_fp8_x4 | 1 | 160 thr | float | fp8 E4M3 | float | yes | 0.403 | 212.62 |
| gemv | gemv_fp8_x4 | 1 | 32 thr | float | fp8 E4M3 | float | yes | 0.414 | 206.87 |
| gemv | gemv_sub4 | 1 | 32 thr | float | int4 | float | yes | 0.252 | 175.18 |
| gemv | gemv_sub4 | 1 | 32 thr | float | int2 | float | yes | 0.203 | 115.70 |
| gemv | gemv_i8f_x4 | 1 | 160 thr | float | int8 | float | no | 0.382 | 217.78 |
| gemv | gemv_i8_x4 | 1 | 160 thr | int8 | int8 | int32 | no | 0.400 | 207.95 |
| gemv | gemv_half_x4 | 1 | 160 thr | float | half | float | no | 1.069 | 155.34 |
| gemv | gemv_bf16_x4 | 1 | 160 thr | float | bfloat | float | no | 1.038 | 160.03 |

| Process | Tile | ms | GiB/s |
|---|---|---:|---:|
| copy | 512 MiB | 4.009 | 249.44 |
| copy | fp4 image | 0.407 | 216.63 |
| copy | int8 image | 0.715 | 232.15 |

The 512 MiB copy is the bandwidth anchor on this machine, 249.44 GiB/s. On scaled fp4 the shader GEMV is 0.241 ms, the 32×128 `matmul2d` is 0.441 ms, the 8×256 `matmul2d` is 1.079 ms, and the 64×64 `matmul2d` is 2.376 ms. On scaled fp8 E4M3 the 160-thread GEMV is 0.403 ms and the 32×128 `matmul2d` is 0.584 ms. The int2 GEMV is the shortest GEMV wall time, 0.203 ms, on the smallest image, at 115.70 GiB/s.

The E8M0 plane on the 64×64 half × fp4 `matmul2d` is the difference between 2.376 ms and 0.759 ms. On the fp4 GEMV the same plane is the difference between 0.241 ms and 0.229 ms. Half × half at 64 rows is 0.948 ms. The half GEMV is 1.069 ms, and the half extent of 1 on the 64×64 descriptor is 1.204 ms. Half × int4 and half × int2 `matmul2d` completed, at 0.585 ms and 0.511 ms. The int4 and int2 GEMVs are 0.252 ms and 0.203 ms.

## Search

25 October 2026. Queries: "Metal matmul2d M multiple of 8 GEMV fp4 17408 5120 Apple M5 decode 2026" and "\"0.241\" fp4 GEMV M5 Pro 17408 5120 matmul2d". The predecessors in this note are the overlaps those queries returned. Cider's 0.241 ms is an int8 tile at K and N of 4096. The search did not find the 0.241 ms fp4 GEMV row, the 0.403 ms fp8 GEMV row, or the 1.079 ms 8×256 half × fp4 row on this gate.

## Corrections

Later finds are appended here. The sections above stay as committed.
