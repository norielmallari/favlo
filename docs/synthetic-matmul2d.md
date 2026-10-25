# Synthetic matmul2d

Measurement of one `mpp::tensor_ops::matmul2d` tile on an M5 Pro with 48 GB. The shape is the Qwen3.8-27B gate, 2048×17408×5120. It is a synthetic dispatch, not a pipeline forward. `half` in the table is binary16. `int8 × int8 → int32` is the product the pipelines call `matmul_i8` or `gemm_i8`.

## Remainder

On an M5 Pro with 48 GB, scaled half × fp4 on this tile measures 15.084 ms (24.20 tera-ops/s) and scaled half × fp8 E4M3 measures 14.928 ms (24.46 tera-ops/s). int8 × int8 on the same tile measures 6.398 ms (57.06 tera-ops/s). A search on 25 October 2026 did not find that table.

## Predecessors

- Method. Apple, [WWDC26 session 330](https://developer.apple.com/videos/play/wwdc2026/330/), specifies `matmul2d` and names int4, int8, fp4, fp8, and int2 as tensor types, with an E8M0 scale plane. The Metal Shading Language Specification, tables 7.3 and 7.4, lists the `char`, `half`, `float`, and `bfloat` products. `char × char → int` is the int8 × int8 → int32 product.
- Measurement. [mtlattn](https://github.com/lastowl/mtlattn/blob/main/docs/PERFORMANCE.md) reports 32.3, 32.2, and 61.3 tera-ops/s for half, bfloat, and int8 on a 64×64×64 tile. [Tungsten](https://tungsten-lang.org/articles/metal4-matmul2d-on-m5-max) reports 18.9 tera-ops/s for half at 1024×2048×2048. [Modular](https://docs.modular.com/api/mojo/linalg/matmul/gpu/apple/fp4_matmul/enqueue_apple_fp4_matmul/) reports about 42 tera-ops/s for a weight-only fp4 `matmul2d` at K of at least 18432. Those figures are other shapes.
- Mechanism. The op dequantizes a quantized right-hand tensor. The rows below pass that tensor straight in. A missing scale plane is a different dispatch from the E8M0 plane the packed records store.

## Rule

One command buffer, then a wait. The sample is the median of five dispatches after one discarded sample. The descriptor is a 64×64 output tile, dynamic K, `transpose_right`, and four simdgroups. The rate is `2 × 2048 × 17408 × 5120` divided by the median seconds. Payloads are constant finite codes. An E8M0 plane, when present, is scale byte 127 on every block of 32 along K. The same-type rows use that plane on both sides.

The compiler accepts half on the left against fp4 E2M1, fp8 E4M3, or fp8 E5M2, with a float or half destination, with or without the scale plane. It also accepts fp4 × fp4, fp8 E4M3 × fp8 E4M3, and fp8 E5M2 × fp8 E5M2 into float when both sides carry the scale plane. It rejects the other pairings tried here with `Unsupported type`: bfloat, float, or int8 on the left against those quantized types; a quantized left side against half, bfloat, float, or int8; and a mixed fp4/fp8 pair. int4 was not dispatched.

## Command

No committed command regenerates these medians. A pipeline `kernels` check is a different tile.

## Table

| Left | Right | Destination | E8M0 | ms | tera-ops/s |
|---|---|---|---|---:|---:|
| int8 | int8 | int32 | no | 6.398 | 57.06 |
| half | fp8 E4M3 | float | no | 11.549 | 31.61 |
| half | fp8 E5M2 | float | no | 11.560 | 31.58 |
| half | fp4 E2M1 | float | no | 11.855 | 30.80 |
| half | half | float | no | 12.221 | 29.87 |
| bfloat | bfloat | float | no | 12.284 | 29.72 |
| half | fp8 E5M2 | float | yes | 14.830 | 24.62 |
| half | fp8 E4M3 | half | yes | 14.826 | 24.62 |
| half | fp8 E5M2 | half | yes | 14.826 | 24.62 |
| half | fp8 E4M3 | float | yes | 14.928 | 24.46 |
| half | fp4 E2M1 | float | yes | 15.084 | 24.20 |
| half | fp4 E2M1 | half | yes | 15.344 | 23.79 |
| fp8 E5M2 | fp8 E5M2 | float | yes | 17.808 | 20.50 |
| fp8 E4M3 | fp8 E4M3 | float | yes | 17.822 | 20.48 |
| fp4 E2M1 | fp4 E2M1 | float | yes | 20.632 | 17.69 |

The packed records store the E8M0 plane. On this tile that plane is the difference between the half × fp4 row at 30.80 and the row at 24.20, and between the half × fp8 E4M3 row at 31.61 and the row at 24.46. A half destination on the same operands stays on that scaled rate.

## Search

25 October 2026. Queries: "Metal matmul2d half fp4 15.084 ms 2048 17408 5120 24.2 TFLOP" and "matmul2d fp4 fp8 int8 TFLOPS 64x64 tile 2026". The predecessors in this note are the overlaps those queries returned. The search did not find the 15.084 ms, 14.928 ms, or 6.398 ms rows.

## Corrections

Later finds are appended here. The sections above stay as committed.
