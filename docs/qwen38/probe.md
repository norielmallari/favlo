# Qwen3.8 probe

The block-of-32 measurement on this M5 Pro, Metal 4.1. `madmetal --qwen38-flash-next probe` writes one block and reads it back on the GPU. It does not pack an expert. Scope is in [ARCHITECTURE.md](../../ARCHITECTURE.md). If this note disagrees with that file, ARCHITECTURE.md wins.

Host `replaceSlice` and `getBytes` copy packed bytes: 32 int4 values are 16 bytes, 32 int2 values are 8 bytes, and int8 / fp8 values are one byte each.

int4 element 0 is the low nibble. Shader `unpack` of `int4b` is signed two's complement, −8..7. Code 7 is 7, code 8 is −8, code 15 is −1. An E8M0 scale of 130 reads back as 130, and `code × 2^(e − 127)` matches the shader (130 is ×8).

`madmetal --qwen38-flash-next matmul` runs one 64×32×32 fp4 tile through `matmul2d` and the same rows through the decode shader dot. The activation is 1 on the first inner column, weight row `n` stores the E2M1 magnitude `(n mod 8)` there (`0, 0.5, 1, 1.5, 2, 3, 4, 6`), and the E8M0 byte is 130. Both kernels return that magnitude times 8. A second prefill, 128×64×64, covers two row tiles and two column tiles and matches the reconstructed fp4 product. An fp8 tile stores E4M3 code `0x38` (1.0) in that column with scale byte 127 and returns 1. The column tile walks the weight's outer extent. Offsetting that slice along K leaves every output past the first 32 columns wrong, and a single 32-column tile does not catch it.

fp8 E4M3 matches OCP E4M3FN, including `0x7E = 448` and `0x7F` NaN. fp8 E5M2 matches OCP E5M2. int8 bytes round-trip, including −128 and 127.

int2 bytes round-trip. Metal 4.1 `unpack` does not accept `int2b`, and this SDK's compute encoder has no tensor argument, so int2 element order and its signed endpoints are not known yet. The shader-dot layout is low two bits first.
