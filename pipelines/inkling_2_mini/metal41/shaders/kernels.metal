#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>

using namespace metal;
using namespace mpp::tensor_ops;

using scales_plane = tensor_blockwise<tensor_plane_scales,
                                      device metal_fp8_ue8m0_format,
                                      32, 1>;

constant float E2M1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
constant float E2M1_SIGNED[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

// OCP E4M3FN. 0x7F and 0xFF are NaN.
constant float E4M3_TABLE[256] = {
    0.0f, 0.001953125f, 0.00390625f, 0.005859375f, 0.0078125f, 0.009765625f, 0.01171875f, 0.013671875f,
    0.015625f, 0.017578125f, 0.01953125f, 0.021484375f, 0.0234375f, 0.025390625f, 0.02734375f, 0.029296875f,
    0.03125f, 0.03515625f, 0.0390625f, 0.04296875f, 0.046875f, 0.05078125f, 0.0546875f, 0.05859375f,
    0.0625f, 0.0703125f, 0.078125f, 0.0859375f, 0.09375f, 0.1015625f, 0.109375f, 0.1171875f,
    0.125f, 0.140625f, 0.15625f, 0.171875f, 0.1875f, 0.203125f, 0.21875f, 0.234375f,
    0.25f, 0.28125f, 0.3125f, 0.34375f, 0.375f, 0.40625f, 0.4375f, 0.46875f,
    0.5f, 0.5625f, 0.625f, 0.6875f, 0.75f, 0.8125f, 0.875f, 0.9375f,
    1.0f, 1.125f, 1.25f, 1.375f, 1.5f, 1.625f, 1.75f, 1.875f,
    2.0f, 2.25f, 2.5f, 2.75f, 3.0f, 3.25f, 3.5f, 3.75f,
    4.0f, 4.5f, 5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.5f,
    8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
    16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f,
    32.0f, 36.0f, 40.0f, 44.0f, 48.0f, 52.0f, 56.0f, 60.0f,
    64.0f, 72.0f, 80.0f, 88.0f, 96.0f, 104.0f, 112.0f, 120.0f,
    128.0f, 144.0f, 160.0f, 176.0f, 192.0f, 208.0f, 224.0f, 240.0f,
    256.0f, 288.0f, 320.0f, 352.0f, 384.0f, 416.0f, 448.0f, NAN,
    -0.0f, -0.001953125f, -0.00390625f, -0.005859375f, -0.0078125f, -0.009765625f, -0.01171875f, -0.013671875f,
    -0.015625f, -0.017578125f, -0.01953125f, -0.021484375f, -0.0234375f, -0.025390625f, -0.02734375f, -0.029296875f,
    -0.03125f, -0.03515625f, -0.0390625f, -0.04296875f, -0.046875f, -0.05078125f, -0.0546875f, -0.05859375f,
    -0.0625f, -0.0703125f, -0.078125f, -0.0859375f, -0.09375f, -0.1015625f, -0.109375f, -0.1171875f,
    -0.125f, -0.140625f, -0.15625f, -0.171875f, -0.1875f, -0.203125f, -0.21875f, -0.234375f,
    -0.25f, -0.28125f, -0.3125f, -0.34375f, -0.375f, -0.40625f, -0.4375f, -0.46875f,
    -0.5f, -0.5625f, -0.625f, -0.6875f, -0.75f, -0.8125f, -0.875f, -0.9375f,
    -1.0f, -1.125f, -1.25f, -1.375f, -1.5f, -1.625f, -1.75f, -1.875f,
    -2.0f, -2.25f, -2.5f, -2.75f, -3.0f, -3.25f, -3.5f, -3.75f,
    -4.0f, -4.5f, -5.0f, -5.5f, -6.0f, -6.5f, -7.0f, -7.5f,
    -8.0f, -9.0f, -10.0f, -11.0f, -12.0f, -13.0f, -14.0f, -15.0f,
    -16.0f, -18.0f, -20.0f, -22.0f, -24.0f, -26.0f, -28.0f, -30.0f,
    -32.0f, -36.0f, -40.0f, -44.0f, -48.0f, -52.0f, -56.0f, -60.0f,
    -64.0f, -72.0f, -80.0f, -88.0f, -96.0f, -104.0f, -112.0f, -120.0f,
    -128.0f, -144.0f, -160.0f, -176.0f, -192.0f, -208.0f, -224.0f, -240.0f,
    -256.0f, -288.0f, -320.0f, -352.0f, -384.0f, -416.0f, -448.0f, NAN
};

constant float GATE_CAP = 4.0f;
constant float UP_CAP = 25.0f;
constant float QK_SCALE = 0.08838834764831845f;
constant float INV_SQRT5 = 0.4472135955f;
constant float CLIP_EPS = 1.0e-6f;

static inline float e8m0_scale(uint scale) {
    uint bits = select(scale << 23, 1u << 22, scale == 0u);
    return as_type<float>(bits);
}

static inline float sigmoid_f(float x) {
    if (x > 20.0f) {
        return 1.0f;
    }
    if (x < -20.0f) {
        return 0.0f;
    }
    return 1.0f / (1.0f + exp(-x));
}

// The sigmoid sees the raw gate. The caps bound the linear gate factor and up.
static inline float swiglu_f(float gate, float up) {
    return GATE_CAP * tanh(gate / GATE_CAP) * sigmoid_f(gate) * UP_CAP * tanh(up / UP_CAP);
}

// tau = 1 + 0.1 log(max((pos + 1) / 8192, 1)) on full-attention layers.
static inline float log_tau(uint pos, uint full) {
    if (full == 0u || pos + 1u <= 8192u) {
        return 1.0f;
    }
    return 1.0f + 0.1f * log(float(pos + 1u) / 8192.0f);
}

// ---------------------------------------------------------------------------
// matmul2d tiles. The right operand is [cols, inner] and transposed.

kernel void matmul_f16_m64(
    device half* activation [[buffer(0)]],
    device half* weights [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    tensor<device half, dextents<int, 2>, tensor_inline> left(
        activation, dextents<int, 2>(int(inner), int(rows)), array<int, 2>{1, int(inner)});
    tensor<device half, dextents<int, 2>, tensor_inline> right(
        weights, dextents<int, 2>(int(inner), int(cols)), array<int, 2>{1, int(inner)});
    tensor<device float, dextents<int, 2>, tensor_inline> out(
        destination, dextents<int, 2>(int(cols), int(rows)), array<int, 2>{1, int(cols)});
    constexpr auto desc = matmul2d_descriptor(
        64, 64, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto tile_left = left.slice(0, int(tgid.y) * 64);
    auto tile_right = right.slice(0, int(tgid.x) * 64);
    auto tile_out = out.slice(int(tgid.x) * 64, int(tgid.y) * 64);
    op.run(tile_left, tile_right, tile_out);
}

// Half matmul of one row band. The destination row stride can be wider than the band.
kernel void matmul_f16_band(
    device half* activation [[buffer(0)]],
    device half* weights [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    constant uint& out_stride [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    tensor<device half, dextents<int, 2>, tensor_inline> left(
        activation, dextents<int, 2>(int(inner), int(rows)), array<int, 2>{1, int(inner)});
    tensor<device half, dextents<int, 2>, tensor_inline> right(
        weights, dextents<int, 2>(int(inner), int(cols)), array<int, 2>{1, int(inner)});
    tensor<device float, dextents<int, 2>, tensor_inline> out(
        destination, dextents<int, 2>(int(cols), int(rows)), array<int, 2>{1, int(out_stride)});
    constexpr auto desc = matmul2d_descriptor(
        64, 64, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto tile_left = left.slice(0, int(tgid.y) * 64);
    auto tile_right = right.slice(0, int(tgid.x) * 64);
    auto tile_out = out.slice(int(tgid.x) * 64, int(tgid.y) * 64);
    op.run(tile_left, tile_right, tile_out);
}

// Scaled fp4 on the 32x128 tile. Output rows are out_stride floats apart.
kernel void matmul_fp4_m32(
    device half* activation [[buffer(0)]],
    device uchar* weights [[buffer(1)]],
    device uchar* scales [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    constant uint& inner [[buffer(6)]],
    constant uint& out_stride [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    tensor<device half, dextents<int, 2>, tensor_inline> left(
        activation, dextents<int, 2>(int(inner), int(rows)), array<int, 2>{1, int(inner)});
    tensor<device metal_fp4_e2m1_format, dextents<int, 2>, tensor_inline, scales_plane> right(
        weights,
        dextents<int, 2>(int(inner), int(cols)),
        array<int, 2>{1, int(inner)},
        scales_plane(scales));
    tensor<device float, dextents<int, 2>, tensor_inline> out(
        destination, dextents<int, 2>(int(cols), int(rows)), array<int, 2>{1, int(out_stride)});
    constexpr auto desc = matmul2d_descriptor(
        32, 128, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto tile_left = left.slice(0, int(tgid.y) * 32);
    auto tile_right = right.slice(0, int(tgid.x) * 128);
    auto tile_out = out.slice(int(tgid.x) * 128, int(tgid.y) * 32);
    op.run(tile_left, tile_right, tile_out);
}

// One 32-row tile of one expert. jobs holds (slot, tile). Empty jobs return.
kernel void expert_fp4_tile(
    device half* pack [[buffer(0)]],
    device uchar* weights [[buffer(1)]],
    device uchar* scales [[buffer(2)]],
    device float* destination [[buffer(3)]],
    device const uint* jobs [[buffer(4)]],
    device const uint* expert_of [[buffer(5)]],
    device const uint* row_base [[buffer(6)]],
    device const uint* info [[buffer(7)]],
    constant uint& inner [[buffer(8)]],
    constant uint& cols [[buffer(9)]],
    constant uint& data_stride [[buffer(10)]],
    constant uint& scale_stride [[buffer(11)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    if (tgid.y >= info[1]) {
        return;
    }
    uint slot = jobs[tgid.y * 2u];
    uint row0 = jobs[tgid.y * 2u + 1u] * 32u;
    uint expert = expert_of[slot];
    uint col0 = tgid.x * 128u;
    uint base_row = row_base[slot] + row0;
    tensor<device half, dextents<int, 2>, tensor_inline> left(
        pack + (ulong)base_row * inner, dextents<int, 2>(int(inner), 32), array<int, 2>{1, int(inner)});
    tensor<device metal_fp4_e2m1_format, dextents<int, 2>, tensor_inline, scales_plane> right(
        weights + (ulong)expert * data_stride,
        dextents<int, 2>(int(inner), int(cols)),
        array<int, 2>{1, int(inner)},
        scales_plane(scales + (ulong)expert * scale_stride));
    tensor<device float, dextents<int, 2>, tensor_inline> out(
        destination + (ulong)base_row * cols, dextents<int, 2>(int(cols), 32), array<int, 2>{1, int(cols)});
    constexpr auto desc = matmul2d_descriptor(
        32, 128, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto tile_left = left.slice(0, 0);
    auto tile_right = right.slice(0, int(col0));
    auto tile_out = out.slice(int(col0), 0);
    op.run(tile_left, tile_right, tile_out);
}

kernel void expert_fp8_tile(
    device half* pack [[buffer(0)]],
    device uchar* weights [[buffer(1)]],
    device uchar* scales [[buffer(2)]],
    device float* destination [[buffer(3)]],
    device const uint* jobs [[buffer(4)]],
    device const uint* expert_of [[buffer(5)]],
    device const uint* row_base [[buffer(6)]],
    device const uint* info [[buffer(7)]],
    constant uint& inner [[buffer(8)]],
    constant uint& cols [[buffer(9)]],
    constant uint& data_stride [[buffer(10)]],
    constant uint& scale_stride [[buffer(11)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    if (tgid.y >= info[1]) {
        return;
    }
    uint slot = jobs[tgid.y * 2u];
    uint row0 = jobs[tgid.y * 2u + 1u] * 32u;
    uint expert = expert_of[slot];
    uint col0 = tgid.x * 128u;
    uint base_row = row_base[slot] + row0;
    tensor<device half, dextents<int, 2>, tensor_inline> left(
        pack + (ulong)base_row * inner, dextents<int, 2>(int(inner), 32), array<int, 2>{1, int(inner)});
    tensor<device metal_fp8_e4m3_format, dextents<int, 2>, tensor_inline, scales_plane> right(
        weights + (ulong)expert * data_stride,
        dextents<int, 2>(int(inner), int(cols)),
        array<int, 2>{1, int(inner)},
        scales_plane(scales + (ulong)expert * scale_stride));
    tensor<device float, dextents<int, 2>, tensor_inline> out(
        destination + (ulong)base_row * cols, dextents<int, 2>(int(cols), 32), array<int, 2>{1, int(cols)});
    constexpr auto desc = matmul2d_descriptor(
        32, 128, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto tile_left = left.slice(0, 0);
    auto tile_right = right.slice(0, int(col0));
    auto tile_out = out.slice(int(col0), 0);
    op.run(tile_left, tile_right, tile_out);
}

// ---------------------------------------------------------------------------
// Single-token GEMVs. Activation is float. One simdgroup owns four rows.

static inline float fp4_word(uint w, float4 a0, float4 a1) {
    float s = E2M1_SIGNED[w & 15u] * a0.x;
    s = fma(E2M1_SIGNED[(w >> 4u) & 15u], a0.y, s);
    s = fma(E2M1_SIGNED[(w >> 8u) & 15u], a0.z, s);
    s = fma(E2M1_SIGNED[(w >> 12u) & 15u], a0.w, s);
    s = fma(E2M1_SIGNED[(w >> 16u) & 15u], a1.x, s);
    s = fma(E2M1_SIGNED[(w >> 20u) & 15u], a1.y, s);
    s = fma(E2M1_SIGNED[(w >> 24u) & 15u], a1.z, s);
    s = fma(E2M1_SIGNED[(w >> 28u) & 15u], a1.w, s);
    return s;
}

static inline float fp8_word(uint w, float4 a) {
    float s = E4M3_TABLE[w & 255u] * a.x;
    s = fma(E4M3_TABLE[(w >> 8u) & 255u], a.y, s);
    s = fma(E4M3_TABLE[(w >> 16u) & 255u], a.z, s);
    s = fma(E4M3_TABLE[w >> 24u], a.w, s);
    return s;
}

// Partial sums of four rows, no reduction. words is the logical inner / 8.
// row_bytes and row_blocks describe the stored row.
static inline float4 fp4_part4(
    device const float* act,
    device const uchar* weights,
    device const uchar* scales,
    uint row0,
    uint words,
    uint row_bytes,
    uint row_blocks,
    uint lane)
{
    device const uint* p0 = (device const uint*)(weights + (ulong)row0 * row_bytes);
    device const uint* p1 = (device const uint*)(weights + (ulong)(row0 + 1u) * row_bytes);
    device const uint* p2 = (device const uint*)(weights + (ulong)(row0 + 2u) * row_bytes);
    device const uint* p3 = (device const uint*)(weights + (ulong)(row0 + 3u) * row_bytes);
    device const uchar* s0 = scales + (ulong)row0 * row_blocks;
    device const uchar* s1 = s0 + row_blocks;
    device const uchar* s2 = s1 + row_blocks;
    device const uchar* s3 = s2 + row_blocks;
    float4 acc = float4(0.0f);
    for (uint i = lane; i < words; i += 32u) {
        uint w0 = p0[i];
        uint w1 = p1[i];
        uint w2 = p2[i];
        uint w3 = p3[i];
        uint b = i >> 2u;
        float f0 = e8m0_scale(s0[b]);
        float f1 = e8m0_scale(s1[b]);
        float f2 = e8m0_scale(s2[b]);
        float f3 = e8m0_scale(s3[b]);
        float4 a0 = *(device const float4*)(act + (i << 3u));
        float4 a1 = *(device const float4*)(act + (i << 3u) + 4u);
        acc.x = fma(f0, fp4_word(w0, a0, a1), acc.x);
        acc.y = fma(f1, fp4_word(w1, a0, a1), acc.y);
        acc.z = fma(f2, fp4_word(w2, a0, a1), acc.z);
        acc.w = fma(f3, fp4_word(w3, a0, a1), acc.w);
    }
    return acc;
}

// vecs is the logical inner / 16. row_bytes is the stored row.
static inline float4 fp8_part4(
    device const float* act,
    device const uchar* weights,
    device const uchar* scales,
    uint row0,
    uint vecs,
    uint row_bytes,
    uint row_blocks,
    uint lane)
{
    device const uint4* p0 = (device const uint4*)(weights + (ulong)row0 * row_bytes);
    device const uint4* p1 = (device const uint4*)(weights + (ulong)(row0 + 1u) * row_bytes);
    device const uint4* p2 = (device const uint4*)(weights + (ulong)(row0 + 2u) * row_bytes);
    device const uint4* p3 = (device const uint4*)(weights + (ulong)(row0 + 3u) * row_bytes);
    device const uchar* s0 = scales + (ulong)row0 * row_blocks;
    device const uchar* s1 = s0 + row_blocks;
    device const uchar* s2 = s1 + row_blocks;
    device const uchar* s3 = s2 + row_blocks;
    float4 acc = float4(0.0f);
    for (uint i = lane; i < vecs; i += 32u) {
        uint4 w0 = p0[i];
        uint4 w1 = p1[i];
        uint4 w2 = p2[i];
        uint4 w3 = p3[i];
        uint b = i >> 1u;
        float f0 = e8m0_scale(s0[b]);
        float f1 = e8m0_scale(s1[b]);
        float f2 = e8m0_scale(s2[b]);
        float f3 = e8m0_scale(s3[b]);
        device const float4* a = (device const float4*)(act + (i << 4u));
        float4 a0 = a[0];
        float4 a1 = a[1];
        float4 a2 = a[2];
        float4 a3 = a[3];
        acc.x = fma(f0, fp8_word(w0.x, a0) + fp8_word(w0.y, a1) + fp8_word(w0.z, a2) + fp8_word(w0.w, a3), acc.x);
        acc.y = fma(f1, fp8_word(w1.x, a0) + fp8_word(w1.y, a1) + fp8_word(w1.z, a2) + fp8_word(w1.w, a3), acc.y);
        acc.z = fma(f2, fp8_word(w2.x, a0) + fp8_word(w2.y, a1) + fp8_word(w2.z, a2) + fp8_word(w2.w, a3), acc.z);
        acc.w = fma(f3, fp8_word(w3.x, a0) + fp8_word(w3.y, a1) + fp8_word(w3.z, a2) + fp8_word(w3.w, a3), acc.w);
    }
    return acc;
}

static inline float4 simd_sum4(float4 v) {
    return float4(simd_sum(v.x), simd_sum(v.y), simd_sum(v.z), simd_sum(v.w));
}

// cols is a multiple of 4. inner is a multiple of 32.
kernel void gemv_fp4(
    device const float* activation [[buffer(0)]],
    device const uchar* weights [[buffer(1)]],
    device const uchar* scales [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint group [[threadgroup_position_in_grid]])
{
    uint row0 = group * 4u;
    if (row0 >= cols) {
        return;
    }
    float4 acc = simd_sum4(fp4_part4(activation, weights, scales, row0, inner >> 3u, inner >> 1u, inner >> 5u, lane));
    if (lane < 4u) {
        destination[row0 + lane] = acc[lane];
    }
}

// The head. Rows at or past valid are -inf. The rest are cap * tanh(x / cap).
kernel void gemv_fp4_logits(
    device const float* activation [[buffer(0)]],
    device const uchar* weights [[buffer(1)]],
    device const uchar* scales [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    constant uint& valid [[buffer(6)]],
    constant float& cap [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint group [[threadgroup_position_in_grid]])
{
    uint row0 = group * 4u;
    if (row0 >= cols) {
        return;
    }
    float4 acc = simd_sum4(fp4_part4(activation, weights, scales, row0, inner >> 3u, inner >> 1u, inner >> 5u, lane));
    if (lane < 4u) {
        uint row = row0 + lane;
        destination[row] = row < valid ? cap * tanh(acc[lane] / cap) : -INFINITY;
    }
}

static inline float4 f16_part4(device const float* act, device const half* weights, uint row0, uint inner, uint lane) {
    device const half4* p0 = (device const half4*)(weights + (ulong)row0 * inner);
    device const half4* p1 = (device const half4*)(weights + (ulong)(row0 + 1u) * inner);
    device const half4* p2 = (device const half4*)(weights + (ulong)(row0 + 2u) * inner);
    device const half4* p3 = (device const half4*)(weights + (ulong)(row0 + 3u) * inner);
    device const float4* a = (device const float4*)act;
    float4 acc = float4(0.0f);
    for (uint i = lane; i < (inner >> 2u); i += 32u) {
        float4 x = a[i];
        acc.x += dot(float4(p0[i]), x);
        acc.y += dot(float4(p1[i]), x);
        acc.z += dot(float4(p2[i]), x);
        acc.w += dot(float4(p3[i]), x);
    }
    return acc;
}

// The head, half weights. Rows at or past valid are -inf. The rest are cap * tanh(x / cap).
kernel void gemv_f16_logits(
    device const float* activation [[buffer(0)]],
    device const half* weights [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& inner [[buffer(4)]],
    constant uint& valid [[buffer(5)]],
    constant float& cap [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint group [[threadgroup_position_in_grid]])
{
    uint row0 = group * 4u;
    if (row0 >= cols) {
        return;
    }
    float4 acc = simd_sum4(f16_part4(activation, weights, row0, inner, lane));
    if (lane < 4u) {
        uint row = row0 + lane;
        destination[row] = row < valid ? cap * tanh(acc[lane] / cap) : -INFINITY;
    }
}

// Residual short convolution of one channel, then the state shift.
// y = (x + w3 x + w2 x[t-1] + w1 x[t-2] + w0 x[t-3]) / sqrt(5).
static inline float conv_step(float x, device const float* w, device float* state, uint c, uint channels) {
    float older = state[c];
    float middle = state[channels + c];
    float newer = state[2u * channels + c];
    float4 wt = *(device const float4*)(w + c * 4u);
    float y = (x + wt.w * x + wt.z * newer + wt.y * middle + wt.x * older) * INV_SQRT5;
    state[c] = middle;
    state[channels + c] = newer;
    state[2u * channels + c] = x;
    return y;
}

// Output projection, its short convolution, and the add into the block sum.
kernel void gemv_fp4_conv_add(
    device const float* activation [[buffer(0)]],
    device const uchar* weights [[buffer(1)]],
    device const uchar* scales [[buffer(2)]],
    device const float* conv_w [[buffer(3)]],
    device float* state [[buffer(4)]],
    device float* sum [[buffer(5)]],
    constant uint& cols [[buffer(6)]],
    constant uint& inner [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint group [[threadgroup_position_in_grid]])
{
    uint row0 = group * 4u;
    if (row0 >= cols) {
        return;
    }
    float4 acc = simd_sum4(fp4_part4(activation, weights, scales, row0, inner >> 3u, inner >> 1u, inner >> 5u, lane));
    if (lane < 4u) {
        uint c = row0 + lane;
        sum[c] += conv_step(acc[lane], conv_w, state, c, cols);
    }
}

// Output projection in half, its short convolution, and the add into the block sum.
kernel void gemv_f16_conv_add(
    device const float* activation [[buffer(0)]],
    device const half* weights [[buffer(1)]],
    device const float* conv_w [[buffer(2)]],
    device float* state [[buffer(3)]],
    device float* sum [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    constant uint& inner [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint group [[threadgroup_position_in_grid]])
{
    uint row0 = group * 4u;
    if (row0 >= cols) {
        return;
    }
    float4 acc = simd_sum4(f16_part4(activation, weights, row0, inner, lane));
    if (lane < 4u) {
        uint c = row0 + lane;
        sum[c] += conv_step(acc[lane], conv_w, state, c, cols);
    }
}

constant uint ATTN_ROW_K = 6528u;
constant uint ATTN_ROW_V = 7552u;

// Every attention input of one layer in one dispatch. The fused half plane
// [q; rel; gate; k; v] reads q_in, k_in, and v_in by row band. Then half rows:
// the sink (6 groups, q_in), and on full layers the key taps (8 groups, k_in)
// and value taps (8 groups, v_in). Outputs follow the fused rows in order.
kernel void gemv_attn(
    device const float* q_in [[buffer(0)]],
    device const float* k_in [[buffer(1)]],
    device const float* v_in [[buffer(2)]],
    device const half* weights [[buffer(3)]],
    device const half* sink_w [[buffer(4)]],
    device const half* k_tap_w [[buffer(5)]],
    device const half* v_tap_w [[buffer(6)]],
    device float* destination [[buffer(7)]],
    constant uint& attn_rows [[buffer(8)]],
    constant uint& inner [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]],
    uint group [[threadgroup_position_in_grid]])
{
    uint row0 = group * 4u;
    float4 acc;
    if (row0 < attn_rows) {
        device const float* act = row0 < ATTN_ROW_K ? q_in : (row0 < ATTN_ROW_V ? k_in : v_in);
        acc = f16_part4(act, weights, row0, inner, lane);
    } else {
        uint local = row0 - attn_rows;
        if (local < 24u) {
            acc = f16_part4(q_in, sink_w, local, inner, lane);
        } else if (local < 56u) {
            acc = f16_part4(k_in, k_tap_w, local - 24u, inner, lane);
        } else {
            acc = f16_part4(v_in, v_tap_w, local - 56u, inner, lane);
        }
    }
    acc = simd_sum4(acc);
    if (lane < 4u) {
        destination[row0 + lane] = acc[lane];
    }
}

// Router logits and the top-8 pick for one token. 32 simdgroups of four rows;
// the last one to finish picks. Sigmoid scores, the bias only for the pick,
// then the picks renormalized and scaled by 8.
kernel void route_gemv(
    device const float* activation [[buffer(0)]],
    device const half* weights [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device uint* ids [[buffer(3)]],
    device float* route [[buffer(4)]],
    device float* scores [[buffer(5)]],
    device atomic_uint* counter [[buffer(6)]],
    constant uint& inner [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint group [[threadgroup_position_in_grid]])
{
    uint row0 = group * 4u;
    float4 acc = simd_sum4(f16_part4(activation, weights, row0, inner, lane));
    if (lane < 4u) {
        scores[row0 + lane] = sigmoid_f(acc[lane]);
    }
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    uint seen = 0u;
    if (lane == 0u) {
        seen = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    }
    seen = simd_broadcast(seen, ushort(0));
    if (seen != 31u) {
        return;
    }
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    float key[4];
    for (uint k = 0u; k < 4u; ++k) {
        key[k] = scores[lane * 4u + k] + bias[lane * 4u + k];
    }
    uint chosen[8];
    float total = 0.0f;
    for (uint pick = 0u; pick < 8u; ++pick) {
        float value = key[0];
        uint ident = lane * 4u;
        for (uint k = 1u; k < 4u; ++k) {
            if (key[k] > value) {
                value = key[k];
                ident = lane * 4u + k;
            }
        }
        for (uint offset = 16u; offset > 0u; offset >>= 1u) {
            float other_v = simd_shuffle_xor(value, offset);
            uint other_i = simd_shuffle_xor(ident, offset);
            if (other_v > value || (other_v == value && other_i < ident)) {
                value = other_v;
                ident = other_i;
            }
        }
        chosen[pick] = ident;
        total += scores[ident];
        if ((ident >> 2u) == lane) {
            key[ident & 3u] = -INFINITY;
        }
    }
    if (lane < 8u) {
        float scale = 8.0f / (total + 1.0e-20f);
        uint pickid = chosen[0];
        for (uint k = 1u; k < 8u; ++k) {
            if (lane == k) {
                pickid = chosen[k];
            }
        }
        ids[lane] = pickid;
        route[lane] = scores[pickid] * scale;
    }
    if (lane == 0u) {
        atomic_store_explicit(counter, 0u, memory_order_relaxed);
    }
}

// One simdgroup per row. inner is a multiple of 8.
kernel void gemv_f16(
    device const float* activation [[buffer(0)]],
    device const half* weights [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& inner [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    if (row >= cols) {
        return;
    }
    uint steps = inner >> 3u;
    device const float4* act = (device const float4*)activation;
    device const half4* rowp = (device const half4*)(weights + (ulong)row * inner);
    float acc = 0.0f;
    for (uint i = lane; i < steps; i += 32u) {
        acc += dot(act[i * 2u], float4(rowp[i * 2u])) + dot(act[i * 2u + 1u], float4(rowp[i * 2u + 1u]));
    }
    float total = simd_sum(acc);
    if (lane == 0u) {
        destination[row] = total;
    }
}

// Eight experts. The fused rows interleave gate (even) and up (odd), so four
// rows are two gated outputs. The grid is (rows / 4, 8).
kernel void moe_gate_fp4(
    device const float* activation [[buffer(0)]],
    device const uchar* weights [[buffer(1)]],
    device const uchar* scales [[buffer(2)]],
    device const uint* ids [[buffer(3)]],
    device float* gated [[buffer(4)]],
    constant uint& data_stride [[buffer(5)]],
    constant uint& scale_stride [[buffer(6)]],
    constant uint& inner [[buffer(7)]],
    constant uint& half_width [[buffer(8)]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    uint row0 = tgid.x * 4u;
    uint slot = tgid.y;
    uint expert = ids[slot];
    float4 acc = simd_sum4(fp4_part4(
        activation,
        weights + (ulong)expert * data_stride,
        scales + (ulong)expert * scale_stride,
        row0, inner >> 3u, inner >> 1u, inner >> 5u, lane));
    if (lane == 0u) {
        uint col = row0 >> 1u;
        gated[slot * half_width + col] = swiglu_f(acc.x, acc.y);
        gated[slot * half_width + col + 1u] = swiglu_f(acc.z, acc.w);
    }
}

kernel void moe_gate_fp8(
    device const float* activation [[buffer(0)]],
    device const uchar* weights [[buffer(1)]],
    device const uchar* scales [[buffer(2)]],
    device const uint* ids [[buffer(3)]],
    device float* gated [[buffer(4)]],
    constant uint& data_stride [[buffer(5)]],
    constant uint& scale_stride [[buffer(6)]],
    constant uint& inner [[buffer(7)]],
    constant uint& half_width [[buffer(8)]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    uint row0 = tgid.x * 4u;
    uint slot = tgid.y;
    uint expert = ids[slot];
    float4 acc = simd_sum4(fp8_part4(
        activation,
        weights + (ulong)expert * data_stride,
        scales + (ulong)expert * scale_stride,
        row0, inner >> 4u, inner, inner >> 5u, lane));
    if (lane == 0u) {
        uint col = row0 >> 1u;
        gated[slot * half_width + col] = swiglu_f(acc.x, acc.y);
        gated[slot * half_width + col + 1u] = swiglu_f(acc.z, acc.w);
    }
}

// Eight weighted downs, the MLP short convolution, and the add into the block sum.
// inner is the logical width. stored is the stored row width.
kernel void moe_down_fp4(
    device const float* gated [[buffer(0)]],
    device const uchar* weights [[buffer(1)]],
    device const uchar* scales [[buffer(2)]],
    device const uint* ids [[buffer(3)]],
    device const float* route [[buffer(4)]],
    device const float* conv_w [[buffer(5)]],
    device float* state [[buffer(6)]],
    device float* sum [[buffer(7)]],
    constant uint& cols [[buffer(8)]],
    constant uint& data_stride [[buffer(9)]],
    constant uint& scale_stride [[buffer(10)]],
    constant uint& inner [[buffer(11)]],
    constant uint& stored [[buffer(12)]],
    uint lane [[thread_index_in_simdgroup]],
    uint group [[threadgroup_position_in_grid]])
{
    uint row0 = group * 4u;
    if (row0 >= cols) {
        return;
    }
    float4 acc = float4(0.0f);
    for (uint slot = 0u; slot < 8u; ++slot) {
        uint expert = ids[slot];
        float4 part = fp4_part4(
            gated + slot * inner,
            weights + (ulong)expert * data_stride,
            scales + (ulong)expert * scale_stride,
            row0, inner >> 3u, stored >> 1u, stored >> 5u, lane);
        acc = fma(float4(route[slot]), part, acc);
    }
    acc = simd_sum4(acc);
    if (lane < 4u) {
        uint c = row0 + lane;
        sum[c] += conv_step(acc[lane], conv_w, state, c, cols);
    }
}

kernel void moe_down_fp8(
    device const float* gated [[buffer(0)]],
    device const uchar* weights [[buffer(1)]],
    device const uchar* scales [[buffer(2)]],
    device const uint* ids [[buffer(3)]],
    device const float* route [[buffer(4)]],
    device const float* conv_w [[buffer(5)]],
    device float* state [[buffer(6)]],
    device float* sum [[buffer(7)]],
    constant uint& cols [[buffer(8)]],
    constant uint& data_stride [[buffer(9)]],
    constant uint& scale_stride [[buffer(10)]],
    constant uint& inner [[buffer(11)]],
    constant uint& stored [[buffer(12)]],
    uint lane [[thread_index_in_simdgroup]],
    uint group [[threadgroup_position_in_grid]])
{
    uint row0 = group * 4u;
    if (row0 >= cols) {
        return;
    }
    float4 acc = float4(0.0f);
    for (uint slot = 0u; slot < 8u; ++slot) {
        uint expert = ids[slot];
        float4 part = fp8_part4(
            gated + slot * inner,
            weights + (ulong)expert * data_stride,
            scales + (ulong)expert * scale_stride,
            row0, inner >> 4u, stored, stored >> 5u, lane);
        acc = fma(float4(route[slot]), part, acc);
    }
    acc = simd_sum4(acc);
    if (lane < 4u) {
        uint c = row0 + lane;
        sum[c] += conv_step(acc[lane], conv_w, state, c, cols);
    }
}

// One threadgroup of 128 per token. Sigmoid scores, the bias only for the
// pick, then the eight picks renormalized and scaled by 8.
kernel void route_rows(
    device const float* logits [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device uint* ids [[buffer(2)]],
    device float* weights [[buffer(3)]],
    constant uint& tokens [[buffer(4)]],
    constant uint& stride [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint token [[threadgroup_position_in_grid]])
{
    if (token >= tokens) {
        return;
    }
    threadgroup float part_v[4];
    threadgroup uint part_i[4];
    threadgroup uint chosen[8];
    float s = sigmoid_f(logits[token * stride + tid]);
    float key = s + bias[tid];
    for (uint pass = 0u; pass < 8u; ++pass) {
        float value = key;
        uint ident = tid;
        for (uint offset = 16u; offset > 0u; offset >>= 1u) {
            float other_v = simd_shuffle_xor(value, offset);
            uint other_i = simd_shuffle_xor(ident, offset);
            if (other_v > value || (other_v == value && other_i < ident)) {
                value = other_v;
                ident = other_i;
            }
        }
        if (lane == 0u) {
            part_v[simd] = value;
            part_i[simd] = ident;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float best = part_v[0];
        uint best_i = part_i[0];
        for (uint g = 1u; g < 4u; ++g) {
            if (part_v[g] > best || (part_v[g] == best && part_i[g] < best_i)) {
                best = part_v[g];
                best_i = part_i[g];
            }
        }
        if (tid == 0u) {
            chosen[pass] = best_i;
        }
        if (tid == best_i) {
            key = -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float picked[8];
    for (uint k = 0u; k < 8u; ++k) {
        if (tid == chosen[k]) {
            picked[k] = s;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float total = 0.0f;
        for (uint k = 0u; k < 8u; ++k) {
            total += picked[k];
        }
        float scale = 8.0f / (total + 1.0e-20f);
        for (uint k = 0u; k < 8u; ++k) {
            ids[token * 8u + k] = chosen[k];
            weights[token * 8u + k] = picked[k] * scale;
        }
    }
}

// ---------------------------------------------------------------------------
// Norms, block mixes, and the message gates.

// One threadgroup of 256 per (token, y). y selects a strided source, gamma, and destination.
kernel void rmsnorm(
    device const float* source [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& tokens [[buffer(3)]],
    constant uint& hidden [[buffer(4)]],
    constant uint& use_gamma [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    constant uint& src_ys [[buffer(7)]],
    constant uint& dst_ys [[buffer(8)]],
    constant uint& gamma_ys [[buffer(9)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    uint row = tgid.x;
    if (row >= tokens) {
        return;
    }
    threadgroup float red[8];
    device const float* in = source + tgid.y * src_ys + row * hidden;
    device float* out = destination + tgid.y * dst_ys + row * hidden;
    device const float* g = gamma + tgid.y * gamma_ys;
    float total = 0.0f;
    for (uint col = tid; col < hidden; col += 256u) {
        float value = in[col];
        total += value * value;
    }
    total = simd_sum(total);
    if (lane == 0u) {
        red[simd] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float all = red[0] + red[1] + red[2] + red[3] + red[4] + red[5] + red[6] + red[7];
    float scale = rsqrt(all / float(hidden) + eps);
    if (use_gamma != 0u) {
        for (uint col = tid; col < hidden; col += 256u) {
            out[col] = in[col] * scale * g[col];
        }
    } else {
        for (uint col = tid; col < hidden; col += 256u) {
            out[col] = in[col] * scale;
        }
    }
}

// Learned queries against RMS-normalized block keys, scaled by q_scale.
// Values are the normalized sources. tgid.y picks query row rows[y]. With
// copies == 0 the mix goes to destination + y * dst_ys. Otherwise the mix is
// RMS-normalized and copy c goes to row y + c with gamma row y + c.
kernel void mix_blocks(
    device const float* query [[buffer(0)]],
    device const float* v0 [[buffer(1)]],
    device const float* v1 [[buffer(2)]],
    device const float* v2 [[buffer(3)]],
    device const float* v3 [[buffer(4)]],
    device const float* v4 [[buffer(5)]],
    device const float* v5 [[buffer(6)]],
    device float* destination [[buffer(7)]],
    device const uint* rows [[buffer(8)]],
    device const float* gamma [[buffer(9)]],
    constant uint& tokens [[buffer(10)]],
    constant uint& hidden [[buffer(11)]],
    constant uint& count [[buffer(12)]],
    constant float& eps [[buffer(13)]],
    constant float& q_scale [[buffer(14)]],
    constant uint& dst_ys [[buffer(15)]],
    constant uint& copies [[buffer(16)]],
    constant uint& gamma_ys [[buffer(17)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    uint row = tgid.x;
    if (row >= tokens || count == 0u) {
        return;
    }
    threadgroup float red[6][8];
    threadgroup float red2[6][8];
    device const float* blocks[6] = {v0, v1, v2, v3, v4, v5};
    device const float* qv = query + rows[tgid.y] * hidden;
    float sq[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float dq[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint col = tid; col < hidden; col += 256u) {
        float q = qv[col];
        for (uint b = 0u; b < 6u; ++b) {
            if (b < count) {
                float value = blocks[b][row * hidden + col];
                sq[b] += value * value;
                dq[b] += q * value;
            }
        }
    }
    for (uint b = 0u; b < 6u; ++b) {
        float s = simd_sum(sq[b]);
        float d = simd_sum(dq[b]);
        if (lane == 0u) {
            red[b][simd] = s;
            red2[b][simd] = d;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv[6];
    float score[6];
    float peak = -INFINITY;
    for (uint b = 0u; b < 6u; ++b) {
        if (b < count) {
            float s = 0.0f;
            float d = 0.0f;
            for (uint k = 0u; k < 8u; ++k) {
                s += red[b][k];
                d += red2[b][k];
            }
            inv[b] = rsqrt(s / float(hidden) + eps);
            score[b] = d * inv[b] * q_scale;
            peak = max(peak, score[b]);
        } else {
            inv[b] = 0.0f;
            score[b] = 0.0f;
        }
    }
    float normal = 0.0f;
    for (uint b = 0u; b < 6u; ++b) {
        if (b < count) {
            score[b] = exp(score[b] - peak);
            normal += score[b];
        }
    }
    float weight[6];
    for (uint b = 0u; b < 6u; ++b) {
        weight[b] = b < count ? score[b] * inv[b] / normal : 0.0f;
    }
    device float* out = destination + tgid.y * dst_ys + row * hidden;
    float energy = 0.0f;
    for (uint col = tid; col < hidden; col += 256u) {
        float mixed = 0.0f;
        for (uint b = 0u; b < 6u; ++b) {
            if (b < count) {
                mixed += weight[b] * blocks[b][row * hidden + col];
            }
        }
        out[col] = mixed;
        energy += mixed * mixed;
    }
    if (copies == 0u) {
        return;
    }
    energy = simd_sum(energy);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
        red[0][simd] = energy;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float all = 0.0f;
    for (uint k = 0u; k < 8u; ++k) {
        all += red[0][k];
    }
    float scale = rsqrt(all / float(hidden) + eps);
    for (uint col = tid; col < hidden; col += 256u) {
        float value = out[col] * scale;
        for (uint c = copies; c > 0u; --c) {
            uint at = tgid.y + c - 1u;
            destination[at * dst_ys + row * hidden + col] = value * gamma[at * gamma_ys + col];
        }
    }
}

// Message gate down projection for one token. h[g][r] = down_g[r] .
// [rms(stream_g), rms(message_g)]. The grid is (64 / 8, gates).
kernel void ctl_down(
    device const float* stream [[buffer(0)]],
    device const float* message [[buffer(1)]],
    device const half* down [[buffer(2)]],
    device float* h [[buffer(3)]],
    constant uint& hidden [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    constant uint& s_ys [[buffer(6)]],
    constant uint& m_ys [[buffer(7)]],
    constant uint& gate0 [[buffer(8)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    uint g = tgid.y;
    device const float* s = stream + g * s_ys;
    device const float* m = message + g * m_ys;
    threadgroup float red[2][8];
    float ss = 0.0f;
    float mm = 0.0f;
    for (uint col = tid; col < hidden; col += 256u) {
        float a = s[col];
        float b = m[col];
        ss += a * a;
        mm += b * b;
    }
    ss = simd_sum(ss);
    mm = simd_sum(mm);
    if (lane == 0u) {
        red[0][simd] = ss;
        red[1][simd] = mm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float st = 0.0f;
    float mt = 0.0f;
    for (uint k = 0u; k < 8u; ++k) {
        st += red[0][k];
        mt += red[1][k];
    }
    float inv_s = rsqrt(st / float(hidden) + eps);
    float inv_m = rsqrt(mt / float(hidden) + eps);
    uint r = tgid.x * 8u + simd;
    device const half4* w = (device const half4*)(down + ((ulong)(gate0 + g) * 64u + r) * (hidden * 2u));
    device const float4* s4 = (device const float4*)s;
    device const float4* m4 = (device const float4*)m;
    uint quads = hidden >> 2u;
    float a0 = 0.0f;
    float a1 = 0.0f;
    for (uint i = lane; i < quads; i += 32u) {
        a0 += dot(float4(w[i]), s4[i]);
        a1 += dot(float4(w[quads + i]), m4[i]);
    }
    float total = simd_sum(a0 * inv_s + a1 * inv_m);
    if (lane == 0u) {
        h[g * 64u + r] = total;
    }
}

// out_g = (1 - tanh(forget)) * stream_g + tanh(gain) * message_g, with
// forget and gain the rank-64 projections of h_g. fg_g is [2 * hidden, 64].
kernel void ctl_blend(
    device const float* stream [[buffer(0)]],
    device const float* message [[buffer(1)]],
    device const half* fg [[buffer(2)]],
    device const float* h [[buffer(3)]],
    device float* destination [[buffer(4)]],
    constant uint& hidden [[buffer(5)]],
    constant uint& s_ys [[buffer(6)]],
    constant uint& m_ys [[buffer(7)]],
    constant uint& gate0 [[buffer(8)]],
    constant uint& d_ys [[buffer(9)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    uint g = tgid.y;
    threadgroup float4 hh[16];
    if (tid < 16u) {
        hh[tid] = *(device const float4*)(h + g * 64u + tid * 4u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint c = tgid.x * 256u + tid;
    if (c >= hidden) {
        return;
    }
    device const half4* fw = (device const half4*)(fg + ((ulong)(gate0 + g) * 2u * hidden + c) * 64u);
    device const half4* gw = (device const half4*)(fg + ((ulong)(gate0 + g) * 2u * hidden + hidden + c) * 64u);
    float f = 0.0f;
    float k = 0.0f;
    for (uint i = 0u; i < 16u; ++i) {
        f += dot(float4(fw[i]), hh[i]);
        k += dot(float4(gw[i]), hh[i]);
    }
    float x = stream[g * s_ys + c];
    float m = message[g * m_ys + c];
    destination[g * d_ys + c] = (1.0f - tanh(f)) * x + tanh(k) * m;
}

// Prefill message gate input: half [tokens, 2 * hidden] of the two RMS-normalized rows.
kernel void ctl_pair_rows(
    device const float* stream [[buffer(0)]],
    device const float* message [[buffer(1)]],
    device half* pair [[buffer(2)]],
    constant uint& tokens [[buffer(3)]],
    constant uint& hidden [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    device half* out = pair + (ulong)row * hidden * 2u;
    if (row >= tokens) {
        for (uint col = tid; col < hidden * 2u; col += 256u) {
            out[col] = half(0.0f);
        }
        return;
    }
    device const float* s = stream + row * hidden;
    device const float* m = message + row * hidden;
    threadgroup float red[2][8];
    float ss = 0.0f;
    float mm = 0.0f;
    for (uint col = tid; col < hidden; col += 256u) {
        ss += s[col] * s[col];
        mm += m[col] * m[col];
    }
    ss = simd_sum(ss);
    mm = simd_sum(mm);
    if (lane == 0u) {
        red[0][simd] = ss;
        red[1][simd] = mm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float st = 0.0f;
    float mt = 0.0f;
    for (uint k = 0u; k < 8u; ++k) {
        st += red[0][k];
        mt += red[1][k];
    }
    float inv_s = rsqrt(st / float(hidden) + eps);
    float inv_m = rsqrt(mt / float(hidden) + eps);
    for (uint col = tid; col < hidden; col += 256u) {
        out[col] = half(s[col] * inv_s);
        out[hidden + col] = half(m[col] * inv_m);
    }
}

// fgout is [tokens, 2 * hidden]: forget then gain.
kernel void ctl_blend_rows(
    device const float* stream [[buffer(0)]],
    device const float* message [[buffer(1)]],
    device const float* fgout [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    constant uint& hidden [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) {
        return;
    }
    uint row = tid / hidden;
    uint c = tid - row * hidden;
    float f = fgout[row * hidden * 2u + c];
    float k = fgout[row * hidden * 2u + hidden + c];
    destination[tid] = (1.0f - tanh(f)) * stream[tid] + tanh(k) * message[tid];
}

// ---------------------------------------------------------------------------
// Attention.

// x[t - lag] from the raw rows, or from the three-row state before the chunk.
static inline float conv_hist(
    device const float* raw, device const float* state, int t, uint lag, uint c, uint channels, uint stride)
{
    int at = t - int(lag);
    if (at >= 0) {
        return raw[uint(at) * stride + c];
    }
    return state[uint(3 + at) * channels + c];
}

// mode 0 copies history [origin, pos0) from a local ring into a contiguous span.
// mode 1 writes the last ring_slots keys of [origin, kend) back onto that ring.
// One thread copies one key row. The grid is that row count, rounded up to 256.
kernel void kv_window(
    device half* stage_k [[buffer(0)]],
    device half* stage_v [[buffer(1)]],
    device half* ring_k [[buffer(2)]],
    device half* ring_v [[buffer(3)]],
    constant uint& origin [[buffer(4)]],
    constant uint& pos0 [[buffer(5)]],
    constant uint& kend [[buffer(6)]],
    constant uint& ring_slots [[buffer(7)]],
    constant uint& mode [[buffer(8)]],
    uint row [[thread_position_in_grid]])
{
    uint mask = ring_slots - 1u;
    if (mode == 0u) {
        if (origin >= pos0 || row >= pos0 - origin) {
            return;
        }
        uint absolute = origin + row;
        uint held = pos0 > ring_slots ? pos0 - ring_slots : 0u;
        device half4* dk = (device half4*)(stage_k + (ulong)row * 1024u);
        device half4* dv = (device half4*)(stage_v + (ulong)row * 1024u);
        if (absolute < held) {
            for (uint i = 0u; i < 256u; ++i) {
                dk[i] = half4(0.0h);
                dv[i] = half4(0.0h);
            }
            return;
        }
        uint slot = absolute & mask;
        device const half4* sk = (device const half4*)(ring_k + (ulong)slot * 1024u);
        device const half4* sv = (device const half4*)(ring_v + (ulong)slot * 1024u);
        for (uint i = 0u; i < 256u; ++i) {
            dk[i] = sk[i];
            dv[i] = sv[i];
        }
        return;
    }
    uint keep = kend > ring_slots ? kend - ring_slots : 0u;
    if (keep < origin) {
        keep = origin;
    }
    if (keep >= kend || row >= kend - keep) {
        return;
    }
    uint absolute = keep + row;
    uint stage_row = absolute - origin;
    uint slot = absolute & mask;
    device const half4* sk = (device const half4*)(stage_k + (ulong)stage_row * 1024u);
    device const half4* sv = (device const half4*)(stage_v + (ulong)stage_row * 1024u);
    device half4* dk = (device half4*)(ring_k + (ulong)slot * 1024u);
    device half4* dv = (device half4*)(ring_v + (ulong)slot * 1024u);
    for (uint i = 0u; i < 256u; ++i) {
        dk[i] = sk[i];
        dv[i] = sv[i];
    }
}

// A local ring stores position p at p & mask. mask 0 keeps the absolute index.
static inline uint kv_slot(uint key, uint mask)
{
    return mask == 0u ? key : (key & mask);
}

// Grid (tokens, 40), 128 threads, one thread per head channel.
// y < 24: query head clip (with its rel clip). 24..31: key head. 32..39: value head.
// Keys: short conv, L2 clip, half store. Values: short conv, RMS, half store.
// full selects the dynamic conv (static [4, C] plus per-token taps) and log scaling.
kernel void attn_prep(
    device float* query [[buffer(0)]],
    device float* rel [[buffer(1)]],
    device const float* k_raw [[buffer(2)]],
    device const float* v_raw [[buffer(3)]],
    device const float* q_scale [[buffer(4)]],
    device const float* rel_scale [[buffer(5)]],
    device const float* k_w [[buffer(6)]],
    device const float* v_w [[buffer(7)]],
    device const float* k_taps [[buffer(8)]],
    device const float* v_taps [[buffer(9)]],
    device float* k_state [[buffer(10)]],
    device float* v_state [[buffer(11)]],
    device half* k_cache [[buffer(12)]],
    device half* v_cache [[buffer(13)]],
    constant uint& tokens [[buffer(14)]],
    constant uint& pos0 [[buffer(15)]],
    constant uint& full [[buffer(16)]],
    constant uint& q_st [[buffer(17)]],
    constant uint& r_st [[buffer(18)]],
    constant uint& kv_st [[buffer(19)]],
    constant uint& tap_st [[buffer(20)]],
    constant float& eps [[buffer(21)]],
    constant float& q_clip [[buffer(22)]],
    constant float& rel_clip [[buffer(23)]],
    constant uint& slot_base [[buffer(24)]],
    constant uint& slot_mask [[buffer(25)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    uint t = tgid.x;
    uint y = tgid.y;
    if (t >= tokens) {
        return;
    }
    threadgroup float red[4];
    uint pos = pos0 + t;
    if (y < 24u) {
        device float* q = query + t * q_st + y * 128u;
        float value = q[tid] * q_scale[y * 128u + tid];
        float part = simd_sum(value * value);
        if (lane == 0u) {
            red[simd] = part;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float norm = sqrt(red[0] + red[1] + red[2] + red[3]);
        float tau = log_tau(pos, full);
        q[tid] = value * min(1.0f, q_clip / (norm + CLIP_EPS)) * tau;
        if (simd == 0u) {
            device float* r = rel + t * r_st + y * 16u;
            float rv = lane < 16u ? r[lane] * rel_scale[y * 16u + lane] : 0.0f;
            float rn = sqrt(simd_sum(rv * rv));
            if (lane < 16u) {
                r[lane] = rv * min(1.0f, rel_clip / (rn + CLIP_EPS)) * tau;
            }
        }
        return;
    }
    bool is_k = y < 32u;
    uint head = is_k ? y - 24u : y - 32u;
    uint c = head * 128u + tid;
    device const float* raw = is_k ? k_raw : v_raw;
    device float* state = is_k ? k_state : v_state;
    device const float* w = is_k ? k_w : v_w;
    float x0 = raw[t * kv_st + c];
    float x1 = conv_hist(raw, state, int(t), 1u, c, 1024u, kv_st);
    float x2 = conv_hist(raw, state, int(t), 2u, c, 1024u, kv_st);
    float x3 = conv_hist(raw, state, int(t), 3u, c, 1024u, kv_st);
    if (tokens == 1u) {
        // One token: this thread alone reads its channel of the state.
        state[c] = x2;
        state[1024u + c] = x1;
        state[2048u + c] = x0;
    }
    float y0;
    if (full != 0u) {
        device const float* taps = (is_k ? k_taps : v_taps) + t * tap_st + head * 4u;
        y0 = (x0 + (w[c] + taps[0]) * x0 + (w[1024u + c] + taps[1]) * x1
              + (w[2048u + c] + taps[2]) * x2 + (w[3072u + c] + taps[3]) * x3) / 3.0f;
    } else {
        float4 wt = *(device const float4*)(w + c * 4u);
        y0 = (x0 + wt.w * x0 + wt.z * x1 + wt.y * x2 + wt.x * x3) * INV_SQRT5;
    }
    float part = simd_sum(y0 * y0);
    if (lane == 0u) {
        red[simd] = part;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sq = red[0] + red[1] + red[2] + red[3];
    float out;
    if (is_k) {
        out = y0 * min(1.0f, q_clip / (sqrt(sq) + CLIP_EPS));
        k_cache[(ulong)kv_slot(pos - slot_base, slot_mask) * 1024u + c] = half(out);
    } else {
        out = y0 * rsqrt(sq / 128.0f + eps);
        v_cache[(ulong)kv_slot(pos - slot_base, slot_mask) * 1024u + c] = half(out);
    }
}

// The last three raw rows become the state. tgid.y picks pair a or b.
kernel void conv_tail(
    device const float* raw_a [[buffer(0)]],
    device float* state_a [[buffer(1)]],
    device const float* raw_b [[buffer(2)]],
    device float* state_b [[buffer(3)]],
    constant uint& tokens [[buffer(4)]],
    constant uint& channels [[buffer(5)]],
    constant uint& stride_a [[buffer(6)]],
    constant uint& stride_b [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint c = gid.x;
    if (c >= channels) {
        return;
    }
    device const float* raw = gid.y == 0u ? raw_a : raw_b;
    device float* state = gid.y == 0u ? state_a : state_b;
    uint stride = gid.y == 0u ? stride_a : stride_b;
    float old0 = state[c];
    float old1 = state[channels + c];
    float old2 = state[2u * channels + c];
    float olds[3] = {old0, old1, old2};
    for (uint k = 0u; k < 3u; ++k) {
        int at = int(tokens) - 3 + int(k);
        float value = at >= 0 ? raw[uint(at) * stride + c] : olds[uint(3 + at)];
        state[k * channels + c] = value;
    }
}

// Prefill residual short conv added into the block sum. One thread per (token, channel).
kernel void conv_add_rows(
    device const float* raw [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* state [[buffer(2)]],
    device float* sum [[buffer(3)]],
    constant uint& tokens [[buffer(4)]],
    constant uint& channels [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= tokens * channels) {
        return;
    }
    uint t = tid / channels;
    uint c = tid - t * channels;
    float x0 = raw[tid];
    float x1 = conv_hist(raw, state, int(t), 1u, c, channels, channels);
    float x2 = conv_hist(raw, state, int(t), 2u, c, channels, channels);
    float x3 = conv_hist(raw, state, int(t), 3u, c, channels, channels);
    float4 wt = *(device const float4*)(weight + c * 4u);
    sum[tid] += (x0 + wt.w * x0 + wt.z * x1 + wt.y * x2 + wt.x * x3) * INV_SQRT5;
}

// Prefill attention on matmul2d. Grid (24 query heads, tokens / 32), four
// simdgroups. Q K^T and P V run as half tiles; the softmax adds the scale,
// the relative bias, the causal window, and the sink in the threadgroup.
constant int FQ = 32;
constant int FK = 64;
constant int FD = 128;

kernel void attend_flash(
    device const float* query [[buffer(0)]],
    device half* k_cache [[buffer(1)]],
    device half* v_cache [[buffer(2)]],
    device const float* rel [[buffer(3)]],
    device const float* table [[buffer(4)]],
    device const float* sink [[buffer(5)]],
    device const float* gate [[buffer(6)]],
    device float* destination [[buffer(7)]],
    constant uint& tokens [[buffer(8)]],
    constant uint& pos0 [[buffer(9)]],
    constant uint& window [[buffer(10)]],
    constant uint& q_st [[buffer(11)]],
    constant uint& r_st [[buffer(12)]],
    constant uint& s_st [[buffer(13)]],
    constant uint& g_st [[buffer(14)]],
    constant uint& origin [[buffer(15)]],
    uint lane [[thread_index_in_simdgroup]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    uint h = tgid.x;
    uint kvh = h / 3u;
    uint t0 = tgid.y * uint(FQ);
    if (t0 >= tokens) {
        return;
    }
    uint qfirst = pos0 + t0;
    uint qlast = pos0 + min(t0 + uint(FQ), tokens) - 1u;
    uint lo = (window != 0u && qfirst + 1u > window) ? qfirst + 1u - window : 0u;
    uint kend = qlast + 1u;
    threadgroup half qtile[FQ * FD];
    threadgroup float scores[FQ * FK];
    threadgroup half prob[FQ * FK];
    threadgroup float relt[FQ * 16];
    threadgroup float mrun[FQ];
    threadgroup float lrun[FQ];
    threadgroup float alpha_row[FQ];
    for (uint i = tid; i < uint(FQ * FD); i += 128u) {
        uint r = i / uint(FD);
        uint t = t0 + r;
        qtile[i] = t < tokens ? half(query[t * q_st + h * uint(FD) + (i - r * uint(FD))]) : half(0.0f);
    }
    for (uint i = tid; i < uint(FQ) * 16u; i += 128u) {
        uint t = t0 + i / 16u;
        relt[i] = t < tokens ? rel[t * r_st + h * 16u + (i & 15u)] : 0.0f;
    }
    if (tid < uint(FQ)) {
        uint t = t0 + tid;
        mrun[tid] = t < tokens ? sink[t * s_st + h] : 0.0f;
        lrun[tid] = 1.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tensor<threadgroup half, dextents<int, 2>, tensor_inline> queries(qtile, dextents<int, 2>(FD, FQ));
    tensor<device half, dextents<int, 2>, tensor_inline> keys(
        k_cache + kvh * uint(FD), dextents<int, 2>(FD, int(kend - origin)), array<int, 2>{1, 1024});
    tensor<device half, dextents<int, 2>, tensor_inline> values(
        v_cache + kvh * uint(FD), dextents<int, 2>(FD, int(kend - origin)), array<int, 2>{1, 1024});
    tensor<threadgroup half, dextents<int, 2>, tensor_inline> tile_p(prob, dextents<int, 2>(FK, FQ));
    auto tile_q = queries.slice(0, 0);
    constexpr auto qk_desc = matmul2d_descriptor(FQ, FK, FD, false, true, false);
    matmul2d<qk_desc, execution_simdgroups<4>> qk_op;
    constexpr auto pv_desc = matmul2d_descriptor(FQ, FD, FK, false, false, false);
    matmul2d<pv_desc, execution_simdgroups<4>> pv_op;
    float oreg[32];
    for (int i = 0; i < 32; ++i) {
        oreg[i] = 0.0f;
    }
    uint row = tid >> 2u;
    uint quarter = tid & 3u;
    uint t = t0 + row;
    uint qp = pos0 + t;
    bool live = t < tokens;
    for (uint k0 = lo & ~uint(FK - 1); k0 < kend; k0 += uint(FK)) {
        auto tile_k = keys.slice(0, int(k0 - origin));
        auto score_tile = qk_op.get_destination_cooperative_tensor<decltype(tile_q), decltype(tile_k), float>();
        qk_op.run(tile_q, tile_k, score_tile);
        for (uint16_t i = 0u; i < score_tile.get_capacity(); ++i) {
            if (score_tile.is_valid_element(i)) {
                auto ix = score_tile.get_multidimensional_index(i);
                scores[uint(ix[1]) * uint(FK) + uint(ix[0])] = score_tile[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool near = qfirst < k0 + uint(FK) + 1023u;
        float vals[FK / 4];
        float peak = -INFINITY;
        for (uint j = 0u; j < uint(FK / 4); ++j) {
            uint c = j * 4u + quarter;
            uint key = k0 + c;
            uint dist = qp - key;
            bool ok = live && key <= qp && key < kend && (window == 0u || dist < window);
            float s = -INFINITY;
            if (ok) {
                s = scores[row * uint(FK) + c] * QK_SCALE;
                if (near && dist < 1024u) {
                    float b = 0.0f;
                    for (uint d = 0u; d < 16u; ++d) {
                        b = fma(relt[row * 16u + d], table[d * 1024u + dist], b);
                    }
                    s += b;
                }
            }
            vals[j] = s;
            peak = max(peak, s);
        }
        peak = max(peak, simd_shuffle_xor(peak, ushort(1)));
        peak = max(peak, simd_shuffle_xor(peak, ushort(2)));
        float mold = mrun[row];
        float next = max(mold, peak);
        float alpha = exp(mold - next);
        float total = 0.0f;
        for (uint j = 0u; j < uint(FK / 4); ++j) {
            float p = vals[j] == -INFINITY ? 0.0f : exp(vals[j] - next);
            prob[row * uint(FK) + j * 4u + quarter] = half(p);
            total += p;
        }
        total += simd_shuffle_xor(total, ushort(1));
        total += simd_shuffle_xor(total, ushort(2));
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (quarter == 0u) {
            alpha_row[row] = alpha;
            lrun[row] = lrun[row] * alpha + total;
            mrun[row] = next;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto tile_v = values.slice(0, int(k0 - origin));
        auto mixed = pv_op.get_destination_cooperative_tensor<decltype(tile_p), decltype(tile_v), float>();
        pv_op.run(tile_p, tile_v, mixed);
        for (uint16_t i = 0u; i < mixed.get_capacity() && i < 32u; ++i) {
            if (mixed.is_valid_element(i)) {
                auto ix = mixed.get_multidimensional_index(i);
                oreg[i] = oreg[i] * alpha_row[uint(ix[1])] + mixed[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto tile_v0 = values.slice(0, 0);
    auto layout = pv_op.get_destination_cooperative_tensor<decltype(tile_p), decltype(tile_v0), float>();
    for (uint16_t i = 0u; i < layout.get_capacity() && i < 32u; ++i) {
        if (!layout.is_valid_element(i)) {
            continue;
        }
        auto ix = layout.get_multidimensional_index(i);
        uint c = uint(ix[0]);
        uint r = uint(ix[1]);
        uint tt = t0 + r;
        if (tt >= tokens || c >= uint(FD)) {
            continue;
        }
        float g = gate[tt * g_st + h * uint(FD) + c];
        destination[tt * 3072u + h * uint(FD) + c] = oreg[i] / lrun[r] * sigmoid_f(g);
    }
}

// Keys [kbeg, kend) for the three query heads of key head kvh at decode
// position qp. Leaves the running (m, l, o) of each head; o holds four
// channels per lane. with_sink starts from the sink logit.
static inline void attend_range(
    device const float* query,
    device const half* k_cache,
    device const half* v_cache,
    device const float* rel,
    device const float* table,
    device const float* sink,
    uint kvh,
    uint qp,
    uint kbeg,
    uint kend,
    uint slot_mask,
    bool with_sink,
    uint lane,
    thread float* m,
    thread float* l,
    thread float4* o)
{
    uint sub = lane & 7u;
    uint kq = lane >> 3u;
    float4 qr[3][4];
    float rv[3];
    for (uint r = 0u; r < 3u; ++r) {
        uint h = kvh * 3u + r;
        for (uint i = 0u; i < 4u; ++i) {
            qr[r][i] = *(device const float4*)(query + h * 128u + sub * 16u + i * 4u);
        }
        o[r] = float4(0.0f);
        m[r] = with_sink ? sink[h] : -1.0e30f;
        l[r] = with_sink ? 1.0f : 0.0f;
        rv[r] = lane < 16u ? rel[h * 16u + lane] : 0.0f;
    }
    ushort src = ushort((lane & 3u) * 8u);
    uint mine = lane >> 2u;
    for (uint kb = kbeg; kb < kend; kb += 32u) {
        float sc[3] = {0.0f, 0.0f, 0.0f};
        for (uint p = 0u; p < 8u; ++p) {
            uint key = kb + p * 4u + kq;
            float d0 = 0.0f;
            float d1 = 0.0f;
            float d2 = 0.0f;
            if (key < kend) {
                uint slot = kv_slot(key, slot_mask);
                device const half4* kp = (device const half4*)(k_cache + (ulong)slot * 1024u + kvh * 128u + sub * 16u);
                for (uint i = 0u; i < 4u; ++i) {
                    float4 kf = float4(kp[i]);
                    d0 += dot(kf, qr[0][i]);
                    d1 += dot(kf, qr[1][i]);
                    d2 += dot(kf, qr[2][i]);
                }
            }
            for (ushort off = 1u; off < 8u; off <<= 1u) {
                d0 += simd_shuffle_xor(d0, off);
                d1 += simd_shuffle_xor(d1, off);
                d2 += simd_shuffle_xor(d2, off);
            }
            d0 = simd_shuffle(d0, src);
            d1 = simd_shuffle(d1, src);
            d2 = simd_shuffle(d2, src);
            if (p == mine) {
                sc[0] = d0;
                sc[1] = d1;
                sc[2] = d2;
            }
        }
        uint key = kb + lane;
        bool ok = key < kend;
        uint dist = qp - key;
        bool near = qp < kb + 32u + 1023u;
        float rb[3] = {0.0f, 0.0f, 0.0f};
        for (uint d = 0u; near && d < 16u; ++d) {
            float tab = (ok && dist < 1024u) ? table[d * 1024u + dist] : 0.0f;
            for (uint r = 0u; r < 3u; ++r) {
                rb[r] = fma(simd_broadcast(rv[r], ushort(d)), tab, rb[r]);
            }
        }
        for (uint r = 0u; r < 3u; ++r) {
            float s = ok ? sc[r] * QK_SCALE + rb[r] : -INFINITY;
            float mn = max(m[r], simd_max(s));
            float alpha = exp(m[r] - mn);
            float p = ok ? exp(s - mn) : 0.0f;
            l[r] = l[r] * alpha + simd_sum(p);
            o[r] *= alpha;
            m[r] = mn;
            sc[r] = p;
        }
        uint nk = min(32u, kend - kb);
        if (slot_mask == 0u) {
            device const half4* vp = (device const half4*)(v_cache + (ulong)kb * 1024u + kvh * 128u) + lane;
            for (uint j = 0u; j < nk; ++j) {
                float4 vf = float4(vp[j * 256u]);
                for (uint r = 0u; r < 3u; ++r) {
                    o[r] = fma(float4(simd_broadcast(sc[r], ushort(j))), vf, o[r]);
                }
            }
        } else {
            for (uint j = 0u; j < nk; ++j) {
                uint slot = kv_slot(kb + j, slot_mask);
                device const half4* vp = (device const half4*)(v_cache + (ulong)slot * 1024u + kvh * 128u) + lane;
                float4 vf = float4(vp[0]);
                for (uint r = 0u; r < 3u; ++r) {
                    o[r] = fma(float4(simd_broadcast(sc[r], ushort(j))), vf, o[r]);
                }
            }
        }
    }
}

// Decode attention. Grid (1, 8 key heads, splits), eight simdgroups. Each
// threadgroup covers span keys, each simdgroup an eighth of them; the
// simdgroups combine in threadgroup memory. With one split the result is
// gated into output. Otherwise each threadgroup writes a (m, l, o) partial of
// 132 floats per head and the last one to finish merges them.
kernel void attend_group(
    device const float* query [[buffer(0)]],
    device const half* k_cache [[buffer(1)]],
    device const half* v_cache [[buffer(2)]],
    device const float* rel [[buffer(3)]],
    device const float* table [[buffer(4)]],
    device const float* sink [[buffer(5)]],
    device const float* gate [[buffer(6)]],
    device float* partial [[buffer(7)]],
    device atomic_uint* counters [[buffer(8)]],
    device float* output [[buffer(9)]],
    constant uint& pos0 [[buffer(10)]],
    constant uint& window [[buffer(11)]],
    constant uint& span [[buffer(12)]],
    constant uint& slot_mask [[buffer(13)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 grid [[threadgroups_per_grid]])
{
    threadgroup float tm[8][3];
    threadgroup float tl[8][3];
    threadgroup float4 to[8][3][32];
    threadgroup uint last;
    uint kvh = tgid.y;
    uint split = tgid.z;
    uint splits = grid.z;
    uint qp = pos0;
    uint lo = (window != 0u && qp + 1u > window) ? qp + 1u - window : 0u;
    uint tbeg = lo + split * span;
    uint tend = min(qp + 1u, tbeg + span);
    uint part = ((span + 255u) / 256u) * 32u;
    uint sbeg = min(tend, tbeg + simd * part);
    uint send = min(tend, sbeg + part);
    float m[3];
    float l[3];
    float4 o[3];
    attend_range(query, k_cache, v_cache, rel, table, sink, kvh, qp, sbeg, send,
                 slot_mask, split == 0u && simd == 0u, lane, m, l, o);
    for (uint r = 0u; r < 3u; ++r) {
        if (lane == 0u) {
            tm[simd][r] = m[r];
            tl[simd][r] = l[r];
        }
        to[simd][r][lane] = o[r];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd < 3u) {
        uint r = simd;
        uint h = kvh * 3u + r;
        float peak = tm[0][r];
        for (uint g = 1u; g < 8u; ++g) {
            peak = max(peak, tm[g][r]);
        }
        float total = 0.0f;
        float4 acc = float4(0.0f);
        for (uint g = 0u; g < 8u; ++g) {
            float scale = exp(tm[g][r] - peak);
            total += tl[g][r] * scale;
            acc += to[g][r][lane] * scale;
        }
        if (splits == 1u) {
            float4 gv = *(device const float4*)(gate + h * 128u + lane * 4u);
            float4 sg = float4(sigmoid_f(gv.x), sigmoid_f(gv.y), sigmoid_f(gv.z), sigmoid_f(gv.w));
            *(device float4*)(output + h * 128u + lane * 4u) = acc / total * sg;
        } else {
            device float* row = partial + (split * 24u + h) * 132u;
            if (lane == 0u) {
                row[0] = peak;
                row[1] = total;
            }
            *(device float4*)(row + 4u + lane * 4u) = acc;
        }
    }
    if (splits == 1u) {
        return;
    }
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (simd == 0u && lane == 0u) {
        last = atomic_fetch_add_explicit(&counters[kvh], 1u, memory_order_relaxed) + 1u == splits ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (last == 0u || simd >= 3u) {
        return;
    }
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    uint h = kvh * 3u + simd;
    float peak = -INFINITY;
    for (uint s = lane; s < splits; s += 32u) {
        peak = max(peak, partial[(s * 24u + h) * 132u]);
    }
    peak = simd_max(peak);
    float total = 0.0f;
    float4 acc = float4(0.0f);
    for (uint s = 0u; s < splits; ++s) {
        device const float* row = partial + (s * 24u + h) * 132u;
        float scale = exp(row[0] - peak);
        total += row[1] * scale;
        acc = fma(*(device const float4*)(row + 4u + lane * 4u), float4(scale), acc);
    }
    float4 gv = *(device const float4*)(gate + h * 128u + lane * 4u);
    float4 sg = float4(sigmoid_f(gv.x), sigmoid_f(gv.y), sigmoid_f(gv.z), sigmoid_f(gv.w));
    *(device float4*)(output + h * 128u + lane * 4u) = acc / total * sg;
    if (simd == 0u && lane == 0u) {
        atomic_store_explicit(&counters[kvh], 0u, memory_order_relaxed);
    }
}

// ---------------------------------------------------------------------------
// Prefill expert grouping on the device.

// One threadgroup of 128. Sixteen slices keep assignment order inside an
// expert. Each expert gets 32-row tiles. An id of 128 or more is skipped.
kernel void moe_group(
    device const uint* ids [[buffer(0)]],
    device uint* csr_i [[buffer(1)]],
    device uint* expert_of [[buffer(2)]],
    device uint* slot_n [[buffer(3)]],
    device uint* slot_csr [[buffer(4)]],
    device uint* row_base [[buffer(5)]],
    device uint* jobs [[buffer(6)]],
    device uint* info [[buffer(7)]],
    device uint* csr_lane [[buffer(8)]],
    device uint* pack_row [[buffer(9)]],
    constant uint& count [[buffer(10)]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup uint histo[16][128];
    threadgroup uint counts[128];
    threadgroup uint offsets[128];
    threadgroup uint row_of[128];
    uint nassign = count * 8u;
    for (uint cell = tid; cell < 16u * 128u; cell += 128u) {
        histo[cell >> 7u][cell & 127u] = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint span = (nassign + 15u) >> 4u;
    if (tid < 16u) {
        uint begin = tid * span;
        uint end = min(begin + span, nassign);
        for (uint index = begin; index < end; ++index) {
            uint expert = ids[index];
            if (expert < 128u) {
                histo[tid][expert] += 1u;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        uint cursor = 0u;
        uint slot = 0u;
        uint row = 0u;
        uint job = 0u;
        for (uint expert = 0u; expert < 128u; ++expert) {
            uint n = 0u;
            for (uint slice = 0u; slice < 16u; ++slice) {
                n += histo[slice][expert];
            }
            counts[expert] = n;
            offsets[expert] = cursor;
            cursor += n;
            row_of[expert] = row;
            if (n == 0u) {
                continue;
            }
            uint pad = (n + 31u) & ~31u;
            expert_of[slot] = expert;
            slot_n[slot] = n;
            slot_csr[slot] = offsets[expert];
            row_base[slot] = row;
            for (uint tile = 0u; tile < (pad >> 5u); ++tile) {
                jobs[job * 2u] = slot;
                jobs[job * 2u + 1u] = tile;
                job += 1u;
            }
            row += pad;
            slot += 1u;
        }
        info[0] = slot;
        info[1] = job;
        info[2] = row;
        info[3] = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 16u) {
        uint rank[128];
        for (uint expert = 0u; expert < 128u; ++expert) {
            uint base = 0u;
            for (uint slice = 0u; slice < tid; ++slice) {
                base += histo[slice][expert];
            }
            rank[expert] = base;
        }
        uint begin = tid * span;
        uint end = min(begin + span, nassign);
        for (uint index = begin; index < end; ++index) {
            uint expert = ids[index];
            if (expert >= 128u) {
                pack_row[index] = 0xffffffffu;
                continue;
            }
            uint at = offsets[expert] + rank[expert];
            rank[expert] += 1u;
            csr_i[at] = index >> 3u;
            csr_lane[at] = index & 7u;
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    uint n = counts[tid];
    uint off = offsets[tid];
    uint row = row_of[tid];
    for (uint local = 0u; local < n; ++local) {
        uint at = off + local;
        pack_row[csr_i[at] * 8u + csr_lane[at]] = row + local;
    }
}

// One job is one 32-row tile. Rows past the expert's count are zero.
kernel void gather_grouped(
    device const float* source [[buffer(0)]],
    device const uint* csr_i [[buffer(1)]],
    device half* pack [[buffer(2)]],
    device const uint* slot_n [[buffer(3)]],
    device const uint* slot_csr [[buffer(4)]],
    device const uint* row_base [[buffer(5)]],
    device const uint* jobs [[buffer(6)]],
    device const uint* info [[buffer(7)]],
    constant uint& inner [[buffer(8)]],
    uint tid [[thread_index_in_threadgroup]],
    uint job [[threadgroup_position_in_grid]])
{
    if (job >= info[1]) {
        return;
    }
    uint slot = jobs[job * 2u];
    uint tile = jobs[job * 2u + 1u];
    uint n = slot_n[slot];
    for (uint r = 0u; r < 32u; ++r) {
        uint local = tile * 32u + r;
        device half* dest = pack + (ulong)(row_base[slot] + local) * inner;
        if (local < n) {
            device const float* src = source + (ulong)csr_i[slot_csr[slot] + local] * inner;
            for (uint i = tid; i < inner; i += 256u) {
                dest[i] = half(src[i]);
            }
        } else {
            for (uint i = tid; i < inner; i += 256u) {
                dest[i] = half(0.0f);
            }
        }
    }
}

// Interleaved gate/up rows to half. Columns past half_width are zero.
kernel void swiglu_grouped(
    device const float* source [[buffer(0)]],
    device half* destination [[buffer(1)]],
    device const uint* info [[buffer(2)]],
    constant uint& half_width [[buffer(3)]],
    constant uint& stride [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    uint row = tid / stride;
    uint col = tid - row * stride;
    if (row >= info[2]) {
        return;
    }
    if (col >= half_width) {
        destination[tid] = half(0.0f);
        return;
    }
    device const float* src = source + (ulong)row * half_width * 2u;
    destination[tid] = half(swiglu_f(src[col * 2u], src[col * 2u + 1u]));
}

// Weighted sum of a token's expert rows. accumulate adds to the destination.
kernel void scatter_grouped(
    device float* destination [[buffer(0)]],
    device const float* partial [[buffer(1)]],
    device const uint* pack_row [[buffer(2)]],
    device const float* route [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& tokens [[buffer(5)]],
    constant uint& accumulate [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint token [[threadgroup_position_in_grid]])
{
    if (token >= tokens) {
        return;
    }
    uint rows[8];
    float weights[8];
    for (uint k = 0u; k < 8u; ++k) {
        rows[k] = pack_row[token * 8u + k];
        weights[k] = route[token * 8u + k];
    }
    device float* dest = destination + (ulong)token * cols;
    for (uint i = tid; i < cols; i += 256u) {
        float acc = accumulate != 0u ? dest[i] : 0.0f;
        for (uint k = 0u; k < 8u; ++k) {
            if (rows[k] != 0xffffffffu) {
                acc += weights[k] * partial[(ulong)rows[k] * cols + i];
            }
        }
        dest[i] = acc;
    }
}

// ---------------------------------------------------------------------------
// Small kernels.

kernel void zero_f32(
    device float* data [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < count) {
        data[tid] = 0.0f;
    }
}

kernel void copy_words(
    device const float4* source [[buffer(0)]],
    device float4* destination [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < count) {
        destination[tid] = source[tid];
    }
}

// Float rows to half, padded rows are zero.
kernel void cast_f16(
    device const float* source [[buffer(0)]],
    device half* destination [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& padded [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= padded * cols) {
        return;
    }
    uint row = tid / cols;
    destination[tid] = row < rows ? half(source[tid]) : half(0.0f);
}

kernel void embed_gather(
    device const half* table [[buffer(0)]],
    device const uint* tokens [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant uint& hidden [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count * hidden) {
        return;
    }
    uint row = tid / hidden;
    uint col = tid - row * hidden;
    destination[tid] = float(table[(ulong)tokens[row] * hidden + col]);
}

static inline uint hash_u32(uint x) {
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

// One threadgroup owns 256 logits. Ties go to the lower index. With
// inv_temp > 0 each logit is scaled and gets Gumbel noise, so the argmax is
// one draw from softmax(logits / temperature). The uniform keeps 23 bits:
// (x + 0.5) / 2^23 is exact in float, so it never rounds to 0 or 1, where
// the noise would be infinite and an arbitrary token would win.
kernel void argmax_span(
    device const float* logits [[buffer(0)]],
    device float* best_value [[buffer(1)]],
    device uint* best_index [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant float& inv_temp [[buffer(4)]],
    constant uint& seed [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]])
{
    uint index = group * 256u + tid;
    float value = index < n ? logits[index] : -INFINITY;
    if (inv_temp > 0.0f && value > -INFINITY) {
        uint bits = hash_u32(index * 0x9e3779b9u ^ hash_u32(seed));
        float u = (float(bits >> 9u) + 0.5f) * (1.0f / 8388608.0f);
        value = value * inv_temp - log(-log(u));
    }
    uint id = index;
    for (uint offset = 1u; offset < 32u; offset <<= 1u) {
        float other = simd_shuffle_xor(value, offset);
        uint other_id = simd_shuffle_xor(id, offset);
        if (other > value || (other == value && other_id < id)) {
            value = other;
            id = other_id;
        }
    }
    threadgroup float vals[8];
    threadgroup uint ids[8];
    if (lane == 0u) {
        vals[simd] = value;
        ids[simd] = id;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float best = vals[0];
        uint best_id = ids[0];
        for (uint i = 1u; i < 8u; ++i) {
            if (vals[i] > best || (vals[i] == best && ids[i] < best_id)) {
                best = vals[i];
                best_id = ids[i];
            }
        }
        best_value[group] = best;
        best_index[group] = best_id;
    }
}

kernel void argmax_index(
    device const float* best_value [[buffer(0)]],
    device const uint* best_index [[buffer(1)]],
    device uint* chosen [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]])
{
    float value = -INFINITY;
    uint id = 0xffffffffu;
    for (uint i = tid; i < count; i += 256u) {
        float other = best_value[i];
        uint other_id = best_index[i];
        if (other > value || (other == value && other_id < id)) {
            value = other;
            id = other_id;
        }
    }
    for (uint offset = 1u; offset < 32u; offset <<= 1u) {
        float other = simd_shuffle_xor(value, offset);
        uint other_id = simd_shuffle_xor(id, offset);
        if (other > value || (other == value && other_id < id)) {
            value = other;
            id = other_id;
        }
    }
    threadgroup float vals[8];
    threadgroup uint ids[8];
    if (lane == 0u) {
        vals[simd] = value;
        ids[simd] = id;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float best = vals[0];
        uint best_id = ids[0];
        for (uint i = 1u; i < 8u; ++i) {
            if (vals[i] > best || (vals[i] == best && ids[i] < best_id)) {
                best = vals[i];
                best_id = ids[i];
            }
        }
        chosen[0] = best_id;
    }
}

// NVFP4 block-16 to Metal fp4 block-32, for a run of experts in one dispatch.
// One thread writes one K-block of 32. Columns past src_k are zeros.
// global[e] is amax / 2688 for expert e.
static uchar fp4_nibble(float value) {
    float mag = fabs(value);
    uint choice = 0u;
    choice += mag > 0.25f ? 1u : 0u;
    choice += mag > 0.75f ? 1u : 0u;
    choice += mag > 1.25f ? 1u : 0u;
    choice += mag > 1.75f ? 1u : 0u;
    choice += mag > 2.5f ? 1u : 0u;
    choice += mag > 3.5f ? 1u : 0u;
    choice += mag > 5.0f ? 1u : 0u;
    constant float* edges = E2M1;
    if (choice < 7u) {
        float edge = (edges[choice] + edges[choice + 1u]) * 0.5f;
        if (mag == edge && (choice & 1u) == 1u) {
            choice += 1u;
        }
    }
    uint sign = value < 0.0f ? 8u : 0u;
    return uchar(choice | sign);
}

kernel void pack_nvfp4_fp4(
    device const uchar* packed [[buffer(0)]],
    device const uchar* scales [[buffer(1)]],
    device const float* global [[buffer(2)]],
    device uchar* dest_packed [[buffer(3)]],
    device uchar* dest_scales [[buffer(4)]],
    constant uint& experts [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& src_k [[buffer(7)]],
    constant uint& dst_k [[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
    uint blocks = dst_k / 32u;
    uint per_expert = rows * blocks;
    if (tid >= experts * per_expert) {
        return;
    }
    uint expert = tid / per_expert;
    uint rest = tid - expert * per_expert;
    uint row = rest / blocks;
    uint block = rest - row * blocks;
    uint k0 = block * 32u;
    device const uchar* src = packed + (ulong)expert * rows * (src_k / 2u) + (ulong)row * (src_k / 2u);
    device const uchar* srcs = scales + (ulong)expert * rows * (src_k / 16u) + (ulong)row * (src_k / 16u);
    float gs = global[expert];
    float vals[32];
    float amax = 0.0f;
    for (uint i = 0u; i < 32u; ++i) {
        uint col = k0 + i;
        float value = 0.0f;
        if (col < src_k) {
            uchar byte = src[col / 2u];
            uint nibble = (col & 1u) ? (byte >> 4) : (byte & 0x0fu);
            value = E2M1_SIGNED[nibble] * E4M3_TABLE[srcs[col / 16u]] * gs;
        }
        vals[i] = value;
        amax = max(amax, fabs(value));
    }
    uchar biased = 0u;
    float factor = 1.0f;
    if (amax > 0.0f) {
        int exponent = 0;
        float mantissa = frexp(max(amax / 6.0f, 1.0e-30f), exponent);
        int power = exponent - (mantissa == 0.5f ? 1 : 0);
        int code = clamp(power + 127, 0, 254);
        biased = uchar(code);
        factor = e8m0_scale(uint(code));
    }
    ulong row_out = (ulong)expert * rows + row;
    dest_scales[row_out * blocks + block] = biased;
    device uchar* out = dest_packed + row_out * (dst_k / 2u) + block * 16u;
    for (uint pair = 0u; pair < 16u; ++pair) {
        uchar lo = fp4_nibble(vals[pair * 2u] / factor);
        uchar hi = fp4_nibble(vals[pair * 2u + 1u] / factor);
        out[pair] = lo | (hi << 4);
    }
}

// One NVFP4 expert to half [rows, cols], for the kernel check.
kernel void dequant_nvfp4(
    device const uchar* packed [[buffer(0)]],
    device const uchar* scales [[buffer(1)]],
    device half* destination [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant float& global_scale [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= rows * cols) {
        return;
    }
    uint row = tid / cols;
    uint col = tid - row * cols;
    uchar byte = packed[row * (cols / 2u) + col / 2u];
    uint nibble = (col & 1u) ? (byte >> 4) : (byte & 0x0fu);
    float scale = E4M3_TABLE[scales[row * (cols / 16u) + col / 16u]];
    destination[tid] = half(E2M1_SIGNED[nibble] * scale * global_scale);
}

// Vision tower. Rank-2 weights are E4M3 with one float scale per 32 along K.
// The scale plane is raw little-endian bytes so the read does not need alignment.

kernel void vision_dequant(
    device const uchar* codes [[buffer(0)]],
    device const uchar* scales [[buffer(1)]],
    device half* destination [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& inner [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= rows * inner) {
        return;
    }
    uint col = tid % inner;
    uint row = tid / inner;
    uint groups = inner >> 5;
    uint byte_index = (row * groups + (col >> 5)) * 4u;
    uint bits = uint(scales[byte_index]) | (uint(scales[byte_index + 1u]) << 8) |
        (uint(scales[byte_index + 2u]) << 16) | (uint(scales[byte_index + 3u]) << 24);
    destination[tid] = half(E4M3_TABLE[codes[tid]] * as_type<float>(bits));
}

kernel void vision_pad(
    device const half* source [[buffer(0)]],
    device half* destination [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& pad_rows [[buffer(4)]],
    constant uint& pad_cols [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= pad_rows * pad_cols) {
        return;
    }
    uint col = tid % pad_cols;
    uint row = tid / pad_cols;
    if (row < rows && col < cols) {
        destination[tid] = source[row * cols + col];
    } else {
        destination[tid] = half(0.0f);
    }
}

kernel void vision_bias_cast(
    device const float* source [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device half* destination [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& pad_cols [[buffer(5)]],
    constant uint& use_bias [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= rows * cols) {
        return;
    }
    uint col = tid % cols;
    uint row = tid / cols;
    float value = source[row * pad_cols + col];
    if (use_bias != 0u) {
        value += bias[col];
    }
    destination[tid] = half(value);
}

kernel void vision_add(
    device const half* left [[buffer(0)]],
    device const half* right [[buffer(1)]],
    device half* destination [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) {
        return;
    }
    destination[tid] = half(float(left[tid]) + float(right[tid]));
}

kernel void vision_ln(
    device const half* source [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device half* destination [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& hidden [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]])
{
    uint row = group.x;
    if (row >= rows) {
        return;
    }
    threadgroup float red[256];
    float sum = 0.0f;
    for (uint i = tid; i < hidden; i += 256u) {
        sum += float(source[row * hidden + i]);
    }
    red[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 128u; step > 0u; step >>= 1u) {
        if (tid < step) {
            red[tid] += red[tid + step];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = red[0] / float(hidden);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float var = 0.0f;
    for (uint i = tid; i < hidden; i += 256u) {
        float delta = float(source[row * hidden + i]) - mean;
        var += delta * delta;
    }
    red[tid] = var;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 128u; step > 0u; step >>= 1u) {
        if (tid < step) {
            red[tid] += red[tid + step];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(red[0] / float(hidden) + eps);
    for (uint i = tid; i < hidden; i += 256u) {
        float value = (float(source[row * hidden + i]) - mean) * inv;
        destination[row * hidden + i] = half(value * weight[i] + bias[i]);
    }
}

// Abramowitz and Stegun 7.1.26. The erf GELU is 0.5 * x * (1 + erf(x / sqrt(2))).
static inline float vision_erf(float value) {
    float sign = value < 0.0f ? -1.0f : 1.0f;
    float ax = fabs(value);
    float t = 1.0f / (1.0f + 0.3275911f * ax);
    float poly = fma(1.061405429f, t, -1.453152027f);
    poly = fma(poly, t, 1.421413741f);
    poly = fma(poly, t, -0.284496736f);
    poly = fma(poly, t, 0.254829592f);
    return sign * (1.0f - poly * t * exp(-ax * ax));
}

kernel void vision_gelu(
    device const half* source [[buffer(0)]],
    device half* destination [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) {
        return;
    }
    float value = float(source[tid]);
    destination[tid] = half(0.5f * value * (1.0f + vision_erf(value * 0.7071067811865476f)));
}

kernel void vision_residual(
    device const half* residual [[buffer(0)]],
    device const half* branch [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device half* destination [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& hidden [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    uint count = rows * hidden;
    if (tid >= count) {
        return;
    }
    uint col = tid % hidden;
    destination[tid] = half(float(residual[tid]) + gamma[col] * float(branch[tid]));
}

// Axial RoPE. cos and sin are width then height, each value repeated for the pair.
kernel void vision_rope(
    device half* states [[buffer(0)]],
    device const float* cos_table [[buffer(1)]],
    device const float* sin_table [[buffer(2)]],
    constant uint& seq [[buffer(3)]],
    constant uint& heads [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    uint pairs = head_dim >> 1;
    if (tid >= seq * heads * pairs) {
        return;
    }
    uint pair = tid % pairs;
    uint tmp = tid / pairs;
    uint head = tmp % heads;
    uint row = tmp / heads;
    uint base = (row * heads + head) * head_dim + pair * 2u;
    float x1 = float(states[base]);
    float x2 = float(states[base + 1u]);
    float cos_v = cos_table[row * head_dim + pair * 2u];
    float sin_v = sin_table[row * head_dim + pair * 2u];
    states[base] = half(x1 * cos_v - x2 * sin_v);
    states[base + 1u] = half(x2 * cos_v + x1 * sin_v);
}

// Bidirectional attention over the real sequence. One simdgroup owns one query row of one head.
kernel void vision_attend(
    device const half* query [[buffer(0)]],
    device const half* key [[buffer(1)]],
    device const half* value [[buffer(2)]],
    device half* destination [[buffer(3)]],
    constant uint& seq [[buffer(4)]],
    constant uint& heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant float& scale [[buffer(7)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]])
{
    uint row = group.x;
    uint head = group.y;
    if (row >= seq || head >= heads || head_dim != 64u) {
        return;
    }
    uint qbase = (row * heads + head) * head_dim;
    float max_score = -INFINITY;
    for (uint key_row = 0u; key_row < seq; ++key_row) {
        uint kbase = (key_row * heads + head) * head_dim;
        float dot = float(query[qbase + lane]) * float(key[kbase + lane]);
        dot += float(query[qbase + 32u + lane]) * float(key[kbase + 32u + lane]);
        dot = simd_sum(dot);
        max_score = max(max_score, dot * scale);
    }
    float sum = 0.0f;
    for (uint key_row = 0u; key_row < seq; ++key_row) {
        uint kbase = (key_row * heads + head) * head_dim;
        float dot = float(query[qbase + lane]) * float(key[kbase + lane]);
        dot += float(query[qbase + 32u + lane]) * float(key[kbase + 32u + lane]);
        dot = simd_sum(dot);
        sum += exp(dot * scale - max_score);
    }
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint key_row = 0u; key_row < seq; ++key_row) {
        uint kbase = (key_row * heads + head) * head_dim;
        float dot = float(query[qbase + lane]) * float(key[kbase + lane]);
        dot += float(query[qbase + 32u + lane]) * float(key[kbase + 32u + lane]);
        dot = simd_sum(dot);
        float prob = exp(dot * scale - max_score) / sum;
        acc0 += prob * float(value[kbase + lane]);
        acc1 += prob * float(value[kbase + 32u + lane]);
    }
    destination[qbase + lane] = half(acc0);
    destination[qbase + 32u + lane] = half(acc1);
}

// Drop the class token and concatenate each group of four patch rows.
kernel void vision_merge(
    device const half* source [[buffer(0)]],
    device half* destination [[buffer(1)]],
    constant uint& tokens [[buffer(2)]],
    constant uint& hidden [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    uint wide = hidden * 4u;
    if (tid >= tokens * wide) {
        return;
    }
    uint col = tid % wide;
    uint token = tid / wide;
    uint which = col / hidden;
    uint inner = col - which * hidden;
    uint src_row = 1u + token * 4u + which;
    destination[tid] = source[src_row * hidden + inner];
}

kernel void vision_scatter(
    device float* hidden [[buffer(0)]],
    device const half* bank [[buffer(1)]],
    device const int* slots [[buffer(2)]],
    constant uint& pos0 [[buffer(3)]],
    constant uint& tokens [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= tokens * width) {
        return;
    }
    uint col = tid % width;
    uint row = tid / width;
    int slot = slots[pos0 + row];
    if (slot < 0) {
        return;
    }
    hidden[tid] = float(bank[(uint)slot * width + col]);
}

// GPTQ column update with the FOEM pull (Zheng et al., AAAI 2026, Eq. 19).
// One thread owns one row. Columns in the block stay in order. fp contract
// stays off so the rank-1 update matches the host multiply-then-subtract.
#pragma clang fp contract(off)

constant float FOEM_EDGE[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
constant float FOEM_GRID[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

static uchar foem_nibble(float scaled) {
    float mag = fabs(scaled);
    uint choice = 0u;
    for (uint edge = 0u; edge < 7u; ++edge) {
        choice += FOEM_EDGE[edge] < mag ? 1u : 0u;
    }
    if (choice < 7u && mag == FOEM_EDGE[choice] && (choice & 1u) == 1u) {
        choice += 1u;
    }
    uint sign = signbit(scaled) ? 8u : 0u;
    return uchar(min(choice, 7u) | sign);
}

static void foem_scale(float amax, thread uchar& biased, thread float& scale) {
    biased = 0u;
    scale = 1.0f;
    if (amax <= 0.0f) {
        return;
    }
    int exponent = 0;
    float mantissa = frexp(max(amax / 6.0f, 1.1754944e-38f), exponent);
    int power = exponent - (mantissa == 0.5f ? 1 : 0);
    int code = clamp(power + 127, 0, 254);
    biased = uchar(code);
    scale = ldexp(1.0f, code - 127);
}

kernel void foem_block(
    device float* work [[buffer(0)]],
    device const float* original [[buffer(1)]],
    device const float* factor [[buffer(2)]],
    device const uchar* dead [[buffer(3)]],
    device uchar* nibbles [[buffer(4)]],
    device uchar* scales [[buffer(5)]],
    device float* errors [[buffer(6)]],
    device uint* flag [[buffer(7)]],
    constant uint& rows [[buffer(8)]],
    constant uint& stride [[buffer(9)]],
    constant uint& column0 [[buffer(10)]],
    constant uint& width [[buffer(11)]],
    constant float& beta [[buffer(12)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= rows) {
        return;
    }
    device float* line = work + (ulong)row * stride + column0;
    device const float* orig = original + (ulong)row * stride + column0;
    device uchar* codes = nibbles + (ulong)row * stride + column0;
    device float* residual = errors + (ulong)row * width;
    float scale = 1.0f;
    for (uint i = 0u; i < width; ++i) {
        uint column = column0 + i;
        if (dead[i]) {
            codes[i] = 0u;
            residual[i] = 0.0f;
            if (column % 32u == 0u) {
                scales[(ulong)row * (stride / 32u) + column / 32u] = 0u;
            }
            continue;
        }
        if (column % 32u == 0u) {
            float amax = 0.0f;
            for (uint k = 0u; k < 32u; ++k) {
                amax = max(amax, fabs(line[i + k]));
            }
            uchar biased = 0u;
            foem_scale(amax, biased, scale);
            scales[(ulong)row * (stride / 32u) + column / 32u] = biased;
        }
        uchar code = foem_nibble(line[i] / scale);
        codes[i] = code;
        float grid = FOEM_GRID[code & 7u];
        float restored = (code & 8u) ? -grid * scale : grid * scale;
        float diagonal = factor[(ulong)i * width + i];
        float drift = line[i] - orig[i];
        float error = ((line[i] - restored) - beta * drift) / diagonal;
        residual[i] = error;
        if (!isfinite(error)) {
            *flag = 1u;
        }
        for (uint j = i + 1u; j < width; ++j) {
            if (!dead[j]) {
                line[j] -= error * factor[(ulong)i * width + j];
            }
        }
        if (i + 1u < width && !dead[i + 1u]) {
            line[i + 1u] -= beta * (line[i + 1u] - orig[i + 1u]);
        }
    }
}

kernel void foem_tail(
    device float* work [[buffer(0)]],
    device const float* original [[buffer(1)]],
    device const float* errors [[buffer(2)]],
    device const float* coeff [[buffer(3)]],
    device uint* flag [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& stride [[buffer(6)]],
    constant uint& column0 [[buffer(7)]],
    constant uint& width [[buffer(8)]],
    constant uint& tail [[buffer(9)]],
    constant uint& pull [[buffer(10)]],
    constant float& beta [[buffer(11)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= rows * tail) {
        return;
    }
    uint col = tid % tail;
    uint row = tid / tail;
    float acc = 0.0f;
    device const float* residual = errors + (ulong)row * width;
    for (uint k = 0u; k < width; ++k) {
        acc += residual[k] * coeff[(ulong)k * tail + col];
    }
    device float* value = work + (ulong)row * stride + column0 + col;
    *value -= acc;
    if (col == 0u && pull != 0u) {
        device const float* source = original + (ulong)row * stride + column0;
        *value -= beta * (*value - *source);
    }
    if (!isfinite(*value)) {
        *flag = 1u;
    }
}
#pragma clang fp contract(on)

