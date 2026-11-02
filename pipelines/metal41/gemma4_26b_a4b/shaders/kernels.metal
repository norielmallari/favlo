#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>

using namespace metal;
using namespace mpp::tensor_ops;

using scales_plane = tensor_blockwise<tensor_plane_scales,
                                      device metal_fp8_ue8m0_format,
                                      32, 1>;

constant float GELU_K = 0.7978845608028654f;
constant float GELU_CUBE = 0.044715f;
constant float FP4_MAG[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
constant float FP4_EDGE[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};

static uchar _fp4_nibble(float unit) {
    float mag = fabs(unit);
    uint index = 0u;
    for (uint edge = 0u; edge < 7u; ++edge) {
        if (mag > FP4_EDGE[edge]) {
            index = edge + 1u;
        }
    }
    if (fabs(mag - 0.75f) <= 1.0e-5f || fabs(mag - 1.75f) <= 1.0e-5f || fabs(mag - 3.5f) <= 1.0e-5f) {
        index = min(index + 1u, 7u);
    }
    uint sign = unit < 0.0f ? 8u : 0u;
    return uchar(sign | index);
}

static float e4m3_mag(uchar code) {
    uchar finite = code & 0x7fu;
    uint exp = (finite >> 3) & 0xfu;
    uint mant = finite & 0x7u;
    if (exp == 0u) {
        return (float(mant) / 8.0f) * exp2(-6.0f);
    }
    return (1.0f + float(mant) / 8.0f) * exp2(float(exp) - 7.0f);
}

kernel void matmul_fp8(
    device half* activation [[buffer(0)]],
    device uchar* weights [[buffer(1)]],
    device uchar* scales [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    constant uint& inner [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    tensor<device half, dextents<int, 2>, tensor_inline> left(
        activation, dextents<int, 2>(int(inner), int(rows)), array<int, 2>{1, int(inner)});
    tensor<device metal_fp8_e4m3_format, dextents<int, 2>, tensor_inline, scales_plane> right(
        weights, dextents<int, 2>(int(inner), int(cols)), array<int, 2>{1, int(inner)},
        scales_plane(scales));
    tensor<device float, dextents<int, 2>, tensor_inline> out(
        destination, dextents<int, 2>(int(cols), int(rows)), array<int, 2>{1, int(cols)});
    constexpr auto desc = matmul2d_descriptor(
        8, 32, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto left_tile = left.slice(0, int(tgid.y) * 8);
    auto right_tile = right.slice(0, int(tgid.x) * 32);
    auto out_tile = out.slice(int(tgid.x) * 32, int(tgid.y) * 8);
    op.run(left_tile, right_tile, out_tile);
}

kernel void matmul_fp4(
    device half* activation [[buffer(0)]],
    device uchar* weights [[buffer(1)]],
    device uchar* scales [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    constant uint& inner [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    tensor<device half, dextents<int, 2>, tensor_inline> left(
        activation, dextents<int, 2>(int(inner), int(rows)), array<int, 2>{1, int(inner)});
    tensor<device metal_fp4_e2m1_format, dextents<int, 2>, tensor_inline, scales_plane> right(
        weights, dextents<int, 2>(int(inner), int(cols)), array<int, 2>{1, int(inner)},
        scales_plane(scales));
    tensor<device float, dextents<int, 2>, tensor_inline> out(
        destination, dextents<int, 2>(int(cols), int(rows)), array<int, 2>{1, int(cols)});
    constexpr auto desc = matmul2d_descriptor(
        8, 32, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto left_tile = left.slice(0, int(tgid.y) * 8);
    auto right_tile = right.slice(0, int(tgid.x) * 32);
    auto out_tile = out.slice(int(tgid.x) * 32, int(tgid.y) * 8);
    op.run(left_tile, right_tile, out_tile);
}

kernel void matmul_half(
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
        8, 32, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto left_tile = left.slice(0, int(tgid.y) * 8);
    auto right_tile = right.slice(0, int(tgid.x) * 32);
    auto out_tile = out.slice(int(tgid.x) * 32, int(tgid.y) * 8);
    op.run(left_tile, right_tile, out_tile);
}

// Prefill tile from the synthetic matmul2d table: 64 by 64, dynamic K,
// transpose_right, four simdgroups. Rows and columns are multiples of 64.
kernel void matmul_half_64(
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
    auto left_tile = left.slice(0, int(tgid.y) * 64);
    auto right_tile = right.slice(0, int(tgid.x) * 64);
    auto out_tile = out.slice(int(tgid.x) * 64, int(tgid.y) * 64);
    op.run(left_tile, right_tile, out_tile);
}

// Same 64×64 product as matmul_half_64, one expert per grid z.
kernel void matmul_half_64_batch(
    device half* activation [[buffer(0)]],
    device half* weights [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
    device half* act = activation + (ulong)tgid.z * rows * inner;
    device half* w = weights + (ulong)tgid.z * cols * inner;
    device float* dest = destination + (ulong)tgid.z * rows * cols;
    tensor<device half, dextents<int, 2>, tensor_inline> left(
        act, dextents<int, 2>(int(inner), int(rows)), array<int, 2>{1, int(inner)});
    tensor<device half, dextents<int, 2>, tensor_inline> right(
        w, dextents<int, 2>(int(inner), int(cols)), array<int, 2>{1, int(inner)});
    tensor<device float, dextents<int, 2>, tensor_inline> out(
        dest, dextents<int, 2>(int(cols), int(rows)), array<int, 2>{1, int(cols)});
    constexpr auto desc = matmul2d_descriptor(
        64, 64, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto left_tile = left.slice(0, int(tgid.y) * 64);
    auto right_tile = right.slice(0, int(tgid.x) * 64);
    auto out_tile = out.slice(int(tgid.x) * 64, int(tgid.y) * 64);
    op.run(left_tile, right_tile, out_tile);
}

// half × fp4 E2M1 with an E8M0 plane. 64×64 tile, one matrix.
kernel void matmul_fp4_64(
    device half* activation [[buffer(0)]],
    device uchar* weights [[buffer(1)]],
    device uchar* scales [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    constant uint& inner [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    tensor<device half, dextents<int, 2>, tensor_inline> left(
        activation, dextents<int, 2>(int(inner), int(rows)), array<int, 2>{1, int(inner)});
    tensor<device metal_fp4_e2m1_format, dextents<int, 2>, tensor_inline, scales_plane> right(
        weights, dextents<int, 2>(int(inner), int(cols)), array<int, 2>{1, int(inner)},
        scales_plane(scales));
    tensor<device float, dextents<int, 2>, tensor_inline> out(
        destination, dextents<int, 2>(int(cols), int(rows)), array<int, 2>{1, int(cols)});
    constexpr auto desc = matmul2d_descriptor(
        64, 64, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto left_tile = left.slice(0, int(tgid.y) * 64);
    auto right_tile = right.slice(0, int(tgid.x) * 64);
    auto out_tile = out.slice(int(tgid.x) * 64, int(tgid.y) * 64);
    op.run(left_tile, right_tile, out_tile);
}

// Same fp4 product, one expert per grid z. Weights and scales are expert-major.
kernel void matmul_fp4_64_batch(
    device half* activation [[buffer(0)]],
    device uchar* weights [[buffer(1)]],
    device uchar* scales [[buffer(2)]],
    device float* destination [[buffer(3)]],
    device const uint* which [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& cols [[buffer(6)]],
    constant uint& inner [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
    uint expert = which[tgid.z];
    device half* act = activation + (ulong)tgid.z * rows * inner;
    device uchar* w = weights + (ulong)expert * cols * (inner / 2u);
    device uchar* scale = scales + (ulong)expert * cols * (inner / 32u);
    device float* dest = destination + (ulong)tgid.z * rows * cols;
    tensor<device half, dextents<int, 2>, tensor_inline> left(
        act, dextents<int, 2>(int(inner), int(rows)), array<int, 2>{1, int(inner)});
    tensor<device metal_fp4_e2m1_format, dextents<int, 2>, tensor_inline, scales_plane> right(
        w, dextents<int, 2>(int(inner), int(cols)), array<int, 2>{1, int(inner)},
        scales_plane(scale));
    tensor<device float, dextents<int, 2>, tensor_inline> out(
        dest, dextents<int, 2>(int(cols), int(rows)), array<int, 2>{1, int(cols)});
    constexpr auto desc = matmul2d_descriptor(
        64, 64, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto left_tile = left.slice(0, int(tgid.y) * 64);
    auto right_tile = right.slice(0, int(tgid.x) * 64);
    auto out_tile = out.slice(int(tgid.x) * 64, int(tgid.y) * 64);
    op.run(left_tile, right_tile, out_tile);
}

// One expert image. A threadgroup returns before matmul2d when the width differs.
kernel void matmul_fp4_64_at(
    device half* activation [[buffer(0)]],
    device uchar* file [[buffer(1)]],
    device float* destination [[buffer(2)]],
    device const uint* which [[buffer(3)]],
    device const ulong* offsets [[buffer(4)]],
    device const uchar* widths [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& cols [[buffer(7)]],
    constant uint& inner [[buffer(8)]],
    constant uint& data_off [[buffer(9)]],
    constant uint& scale_off [[buffer(10)]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
    uint expert = which[tgid.z];
    if (widths[expert] != 4u) {
        return;
    }
    device uchar* image = file + offsets[expert];
    device half* act = activation + (ulong)tgid.z * rows * inner;
    device float* dest = destination + (ulong)tgid.z * rows * cols;
    tensor<device half, dextents<int, 2>, tensor_inline> left(
        act, dextents<int, 2>(int(inner), int(rows)), array<int, 2>{1, int(inner)});
    tensor<device metal_fp4_e2m1_format, dextents<int, 2>, tensor_inline, scales_plane> right(
        image + data_off, dextents<int, 2>(int(inner), int(cols)), array<int, 2>{1, int(inner)},
        scales_plane(image + scale_off));
    tensor<device float, dextents<int, 2>, tensor_inline> out(
        dest, dextents<int, 2>(int(cols), int(rows)), array<int, 2>{1, int(cols)});
    constexpr auto desc = matmul2d_descriptor(
        64, 64, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto left_tile = left.slice(0, int(tgid.y) * 64);
    auto right_tile = right.slice(0, int(tgid.x) * 64);
    auto out_tile = out.slice(int(tgid.x) * 64, int(tgid.y) * 64);
    op.run(left_tile, right_tile, out_tile);
}

kernel void matmul_fp8_64_at(
    device half* activation [[buffer(0)]],
    device uchar* file [[buffer(1)]],
    device float* destination [[buffer(2)]],
    device const uint* which [[buffer(3)]],
    device const ulong* offsets [[buffer(4)]],
    device const uchar* widths [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& cols [[buffer(7)]],
    constant uint& inner [[buffer(8)]],
    constant uint& data_off [[buffer(9)]],
    constant uint& scale_off [[buffer(10)]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
    uint expert = which[tgid.z];
    if (widths[expert] != 8u) {
        return;
    }
    device uchar* image = file + offsets[expert];
    device half* act = activation + (ulong)tgid.z * rows * inner;
    device float* dest = destination + (ulong)tgid.z * rows * cols;
    tensor<device half, dextents<int, 2>, tensor_inline> left(
        act, dextents<int, 2>(int(inner), int(rows)), array<int, 2>{1, int(inner)});
    tensor<device metal_fp8_e4m3_format, dextents<int, 2>, tensor_inline, scales_plane> right(
        image + data_off, dextents<int, 2>(int(inner), int(cols)), array<int, 2>{1, int(inner)},
        scales_plane(image + scale_off));
    tensor<device float, dextents<int, 2>, tensor_inline> out(
        dest, dextents<int, 2>(int(cols), int(rows)), array<int, 2>{1, int(cols)});
    constexpr auto desc = matmul2d_descriptor(
        64, 64, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto left_tile = left.slice(0, int(tgid.y) * 64);
    auto right_tile = right.slice(0, int(tgid.x) * 64);
    auto out_tile = out.slice(int(tgid.x) * 64, int(tgid.y) * 64);
    op.run(left_tile, right_tile, out_tile);
}

// 8×32 fp4 product, one expert per grid z. Token counts stay a multiple of 8.
kernel void matmul_fp4_batch(
    device half* activation [[buffer(0)]],
    device uchar* weights [[buffer(1)]],
    device uchar* scales [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    constant uint& inner [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
    device half* act = activation + (ulong)tgid.z * rows * inner;
    device uchar* w = weights + (ulong)tgid.z * cols * (inner / 2u);
    device uchar* scale = scales + (ulong)tgid.z * cols * (inner / 32u);
    device float* dest = destination + (ulong)tgid.z * rows * cols;
    tensor<device half, dextents<int, 2>, tensor_inline> left(
        act, dextents<int, 2>(int(inner), int(rows)), array<int, 2>{1, int(inner)});
    tensor<device metal_fp4_e2m1_format, dextents<int, 2>, tensor_inline, scales_plane> right(
        w, dextents<int, 2>(int(inner), int(cols)), array<int, 2>{1, int(inner)},
        scales_plane(scale));
    tensor<device float, dextents<int, 2>, tensor_inline> out(
        dest, dextents<int, 2>(int(cols), int(rows)), array<int, 2>{1, int(cols)});
    constexpr auto desc = matmul2d_descriptor(
        8, 32, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto left_tile = left.slice(0, int(tgid.y) * 8);
    auto right_tile = right.slice(0, int(tgid.x) * 32);
    auto out_tile = out.slice(int(tgid.x) * 32, int(tgid.y) * 8);
    op.run(left_tile, right_tile, out_tile);
}

kernel void dequant_q4_0_batch(
    device const uchar* packed [[buffer(0)]],
    device const ulong* origins [[buffer(1)]],
    device half* weights [[buffer(2)]],
    constant uint& k_pad [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint blocks = k_pad >> 5;
    uint span = rows * blocks;
    if (gid.x >= span) {
        return;
    }
    uint row = gid.x / blocks;
    uint block = gid.x - row * blocks;
    device const uchar* src = packed + origins[gid.y] + (ulong)gid.x * 18u;
    uint bits = uint(src[0]) | (uint(src[1]) << 8);
    float scale = as_type<float>(bits << 16);
    device half* base = weights + ((ulong)gid.y * rows + row) * k_pad + block * 32u;
    for (uint j = 0u; j < 16u; ++j) {
        uchar byte = src[2u + j];
        base[j] = half((float(byte & 0x0fu) - 8.0f) * scale);
        base[16u + j] = half((float(byte >> 4) - 8.0f) * scale);
    }
}

// One q4_0 block becomes one fp4 E2M1 group. The fp4 K pad may be wider;
// the caller zeros the destination so the extra groups stay at code zero.
kernel void repack_q4_fp4(
    device const uchar* packed [[buffer(0)]],
    device const ulong* origins [[buffer(1)]],
    device uchar* codes [[buffer(2)]],
    device uchar* scales [[buffer(3)]],
    constant uint& k_q4 [[buffer(4)]],
    constant uint& k_fp4 [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& extra [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint blocks = k_q4 >> 5;
    if (gid.x >= rows * blocks) {
        return;
    }
    uint row = gid.x / blocks;
    uint block = gid.x - row * blocks;
    device const uchar* src = packed + origins[gid.y] + ulong(extra) + (ulong)gid.x * 18u;
    uint bits = uint(src[0]) | (uint(src[1]) << 8);
    float qscale = as_type<float>(bits << 16);
    uchar code[16];
    float peak = 0.0f;
    for (uint j = 0u; j < 16u; ++j) {
        code[j] = src[2u + j];
        float low = fabs((float(code[j] & 0x0fu) - 8.0f) * qscale);
        float high = fabs((float(code[j] >> 4) - 8.0f) * qscale);
        peak = max(peak, max(low, high));
    }
    int exponent = 0;
    int unbiased = -127;
    if (peak > 0.0f) {
        float mantissa = frexp(peak / 6.0f, exponent);
        unbiased = mantissa == 0.5f ? exponent - 1 : exponent;
        unbiased = clamp(unbiased, -127, 127);
    }
    float inv = exp2(-float(unbiased));
    uchar packed_byte[16];
    for (uint pair = 0u; pair < 16u; ++pair) {
        uint left = pair * 2u;
        uint right = left + 1u;
        uint left_byte = left < 16u ? left : left - 16u;
        uint right_byte = right < 16u ? right : right - 16u;
        float left_n = left < 16u ? float(code[left_byte] & 0x0fu) : float(code[left_byte] >> 4);
        float right_n = right < 16u ? float(code[right_byte] & 0x0fu) : float(code[right_byte] >> 4);
        float left_v = (left_n - 8.0f) * qscale * inv;
        float right_v = (right_n - 8.0f) * qscale * inv;
        packed_byte[pair] = _fp4_nibble(left_v) | (_fp4_nibble(right_v) << 4);
    }
    uint scale_stride = k_fp4 >> 5;
    uint code_stride = k_fp4 >> 1;
    ulong base = (ulong)gid.y * rows + row;
    scales[base * scale_stride + block] = uchar(unbiased + 127);
    device uchar* dst = codes + base * code_stride + block * 16u;
    for (uint pair = 0u; pair < 16u; ++pair) {
        dst[pair] = packed_byte[pair];
    }
}

kernel void dequant_q4_0(
    device const uchar* packed [[buffer(0)]],
    device half* weights [[buffer(1)]],
    constant uint& k_pad [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint blocks = k_pad >> 5;
    if (gid >= rows * blocks) {
        return;
    }
    uint row = gid / blocks;
    uint block = gid - row * blocks;
    device const uchar* src = packed + gid * 18u;
    uint bits = uint(src[0]) | (uint(src[1]) << 8);
    float scale = as_type<float>(bits << 16);
    uint base = row * k_pad + block * 32u;
    for (uint j = 0u; j < 16u; ++j) {
        uchar byte = src[2u + j];
        weights[base + j] = half((float(byte & 0x0fu) - 8.0f) * scale);
        weights[base + 16u + j] = half((float(byte >> 4) - 8.0f) * scale);
    }
}

kernel void write_row(
    device const float* source [[buffer(0)]],
    device half* destination [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant uint& stride [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= stride) {
        return;
    }
    destination[index] = index < count ? half(source[index]) : half(0.0f);
}

// Float rows to half, row-major, stride k_pad. Rows at and after `valid` are zero
// so a matmul tile can pad the token count.
kernel void write_rows(
    device const float* source [[buffer(0)]],
    device half* destination [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant uint& stride [[buffer(3)]],
    constant uint& valid [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= stride) {
        return;
    }
    float value = 0.0f;
    if (gid.y < valid && gid.x < count) {
        value = source[(ulong)gid.y * stride + gid.x];
    }
    destination[(ulong)gid.y * stride + gid.x] = half(value);
}

kernel void rmsnorm(
    device const float* source [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint lane [[thread_index_in_threadgroup]],
    uint span [[threads_per_threadgroup]])
{
    float sum = 0.0f;
    for (uint index = lane; index < count; index += span) {
        float value = source[index];
        sum += value * value;
    }
    float grouped = simd_sum(sum);
    threadgroup float parts[32];
    if ((lane & 31u) == 0u) {
        parts[lane >> 5] = grouped;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
        float total = 0.0f;
        uint groups = span >> 5;
        for (uint index = 0u; index < groups; ++index) {
            total += parts[index];
        }
        parts[0] = rsqrt(total / float(count) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = parts[0];
    for (uint index = lane; index < count; index += span) {
        destination[index] = source[index] * weight[index] * inv;
    }
}

kernel void rmsnorm_rows(
    device const float* source [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint span [[threads_per_threadgroup]])
{
    if (row >= rows) {
        return;
    }
    device const float* in = source + row * count;
    device float* out = destination + row * count;
    float sum = 0.0f;
    for (uint index = lane; index < count; index += span) {
        float value = in[index];
        sum += value * value;
    }
    float grouped = simd_sum(sum);
    threadgroup float parts[32];
    if ((lane & 31u) == 0u) {
        parts[lane >> 5] = grouped;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
        float total = 0.0f;
        uint groups = span >> 5;
        for (uint index = 0u; index < groups; ++index) {
            total += parts[index];
        }
        parts[0] = rsqrt(total / float(count) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = parts[0];
    for (uint index = lane; index < count; index += span) {
        out[index] = in[index] * weight[index] * inv;
    }
}

kernel void rmsnorm_heads(
    device float* values [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    constant uint& dim [[buffer(2)]],
    constant float& eps [[buffer(3)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint span [[threads_per_threadgroup]])
{
    device float* row = values + head * dim;
    float sum = 0.0f;
    for (uint index = lane; index < dim; index += span) {
        float value = row[index];
        sum += value * value;
    }
    float grouped = simd_sum(sum);
    threadgroup float parts[32];
    if ((lane & 31u) == 0u) {
        parts[lane >> 5] = grouped;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
        float total = 0.0f;
        uint groups = span >> 5;
        for (uint index = 0u; index < groups; ++index) {
            total += parts[index];
        }
        parts[0] = rsqrt(total / float(dim) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = parts[0];
    for (uint index = lane; index < dim; index += span) {
        row[index] = row[index] * weight[index] * inv;
    }
}

kernel void zero_buffer(
    device uchar* destination [[buffer(0)]],
    constant uint& nbytes [[buffer(1)]],
    uint index [[thread_position_in_grid]])
{
    if (index < nbytes) {
        destination[index] = 0;
    }
}

kernel void expand_half(
    device const half* source [[buffer(0)]],
    device float* destination [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (index < count) {
        destination[index] = float(source[index]);
    }
}

struct _q4_half {
    float scale;
    ushort word[4];
};

// One half of a q4_0 block. The scale is bfloat16. The four ushorts hold eight
// code bytes. Nibbles stay in place: bits 0, 4, 8, and 12. The caller scales
// the matching activation by 1, 1/16, 1/256, and 1/4096.
static _q4_half _q4_load(device const uchar* block, uint part) {
    _q4_half out;
    uint bits = uint(block[0]) | (uint(block[1]) << 8);
    out.scale = as_type<float>(bits << 16);
    device const ushort* qs = ((device const ushort*)(block + 2)) + part * 4u;
    out.word[0] = qs[0];
    out.word[1] = qs[1];
    out.word[2] = qs[2];
    out.word[3] = qs[3];
    return out;
}

struct _q4_act {
    float sum;
    float y0[4];
    float y1[4];
    float y16[4];
    float y17[4];
};

// Scale the sixteen activations once. The four rows of a simd reuse this.
static _q4_act _q4_prep(device const float* act) {
    _q4_act out;
    out.sum = 0.0f;
    for (uint i = 0u; i < 4u; ++i) {
        uint base = i * 2u;
        float a0 = act[base];
        float a1 = act[base + 1u];
        float a16 = act[base + 16u];
        float a17 = act[base + 17u];
        out.sum += a0 + a1 + a16 + a17;
        out.y0[i] = a0;
        out.y1[i] = a1 * (1.0f / 256.0f);
        out.y16[i] = a16 * (1.0f / 16.0f);
        out.y17[i] = a17 * (1.0f / 4096.0f);
    }
    return out;
}

static float _q4_dot(_q4_half block, _q4_act act) {
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    for (uint i = 0u; i < 4u; ++i) {
        ushort word = block.word[i];
        acc0 += act.y0[i] * float(word & 0x000Fu);
        acc1 += act.y1[i] * float(word & 0x0F00u);
        acc2 += act.y16[i] * float(word & 0x00F0u);
        acc3 += act.y17[i] * float(word & 0xF000u);
    }
    return block.scale * (act.sum * -8.0f + acc0 + acc1 + acc2 + acc3);
}

// Two simdgroups, four rows each. A thread owns half of one 32-value block
// and the same half of the next block sixteen steps along K.
kernel void gemv_q4_0(
    device const uchar* packed [[buffer(0)]],
    device const float* activation [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& k_pad [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& active [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint span [[threads_per_threadgroup]],
    uint group [[threadgroup_position_in_grid]])
{
    uint row0 = (group * (span >> 5) + simd) * 4u;
    if (row0 >= rows) {
        return;
    }
    uint live = min(4u, rows - row0);
    uint blocks = k_pad >> 5;
    uint stride = blocks * 18u;
    uint step = lane >> 1;
    uint part = lane & 1u;
    device const uchar* row_ptr[4];
    for (uint row = 0u; row < 4u; ++row) {
        uint which = row0 + row;
        if (which >= rows) {
            which = row0;
        }
        row_ptr[row] = packed + (ulong)which * stride;
    }
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint block = step; block < blocks; block += 16u) {
        uint k = block << 5;
        if (k >= active) {
            continue;
        }
        _q4_act act = _q4_prep(activation + k + part * 8u);
        for (uint row = 0u; row < live; ++row) {
            acc[row] += _q4_dot(_q4_load(row_ptr[row] + block * 18u, part), act);
        }
    }
    float out0 = simd_sum(acc[0]);
    float out1 = simd_sum(acc[1]);
    float out2 = simd_sum(acc[2]);
    float out3 = simd_sum(acc[3]);
    if (lane == 0u) {
        destination[row0] = out0;
        if (live > 1u) {
            destination[row0 + 1u] = out1;
        }
        if (live > 2u) {
            destination[row0 + 2u] = out2;
        }
        if (live > 3u) {
            destination[row0 + 3u] = out3;
        }
    }
}

// The eight routed experts of one token. Each group owns four rows of one
// expert. The byte offset comes from the layer table. `extra` is the down
// projection's shift past the gate-up image. A stride of zero reuses one
// activation for every expert.
kernel void gemv_q4_0_routed(
    device const uchar* packed [[buffer(0)]],
    device const float* activation [[buffer(1)]],
    device float* destination [[buffer(2)]],
    device const ulong* offsets [[buffer(3)]],
    device const uint* expert_ids [[buffer(4)]],
    constant uint& k_pad [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& active [[buffer(7)]],
    constant uint& groups_per [[buffer(8)]],
    constant uint& extra [[buffer(9)]],
    constant uint& act_stride [[buffer(10)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint span [[threads_per_threadgroup]],
    uint group [[threadgroup_position_in_grid]])
{
    uint which = group / groups_per;
    uint local = group - which * groups_per;
    uint expert = expert_ids[which];
    ulong origin = offsets[expert] + ulong(extra);
    uint row0 = (local * (span >> 5) + simd) * 4u;
    if (row0 >= rows) {
        return;
    }
    uint live = min(4u, rows - row0);
    uint blocks = k_pad >> 5;
    uint stride = blocks * 18u;
    uint step = lane >> 1;
    uint part = lane & 1u;
    device const uchar* base = packed + origin;
    device const float* act_base = activation + (ulong)which * act_stride;
    device const uchar* row_ptr[4];
    for (uint row = 0u; row < 4u; ++row) {
        uint at = row0 + row;
        if (at >= rows) {
            at = row0;
        }
        row_ptr[row] = base + (ulong)at * stride;
    }
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint block = step; block < blocks; block += 16u) {
        uint k = block << 5;
        if (k >= active) {
            continue;
        }
        _q4_act act = _q4_prep(act_base + k + part * 8u);
        for (uint row = 0u; row < live; ++row) {
            acc[row] += _q4_dot(_q4_load(row_ptr[row] + block * 18u, part), act);
        }
    }
    float out0 = simd_sum(acc[0]);
    float out1 = simd_sum(acc[1]);
    float out2 = simd_sum(acc[2]);
    float out3 = simd_sum(acc[3]);
    if (lane == 0u) {
        uint dest = which * rows + row0;
        destination[dest] = out0;
        if (live > 1u) {
            destination[dest + 1u] = out1;
        }
        if (live > 2u) {
            destination[dest + 2u] = out2;
        }
        if (live > 3u) {
            destination[dest + 3u] = out3;
        }
    }
}

kernel void mix_experts(
    device float* destination [[buffer(0)]],
    device const float* source [[buffer(1)]],
    device const float* weights [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= width) {
        return;
    }
    float acc = 0.0f;
    for (uint slot = 0u; slot < count; ++slot) {
        acc += weights[slot] * source[(ulong)slot * width + index];
    }
    destination[index] += acc;
}

// Same dot as gemv_q4_0, applied to every token in a short batch. The weight
// half is loaded once per block and reused across those tokens.
kernel void gemm_q4_0_n(
    device const uchar* packed [[buffer(0)]],
    device const float* activation [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& k_pad [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& tokens [[buffer(5)]],
    constant uint& active [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]])
{
    uint row0 = (group * 2u + simd) * 4u;
    if (row0 >= rows || tokens == 0u) {
        return;
    }
    uint live = min(4u, rows - row0);
    uint blocks = k_pad >> 5;
    uint stride = blocks * 18u;
    uint step = lane >> 1;
    uint part = lane & 1u;
    device const uchar* row_ptr[4];
    for (uint row = 0u; row < 4u; ++row) {
        uint which = row0 + row;
        if (which >= rows) {
            which = row0;
        }
        row_ptr[row] = packed + (ulong)which * stride;
    }
    float acc[4][8];
    for (uint row = 0u; row < 4u; ++row) {
        for (uint token = 0u; token < 8u; ++token) {
            acc[row][token] = 0.0f;
        }
    }
    for (uint block = step; block < blocks; block += 16u) {
        uint k = block << 5;
        if (k >= active) {
            continue;
        }
        _q4_half weight[4];
        for (uint row = 0u; row < live; ++row) {
            weight[row] = _q4_load(row_ptr[row] + block * 18u, part);
        }
        uint ntok = min(tokens, 8u);
        for (uint token = 0u; token < ntok; ++token) {
            _q4_act act = _q4_prep(activation + (ulong)token * k_pad + k + part * 8u);
            for (uint row = 0u; row < live; ++row) {
                acc[row][token] += _q4_dot(weight[row], act);
            }
        }
    }
    uint ntok = min(tokens, 8u);
    for (uint token = 0u; token < ntok; ++token) {
        float out0 = simd_sum(acc[0][token]);
        float out1 = simd_sum(acc[1][token]);
        float out2 = simd_sum(acc[2][token]);
        float out3 = simd_sum(acc[3][token]);
        if (lane == 0u) {
            device float* dest = destination + (ulong)token * rows + row0;
            dest[0] = out0;
            if (live > 1u) {
                dest[1] = out1;
            }
            if (live > 2u) {
                dest[2] = out2;
            }
            if (live > 3u) {
                dest[3] = out3;
            }
        }
    }
}

// Sorted expert rows. One simd owns four rows and up to 32 tokens.
// Each lane keeps one accumulator per row and walks K the way gemv_q4_0 does,
// so the weight block is expanded once and reused across that lane's token.
kernel void qmm_q4_0(
    device const uchar* packed [[buffer(0)]],
    device const ulong* origins [[buffer(1)]],
    device const uint* jobs [[buffer(2)]],
    device const float* activation [[buffer(3)]],
    device float* destination [[buffer(4)]],
    constant uint& k_pad [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& active [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 gid [[threadgroup_position_in_grid]])
{
    uint slot = gid.y * 4u;
    uint ntok = jobs[slot + 2u];
    uint act_row = jobs[slot + 3u];
    uint expert = jobs[slot];
    uint row0 = gid.x * 4u;
    if (row0 >= rows || ntok == 0u) {
        return;
    }
    uint live = min(4u, rows - row0);
    uint blocks = k_pad >> 5;
    uint stride = blocks * 18u;
    device const uchar* base = packed + origins[expert];
    device const float* act = activation + (ulong)(act_row + lane) * k_pad;
    bool use = lane < ntok;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    device const uchar* row_ptr0 = base + (ulong)row0 * stride;
    device const uchar* row_ptr1 = base + (ulong)min(row0 + 1u, rows - 1u) * stride;
    device const uchar* row_ptr2 = base + (ulong)min(row0 + 2u, rows - 1u) * stride;
    device const uchar* row_ptr3 = base + (ulong)min(row0 + 3u, rows - 1u) * stride;
    for (uint block = 0u; block < blocks; ++block) {
        uint k = block << 5;
        if (k >= active) {
            break;
        }
        float partial = 0.0f;
        device const uchar* src = row_ptr0 + block * 18u;
        uint bits = uint(src[0]) | (uint(src[1]) << 8);
        float scale = as_type<float>(bits << 16);
        for (uint j = 0u; j < 16u; ++j) {
            uchar byte = src[2u + j];
            float low = use ? act[k + j] : 0.0f;
            float high = use ? act[k + 16u + j] : 0.0f;
            partial += (float(byte & 0x0fu) - 8.0f) * low;
            partial += (float(byte >> 4) - 8.0f) * high;
        }
        acc0 += partial * scale;
        if (live > 1u) {
            partial = 0.0f;
            src = row_ptr1 + block * 18u;
            bits = uint(src[0]) | (uint(src[1]) << 8);
            scale = as_type<float>(bits << 16);
            for (uint j = 0u; j < 16u; ++j) {
                uchar byte = src[2u + j];
                float low = use ? act[k + j] : 0.0f;
                float high = use ? act[k + 16u + j] : 0.0f;
                partial += (float(byte & 0x0fu) - 8.0f) * low;
                partial += (float(byte >> 4) - 8.0f) * high;
            }
            acc1 += partial * scale;
        }
        if (live > 2u) {
            partial = 0.0f;
            src = row_ptr2 + block * 18u;
            bits = uint(src[0]) | (uint(src[1]) << 8);
            scale = as_type<float>(bits << 16);
            for (uint j = 0u; j < 16u; ++j) {
                uchar byte = src[2u + j];
                float low = use ? act[k + j] : 0.0f;
                float high = use ? act[k + 16u + j] : 0.0f;
                partial += (float(byte & 0x0fu) - 8.0f) * low;
                partial += (float(byte >> 4) - 8.0f) * high;
            }
            acc2 += partial * scale;
        }
        if (live > 3u) {
            partial = 0.0f;
            src = row_ptr3 + block * 18u;
            bits = uint(src[0]) | (uint(src[1]) << 8);
            scale = as_type<float>(bits << 16);
            for (uint j = 0u; j < 16u; ++j) {
                uchar byte = src[2u + j];
                float low = use ? act[k + j] : 0.0f;
                float high = use ? act[k + 16u + j] : 0.0f;
                partial += (float(byte & 0x0fu) - 8.0f) * low;
                partial += (float(byte >> 4) - 8.0f) * high;
            }
            acc3 += partial * scale;
        }
    }
    if (use) {
        device float* dest = destination + (ulong)(act_row + lane) * rows + row0;
        dest[0] = acc0;
        if (live > 1u) {
            dest[1] = acc1;
        }
        if (live > 2u) {
            dest[2] = acc2;
        }
        if (live > 3u) {
            dest[3] = acc3;
        }
    }
}

static float _q4_lane(
    device const uchar* packed,
    uint row,
    uint blocks,
    uint block,
    uint nibble,
    bool high)
{
    device const uchar* src = packed + ((ulong)row * blocks + block) * 18u;
    uint bits = uint(src[0]) | (uint(src[1]) << 8);
    uchar code = src[2u + nibble];
    float mag = high ? float(code >> 4) : float(code & 0x0fu);
    return (mag - 8.0f) * as_type<float>(bits << 16);
}

// Four simds own 32 output rows. A K tile of 64 is dequantized once and applied
// to 32 tokens through simdgroup 8×8 products.
kernel void gemm_q4_0(
    device const uchar* packed [[buffer(0)]],
    device const float* activation [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& k_pad [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& tokens [[buffer(5)]],
    constant uint& active [[buffer(6)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    uint simd = lane >> 5;
    uint local = lane & 31u;
    uint row0 = gid.x * 32u + simd * 8u;
    uint token0 = gid.y * 32u;
    uint blocks = k_pad >> 5;
    threadgroup half weights[2048];
    threadgroup half acts[2048];
    threadgroup float tile[1024];
    simdgroup_matrix<float, 8, 8> acc0(0.0f);
    simdgroup_matrix<float, 8, 8> acc1(0.0f);
    simdgroup_matrix<float, 8, 8> acc2(0.0f);
    simdgroup_matrix<float, 8, 8> acc3(0.0f);
    for (uint block0 = 0u; block0 < blocks; block0 += 2u) {
        for (uint item = lane; item < 2048u; item += 128u) {
            uint local_token = item >> 6;
            uint k = item & 63u;
            uint token = token0 + local_token;
            uint index = (block0 << 5) + k;
            float value = 0.0f;
            if (token < tokens && index < active) {
                value = activation[token * k_pad + index];
            }
            acts[item] = half(value);
        }
        for (uint item = local; item < 512u; item += 32u) {
            uint local_row = item >> 6;
            uint k = item & 63u;
            uint row = row0 + local_row;
            uint block = block0 + (k >> 5);
            float weight = 0.0f;
            if (row < rows && block < blocks && ((block0 << 5) + k) < active) {
                weight = _q4_lane(packed, row, blocks, block, k & 15u, (k & 31u) >= 16u);
            }
            weights[k * 32u + simd * 8u + local_row] = half(weight);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (row0 < rows) {
            for (uint step = 0u; step < 8u; ++step) {
                simdgroup_matrix<half, 8, 8> right;
                simdgroup_load(right, weights + step * 256u + simd * 8u, 32);
                simdgroup_matrix<half, 8, 8> left0;
                simdgroup_matrix<half, 8, 8> left1;
                simdgroup_matrix<half, 8, 8> left2;
                simdgroup_matrix<half, 8, 8> left3;
                simdgroup_load(left0, acts + step * 8u, 64);
                simdgroup_load(left1, acts + 512u + step * 8u, 64);
                simdgroup_load(left2, acts + 1024u + step * 8u, 64);
                simdgroup_load(left3, acts + 1536u + step * 8u, 64);
                simdgroup_multiply_accumulate(acc0, left0, right, acc0);
                simdgroup_multiply_accumulate(acc1, left1, right, acc1);
                simdgroup_multiply_accumulate(acc2, left2, right, acc2);
                simdgroup_multiply_accumulate(acc3, left3, right, acc3);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row0 < rows) {
        simdgroup_store(acc0, tile + simd * 256u, 8);
        simdgroup_store(acc1, tile + simd * 256u + 64u, 8);
        simdgroup_store(acc2, tile + simd * 256u + 128u, 8);
        simdgroup_store(acc3, tile + simd * 256u + 192u, 8);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row0 < rows) {
        for (uint item = local; item < 256u; item += 32u) {
            uint sub = item >> 6;
            uint inner = item & 63u;
            uint token = token0 + sub * 8u + (inner >> 3);
            uint row = row0 + (inner & 7u);
            if (token < tokens && row < rows) {
                destination[token * rows + row] = tile[simd * 256u + item];
            }
        }
    }
}


kernel void embed_fp8(
    device const uchar* codes [[buffer(0)]],
    device const uchar* scales [[buffer(1)]],
    device float* hidden [[buffer(2)]],
    constant uint& token [[buffer(3)]],
    constant uint& k_pad [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    constant float& embed_scale [[buffer(6)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= k) {
        return;
    }
    uchar code = codes[(ulong)token * k_pad + i];
    float mag = e4m3_mag(code);
    if (code & 0x80u) {
        mag = -mag;
    }
    uint block = i >> 5;
    float scale = as_type<float>(uint(scales[(ulong)token * (k_pad >> 5) + block]) << 23);
    hidden[i] = mag * scale * embed_scale;
}

kernel void embed_q4_0(
    device const uchar* table [[buffer(0)]],
    device float* hidden [[buffer(1)]],
    constant uint& token [[buffer(2)]],
    constant uint& k_pad [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    constant float& embed_scale [[buffer(5)]],
    uint block [[thread_position_in_grid]])
{
    uint blocks = k_pad >> 5;
    if (block >= blocks) {
        return;
    }
    device const uchar* src = table + (token * blocks + block) * 18u;
    uint bits = uint(src[0]) | (uint(src[1]) << 8);
    float scale = as_type<float>(bits << 16);
    for (uint j = 0u; j < 16u; ++j) {
        uchar byte = src[2u + j];
        uint low = block * 32u + j;
        uint high = low + 16u;
        if (low < k) {
            hidden[low] = (float(byte & 0x0fu) - 8.0f) * scale * embed_scale;
        }
        if (high < k) {
            hidden[high] = (float(byte >> 4) - 8.0f) * scale * embed_scale;
        }
    }
}

kernel void rope_pairs(
    device float* queries [[buffer(0)]],
    device float* keys [[buffer(1)]],
    constant uint& query_heads [[buffer(2)]],
    constant uint& key_heads [[buffer(3)]],
    constant uint& dim [[buffer(4)]],
    constant uint& rotating [[buffer(5)]],
    constant float& theta [[buffer(6)]],
    constant uint& position [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    uint pairs = dim >> 1;
    uint query_count = query_heads * pairs;
    uint key_count = key_heads * pairs;
    device float* plane = queries;
    uint local = gid;
    if (gid >= query_count) {
        if (gid >= query_count + key_count) {
            return;
        }
        plane = keys;
        local = gid - query_count;
    }
    uint head = local / pairs;
    uint pair = local - head * pairs;
    float cosine = 1.0f;
    float sine = 0.0f;
    if (pair < rotating) {
        float fraction = (2.0f * float(pair)) / float(dim);
        float angle = float(position) / pow(theta, fraction);
        cosine = cos(angle);
        sine = sin(angle);
    }
    uint base = head * dim;
    float left = plane[base + pair];
    float right = plane[base + pairs + pair];
    plane[base + pair] = left * cosine - right * sine;
    plane[base + pairs + pair] = right * cosine + left * sine;
}

kernel void rope_rows(
    device float* queries [[buffer(0)]],
    device float* keys [[buffer(1)]],
    constant uint& query_heads [[buffer(2)]],
    constant uint& key_heads [[buffer(3)]],
    constant uint& dim [[buffer(4)]],
    constant uint& rotating [[buffer(5)]],
    constant float& theta [[buffer(6)]],
    constant uint& position [[buffer(7)]],
    constant uint& tokens [[buffer(8)]],
    uint token [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint span [[threads_per_threadgroup]])
{
    if (token >= tokens) {
        return;
    }
    uint pairs = dim >> 1;
    uint query_count = query_heads * pairs;
    uint total = query_count + key_heads * pairs;
    device float* q = queries + token * query_heads * dim;
    device float* k = keys + token * key_heads * dim;
    uint pos = position + token;
    for (uint gid = lane; gid < total; gid += span) {
        device float* plane = q;
        uint local = gid;
        if (gid >= query_count) {
            plane = k;
            local = gid - query_count;
        }
        uint head = local / pairs;
        uint pair = local - head * pairs;
        float cosine = 1.0f;
        float sine = 0.0f;
        if (pair < rotating) {
            float fraction = (2.0f * float(pair)) / float(dim);
            float angle = float(pos) / pow(theta, fraction);
            cosine = cos(angle);
            sine = sin(angle);
        }
        uint base = head * dim;
        float left = plane[base + pair];
        float right = plane[base + pairs + pair];
        plane[base + pair] = left * cosine - right * sine;
        plane[base + pairs + pair] = right * cosine + left * sine;
    }
}

kernel void attend(
    device const float* queries [[buffer(0)]],
    device const half* keys [[buffer(1)]],
    device const half* values [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& key_heads [[buffer(4)]],
    constant uint& dim [[buffer(5)]],
    constant uint& group [[buffer(6)]],
    constant uint& length [[buffer(7)]],
    constant uint& window [[buffer(8)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint span [[threads_per_threadgroup]])
{
    threadgroup float parts[8];
    threadgroup float stats[2];
    threadgroup float inv_box[1];
    uint key_head = head / group;
    float running = -INFINITY;
    float mass = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    uint start = 0u;
    if (window > 0u && length > window) {
        start = length - window;
    }
    uint query_base = head * dim;
    for (uint token = start; token < length; ++token) {
        float dot = 0.0f;
        uint key_base = (token * key_heads + key_head) * dim;
        for (uint slot = 0u; slot < 4u; ++slot) {
            uint index = lane + slot * span;
            if (index < dim) {
                dot += queries[query_base + index] * float(keys[key_base + index]);
            }
        }
        dot = simd_sum(dot);
        if ((lane & 31u) == 0u) {
            parts[lane >> 5] = dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0u) {
            float total = 0.0f;
            uint groups = span >> 5;
            for (uint index = 0u; index < groups; ++index) {
                total += parts[index];
            }
            parts[0] = total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float scaled = parts[0];
        if (lane == 0u) {
            float next = max(running, scaled);
            float fix = exp(running - next);
            float weight = exp(scaled - next);
            stats[0] = fix;
            stats[1] = weight;
            running = next;
            mass = mass * fix + weight;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float fix = stats[0];
        float weight = stats[1];
        for (uint slot = 0u; slot < 4u; ++slot) {
            uint index = lane + slot * span;
            if (index < dim) {
                float value = weight * float(values[key_base + index]);
                if (slot == 0u) {
                    acc0 = acc0 * fix + value;
                } else if (slot == 1u) {
                    acc1 = acc1 * fix + value;
                } else if (slot == 2u) {
                    acc2 = acc2 * fix + value;
                } else {
                    acc3 = acc3 * fix + value;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) {
        inv_box[0] = mass > 0.0f ? 1.0f / mass : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = inv_box[0];
    float acc[4] = {acc0, acc1, acc2, acc3};
    for (uint slot = 0u; slot < 4u; ++slot) {
        uint index = lane + slot * span;
        if (index < dim) {
            destination[query_base + index] = acc[slot] * inv;
        }
    }
}

// Every query in the chunk. One simd owns one query head and walks that
// head's keys. The score is the raw dot. A query at position pos0 + token
// attends keys [start, pos0 + token].
kernel void attend_span(
    device const float* queries [[buffer(0)]],
    device const half* keys [[buffer(1)]],
    device const half* values [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& query_heads [[buffer(4)]],
    constant uint& key_heads [[buffer(5)]],
    constant uint& dim [[buffer(6)]],
    constant uint& group [[buffer(7)]],
    constant uint& pos0 [[buffer(8)]],
    constant uint& count [[buffer(9)]],
    constant uint& window [[buffer(10)]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    uint token = tgid.x;
    uint head = tgid.y;
    if (token >= count || head >= query_heads) {
        return;
    }
    uint width = dim >> 5;
    uint length = pos0 + token + 1u;
    uint start = 0u;
    if (window > 0u && length > window) {
        start = length - window;
    }
    uint key_head = head / group;
    device const float* q = queries + ((ulong)token * query_heads + head) * dim;
    float qreg[16];
    for (uint i = 0u; i < 16u; ++i) {
        qreg[i] = i < width ? q[lane * width + i] : 0.0f;
    }
    float running = -INFINITY;
    float mass = 0.0f;
    float outv[16];
    for (uint i = 0u; i < 16u; ++i) {
        outv[i] = 0.0f;
    }
    uint kv_span = key_heads * dim;
    for (uint key = start; key < length; ++key) {
        uint base = (key * key_heads + key_head) * dim + lane * width;
        float partial = 0.0f;
        for (uint i = 0u; i < width; i += 4u) {
            half4 packed = *(device const half4*)(keys + base + i);
            partial += qreg[i] * float(packed.x);
            partial += qreg[i + 1u] * float(packed.y);
            partial += qreg[i + 2u] * float(packed.z);
            partial += qreg[i + 3u] * float(packed.w);
        }
        float score = simd_sum(partial);
        float next = max(running, score);
        float fix = exp(running - next);
        float weight = exp(score - next);
        running = next;
        mass = mass * fix + weight;
        device const half* value = values + base;
        for (uint i = 0u; i < width; i += 4u) {
            half4 packed = *(device const half4*)(value + i);
            outv[i] = outv[i] * fix + weight * float(packed.x);
            outv[i + 1u] = outv[i + 1u] * fix + weight * float(packed.y);
            outv[i + 2u] = outv[i + 2u] * fix + weight * float(packed.z);
            outv[i + 3u] = outv[i + 3u] * fix + weight * float(packed.w);
        }
    }
    float inv = mass > 0.0f ? 1.0f / mass : 0.0f;
    device float* dest = destination + ((ulong)token * query_heads + head) * dim;
    for (uint i = 0u; i < width; ++i) {
        dest[lane * width + i] = outv[i] * inv;
    }
}

// Decode attention. Simdgroups split the key range and merge one online softmax.
kernel void attend_decode(
    device const float* queries [[buffer(0)]],
    device const half* keys [[buffer(1)]],
    device const half* values [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& query_heads [[buffer(4)]],
    constant uint& key_heads [[buffer(5)]],
    constant uint& dim [[buffer(6)]],
    constant uint& group [[buffer(7)]],
    constant uint& pos0 [[buffer(8)]],
    constant uint& count [[buffer(9)]],
    constant uint& window [[buffer(10)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint simds [[simdgroups_per_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    uint token = tgid.x;
    uint head = tgid.y;
    uint width = dim >> 5;
    uint length = pos0 + token + 1u;
    uint start = 0u;
    if (window > 0u && length > window) {
        start = length - window;
    }
    threadgroup float red_max[8];
    threadgroup float red_mass[8];
    threadgroup float red_out[8][32][16];
    if (token >= count || head >= query_heads || simd >= 8u) {
        return;
    }
    uint key_head = head / group;
    device const float* q = queries + ((ulong)token * query_heads + head) * dim;
    float qreg[16];
    for (uint i = 0u; i < 16u; ++i) {
        qreg[i] = i < width ? q[lane * width + i] : 0.0f;
    }
    float running = -INFINITY;
    float mass = 0.0f;
    float outv[16];
    for (uint i = 0u; i < 16u; ++i) {
        outv[i] = 0.0f;
    }
    uint span = length - start;
    uint chunk = (span + simds - 1u) / simds;
    uint key0 = start + simd * chunk;
    uint key1 = key0 + chunk;
    if (key0 > length) {
        key0 = length;
    }
    if (key1 > length) {
        key1 = length;
    }
    for (uint key = key0; key < key1; ++key) {
        uint base = (key * key_heads + key_head) * dim + lane * width;
        float partial = 0.0f;
        for (uint i = 0u; i < width; i += 4u) {
            half4 packed = *(device const half4*)(keys + base + i);
            partial += qreg[i] * float(packed.x);
            partial += qreg[i + 1u] * float(packed.y);
            partial += qreg[i + 2u] * float(packed.z);
            partial += qreg[i + 3u] * float(packed.w);
        }
        float score = simd_sum(partial);
        float next = max(running, score);
        float fix = exp(running - next);
        float weight = exp(score - next);
        running = next;
        mass = mass * fix + weight;
        device const half* value = values + base;
        for (uint i = 0u; i < width; i += 4u) {
            half4 packed = *(device const half4*)(value + i);
            outv[i] = outv[i] * fix + weight * float(packed.x);
            outv[i + 1u] = outv[i + 1u] * fix + weight * float(packed.y);
            outv[i + 2u] = outv[i + 2u] * fix + weight * float(packed.z);
            outv[i + 3u] = outv[i + 3u] * fix + weight * float(packed.w);
        }
    }
    if (lane == 0u) {
        red_max[simd] = running;
        red_mass[simd] = mass;
    }
    for (uint i = 0u; i < 16u; ++i) {
        red_out[simd][lane][i] = outv[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd != 0u) {
        return;
    }
    float global = -INFINITY;
    for (uint part = 0u; part < simds; ++part) {
        global = max(global, red_max[part]);
    }
    float acc[16];
    for (uint i = 0u; i < 16u; ++i) {
        acc[i] = 0.0f;
    }
    float total = 0.0f;
    for (uint part = 0u; part < simds; ++part) {
        if (red_mass[part] == 0.0f) {
            continue;
        }
        float fix = exp(red_max[part] - global);
        total += red_mass[part] * fix;
        for (uint i = 0u; i < width; ++i) {
            acc[i] += red_out[part][lane][i] * fix;
        }
    }
    float inv = total > 0.0f ? 1.0f / total : 0.0f;
    device float* dest = destination + ((ulong)token * query_heads + head) * dim;
    for (uint i = 0u; i < width; ++i) {
        dest[lane * width + i] = acc[i] * inv;
    }
}

// Four consecutive queries share each key load. Head dim is 256.
kernel void attend_span4(
    device const float* queries [[buffer(0)]],
    device const half* keys [[buffer(1)]],
    device const half* values [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& query_heads [[buffer(4)]],
    constant uint& key_heads [[buffer(5)]],
    constant uint& dim [[buffer(6)]],
    constant uint& group [[buffer(7)]],
    constant uint& pos0 [[buffer(8)]],
    constant uint& count [[buffer(9)]],
    constant uint& window [[buffer(10)]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    uint token0 = tgid.x * 4u;
    uint head = tgid.y;
    uint width = dim >> 5;
    uint key_head = head / group;
    float qreg[4][8];
    float outv[4][8];
    float running[4];
    float mass[4];
    uint start_k[4];
    uint end_k[4];
    for (uint q = 0u; q < 4u; ++q) {
        running[q] = -INFINITY;
        mass[q] = 0.0f;
        for (uint i = 0u; i < 8u; ++i) {
            qreg[q][i] = 0.0f;
            outv[q][i] = 0.0f;
        }
        uint token = token0 + q;
        end_k[q] = pos0 + token + 1u;
        start_k[q] = 0u;
        if (window > 0u && end_k[q] > window) {
            start_k[q] = end_k[q] - window;
        }
        if (token < count && head < query_heads) {
            device const float* query = queries + ((ulong)token * query_heads + head) * dim;
            for (uint i = 0u; i < width && i < 8u; ++i) {
                qreg[q][i] = query[lane * width + i];
            }
        } else {
            end_k[q] = 0u;
        }
    }
    uint key_begin = start_k[0];
    uint key_end = end_k[0];
    for (uint q = 1u; q < 4u; ++q) {
        key_begin = min(key_begin, start_k[q]);
        key_end = max(key_end, end_k[q]);
    }
    for (uint key = key_begin; key < key_end; ++key) {
        float partial0 = 0.0f;
        float partial1 = 0.0f;
        float partial2 = 0.0f;
        float partial3 = 0.0f;
        uint base = (key * key_heads + key_head) * dim + lane * width;
        for (uint i = 0u; i < width && i < 8u; i += 4u) {
            half4 packed = *(device const half4*)(keys + base + i);
            partial0 += qreg[0][i] * float(packed.x);
            partial0 += qreg[0][i + 1u] * float(packed.y);
            partial0 += qreg[0][i + 2u] * float(packed.z);
            partial0 += qreg[0][i + 3u] * float(packed.w);
            partial1 += qreg[1][i] * float(packed.x);
            partial1 += qreg[1][i + 1u] * float(packed.y);
            partial1 += qreg[1][i + 2u] * float(packed.z);
            partial1 += qreg[1][i + 3u] * float(packed.w);
            partial2 += qreg[2][i] * float(packed.x);
            partial2 += qreg[2][i + 1u] * float(packed.y);
            partial2 += qreg[2][i + 2u] * float(packed.z);
            partial2 += qreg[2][i + 3u] * float(packed.w);
            partial3 += qreg[3][i] * float(packed.x);
            partial3 += qreg[3][i + 1u] * float(packed.y);
            partial3 += qreg[3][i + 2u] * float(packed.z);
            partial3 += qreg[3][i + 3u] * float(packed.w);
        }
        float score0 = simd_sum(partial0);
        float score1 = simd_sum(partial1);
        float score2 = simd_sum(partial2);
        float score3 = simd_sum(partial3);
        float score[4] = {score0, score1, score2, score3};
        for (uint q = 0u; q < 4u; ++q) {
            if (key < start_k[q] || key >= end_k[q]) {
                continue;
            }
            float next = max(running[q], score[q]);
            float fix = exp(running[q] - next);
            float weight = exp(score[q] - next);
            running[q] = next;
            mass[q] = mass[q] * fix + weight;
            device const half* value = values + base;
            for (uint i = 0u; i < width && i < 8u; i += 4u) {
                half4 packed = *(device const half4*)(value + i);
                outv[q][i] = outv[q][i] * fix + weight * float(packed.x);
                outv[q][i + 1u] = outv[q][i + 1u] * fix + weight * float(packed.y);
                outv[q][i + 2u] = outv[q][i + 2u] * fix + weight * float(packed.z);
                outv[q][i + 3u] = outv[q][i + 3u] * fix + weight * float(packed.w);
            }
        }
    }
    for (uint q = 0u; q < 4u; ++q) {
        uint token = token0 + q;
        if (token >= count || head >= query_heads || mass[q] == 0.0f) {
            continue;
        }
        float inv = 1.0f / mass[q];
        device float* dest = destination + ((ulong)token * query_heads + head) * dim;
        for (uint i = 0u; i < width && i < 8u; ++i) {
            dest[lane * width + i] = outv[q][i] * inv;
        }
    }
}

kernel void store_kv(
    device const float* source [[buffer(0)]],
    device half* cache [[buffer(1)]],
    constant uint& heads [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    constant uint& token [[buffer(4)]],
    uint index [[thread_position_in_grid]])
{
    uint count = heads * dim;
    if (index >= count) {
        return;
    }
    cache[token * count + index] = half(source[index]);
}

kernel void store_kv_rows(
    device const float* source [[buffer(0)]],
    device half* cache [[buffer(1)]],
    constant uint& heads [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    constant uint& token0 [[buffer(4)]],
    constant uint& tokens [[buffer(5)]],
    uint index [[thread_position_in_grid]])
{
    uint count = heads * dim;
    if (index >= tokens * count) {
        return;
    }
    uint token = index / count;
    uint local = index - token * count;
    cache[(token0 + token) * count + local] = half(source[index]);
}

// Block-peak E8M0 and nearest E4M3. A tie keeps the even magnitude code.
static int _fp8_unbiased(float peak) {
    if (peak <= 0.0f) {
        return -127;
    }
    int exp = 0;
    float mant = frexp(peak / 448.0f, exp);
    int unbiased = mant == 0.5f ? exp - 1 : exp;
    if (unbiased < -127) {
        unbiased = -127;
    }
    if (unbiased > 127) {
        unbiased = 127;
    }
    float limit = 448.0f * exp2(float(unbiased));
    if (peak > limit && unbiased < 127) {
        unbiased += 1;
    } else if (unbiased > -127) {
        float lower = 448.0f * exp2(float(unbiased - 1));
        if (peak <= lower) {
            unbiased -= 1;
        }
    }
    return unbiased;
}

static uchar _fp8_code(float value, float scale) {
    float unit = scale == 0.0f ? 0.0f : value / scale;
    float mag = fabs(unit);
    if (mag > 448.0f) {
        mag = 448.0f;
    }
    uint code;
    if (mag < 0.015625f) {
        float bins = rint(mag * 512.0f);
        code = bins >= 8.0f ? 0x08u : uint(bins);
    } else {
        int exp = 0;
        float mant = frexp(mag, exp);
        float significand = mant * 2.0f;
        int field = exp - 1 + 7;
        int mant_i = int(rint((significand - 1.0f) * 8.0f));
        if (mant_i == 8) {
            field += 1;
            mant_i = 0;
        }
        if (field > 15 || (field == 15 && mant_i > 6)) {
            code = 126u;
        } else {
            code = uint((field << 3) | mant_i);
        }
    }
    if (value < 0.0f) {
        code |= 0x80u;
    }
    return uchar(code);
}

// One scale block of a filled cache. ``transposed`` packs local values as [dim, count].
kernel void pack_kv_fp8(
    device const half* source [[buffer(0)]],
    device uchar* destination [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant uint& heads [[buffer(3)]],
    constant uint& dim [[buffer(4)]],
    constant uint& k_pad [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& transposed [[buffer(7)]],
    uint index [[thread_position_in_grid]])
{
    uint blocks = k_pad >> 5;
    uint span = heads * rows * blocks;
    if (index >= span || blocks == 0u) {
        return;
    }
    uint per_head = rows * blocks;
    uint head = index / per_head;
    uint rest = index - head * per_head;
    uint row = rest / blocks;
    uint block = rest - row * blocks;
    float values[32];
    float peak = 0.0f;
    uint active = transposed ? count : dim;
    for (uint lane = 0u; lane < 32u; ++lane) {
        uint col = block * 32u + lane;
        float value = 0.0f;
        if (col < active) {
            uint token = transposed ? col : row;
            uint channel = transposed ? row : col;
            ulong src = (ulong(token) * heads + head) * dim + channel;
            value = float(source[src]);
        }
        values[lane] = value;
        peak = max(peak, fabs(value));
    }
    int unbiased = _fp8_unbiased(peak);
    float scale = exp2(float(unbiased));
    ulong data_bytes = ulong(rows) * k_pad;
    ulong head_bytes = data_bytes + ulong(rows) * blocks;
    ulong base = ulong(head) * head_bytes + ulong(row) * k_pad + ulong(block) * 32u;
    for (uint lane = 0u; lane < 32u; ++lane) {
        destination[base + lane] = _fp8_code(values[lane], scale);
    }
    destination[ulong(head) * head_bytes + data_bytes + ulong(row) * blocks + block] =
        uchar(unbiased + 127);
}

kernel void unpack_kv_fp8(
    device const uchar* packed [[buffer(0)]],
    device half* cache [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant uint& heads [[buffer(3)]],
    constant uint& dim [[buffer(4)]],
    constant uint& k_pad [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& transposed [[buffer(7)]],
    uint index [[thread_position_in_grid]])
{
    uint width = heads * dim;
    if (index >= count * width || k_pad == 0u) {
        return;
    }
    uint token = index / width;
    uint rem = index - token * width;
    uint head = rem / dim;
    uint channel = rem - head * dim;
    uint row = transposed ? channel : token;
    uint col = transposed ? token : channel;
    uint blocks = k_pad >> 5;
    ulong data_bytes = ulong(rows) * k_pad;
    ulong head_bytes = data_bytes + ulong(rows) * blocks;
    ulong base = ulong(head) * head_bytes;
    uchar code = packed[base + ulong(row) * k_pad + col];
    uchar stored = packed[base + data_bytes + ulong(row) * blocks + (col >> 5)];
    float mag = e4m3_mag(code);
    float sign = (code & 0x80u) != 0u ? -1.0f : 1.0f;
    cache[index] = half(sign * mag * exp2(float(int(stored) - 127)));
}

kernel void geglu_pair(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= width) {
        return;
    }
    float value = gate[index];
    float hidden = 0.0f;
    if (value > 0.0f && fabs(value) >= 8.0f) {
        hidden = value;
    } else if (fabs(value) < 8.0f) {
        float inner = GELU_K * (value + GELU_CUBE * value * value * value);
        hidden = 0.5f * value * (1.0f + tanh(inner));
    }
    destination[index] = hidden * up[index];
}

kernel void geglu(
    device const float* fused [[buffer(0)]],
    device float* destination [[buffer(1)]],
    constant uint& width [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= width) {
        return;
    }
    float gate = fused[index];
    float up = fused[width + index];
    float hidden = 0.0f;
    if (gate > 0.0f && fabs(gate) >= 8.0f) {
        hidden = gate;
    } else if (fabs(gate) < 8.0f) {
        float inner = GELU_K * (gate + GELU_CUBE * gate * gate * gate);
        hidden = 0.5f * gate * (1.0f + tanh(inner));
    }
    destination[index] = hidden * up;
}

kernel void geglu_rows(
    device const float* fused [[buffer(0)]],
    device float* destination [[buffer(1)]],
    constant uint& width [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= width * rows) {
        return;
    }
    uint row = index / width;
    uint col = index - row * width;
    uint base = row * (width * 2u) + col;
    float gate = fused[base];
    float up = fused[base + width];
    float hidden = 0.0f;
    if (gate > 0.0f && fabs(gate) >= 8.0f) {
        hidden = gate;
    } else if (fabs(gate) < 8.0f) {
        float inner = GELU_K * (gate + GELU_CUBE * gate * gate * gate);
        hidden = 0.5f * gate * (1.0f + tanh(inner));
    }
    destination[index] = hidden * up;
}

kernel void gather_rows(
    device const float* source [[buffer(0)]],
    device float* destination [[buffer(1)]],
    device const uint* indices [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= count * width) {
        return;
    }
    uint row = index / width;
    uint col = index - row * width;
    destination[index] = source[indices[row] * width + col];
}

kernel void scatter_axpy(
    device float* destination [[buffer(0)]],
    device const float* source [[buffer(1)]],
    device const uint* indices [[buffer(2)]],
    device const float* scales [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    constant uint& count [[buffer(5)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= count * width) {
        return;
    }
    uint row = index / width;
    uint col = index - row * width;
    destination[indices[row] * width + col] += scales[row] * source[index];
}

// Token rows into a padded half slot. A sentinel index writes zeros.
kernel void gather_half_slots(
    device const float* source [[buffer(0)]],
    device half* destination [[buffer(1)]],
    device const uint* indices [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    constant uint& stride [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.y >= rows || gid.x >= stride) {
        return;
    }
    float value = 0.0f;
    uint token = indices[gid.y];
    if (gid.x < width && token != 0xFFFFFFFFu) {
        value = source[(ulong)token * width + gid.x];
    }
    destination[(ulong)gid.y * stride + gid.x] = half(value);
}

// Float rows of `count` columns into a wider half row. The tail is zero.
kernel void pad_half_rows(
    device const float* source [[buffer(0)]],
    device half* destination [[buffer(1)]],
    constant uint& src_stride [[buffer(2)]],
    constant uint& dst_stride [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.y >= rows || gid.x >= dst_stride) {
        return;
    }
    float value = 0.0f;
    if (gid.x < count) {
        value = source[(ulong)gid.y * src_stride + gid.x];
    }
    destination[(ulong)gid.y * dst_stride + gid.x] = half(value);
}

// Valid slot rows, scattered with the router scale. A sentinel token is padding.
kernel void scatter_slot_rows(
    device float* destination [[buffer(0)]],
    device const float* source [[buffer(1)]],
    device const uint* token [[buffer(2)]],
    device const float* scales [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    constant uint& count [[buffer(5)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= count * width) {
        return;
    }
    uint row = index / width;
    uint col = index - row * width;
    uint dest = token[row];
    if (dest == 0xFFFFFFFFu) {
        return;
    }
    atomic_fetch_add_explicit(
        (device atomic_float*)(destination + (ulong)dest * width + col),
        scales[row] * source[(ulong)row * width + col],
        memory_order_relaxed);
}

// Group one chunk's route into slots of 64. The grid records are indirect
// dispatches: gather, gate, geglu, pad, down, scatter. The host does not
// read the ids.
kernel void group_expert_slots(
    device const uint* ids [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device uint* slot_which [[buffer(2)]],
    device uint* slot_index [[buffer(3)]],
    device float* slot_scale [[buffer(4)]],
    device uint* grids [[buffer(5)]],
    constant uint& count [[buffer(6)]],
    constant uint& hidden [[buffer(7)]],
    constant uint& gate_cols [[buffer(8)]],
    constant uint& expert_width [[buffer(9)]],
    constant uint& down_k [[buffer(10)]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup uint histo[16][128];
    threadgroup uint slot_of[128];
    threadgroup uint place[16][128];
    threadgroup uint nslot;
    for (uint cell = tid; cell < 16u * 128u; cell += 256u) {
        histo[cell >> 7u][cell & 127u] = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint nassign = count * 8u;
    uint span = (nassign + 15u) >> 4u;
    if (tid < 16u) {
        uint begin = tid * span;
        uint end = begin + span;
        if (end > nassign) {
            end = nassign;
        }
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
        for (uint expert = 0u; expert < 128u; ++expert) {
            uint n = 0u;
            for (uint slice = 0u; slice < 16u; ++slice) {
                n += histo[slice][expert];
            }
            slot_of[expert] = cursor;
            if (n == 0u) {
                continue;
            }
            uint tiles = (n + 63u) >> 6u;
            for (uint tile = 0u; tile < tiles; ++tile) {
                slot_which[cursor + tile] = expert;
            }
            cursor += tiles;
        }
        uint slots = cursor == 0u ? 1u : cursor;
        if (cursor == 0u) {
            slot_which[0] = 0u;
        }
        nslot = slots;
        uint tiled = slots * 64u;
        device uint* gather = grids;
        gather[0] = (hidden + 255u) >> 8u;
        gather[1] = tiled;
        gather[2] = 1u;
        device uint* gate = grids + 4u;
        gate[0] = gate_cols >> 6u;
        gate[1] = 1u;
        gate[2] = slots;
        device uint* geglu = grids + 8u;
        geglu[0] = (tiled * expert_width + 255u) >> 8u;
        geglu[1] = 1u;
        geglu[2] = 1u;
        device uint* pad = grids + 12u;
        pad[0] = (down_k + 255u) >> 8u;
        pad[1] = tiled;
        pad[2] = 1u;
        device uint* down = grids + 16u;
        down[0] = hidden >> 6u;
        down[1] = 1u;
        down[2] = slots;
        device uint* scatter = grids + 20u;
        scatter[0] = (tiled * hidden + 255u) >> 8u;
        scatter[1] = 1u;
        scatter[2] = 1u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = tid; row < 256u * 64u; row += 256u) {
        slot_index[row] = 0xffffffffu;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 16u) {
        for (uint expert = 0u; expert < 128u; ++expert) {
            uint base = 0u;
            for (uint slice = 0u; slice < tid; ++slice) {
                base += histo[slice][expert];
            }
            place[tid][expert] = base;
        }
        uint begin = tid * span;
        uint end = begin + span;
        if (end > nassign) {
            end = nassign;
        }
        for (uint index = begin; index < end; ++index) {
            uint expert = ids[index];
            if (expert >= 128u) {
                continue;
            }
            uint local = place[tid][expert];
            place[tid][expert] = local + 1u;
            uint row = (slot_of[expert] + (local >> 6u)) * 64u + (local & 63u);
            slot_index[row] = index >> 3u;
            slot_scale[row] = weights[index];
        }
    }
}

// Sixteen fp4 values, eight code bytes. The low nibble is the even element.
static float _fp4_pair(uchar packed, float even, float odd) {
    float left = FP4_MAG[packed & 7u];
    float right = FP4_MAG[(packed >> 4) & 7u];
    if (packed & 8u) {
        left = -left;
    }
    if (packed & 0x80u) {
        right = -right;
    }
    return left * even + right * odd;
}

static float _fp8_dot16(device const uchar* bytes, thread float act[16], float scale) {
    float sum = 0.0f;
    for (uint i = 0u; i < 16u; ++i) {
        uchar code = bytes[i];
        float mag = e4m3_mag(code);
        if (code & 0x80u) {
            mag = -mag;
        }
        sum += mag * act[i];
    }
    return sum * scale;
}

static float _fp4_dot16(device const uchar* bytes, thread float act[16], float scale) {
    uint2 word = *((device const uint2*)bytes);
    uchar packed[8] = {
        uchar(word.x),
        uchar(word.x >> 8),
        uchar(word.x >> 16),
        uchar(word.x >> 24),
        uchar(word.y),
        uchar(word.y >> 8),
        uchar(word.y >> 16),
        uchar(word.y >> 24),
    };
    float acc = _fp4_pair(packed[0], act[0], act[1]);
    acc += _fp4_pair(packed[1], act[2], act[3]);
    acc += _fp4_pair(packed[2], act[4], act[5]);
    acc += _fp4_pair(packed[3], act[6], act[7]);
    acc += _fp4_pair(packed[4], act[8], act[9]);
    acc += _fp4_pair(packed[5], act[10], act[11]);
    acc += _fp4_pair(packed[6], act[12], act[13]);
    acc += _fp4_pair(packed[7], act[14], act[15]);
    return acc * scale;
}

// One token, eight experts, fp4 E2M1 weights. Two simdgroups own eight output
// rows of one expert. Each lane dots one half of a 32-value block and reuses
// that activation across the four rows.
kernel void gemv_fp4_routed(
    device const uchar* codes [[buffer(0)]],
    device const uchar* scales [[buffer(1)]],
    device const float* activation [[buffer(2)]],
    device float* destination [[buffer(3)]],
    device const uint* expert_ids [[buffer(4)]],
    constant uint& k_pad [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& active [[buffer(7)]],
    constant uint& act_stride [[buffer(8)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint span [[threads_per_threadgroup]],
    uint group [[threadgroup_position_in_grid]])
{
    uint rows_per = (span >> 5) * 4u;
    uint groups_per = (rows + rows_per - 1u) / rows_per;
    uint which = group / groups_per;
    uint local = group - which * groups_per;
    uint row0 = (local * (span >> 5) + simd) * 4u;
    if (row0 >= rows) {
        return;
    }
    uint live = min(4u, rows - row0);
    uint expert = expert_ids[which];
    uint blocks = k_pad >> 5;
    ulong code_stride = (ulong)(k_pad >> 1);
    ulong scale_stride = (ulong)blocks;
    device const uchar* code_base = codes + (ulong)expert * rows * code_stride;
    device const uchar* scale_base = scales + (ulong)expert * rows * scale_stride;
    device const float* act = activation;
    if (act_stride > 0u) {
        act = activation + (ulong)which * act_stride;
    }
    uint step = lane >> 1;
    uint part = lane & 1u;
    device const uchar* code_row[4];
    device const uchar* scale_row[4];
    for (uint row = 0u; row < 4u; ++row) {
        uint at = row0 + row;
        if (at >= rows) {
            at = row0;
        }
        code_row[row] = code_base + (ulong)at * code_stride;
        scale_row[row] = scale_base + (ulong)at * scale_stride;
    }
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint block = step; block < blocks; block += 16u) {
        uint k = (block << 5) + part * 16u;
        if (k >= active) {
            continue;
        }
        float sample[16];
        if (k + 16u <= active) {
            float4 a0 = *((device const float4*)(act + k));
            float4 a1 = *((device const float4*)(act + k + 4u));
            float4 a2 = *((device const float4*)(act + k + 8u));
            float4 a3 = *((device const float4*)(act + k + 12u));
            sample[0] = a0.x;
            sample[1] = a0.y;
            sample[2] = a0.z;
            sample[3] = a0.w;
            sample[4] = a1.x;
            sample[5] = a1.y;
            sample[6] = a1.z;
            sample[7] = a1.w;
            sample[8] = a2.x;
            sample[9] = a2.y;
            sample[10] = a2.z;
            sample[11] = a2.w;
            sample[12] = a3.x;
            sample[13] = a3.y;
            sample[14] = a3.z;
            sample[15] = a3.w;
        } else {
            uint remain = active - k;
            for (uint i = 0u; i < 16u; ++i) {
                sample[i] = i < remain ? act[k + i] : 0.0f;
            }
        }
        for (uint row = 0u; row < live; ++row) {
            float scale = as_type<float>(uint(scale_row[row][block]) << 23);
            acc[row] += _fp4_dot16(code_row[row] + block * 16u + part * 8u, sample, scale);
        }
    }
    float out0 = simd_sum(acc[0]);
    float out1 = simd_sum(acc[1]);
    float out2 = simd_sum(acc[2]);
    float out3 = simd_sum(acc[3]);
    if (lane == 0u) {
        uint dest = which * rows + row0;
        destination[dest] = out0;
        if (live > 1u) {
            destination[dest + 1u] = out1;
        }
        if (live > 2u) {
            destination[dest + 2u] = out2;
        }
        if (live > 3u) {
            destination[dest + 3u] = out3;
        }
    }
}

// One expert image. Width 4 is fp4. A mismatched expert writes nothing.
kernel void gemv_fp4_at(
    device const uchar* file [[buffer(0)]],
    device const float* activation [[buffer(1)]],
    device float* destination [[buffer(2)]],
    device const uint* expert_ids [[buffer(3)]],
    device const ulong* offsets [[buffer(4)]],
    device const uchar* widths [[buffer(5)]],
    constant uint& k_pad [[buffer(6)]],
    constant uint& rows [[buffer(7)]],
    constant uint& active [[buffer(8)]],
    constant uint& act_stride [[buffer(9)]],
    constant uint& data_off [[buffer(10)]],
    constant uint& scale_off [[buffer(11)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint span [[threads_per_threadgroup]],
    uint group [[threadgroup_position_in_grid]])
{
    uint rows_per = (span >> 5) * 4u;
    uint groups_per = (rows + rows_per - 1u) / rows_per;
    uint which = group / groups_per;
    uint expert = expert_ids[which];
    if (widths[expert] != 4u) {
        return;
    }
    uint local = group - which * groups_per;
    uint row0 = (local * (span >> 5) + simd) * 4u;
    if (row0 >= rows) {
        return;
    }
    uint live = min(4u, rows - row0);
    uint blocks = k_pad >> 5;
    ulong code_stride = (ulong)(k_pad >> 1);
    ulong scale_stride = (ulong)blocks;
    device const uchar* image = file + offsets[expert];
    device const uchar* code_base = image + data_off;
    device const uchar* scale_base = image + scale_off;
    device const float* act = activation;
    if (act_stride > 0u) {
        act = activation + (ulong)which * act_stride;
    }
    uint step = lane >> 1;
    uint part = lane & 1u;
    device const uchar* code_row[4];
    device const uchar* scale_row[4];
    for (uint row = 0u; row < 4u; ++row) {
        uint at = row0 + row;
        if (at >= rows) {
            at = row0;
        }
        code_row[row] = code_base + (ulong)at * code_stride;
        scale_row[row] = scale_base + (ulong)at * scale_stride;
    }
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint block = step; block < blocks; block += 16u) {
        uint k = (block << 5) + part * 16u;
        if (k >= active) {
            continue;
        }
        float sample[16];
        if (k + 16u <= active) {
            float4 a0 = *((device const float4*)(act + k));
            float4 a1 = *((device const float4*)(act + k + 4u));
            float4 a2 = *((device const float4*)(act + k + 8u));
            float4 a3 = *((device const float4*)(act + k + 12u));
            sample[0] = a0.x;
            sample[1] = a0.y;
            sample[2] = a0.z;
            sample[3] = a0.w;
            sample[4] = a1.x;
            sample[5] = a1.y;
            sample[6] = a1.z;
            sample[7] = a1.w;
            sample[8] = a2.x;
            sample[9] = a2.y;
            sample[10] = a2.z;
            sample[11] = a2.w;
            sample[12] = a3.x;
            sample[13] = a3.y;
            sample[14] = a3.z;
            sample[15] = a3.w;
        } else {
            uint remain = active - k;
            for (uint i = 0u; i < 16u; ++i) {
                sample[i] = i < remain ? act[k + i] : 0.0f;
            }
        }
        for (uint row = 0u; row < live; ++row) {
            float scale = as_type<float>(uint(scale_row[row][block]) << 23);
            acc[row] += _fp4_dot16(code_row[row] + block * 16u + part * 8u, sample, scale);
        }
    }
    float out0 = simd_sum(acc[0]);
    float out1 = simd_sum(acc[1]);
    float out2 = simd_sum(acc[2]);
    float out3 = simd_sum(acc[3]);
    if (lane == 0u) {
        uint dest = which * rows + row0;
        destination[dest] = out0;
        if (live > 1u) {
            destination[dest + 1u] = out1;
        }
        if (live > 2u) {
            destination[dest + 2u] = out2;
        }
        if (live > 3u) {
            destination[dest + 3u] = out3;
        }
    }
}

// Width 8 is fp8 E4M3. The scale plane is the same E8M0 byte per 32 values.
kernel void gemv_fp8_at(
    device const uchar* file [[buffer(0)]],
    device const float* activation [[buffer(1)]],
    device float* destination [[buffer(2)]],
    device const uint* expert_ids [[buffer(3)]],
    device const ulong* offsets [[buffer(4)]],
    device const uchar* widths [[buffer(5)]],
    constant uint& k_pad [[buffer(6)]],
    constant uint& rows [[buffer(7)]],
    constant uint& active [[buffer(8)]],
    constant uint& act_stride [[buffer(9)]],
    constant uint& data_off [[buffer(10)]],
    constant uint& scale_off [[buffer(11)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint span [[threads_per_threadgroup]],
    uint group [[threadgroup_position_in_grid]])
{
    uint rows_per = (span >> 5) * 4u;
    uint groups_per = (rows + rows_per - 1u) / rows_per;
    uint which = group / groups_per;
    uint expert = expert_ids[which];
    if (widths[expert] != 8u) {
        return;
    }
    uint local = group - which * groups_per;
    uint row0 = (local * (span >> 5) + simd) * 4u;
    if (row0 >= rows) {
        return;
    }
    uint live = min(4u, rows - row0);
    uint blocks = k_pad >> 5;
    ulong code_stride = (ulong)k_pad;
    ulong scale_stride = (ulong)blocks;
    device const uchar* image = file + offsets[expert];
    device const uchar* code_base = image + data_off;
    device const uchar* scale_base = image + scale_off;
    device const float* act = activation;
    if (act_stride > 0u) {
        act = activation + (ulong)which * act_stride;
    }
    uint step = lane >> 1;
    uint part = lane & 1u;
    device const uchar* code_row[4];
    device const uchar* scale_row[4];
    for (uint row = 0u; row < 4u; ++row) {
        uint at = row0 + row;
        if (at >= rows) {
            at = row0;
        }
        code_row[row] = code_base + (ulong)at * code_stride;
        scale_row[row] = scale_base + (ulong)at * scale_stride;
    }
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint block = step; block < blocks; block += 16u) {
        uint k = (block << 5) + part * 16u;
        if (k >= active) {
            continue;
        }
        float sample[16];
        if (k + 16u <= active) {
            float4 a0 = *((device const float4*)(act + k));
            float4 a1 = *((device const float4*)(act + k + 4u));
            float4 a2 = *((device const float4*)(act + k + 8u));
            float4 a3 = *((device const float4*)(act + k + 12u));
            sample[0] = a0.x;
            sample[1] = a0.y;
            sample[2] = a0.z;
            sample[3] = a0.w;
            sample[4] = a1.x;
            sample[5] = a1.y;
            sample[6] = a1.z;
            sample[7] = a1.w;
            sample[8] = a2.x;
            sample[9] = a2.y;
            sample[10] = a2.z;
            sample[11] = a2.w;
            sample[12] = a3.x;
            sample[13] = a3.y;
            sample[14] = a3.z;
            sample[15] = a3.w;
        } else {
            uint remain = active - k;
            for (uint i = 0u; i < 16u; ++i) {
                sample[i] = i < remain ? act[k + i] : 0.0f;
            }
        }
        for (uint row = 0u; row < live; ++row) {
            float scale = as_type<float>(uint(scale_row[row][block]) << 23);
            acc[row] += _fp8_dot16(code_row[row] + (ulong)block * 32u + (ulong)part * 16u, sample, scale);
        }
    }
    float out0 = simd_sum(acc[0]);
    float out1 = simd_sum(acc[1]);
    float out2 = simd_sum(acc[2]);
    float out3 = simd_sum(acc[3]);
    if (lane == 0u) {
        uint dest = which * rows + row0;
        destination[dest] = out0;
        if (live > 1u) {
            destination[dest + 1u] = out1;
        }
        if (live > 2u) {
            destination[dest + 2u] = out2;
        }
        if (live > 3u) {
            destination[dest + 3u] = out3;
        }
    }
}

kernel void gemv_fp8(
    device const uchar* codes [[buffer(0)]],
    device const uchar* scales [[buffer(1)]],
    device const float* activation [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& k_pad [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& active [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint span [[threads_per_threadgroup]],
    uint group [[threadgroup_position_in_grid]])
{
    uint row0 = (group * (span >> 5) + simd) * 4u;
    if (row0 >= rows) {
        return;
    }
    uint live = min(4u, rows - row0);
    uint blocks = k_pad >> 5;
    ulong code_stride = (ulong)k_pad;
    ulong scale_stride = (ulong)blocks;
    uint step = lane >> 1;
    uint part = lane & 1u;
    device const uchar* code_row[4];
    device const uchar* scale_row[4];
    for (uint row = 0u; row < 4u; ++row) {
        uint at = row0 + row;
        if (at >= rows) {
            at = row0;
        }
        code_row[row] = codes + (ulong)at * code_stride;
        scale_row[row] = scales + (ulong)at * scale_stride;
    }
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint block = step; block < blocks; block += 16u) {
        uint k = (block << 5) + part * 16u;
        if (k >= active) {
            continue;
        }
        float sample[16];
        if (k + 16u <= active) {
            float4 a0 = *((device const float4*)(activation + k));
            float4 a1 = *((device const float4*)(activation + k + 4u));
            float4 a2 = *((device const float4*)(activation + k + 8u));
            float4 a3 = *((device const float4*)(activation + k + 12u));
            sample[0] = a0.x;
            sample[1] = a0.y;
            sample[2] = a0.z;
            sample[3] = a0.w;
            sample[4] = a1.x;
            sample[5] = a1.y;
            sample[6] = a1.z;
            sample[7] = a1.w;
            sample[8] = a2.x;
            sample[9] = a2.y;
            sample[10] = a2.z;
            sample[11] = a2.w;
            sample[12] = a3.x;
            sample[13] = a3.y;
            sample[14] = a3.z;
            sample[15] = a3.w;
        } else {
            uint remain = active - k;
            for (uint i = 0u; i < 16u; ++i) {
                sample[i] = i < remain ? activation[k + i] : 0.0f;
            }
        }
        for (uint row = 0u; row < live; ++row) {
            float scale = as_type<float>(uint(scale_row[row][block]) << 23);
            acc[row] += _fp8_dot16(
                code_row[row] + (ulong)block * 32u + (ulong)part * 16u, sample, scale);
        }
    }
    float out0 = simd_sum(acc[0]);
    float out1 = simd_sum(acc[1]);
    float out2 = simd_sum(acc[2]);
    float out3 = simd_sum(acc[3]);
    if (lane == 0u) {
        destination[row0] = out0;
        if (live > 1u) {
            destination[row0 + 1u] = out1;
        }
        if (live > 2u) {
            destination[row0 + 2u] = out2;
        }
        if (live > 3u) {
            destination[row0 + 3u] = out3;
        }
    }
}

kernel void residual_scale(
    device float* hidden [[buffer(0)]],
    device const float* update [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index < count) {
        hidden[index] += scale * update[index];
    }
}

kernel void add_scaled(
    device float* hidden [[buffer(0)]],
    device const float* left [[buffer(1)]],
    device const float* right [[buffer(2)]],
    constant float& scale [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint index [[thread_position_in_grid]])
{
    if (index < count) {
        hidden[index] += scale * (left[index] + right[index]);
    }
}

kernel void scale_copy(
    device const float* source [[buffer(0)]],
    device float* destination [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index < count) {
        destination[index] = source[index] * scale;
    }
}

kernel void softcap(
    device float* values [[buffer(0)]],
    constant float& cap [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (index < count) {
        values[index] = cap * tanh(values[index] / cap);
    }
}

kernel void router_norm(
    device const float* source [[buffer(0)]],
    device float* destination [[buffer(1)]],
    constant float& gain [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint lane [[thread_index_in_threadgroup]],
    uint span [[threads_per_threadgroup]])
{
    float sum = 0.0f;
    for (uint index = lane; index < count; index += span) {
        float value = source[index];
        sum += value * value;
    }
    float grouped = simd_sum(sum);
    threadgroup float parts[32];
    if ((lane & 31u) == 0u) {
        parts[lane >> 5] = grouped;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
        float total = 0.0f;
        uint groups = span >> 5;
        for (uint index = 0u; index < groups; ++index) {
            total += parts[index];
        }
        parts[0] = rsqrt(total / float(count) + eps) * gain;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = parts[0];
    for (uint index = lane; index < count; index += span) {
        destination[index] = source[index] * inv;
    }
}

static bool better(float left, uint left_id, float right, uint right_id) {
    if (left > right) {
        return true;
    }
    if (left < right) {
        return false;
    }
    return left_id < right_id;
}

kernel void presence_penalty(
    device float* logits [[buffer(0)]],
    device const uint* seen [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant float& penalty [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index < count) {
        logits[seen[index]] -= penalty;
    }
}

kernel void router_topk(
    device const float* logits [[buffer(0)]],
    device const float* expert_scale [[buffer(1)]],
    device uint* indices [[buffer(2)]],
    device float* weights [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    constant uint& keep [[buffer(5)]],
    uint token [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    device const float* row_logits = logits + token * count;
    device uint* row_ids = indices + token * keep;
    device float* row_weights = weights + token * keep;
    threadgroup float scores[128];
    threadgroup uint alive[128];
    threadgroup float red_v[128];
    threadgroup uint red_i[128];
    threadgroup float picked[8];
    scores[lane] = lane < count ? row_logits[lane] : -INFINITY;
    alive[lane] = lane < count ? 1u : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint rank = 0u; rank < keep; ++rank) {
        red_v[lane] = alive[lane] != 0u ? scores[lane] : -INFINITY;
        red_i[lane] = lane;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 64u; stride > 0u; stride >>= 1) {
            if (lane < stride) {
                if (better(red_v[lane + stride], red_i[lane + stride], red_v[lane], red_i[lane])) {
                    red_v[lane] = red_v[lane + stride];
                    red_i[lane] = red_i[lane + stride];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0u) {
            uint winner = red_i[0];
            alive[winner] = 0u;
            row_ids[rank] = winner;
            picked[rank] = scores[winner];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) {
        float peak = picked[0];
        for (uint rank = 1u; rank < keep; ++rank) {
            peak = max(peak, picked[rank]);
        }
        float sum = 0.0f;
        float chosen[8];
        for (uint rank = 0u; rank < keep; ++rank) {
            chosen[rank] = exp(picked[rank] - peak);
            sum += chosen[rank];
        }
        float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        for (uint rank = 0u; rank < keep; ++rank) {
            uint expert = row_ids[rank];
            row_weights[rank] = chosen[rank] * inv * expert_scale[expert];
        }
    }
}

kernel void topk_groups(
    device const float* scores [[buffer(0)]],
    device const uint* ids [[buffer(1)]],
    device float* out_scores [[buffer(2)]],
    device uint* out_ids [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    constant uint& keep [[buffer(5)]],
    constant float& temperature [[buffer(6)]],
    constant uint& identity [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    threadgroup float keys[256];
    threadgroup uint vals[256];
    uint span = 256u;
    uint index = group * span + lane;
    float key = -INFINITY;
    uint id = 0u;
    if (index < count) {
        key = scores[index];
        if (temperature != 1.0f) {
            key /= temperature;
        }
        id = identity != 0u ? index : ids[index];
    }
    keys[lane] = key;
    vals[lane] = id;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width = 2u; width <= span; width <<= 1) {
        for (uint step = width >> 1; step > 0u; step >>= 1) {
            uint partner = lane ^ step;
            if (partner > lane) {
                bool partner_better = better(keys[partner], vals[partner], keys[lane], vals[lane]);
                bool lane_better = better(keys[lane], vals[lane], keys[partner], vals[partner]);
                bool swap = ((lane & width) == 0u) ? partner_better : lane_better;
                if (swap) {
                    float saved = keys[lane];
                    keys[lane] = keys[partner];
                    keys[partner] = saved;
                    uint saved_id = vals[lane];
                    vals[lane] = vals[partner];
                    vals[partner] = saved_id;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (lane < keep) {
        uint slot = group * keep + lane;
        out_scores[slot] = keys[lane];
        out_ids[slot] = vals[lane];
    }
}

kernel void sample_topk(
    device const float* scores [[buffer(0)]],
    device const uint* ids [[buffer(1)]],
    device uint* chosen [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant float& top_p [[buffer(4)]],
    constant float& draw [[buffer(5)]])
{
    float weights[64];
    uint order[64];
    float peak = scores[0];
    uint best = ids[0];
    float best_score = scores[0];
    for (uint index = 0u; index < count; ++index) {
        float score = scores[index];
        if (better(score, ids[index], best_score, best)) {
            best_score = score;
            best = ids[index];
        }
        peak = max(peak, score);
    }
    float total = 0.0f;
    for (uint index = 0u; index < count; ++index) {
        float weight = exp(scores[index] - peak);
        weights[index] = weight;
        total += weight;
        order[index] = index;
    }
    if (!(total > 0.0f)) {
        chosen[0] = best;
        return;
    }
    for (uint index = 0u; index < count; ++index) {
        weights[index] /= total;
    }
    if (top_p > 0.0f && top_p < 1.0f) {
        for (uint index = 1u; index < count; ++index) {
            uint key = order[index];
            uint cursor = index;
            while (cursor > 0u && weights[order[cursor - 1u]] < weights[key]) {
                order[cursor] = order[cursor - 1u];
                cursor -= 1u;
            }
            order[cursor] = key;
        }
        float cumulative = 0.0f;
        uint cut = count;
        for (uint index = 0u; index < count; ++index) {
            cumulative += weights[order[index]];
            if (cumulative >= top_p) {
                cut = index;
                break;
            }
        }
        for (uint index = cut + 1u; index < count; ++index) {
            weights[order[index]] = 0.0f;
        }
        float sum = 0.0f;
        for (uint index = 0u; index < count; ++index) {
            sum += weights[index];
        }
        if (sum > 0.0f) {
            for (uint index = 0u; index < count; ++index) {
                weights[index] /= sum;
            }
        }
    }
    for (uint index = 0u; index < count; ++index) {
        order[index] = index;
    }
    for (uint index = 1u; index < count; ++index) {
        uint key = order[index];
        uint cursor = index;
        while (cursor > 0u && ids[order[cursor - 1u]] > ids[key]) {
            order[cursor] = order[cursor - 1u];
            cursor -= 1u;
        }
        order[cursor] = key;
    }
    float cursor = 0.0f;
    uint pick = best;
    for (uint index = 0u; index < count; ++index) {
        uint slot = order[index];
        if (weights[slot] == 0.0f) {
            continue;
        }
        cursor += weights[slot];
        if (cursor >= draw) {
            pick = ids[slot];
            break;
        }
    }
    chosen[0] = pick;
}

// Float rows of width `count` into half rows of width `stride`. Columns at
// and after `count`, and rows at and after `valid`, are zero.
kernel void write_rows_tight(
    device const float* source [[buffer(0)]],
    device half* destination [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant uint& stride [[buffer(3)]],
    constant uint& valid [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= stride) {
        return;
    }
    float value = 0.0f;
    if (gid.y < valid && gid.x < count) {
        value = source[(ulong)gid.y * count + gid.x];
    }
    destination[(ulong)gid.y * stride + gid.x] = half(value);
}

// Per-token mean and variance, then the learned scale and bias. ε is 10⁻⁶.
kernel void vision_standardize(
    device float* rows [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    constant uint& count [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint span [[threads_per_threadgroup]])
{
    if (row >= count) {
        return;
    }
    threadgroup float parts[32];
    device float* values = rows + (ulong)row * width;
    float sum = 0.0f;
    for (uint index = lane; index < width; index += span) {
        sum += values[index];
    }
    float grouped = simd_sum(sum);
    if ((lane & 31u) == 0u) {
        parts[lane >> 5] = grouped;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
        float total = 0.0f;
        uint groups = span >> 5;
        for (uint index = 0u; index < groups; ++index) {
            total += parts[index];
        }
        parts[0] = total / float(width);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = parts[0];
    float var = 0.0f;
    for (uint index = lane; index < width; index += span) {
        float delta = values[index] - mean;
        var += delta * delta;
    }
    grouped = simd_sum(var);
    if ((lane & 31u) == 0u) {
        parts[lane >> 5] = grouped;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
        float total = 0.0f;
        uint groups = span >> 5;
        for (uint index = 0u; index < groups; ++index) {
            total += parts[index];
        }
        parts[0] = rsqrt(total / float(width) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = parts[0];
    for (uint index = lane; index < width; index += span) {
        values[index] = (values[index] - mean) * inv * scale[index] + bias[index];
    }
}

// Add axis 0 at the row and axis 1 at the column.
kernel void vision_positions(
    device float* tokens [[buffer(0)]],
    device const float* row_table [[buffer(1)]],
    device const float* col_table [[buffer(2)]],
    constant uint& grid_w [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    constant uint& count [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= width || gid.y >= count) {
        return;
    }
    uint row = gid.y / grid_w;
    uint col = gid.y - row * grid_w;
    tokens[(ulong)gid.y * width + gid.x] +=
        row_table[(ulong)row * width + gid.x] + col_table[(ulong)col * width + gid.x];
}

// Axial 2D-RoPE. The first half of the head rotates with the row, the second
// half with the column. Each half pairs its first 18 values with its second 18.
kernel void rope_axial(
    device float* queries [[buffer(0)]],
    device float* keys [[buffer(1)]],
    constant uint& heads [[buffer(2)]],
    constant uint& grid_w [[buffer(3)]],
    constant float& theta [[buffer(4)]],
    constant uint& count [[buffer(5)]],
    uint token [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint span [[threads_per_threadgroup]])
{
    if (token >= count) {
        return;
    }
    uint row = token / grid_w;
    uint col = token - row * grid_w;
    uint pairs = 18u;
    uint total = heads * pairs * 2u;
    for (uint gid = lane; gid < total; gid += span) {
        uint plane = gid / (heads * pairs);
        uint local = gid - plane * heads * pairs;
        uint head = local / pairs;
        uint pair = local - head * pairs;
        float fraction = (2.0f * float(pair)) / 36.0f;
        float angle_scale = pow(theta, fraction);
        device float* base = queries;
        if (plane != 0u) {
            base = keys;
        }
        base += ((ulong)token * heads + head) * 72u;
        float sine_y = sin(float(row) / angle_scale);
        float cosine_y = cos(float(row) / angle_scale);
        float left = base[pair];
        float right = base[18u + pair];
        base[pair] = left * cosine_y - right * sine_y;
        base[18u + pair] = right * cosine_y + left * sine_y;
        float sine_x = sin(float(col) / angle_scale);
        float cosine_x = cos(float(col) / angle_scale);
        left = base[36u + pair];
        right = base[54u + pair];
        base[36u + pair] = left * cosine_x - right * sine_x;
        base[54u + pair] = right * cosine_x + left * sine_x;
    }
}

// Non-causal attention for head dimension 72. The score is the dot times `scale`.
kernel void attend_vision(
    device const float* queries [[buffer(0)]],
    device const float* keys [[buffer(1)]],
    device const float* values [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& heads [[buffer(4)]],
    constant uint& count [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    uint token = tgid.x;
    uint head = tgid.y;
    if (token >= count || head >= heads) {
        return;
    }
    device const float* q = queries + ((ulong)token * heads + head) * 72u;
    float q0 = q[lane];
    float q1 = q[lane + 32u];
    float q2 = lane + 64u < 72u ? q[lane + 64u] : 0.0f;
    float running = -INFINITY;
    float mass = 0.0f;
    float out0 = 0.0f;
    float out1 = 0.0f;
    float out2 = 0.0f;
    for (uint key = 0u; key < count; ++key) {
        device const float* packed = keys + ((ulong)key * heads + head) * 72u;
        float partial = q0 * packed[lane] + q1 * packed[lane + 32u];
        if (lane + 64u < 72u) {
            partial += q2 * packed[lane + 64u];
        }
        float score = simd_sum(partial) * scale;
        float next = max(running, score);
        float fix = exp(running - next);
        float weight = exp(score - next);
        running = next;
        mass = mass * fix + weight;
        device const float* value = values + ((ulong)key * heads + head) * 72u;
        out0 = out0 * fix + weight * value[lane];
        out1 = out1 * fix + weight * value[lane + 32u];
        if (lane + 64u < 72u) {
            out2 = out2 * fix + weight * value[lane + 64u];
        }
    }
    float inv = mass > 0.0f ? 1.0f / mass : 0.0f;
    device float* dest = destination + ((ulong)token * heads + head) * 72u;
    dest[lane] = out0 * inv;
    dest[lane + 32u] = out1 * inv;
    if (lane + 64u < 72u) {
        dest[lane + 64u] = out2 * inv;
    }
}

// Non-overlapping 3×3 average. `grid_w` is the patch columns.
kernel void pool_patches(
    device const float* source [[buffer(0)]],
    device float* destination [[buffer(1)]],
    constant uint& grid_w [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    constant uint& soft [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.y >= soft || gid.x >= width) {
        return;
    }
    uint pooled_w = grid_w / 3u;
    uint pr = gid.y / pooled_w;
    uint pc = gid.y - pr * pooled_w;
    float sum = 0.0f;
    for (uint dy = 0u; dy < 3u; ++dy) {
        for (uint dx = 0u; dx < 3u; ++dx) {
            uint row = pr * 3u + dy;
            uint col = pc * 3u + dx;
            sum += source[((ulong)row * grid_w + col) * width + gid.x];
        }
    }
    destination[(ulong)gid.y * width + gid.x] = sum * (1.0f / 9.0f);
}

// Copy `dst_stride` columns out of a wider matmul row.
kernel void compact_rows(
    device const float* source [[buffer(0)]],
    device float* destination [[buffer(1)]],
    constant uint& src_stride [[buffer(2)]],
    constant uint& dst_stride [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.y >= rows || gid.x >= dst_stride) {
        return;
    }
    destination[(ulong)gid.y * dst_stride + gid.x] =
        source[(ulong)gid.y * src_stride + gid.x];
}
