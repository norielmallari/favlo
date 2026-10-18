#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>

using namespace metal;
using namespace mpp::tensor_ops;

using scales_plane = tensor_blockwise<tensor_plane_scales,
                                      device metal_fp8_ue8m0_format,
                                      32, 1>;

constant float INV_SQRT_128 = 0.08838834764831845f;
constant float INV_SQRT_256 = 0.0625f;
constant uint Q_HEADS = 24u;
constant uint KV_HEADS = 4u;
constant uint GROUP = 6u;
constant uint LINEAR_V = 48u;
constant uint MIX = 10240u;
constant uint Z_WIDTH = 6144u;
constant uint V_OFFSET = 4096u;
constant uint AB_STRIDE = 64u;
constant uint REPEAT = 3u;

static float sigmoid_f(float x) {
    return 1.0f / (1.0f + exp(-x));
}

static float softplus_f(float x) {
    if (x > 20.0f) {
        return x;
    }
    return log(1.0f + exp(x));
}

static float silu_f(float x) {
    return x * sigmoid_f(x);
}

constant float FP4_TABLE[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

static float fp4_e2m1(uint nibble) {
    return FP4_TABLE[nibble & 15u];
}

// One simdgroup, one output row. Inner is a multiple of 256 so a 32-bit word
// stays inside one scale block of 32 elements.
static float fp4_vec_dot(
    device const float* activation,
    device const uchar* weights,
    device const uchar* scales,
    uint row,
    uint inner,
    uint lane)
{
    uint words = inner >> 3;
    device const uint* packed = (device const uint*)(weights + row * (inner >> 1));
    device const uchar* row_scale = scales + row * (inner >> 5);
    float acc = 0.0f;
    for (uint i = lane; i < words; i += 32u) {
        uint word = packed[i];
        float factor = exp2(float(row_scale[i >> 2]) - 127.0f);
        uint base = i << 3;
        for (uint n = 0u; n < 8u; ++n) {
            uint nibble = (word >> (n * 4u)) & 0xFu;
            acc += fp4_e2m1(nibble) * factor * activation[base + n];
        }
    }
    return simd_sum(acc);
}

// Four rows share one activation. ``row0`` is a multiple of 4 and ``inner`` of 256.
static void fp4_rows4(
    device const float* activation,
    device const uchar* weights,
    device const uchar* scales,
    uint row0,
    uint inner,
    uint lane,
    thread float& a0,
    thread float& a1,
    thread float& a2,
    thread float& a3)
{
    uint words = inner >> 3;
    uint stride = inner >> 1;
    uint blocks = inner >> 5;
    device const uint* p0 = (device const uint*)(weights + row0 * stride);
    device const uint* p1 = (device const uint*)(weights + (row0 + 1u) * stride);
    device const uint* p2 = (device const uint*)(weights + (row0 + 2u) * stride);
    device const uint* p3 = (device const uint*)(weights + (row0 + 3u) * stride);
    device const uchar* s0 = scales + row0 * blocks;
    device const uchar* s1 = scales + (row0 + 1u) * blocks;
    device const uchar* s2 = scales + (row0 + 2u) * blocks;
    device const uchar* s3 = scales + (row0 + 3u) * blocks;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    for (uint i = lane; i < words; i += 32u) {
        uint w0 = p0[i];
        uint w1 = p1[i];
        uint w2 = p2[i];
        uint w3 = p3[i];
        float f0 = exp2(float(s0[i >> 2]) - 127.0f);
        float f1 = exp2(float(s1[i >> 2]) - 127.0f);
        float f2 = exp2(float(s2[i >> 2]) - 127.0f);
        float f3 = exp2(float(s3[i >> 2]) - 127.0f);
        uint base = i << 3;
        for (uint n = 0u; n < 8u; ++n) {
            float act = activation[base + n];
            uint shift = n * 4u;
            acc0 += fp4_e2m1((w0 >> shift) & 0xFu) * f0 * act;
            acc1 += fp4_e2m1((w1 >> shift) & 0xFu) * f1 * act;
            acc2 += fp4_e2m1((w2 >> shift) & 0xFu) * f2 * act;
            acc3 += fp4_e2m1((w3 >> shift) & 0xFu) * f3 * act;
        }
    }
    a0 = acc0;
    a1 = acc1;
    a2 = acc2;
    a3 = acc3;
}

// One weight row, two activations. The weight bytes are loaded once.
static void fp4_span2_row(
    device const float* act0,
    device const float* act1,
    device const uchar* weights,
    device const uchar* scales,
    uint row,
    uint inner,
    uint lane,
    thread float& out0,
    thread float& out1)
{
    uint words = inner >> 3;
    device const uint* packed = (device const uint*)(weights + row * (inner >> 1));
    device const uchar* row_scale = scales + row * (inner >> 5);
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint i = lane; i < words; i += 32u) {
        uint word = packed[i];
        float factor = exp2(float(row_scale[i >> 2]) - 127.0f);
        uint base = i << 3;
        for (uint n = 0u; n < 8u; ++n) {
            uint nibble = (word >> (n * 4u)) & 0xFu;
            acc0 += fp4_e2m1(nibble) * factor * act0[base + n];
            acc1 += fp4_e2m1(nibble) * factor * act1[base + n];
        }
    }
    out0 = simd_sum(acc0);
    out1 = simd_sum(acc1);
}

static float e4m3_fn(uint code) {
    uint sign = code & 0x80u;
    uint exponent = (code >> 3) & 0xFu;
    uint mantissa = code & 0x7u;
    float value = 0.0f;
    if (exponent == 0u) {
        value = (float(mantissa) / 8.0f) * exp2(-6.0f);
    } else if (exponent == 0xFu && mantissa == 0x7u) {
        return NAN;
    } else {
        value = (1.0f + float(mantissa) / 8.0f) * exp2(float(exponent) - 7.0f);
    }
    return sign != 0u ? -value : value;
}

kernel void prefill_fp4_n64(
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
        activation,
        dextents<int, 2>(int(inner), int(rows)),
        array<int, 2>{1, int(inner)});
    tensor<device metal_fp4_e2m1_format, dextents<int, 2>, tensor_inline, scales_plane> right(
        weights,
        dextents<int, 2>(int(inner), int(cols)),
        array<int, 2>{1, int(inner)},
        scales_plane(scales));
    tensor<device float, dextents<int, 2>, tensor_inline> out(
        destination,
        dextents<int, 2>(int(cols), int(rows)),
        array<int, 2>{1, int(cols)});
    constexpr auto desc = matmul2d_descriptor(
        64, 64, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto tile_left = left.slice(0, int(tgid.y) * 64);
    auto tile_right = right.slice(0, int(tgid.x) * 64);
    auto tile_out = out.slice(int(tgid.x) * 64, int(tgid.y) * 64);
    op.run(tile_left, tile_right, tile_out);
}

kernel void prefill_fp4_e8m0(
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
        activation,
        dextents<int, 2>(int(inner), int(rows)),
        array<int, 2>{1, int(inner)});
    tensor<device metal_fp4_e2m1_format, dextents<int, 2>, tensor_inline, scales_plane> right(
        weights,
        dextents<int, 2>(int(inner), int(cols)),
        array<int, 2>{1, int(inner)},
        scales_plane(scales));
    tensor<device float, dextents<int, 2>, tensor_inline> out(
        destination,
        dextents<int, 2>(int(cols), int(rows)),
        array<int, 2>{1, int(cols)});
    constexpr auto desc = matmul2d_descriptor(
        64, 32, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto tile_left = left.slice(0, int(tgid.y) * 64);
    auto tile_right = right.slice(0, int(tgid.x) * 32);
    auto tile_out = out.slice(int(tgid.x) * 32, int(tgid.y) * 64);
    op.run(tile_left, tile_right, tile_out);
}

kernel void prefill_fp8_n64(
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
        activation,
        dextents<int, 2>(int(inner), int(rows)),
        array<int, 2>{1, int(inner)});
    tensor<device metal_fp8_e4m3_format, dextents<int, 2>, tensor_inline, scales_plane> right(
        weights,
        dextents<int, 2>(int(inner), int(cols)),
        array<int, 2>{1, int(inner)},
        scales_plane(scales));
    tensor<device float, dextents<int, 2>, tensor_inline> out(
        destination,
        dextents<int, 2>(int(cols), int(rows)),
        array<int, 2>{1, int(cols)});
    constexpr auto desc = matmul2d_descriptor(
        64, 64, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto tile_left = left.slice(0, int(tgid.y) * 64);
    auto tile_right = right.slice(0, int(tgid.x) * 64);
    auto tile_out = out.slice(int(tgid.x) * 64, int(tgid.y) * 64);
    op.run(tile_left, tile_right, tile_out);
}

kernel void prefill_fp8_n64_stride(
    device half* activation [[buffer(0)]],
    device uchar* weights [[buffer(1)]],
    device uchar* scales [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    constant uint& inner [[buffer(6)]],
    constant uint& stride [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    tensor<device half, dextents<int, 2>, tensor_inline> left(
        activation,
        dextents<int, 2>(int(inner), int(rows)),
        array<int, 2>{1, int(inner)});
    tensor<device metal_fp8_e4m3_format, dextents<int, 2>, tensor_inline, scales_plane> right(
        weights,
        dextents<int, 2>(int(inner), int(cols)),
        array<int, 2>{1, int(stride)},
        scales_plane(scales));
    tensor<device float, dextents<int, 2>, tensor_inline> out(
        destination,
        dextents<int, 2>(int(cols), int(rows)),
        array<int, 2>{1, int(cols)});
    constexpr auto desc = matmul2d_descriptor(
        64, 64, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto tile_left = left.slice(0, int(tgid.y) * 64);
    auto tile_right = right.slice(0, int(tgid.x) * 64);
    auto tile_out = out.slice(int(tgid.x) * 64, int(tgid.y) * 64);
    op.run(tile_left, tile_right, tile_out);
}

// Eight query rows. Two of them are padding when a key/value head has six queries.
kernel void prefill_fp8_m8(
    device half* activation [[buffer(0)]],
    device uchar* weights [[buffer(1)]],
    device uchar* scales [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    constant uint& inner [[buffer(6)]],
    uint gid [[threadgroup_position_in_grid]])
{
    tensor<device half, dextents<int, 2>, tensor_inline> left(
        activation,
        dextents<int, 2>(int(inner), int(rows)),
        array<int, 2>{1, int(inner)});
    tensor<device metal_fp8_e4m3_format, dextents<int, 2>, tensor_inline, scales_plane> right(
        weights,
        dextents<int, 2>(int(inner), int(cols)),
        array<int, 2>{1, int(inner)},
        scales_plane(scales));
    tensor<device float, dextents<int, 2>, tensor_inline> out(
        destination,
        dextents<int, 2>(int(cols), int(rows)),
        array<int, 2>{1, int(cols)});
    constexpr auto desc = matmul2d_descriptor(8, 256, 256, false, true, false);
    matmul2d<desc, execution_simdgroups<2>> op;
    auto tile_left = left.slice(0, 0);
    auto tile_right = right.slice(0, int(gid) * 256);
    auto tile_out = out.slice(int(gid) * 256, 0);
    op.run(tile_left, tile_right, tile_out);
}

kernel void gemv_fp4(
    device const half* activation [[buffer(0)]],
    device const uchar* weights [[buffer(1)]],
    device const uchar* scales [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    if (row >= cols) {
        return;
    }
    uint blocks = inner >> 5;
    device const uchar* packed = weights + row * (inner >> 1);
    device const uchar* row_scale = scales + row * blocks;
    float acc = 0.0f;
    for (uint block = 0; block < blocks; ++block) {
        float factor = exp2(float(row_scale[block]) - 127.0f);
        uint element = (block << 5) + lane;
        uchar byte = packed[element >> 1];
        uint nibble = (element & 1u) ? (byte >> 4) : (byte & 0x0Fu);
        acc += fp4_e2m1(nibble) * factor * float(activation[element]);
    }
    float sum = simd_sum(acc);
    if (lane == 0u) {
        destination[row] = sum;
    }
}

kernel void gemv_fp4_vec(
    device const float* activation [[buffer(0)]],
    device const uchar* weights [[buffer(1)]],
    device const uchar* scales [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    if (row >= cols) {
        return;
    }
    float sum = fp4_vec_dot(activation, weights, scales, row, inner, lane);
    if (lane == 0u) {
        destination[row] = sum;
    }
}

// Four output rows share the activation. cols is a multiple of 4 and inner of 256.
kernel void gemv_fp4_x4(
    device const float* activation [[buffer(0)]],
    device const uchar* weights [[buffer(1)]],
    device const uchar* scales [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint group [[threadgroup_position_in_grid]])
{
    uint row0 = group << 2;
    if (row0 >= cols) {
        return;
    }
    float acc0, acc1, acc2, acc3;
    fp4_rows4(activation, weights, scales, row0, inner, lane, acc0, acc1, acc2, acc3);
    float o0 = simd_sum(acc0);
    float o1 = simd_sum(acc1);
    float o2 = simd_sum(acc2);
    float o3 = simd_sum(acc3);
    if (lane == 0u) {
        destination[row0] = o0;
        destination[row0 + 1u] = o1;
        destination[row0 + 2u] = o2;
        destination[row0 + 3u] = o3;
    }
}

// Four output rows, two activations. Each weight row is read once.
kernel void gemv_fp4_span2(
    device const float* act0 [[buffer(0)]],
    device const float* act1 [[buffer(1)]],
    device const uchar* weights [[buffer(2)]],
    device const uchar* scales [[buffer(3)]],
    device float* dest0 [[buffer(4)]],
    device float* dest1 [[buffer(5)]],
    constant uint& cols [[buffer(6)]],
    constant uint& inner [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint group [[threadgroup_position_in_grid]])
{
    uint row0 = group << 2;
    if (row0 >= cols) {
        return;
    }
    uint words = inner >> 3;
    uint stride = inner >> 1;
    uint blocks = inner >> 5;
    device const uint* p0 = (device const uint*)(weights + row0 * stride);
    device const uint* p1 = (device const uint*)(weights + (row0 + 1u) * stride);
    device const uint* p2 = (device const uint*)(weights + (row0 + 2u) * stride);
    device const uint* p3 = (device const uint*)(weights + (row0 + 3u) * stride);
    device const uchar* s0 = scales + row0 * blocks;
    device const uchar* s1 = scales + (row0 + 1u) * blocks;
    device const uchar* s2 = scales + (row0 + 2u) * blocks;
    device const uchar* s3 = scales + (row0 + 3u) * blocks;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
    for (uint i = lane; i < words; i += 32u) {
        uint w0 = p0[i];
        uint w1 = p1[i];
        uint w2 = p2[i];
        uint w3 = p3[i];
        float f0 = exp2(float(s0[i >> 2]) - 127.0f);
        float f1 = exp2(float(s1[i >> 2]) - 127.0f);
        float f2 = exp2(float(s2[i >> 2]) - 127.0f);
        float f3 = exp2(float(s3[i >> 2]) - 127.0f);
        uint base = i << 3;
        for (uint n = 0u; n < 8u; ++n) {
            float x0 = act0[base + n];
            float x1 = act1[base + n];
            uint shift = n * 4u;
            a0 += fp4_e2m1((w0 >> shift) & 0xFu) * f0 * x0;
            a1 += fp4_e2m1((w1 >> shift) & 0xFu) * f1 * x0;
            a2 += fp4_e2m1((w2 >> shift) & 0xFu) * f2 * x0;
            a3 += fp4_e2m1((w3 >> shift) & 0xFu) * f3 * x0;
            b0 += fp4_e2m1((w0 >> shift) & 0xFu) * f0 * x1;
            b1 += fp4_e2m1((w1 >> shift) & 0xFu) * f1 * x1;
            b2 += fp4_e2m1((w2 >> shift) & 0xFu) * f2 * x1;
            b3 += fp4_e2m1((w3 >> shift) & 0xFu) * f3 * x1;
        }
    }
    float o0 = simd_sum(a0);
    float o1 = simd_sum(a1);
    float o2 = simd_sum(a2);
    float o3 = simd_sum(a3);
    float p0s = simd_sum(b0);
    float p1s = simd_sum(b1);
    float p2s = simd_sum(b2);
    float p3s = simd_sum(b3);
    if (lane == 0u) {
        dest0[row0] = o0;
        dest0[row0 + 1u] = o1;
        dest0[row0 + 2u] = o2;
        dest0[row0 + 3u] = o3;
        dest1[row0] = p0s;
        dest1[row0 + 1u] = p1s;
        dest1[row0 + 2u] = p2s;
        dest1[row0 + 3u] = p3s;
    }
}

kernel void copy_f32(
    device const float* source [[buffer(0)]],
    device float* destination [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < count) {
        destination[tid] = source[tid];
    }
}

// Qwen3.5 RMSNorm: x / rms * (1 + w). The stored vector is centered at zero.
kernel void rmsnorm_f32(
    device const float* x [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[8];
    float sum = 0.0f;
    for (uint i = tid; i < width; i += 256u) {
        float value = x[i];
        sum += value * value;
    }
    float part = simd_sum(sum);
    if (lane == 0u) {
        red[simd] = part;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float total = red[0] + red[1] + red[2] + red[3] + red[4] + red[5] + red[6] + red[7];
        red[0] = rsqrt(total / float(width) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = red[0];
    for (uint i = tid; i < width; i += 256u) {
        y[i] = x[i] * inv * (1.0f + float(weight[i]));
    }
}

kernel void rmsnorm_rows(
    device const float* x [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    if (row >= rows) {
        return;
    }
    threadgroup float red[8];
    x += row * width;
    y += row * width;
    float sum = 0.0f;
    for (uint i = tid; i < width; i += 256u) {
        float value = x[i];
        sum += value * value;
    }
    float part = simd_sum(sum);
    if (lane == 0u) {
        red[simd] = part;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float total = red[0] + red[1] + red[2] + red[3] + red[4] + red[5] + red[6] + red[7];
        red[0] = rsqrt(total / float(width) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = red[0];
    for (uint i = tid; i < width; i += 256u) {
        y[i] = x[i] * inv * (1.0f + float(weight[i]));
    }
}

kernel void swiglu_f32(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* y [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    y[tid] = silu_f(gate[tid]) * up[tid];
}

kernel void residual_rows(
    device const float* left [[buffer(0)]],
    device const float* right [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < count) {
        destination[tid] = left[tid] + right[tid];
    }
}

kernel void concat_pair(
    device const float* left [[buffer(0)]],
    device const float* right [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    if (row >= rows) {
        return;
    }
    for (uint col = tid; col < width; col += 256u) {
        destination[row * width * 2u + col] = left[row * width + col];
        destination[row * width * 2u + width + col] = right[row * width + col];
    }
}

kernel void cast_f32_f16(
    device const float* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    dst[tid] = half(src[tid]);
}

kernel void zero_f32(
    device float* data [[buffer(0)]],
    uint tid [[thread_position_in_grid]])
{
    data[tid] = 0.0f;
}

kernel void embed_fp4_rows(
    device const uchar* weights [[buffer(0)]],
    device const uchar* scales [[buffer(1)]],
    device const uint* tokens [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    if (row >= count) {
        return;
    }
    uint blocks = inner >> 5;
    device const uchar* packed = weights + tokens[row] * (inner >> 1);
    device const uchar* row_scale = scales + tokens[row] * blocks;
    device float* dest = destination + row * inner;
    for (uint block = 0u; block < blocks; ++block) {
        float factor = exp2(float(row_scale[block]) - 127.0f);
        uint element = (block << 5) + lane;
        uchar byte = packed[element >> 1];
        uint nibble = (element & 1u) ? (byte >> 4) : (byte & 0x0Fu);
        dest[element] = fp4_e2m1(nibble) * factor;
    }
}

kernel void gather_fp4_row(
    device const uchar* weights [[buffer(0)]],
    device const uchar* scales [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& row [[buffer(3)]],
    constant uint& inner [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint blocks = inner >> 5;
    device const uchar* packed = weights + row * (inner >> 1);
    device const uchar* row_scale = scales + row * blocks;
    for (uint block = 0u; block < blocks; ++block) {
        float factor = exp2(float(row_scale[block]) - 127.0f);
        uint element = (block << 5) + lane;
        uchar byte = packed[element >> 1];
        uint nibble = (element & 1u) ? (byte >> 4) : (byte & 0x0Fu);
        destination[element] = fp4_e2m1(nibble) * factor;
    }
}

// qkv, z, a, and b of one Gated DeltaNet layer. One launch.
kernel void gdn_in_proj(
    device const float* activation [[buffer(0)]],
    device const uchar* qkv_w [[buffer(1)]],
    device const uchar* qkv_s [[buffer(2)]],
    device const uchar* z_w [[buffer(3)]],
    device const uchar* z_s [[buffer(4)]],
    device const uchar* a_w [[buffer(5)]],
    device const uchar* a_s [[buffer(6)]],
    device const uchar* b_w [[buffer(7)]],
    device const uchar* b_s [[buffer(8)]],
    device float* qkv [[buffer(9)]],
    device float* z [[buffer(10)]],
    device float* a [[buffer(11)]],
    device float* b [[buffer(12)]],
    uint lane [[thread_index_in_simdgroup]],
    uint job [[threadgroup_position_in_grid]])
{
    if (job < MIX) {
        float sum = fp4_vec_dot(activation, qkv_w, qkv_s, job, 5120u, lane);
        if (lane == 0u) {
            qkv[job] = sum;
        }
        return;
    }
    if (job < MIX + Z_WIDTH) {
        uint row = job - MIX;
        float sum = fp4_vec_dot(activation, z_w, z_s, row, 5120u, lane);
        if (lane == 0u) {
            z[row] = sum;
        }
        return;
    }
    if (job < MIX + Z_WIDTH + AB_STRIDE) {
        uint row = job - MIX - Z_WIDTH;
        float sum = fp4_vec_dot(activation, a_w, a_s, row, 5120u, lane);
        if (lane == 0u) {
            a[row] = sum;
        }
        return;
    }
    uint row = job - MIX - Z_WIDTH - AB_STRIDE;
    if (row < AB_STRIDE) {
        float sum = fp4_vec_dot(activation, b_w, b_s, row, 5120u, lane);
        if (lane == 0u) {
            b[row] = sum;
        }
    }
}

// Query, key, and value of one full-attention layer.
kernel void attn_qkv(
    device const float* activation [[buffer(0)]],
    device const uchar* q_w [[buffer(1)]],
    device const uchar* q_s [[buffer(2)]],
    device const uchar* k_w [[buffer(3)]],
    device const uchar* k_s [[buffer(4)]],
    device const uchar* v_w [[buffer(5)]],
    device const uchar* v_s [[buffer(6)]],
    device float* query [[buffer(7)]],
    device float* key [[buffer(8)]],
    device float* value [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]],
    uint job [[threadgroup_position_in_grid]])
{
    if (job < 12288u) {
        float sum = fp4_vec_dot(activation, q_w, q_s, job, 5120u, lane);
        if (lane == 0u) {
            query[job] = sum;
        }
        return;
    }
    if (job < 13312u) {
        uint row = job - 12288u;
        float sum = fp4_vec_dot(activation, k_w, k_s, row, 5120u, lane);
        if (lane == 0u) {
            key[row] = sum;
        }
        return;
    }
    if (job < 14336u) {
        uint row = job - 13312u;
        float sum = fp4_vec_dot(activation, v_w, v_s, row, 5120u, lane);
        if (lane == 0u) {
            value[row] = sum;
        }
    }
}

// Gate and up of one feed-forward. The two planes share the activation.
kernel void ffn_gate_up(
    device const float* activation [[buffer(0)]],
    device const uchar* gate_w [[buffer(1)]],
    device const uchar* gate_s [[buffer(2)]],
    device const uchar* up_w [[buffer(3)]],
    device const uchar* up_s [[buffer(4)]],
    device float* gate [[buffer(5)]],
    device float* up [[buffer(6)]],
    constant uint& rows [[buffer(7)]],
    constant uint& inner [[buffer(8)]],
    uint lane [[thread_index_in_simdgroup]],
    uint job [[threadgroup_position_in_grid]])
{
    if (job < rows) {
        float sum = fp4_vec_dot(activation, gate_w, gate_s, job, inner, lane);
        if (lane == 0u) {
            gate[job] = sum;
        }
        return;
    }
    uint row = job - rows;
    if (row < rows) {
        float sum = fp4_vec_dot(activation, up_w, up_s, row, inner, lane);
        if (lane == 0u) {
            up[row] = sum;
        }
    }
}

// Same projections as gdn_in_proj. Two activations, one read of each weight row.
kernel void gdn_in_proj_span2(
    device const float* act0 [[buffer(0)]],
    device const float* act1 [[buffer(1)]],
    device const uchar* qkv_w [[buffer(2)]],
    device const uchar* qkv_s [[buffer(3)]],
    device const uchar* z_w [[buffer(4)]],
    device const uchar* z_s [[buffer(5)]],
    device const uchar* a_w [[buffer(6)]],
    device const uchar* a_s [[buffer(7)]],
    device const uchar* b_w [[buffer(8)]],
    device const uchar* b_s [[buffer(9)]],
    device float* qkv [[buffer(10)]],
    device float* z [[buffer(11)]],
    device float* a [[buffer(12)]],
    device float* b [[buffer(13)]],
    uint lane [[thread_index_in_simdgroup]],
    uint job [[threadgroup_position_in_grid]])
{
    float s0, s1;
    if (job < MIX) {
        fp4_span2_row(act0, act1, qkv_w, qkv_s, job, 5120u, lane, s0, s1);
        if (lane == 0u) {
            qkv[job] = s0;
            qkv[MIX + job] = s1;
        }
        return;
    }
    if (job < MIX + Z_WIDTH) {
        uint row = job - MIX;
        fp4_span2_row(act0, act1, z_w, z_s, row, 5120u, lane, s0, s1);
        if (lane == 0u) {
            z[row] = s0;
            z[Z_WIDTH + row] = s1;
        }
        return;
    }
    if (job < MIX + Z_WIDTH + AB_STRIDE) {
        uint row = job - MIX - Z_WIDTH;
        fp4_span2_row(act0, act1, a_w, a_s, row, 5120u, lane, s0, s1);
        if (lane == 0u) {
            a[row] = s0;
            a[AB_STRIDE + row] = s1;
        }
        return;
    }
    uint row = job - MIX - Z_WIDTH - AB_STRIDE;
    if (row < AB_STRIDE) {
        fp4_span2_row(act0, act1, b_w, b_s, row, 5120u, lane, s0, s1);
        if (lane == 0u) {
            b[row] = s0;
            b[AB_STRIDE + row] = s1;
        }
    }
}

kernel void attn_qkv_span2(
    device const float* act0 [[buffer(0)]],
    device const float* act1 [[buffer(1)]],
    device const uchar* q_w [[buffer(2)]],
    device const uchar* q_s [[buffer(3)]],
    device const uchar* k_w [[buffer(4)]],
    device const uchar* k_s [[buffer(5)]],
    device const uchar* v_w [[buffer(6)]],
    device const uchar* v_s [[buffer(7)]],
    device float* query [[buffer(8)]],
    device float* key [[buffer(9)]],
    device float* value [[buffer(10)]],
    uint lane [[thread_index_in_simdgroup]],
    uint job [[threadgroup_position_in_grid]])
{
    float s0, s1;
    if (job < 12288u) {
        fp4_span2_row(act0, act1, q_w, q_s, job, 5120u, lane, s0, s1);
        if (lane == 0u) {
            query[job] = s0;
            query[12288u + job] = s1;
        }
        return;
    }
    if (job < 13312u) {
        uint row = job - 12288u;
        fp4_span2_row(act0, act1, k_w, k_s, row, 5120u, lane, s0, s1);
        if (lane == 0u) {
            key[row] = s0;
            key[1024u + row] = s1;
        }
        return;
    }
    if (job < 14336u) {
        uint row = job - 13312u;
        fp4_span2_row(act0, act1, v_w, v_s, row, 5120u, lane, s0, s1);
        if (lane == 0u) {
            value[row] = s0;
            value[1024u + row] = s1;
        }
    }
}

kernel void ffn_gate_up_span2(
    device const float* act0 [[buffer(0)]],
    device const float* act1 [[buffer(1)]],
    device const uchar* gate_w [[buffer(2)]],
    device const uchar* gate_s [[buffer(3)]],
    device const uchar* up_w [[buffer(4)]],
    device const uchar* up_s [[buffer(5)]],
    device float* gate [[buffer(6)]],
    device float* up [[buffer(7)]],
    constant uint& rows [[buffer(8)]],
    constant uint& inner [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]],
    uint job [[threadgroup_position_in_grid]])
{
    float s0, s1;
    if (job < rows) {
        fp4_span2_row(act0, act1, gate_w, gate_s, job, inner, lane, s0, s1);
        if (lane == 0u) {
            gate[job] = s0;
            gate[rows + job] = s1;
        }
        return;
    }
    uint row = job - rows;
    if (row < rows) {
        fp4_span2_row(act0, act1, up_w, up_s, row, inner, lane, s0, s1);
        if (lane == 0u) {
            up[row] = s0;
            up[rows + row] = s1;
        }
    }
}

kernel void gdn_conv_rows(
    device half* history [[buffer(0)]],
    device const float* qkv [[buffer(1)]],
    device const half* weight [[buffer(2)]],
    device float* mixed [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    constant uint& channels [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= channels) {
        return;
    }
    for (uint token = 0u; token < count; ++token) {
        device const float* current = qkv + token * channels;
        float raw = current[tid];
        float acc = raw * float(weight[tid * 4u + 3u]);
        for (uint tap = 0u; tap < 3u; ++tap) {
            acc += float(history[tap * channels + tid]) * float(weight[tid * 4u + tap]);
        }
        mixed[token * channels + tid] = silu_f(acc);
        history[tid] = history[channels + tid];
        history[channels + tid] = history[2u * channels + tid];
        history[2u * channels + tid] = half(raw);
    }
}

kernel void gdn_conv_shift(
    device half* history [[buffer(0)]],
    device const float* current [[buffer(1)]],
    device const half* weight [[buffer(2)]],
    device float* mixed [[buffer(3)]],
    constant uint& channels [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= channels) {
        return;
    }
    float acc = float(current[tid]) * float(weight[tid * 4u + 3u]);
    for (uint tap = 0u; tap < 3u; ++tap) {
        acc += float(history[tap * channels + tid]) * float(weight[tid * 4u + tap]);
    }
    mixed[tid] = silu_f(acc);
    history[tid] = history[channels + tid];
    history[channels + tid] = history[2u * channels + tid];
    history[2u * channels + tid] = half(current[tid]);
}

// Forty-eight value heads. Each query/key head is repeated three times.
// The state decays by alpha, then writes beta (v - S k) k^T, then reads with q.
kernel void gdn_state_rows(
    device float* state [[buffer(0)]],
    device const float* mixed [[buffer(1)]],
    device const float* raw_a [[buffer(2)]],
    device const float* raw_b [[buffer(3)]],
    device const float* gate [[buffer(4)]],
    device const half* a_log [[buffer(5)]],
    device const half* dt_bias [[buffer(6)]],
    device const half* norm_w [[buffer(7)]],
    device float* destination [[buffer(8)]],
    constant float& eps [[buffer(9)]],
    constant uint& count [[buffer(10)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint head [[threadgroup_position_in_grid]])
{
    if (head >= LINEAR_V) {
        return;
    }
    threadgroup float red[4];
    threadgroup float qshare[128];
    threadgroup float kshare[128];
    uint qhead = head / REPEAT;
    for (uint token = 0u; token < count; ++token) {
        device const float* mix = mixed + token * MIX;
        float qv = mix[qhead * 128u + tid];
        float kv = mix[2048u + qhead * 128u + tid];
        float qpart = simd_sum(qv * qv);
        float kpart = simd_sum(kv * kv);
        if (lane == 0u) {
            red[simd] = qpart;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float qtot = red[0] + red[1] + red[2] + red[3];
        if (lane == 0u) {
            red[simd] = kpart;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float ktot = red[0] + red[1] + red[2] + red[3];
        qshare[tid] = qv * rsqrt(qtot + eps) * INV_SQRT_128;
        kshare[tid] = kv * rsqrt(ktot + eps);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        device const float* avec = raw_a + token * AB_STRIDE;
        device const float* bvec = raw_b + token * AB_STRIDE;
        float along = float(a_log[head]);
        float g = -exp(along) * softplus_f(avec[head] + float(dt_bias[head]));
        float alpha = exp(g);
        float beta = sigmoid_f(bvec[head]);
        device float* row = state + (head * 128u + tid) * 128u;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        float acc3 = 0.0f;
        for (uint j = 0u; j < 128u; j += 4u) {
            float r0 = row[j] * alpha;
            float r1 = row[j + 1u] * alpha;
            float r2 = row[j + 2u] * alpha;
            float r3 = row[j + 3u] * alpha;
            row[j] = r0;
            row[j + 1u] = r1;
            row[j + 2u] = r2;
            row[j + 3u] = r3;
            acc0 += r0 * kshare[j];
            acc1 += r1 * kshare[j + 1u];
            acc2 += r2 * kshare[j + 2u];
            acc3 += r3 * kshare[j + 3u];
        }
        float retrieved = acc0 + acc1 + acc2 + acc3;
        float delta = beta * (mix[V_OFFSET + head * 128u + tid] - retrieved);
        float y0 = 0.0f;
        float y1 = 0.0f;
        float y2 = 0.0f;
        float y3 = 0.0f;
        for (uint j = 0u; j < 128u; j += 4u) {
            float r0 = row[j] + delta * kshare[j];
            float r1 = row[j + 1u] + delta * kshare[j + 1u];
            float r2 = row[j + 2u] + delta * kshare[j + 2u];
            float r3 = row[j + 3u] + delta * kshare[j + 3u];
            row[j] = r0;
            row[j + 1u] = r1;
            row[j + 2u] = r2;
            row[j + 3u] = r3;
            y0 += r0 * qshare[j];
            y1 += r1 * qshare[j + 1u];
            y2 += r2 * qshare[j + 2u];
            y3 += r3 * qshare[j + 3u];
        }
        float y = y0 + y1 + y2 + y3;
        float ypart = simd_sum(y * y);
        if (lane == 0u) {
            red[simd] = ypart;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float ytot = red[0] + red[1] + red[2] + red[3];
        float inv = rsqrt(ytot / 128.0f + eps);
        float gated = y * inv * float(norm_w[tid]);
        device const float* z = gate + token * Z_WIDTH;
        destination[token * Z_WIDTH + head * 128u + tid] = gated * silu_f(z[head * 128u + tid]);
    }
}

// Twenty-four heads. Each head is 256 query channels then 256 gate channels.
kernel void split_q_gate_rows(
    device const float* mixed [[buffer(0)]],
    device float* query [[buffer(1)]],
    device float* gate [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    uint span = count * 6144u;
    if (tid >= span) {
        return;
    }
    uint token = tid / 6144u;
    uint local = tid % 6144u;
    uint head = local >> 8;
    uint lane = local & 255u;
    uint base = token * 12288u + head * 512u + lane;
    query[tid] = mixed[base];
    gate[tid] = mixed[base + 256u];
}

// Text uses the token index on every mRoPE axis, so the 64 channels are plain RoPE.
kernel void qk_norm_rope(
    device float* query [[buffer(0)]],
    device float* key [[buffer(1)]],
    device const half* q_weight [[buffer(2)]],
    device const half* k_weight [[buffer(3)]],
    constant float& theta [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    constant uint& position [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint head [[threadgroup_position_in_grid]])
{
    threadgroup float red[8];
    device float* row = head < Q_HEADS ? query + head * 256u : key + (head - Q_HEADS) * 256u;
    device const half* weight = head < Q_HEADS ? q_weight : k_weight;
    float value = row[tid];
    float part = simd_sum(value * value);
    if (lane == 0u) {
        red[simd] = part;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float total = 0.0f;
        for (uint i = 0u; i < 8u; ++i) {
            total += red[i];
        }
        red[0] = rsqrt(total / 256.0f + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    row[tid] = value * red[0] * (1.0f + float(weight[tid]));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid >= 32u) {
        return;
    }
    float angle = float(position) * pow(theta, -float(tid) / 32.0f);
    float c = cos(angle);
    float s = sin(angle);
    uint i0 = tid * 2u;
    float x0 = row[i0];
    float x1 = row[i0 + 1u];
    row[i0] = x0 * c - x1 * s;
    row[i0 + 1u] = x0 * s + x1 * c;
}

kernel void qk_norm_rope_rows(
    device float* query [[buffer(0)]],
    device float* key [[buffer(1)]],
    device const half* q_weight [[buffer(2)]],
    device const half* k_weight [[buffer(3)]],
    constant float& theta [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    constant uint& pos0 [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    uint token = tgid.x;
    uint head = tgid.y;
    threadgroup float red[8];
    device float* row = head < Q_HEADS ? query + (token * Q_HEADS + head) * 256u
                                       : key + (token * KV_HEADS + (head - Q_HEADS)) * 256u;
    device const half* weight = head < Q_HEADS ? q_weight : k_weight;
    float value = row[tid];
    float part = simd_sum(value * value);
    if (lane == 0u) {
        red[simd] = part;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float total = 0.0f;
        for (uint i = 0u; i < 8u; ++i) {
            total += red[i];
        }
        red[0] = rsqrt(total / 256.0f + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    row[tid] = value * red[0] * (1.0f + float(weight[tid]));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid >= 32u) {
        return;
    }
    float angle = float(pos0 + token) * pow(theta, -float(tid) / 32.0f);
    float c = cos(angle);
    float s = sin(angle);
    uint i0 = tid * 2u;
    float x0 = row[i0];
    float x1 = row[i0 + 1u];
    row[i0] = x0 * c - x1 * s;
    row[i0 + 1u] = x0 * s + x1 * c;
}

static int kv_e8m0_power(float peak, float qmax) {
    if (!(peak > 0.0f)) {
        return -127;
    }
    float ratio = peak / qmax;
    if (ratio < 1.1754944e-38f) {
        ratio = 1.1754944e-38f;
    }
    int exponent = 0;
    float mantissa = frexp(ratio, exponent);
    int power = exponent - (mantissa == 0.5f ? 1 : 0);
    if (power < -127) {
        return -127;
    }
    if (power > 127) {
        return 127;
    }
    return power;
}

static uint kv_e4m3_code(float unit) {
    float ax = fabs(unit);
    uint sign = unit < 0.0f ? 0x80u : 0u;
    if (ax >= 448.0f) {
        return sign | 0x7Eu;
    }
    if (ax < exp2(-9.0f)) {
        return sign;
    }
    int exponent = 0;
    float mantissa = frexp(ax, exponent);
    float significand = mantissa * 2.0f;
    int biased = exponent - 1 + 7;
    if (biased <= 0) {
        uint sub = uint(rint(ax * exp2(9.0f)));
        if (sub > 7u) {
            sub = 7u;
        }
        return sign | sub;
    }
    float fraction = (significand - 1.0f) * 8.0f;
    int mant = int(rint(fraction));
    if (mant == 8) {
        mant = 0;
        biased += 1;
    }
    if (biased > 15 || (biased == 15 && mant > 6)) {
        return sign | 0x7Eu;
    }
    if (mant < 0) {
        mant = 0;
    }
    return sign | (uint(biased) << 3) | uint(mant);
}

// Keys are [head, token, channel], E8M0 every 32 channels.
kernel void kv_plane_k(
    device uchar* data [[buffer(0)]],
    device uchar* scales [[buffer(1)]],
    device const float* key [[buffer(2)]],
    constant uint& pos0 [[buffer(3)]],
    constant uint& max_pos [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 local [[thread_position_in_threadgroup]])
{
    uint channel = local.x;
    uint head = tgid.y;
    uint token = pos0 + tgid.x;
    float value = key[(tgid.x * KV_HEADS + head) * 256u + channel];
    float peak = simd_max(fabs(value));
    int power = kv_e8m0_power(peak, 448.0f);
    float unit = value * exp2(-float(power));
    uint block = channel >> 5;
    if ((channel & 31u) == 0u) {
        scales[(head * max_pos + token) * 8u + block] = uchar(power + 127);
    }
    data[(head * max_pos + token) * 256u + channel] = uchar(kv_e4m3_code(unit));
}

kernel void kv_tail_store(
    device const float* values [[buffer(0)]],
    device float* tail [[buffer(1)]],
    constant uint& src [[buffer(2)]],
    constant uint& slot [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 local [[thread_position_in_threadgroup]])
{
    uint channel = local.x;
    uint head = tgid.y;
    tail[(head * 32u + slot + tgid.x) * 256u + channel] =
        values[((src + tgid.x) * KV_HEADS + head) * 256u + channel];
}

// Values are [head, channel, token], E8M0 every 32 tokens.
kernel void kv_plane_v(
    device uchar* data [[buffer(0)]],
    device uchar* scales [[buffer(1)]],
    device const float* tail [[buffer(2)]],
    constant uint& block_start [[buffer(3)]],
    constant uint& valid [[buffer(4)]],
    constant uint& max_pos [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    uint channel = tgid.x;
    uint head = tgid.y;
    float value = 0.0f;
    if (lane < valid) {
        value = tail[(head * 32u + lane) * 256u + channel];
    }
    float peak = simd_max(fabs(value));
    int power = kv_e8m0_power(peak, 448.0f);
    uint token = block_start + lane;
    uint scale_index = (head * 256u + channel) * (max_pos >> 5) + (block_start >> 5);
    if (lane == 0u) {
        scales[scale_index] = uchar(power + 127);
    }
    uchar code = 0u;
    if (lane < valid) {
        code = uchar(kv_e4m3_code(value * exp2(-float(power))));
    }
    data[(head * 256u + channel) * max_pos + token] = code;
}

kernel void kv_cast_q(
    device const float* query [[buffer(0)]],
    device half* destination [[buffer(1)]],
    constant uint& pad_count [[buffer(2)]],
    constant uint& kv_head [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 lane [[thread_position_in_threadgroup]])
{
    uint channel = lane.x;
    uint token = tgid.x;
    uint local = tgid.y;
    uint row = local * pad_count + token;
    if (local >= GROUP) {
        destination[row * 256u + channel] = half(0.0f);
        return;
    }
    uint global = kv_head * GROUP + local;
    destination[row * 256u + channel] = half(query[(token * Q_HEADS + global) * 256u + channel]);
}

kernel void kv_row_init(
    device float* row_max [[buffer(0)]],
    device float* row_sum [[buffer(1)]],
    device float* acc [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint channel [[thread_position_in_threadgroup]])
{
    if (channel == 0u) {
        row_max[row] = -INFINITY;
        row_sum[row] = 0.0f;
    }
    acc[row * 256u + channel] = 0.0f;
}

kernel void kv_softmax_tile(
    device const float* scores [[buffer(0)]],
    device float* probs [[buffer(1)]],
    device float* acc [[buffer(2)]],
    device float* row_max [[buffer(3)]],
    device float* row_sum [[buffer(4)]],
    constant uint& pad_count [[buffer(5)]],
    constant uint& count [[buffer(6)]],
    constant uint& pos0 [[buffer(7)]],
    constant uint& tile_start [[buffer(8)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint token = row % pad_count;
    bool live = token < count;
    if (!live) {
        for (uint step = 0u; step < 8u; ++step) {
            probs[row * 256u + lane + step * 32u] = 0.0f;
        }
        return;
    }
    uint qpos = pos0 + token;
    uint limit = pos0 + count;
    float local[8];
    float tile_max = -INFINITY;
    uint seen = 0u;
    for (uint step = 0u; step < 8u; ++step) {
        uint col = lane + step * 32u;
        uint kpos = tile_start + col;
        bool valid = kpos <= qpos && kpos < limit;
        float score = valid ? scores[row * 256u + col] * INV_SQRT_256 : -INFINITY;
        local[step] = score;
        if (valid) {
            seen = 1u;
            tile_max = max(tile_max, score);
        }
    }
    seen = simd_sum(seen);
    tile_max = simd_max(tile_max);
    if (seen == 0u) {
        for (uint step = 0u; step < 8u; ++step) {
            probs[row * 256u + lane + step * 32u] = 0.0f;
        }
        return;
    }
    float old_m = row_max[row];
    float new_m = max(old_m, tile_max);
    float alpha = isinf(old_m) ? 0.0f : exp(old_m - new_m);
    float psum = 0.0f;
    for (uint step = 0u; step < 8u; ++step) {
        float prob = local[step] == -INFINITY ? 0.0f : exp(local[step] - new_m);
        psum += prob;
        probs[row * 256u + lane + step * 32u] = prob;
    }
    psum = simd_sum(psum);
    for (uint channel = lane; channel < 256u; channel += 32u) {
        acc[row * 256u + channel] *= alpha;
    }
    if (lane == 0u) {
        row_sum[row] = row_sum[row] * alpha + psum;
        row_max[row] = new_m;
    }
}

kernel void kv_softmax_span(
    device const float* scores [[buffer(0)]],
    device float* probs [[buffer(1)]],
    device float* acc [[buffer(2)]],
    device float* row_max [[buffer(3)]],
    device float* row_sum [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    constant uint& pad_count [[buffer(6)]],
    constant uint& count [[buffer(7)]],
    constant uint& pos0 [[buffer(8)]],
    constant uint& tile_start [[buffer(9)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint token = row % pad_count;
    bool live = token < count;
    if (!live) {
        for (uint col = lane; col < cols; col += 32u) {
            probs[row * cols + col] = 0.0f;
        }
        return;
    }
    uint qpos = pos0 + token;
    uint limit = pos0 + count;
    float tile_max = -INFINITY;
    uint seen = 0u;
    for (uint col = lane; col < cols; col += 32u) {
        uint kpos = tile_start + col;
        bool valid = kpos <= qpos && kpos < limit;
        if (!valid) {
            continue;
        }
        seen = 1u;
        tile_max = max(tile_max, scores[row * cols + col] * INV_SQRT_256);
    }
    seen = simd_sum(seen);
    tile_max = simd_max(tile_max);
    if (seen == 0u) {
        for (uint col = lane; col < cols; col += 32u) {
            probs[row * cols + col] = 0.0f;
        }
        return;
    }
    float old_m = row_max[row];
    float new_m = max(old_m, tile_max);
    float alpha = isinf(old_m) ? 0.0f : exp(old_m - new_m);
    float psum = 0.0f;
    for (uint col = lane; col < cols; col += 32u) {
        uint kpos = tile_start + col;
        bool valid = kpos <= qpos && kpos < limit;
        float prob = 0.0f;
        if (valid) {
            prob = exp(scores[row * cols + col] * INV_SQRT_256 - new_m);
        }
        psum += prob;
        probs[row * cols + col] = prob;
    }
    psum = simd_sum(psum);
    for (uint channel = lane; channel < 256u; channel += 32u) {
        acc[row * 256u + channel] *= alpha;
    }
    if (lane == 0u) {
        row_sum[row] = row_sum[row] * alpha + psum;
        row_max[row] = new_m;
    }
}

kernel void kv_softmax_max(
    device const float* scores [[buffer(0)]],
    device float* partial_max [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    constant uint& chunks [[buffer(3)]],
    constant uint& pos0 [[buffer(4)]],
    constant uint& tile_start [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint chunk = tgid.x;
    uint row = tgid.y;
    uint base = chunk * 256u;
    float tile_max = -INFINITY;
    for (uint step = 0u; step < 8u; ++step) {
        uint col = base + lane + step * 32u;
        if (col >= cols) {
            continue;
        }
        uint kpos = tile_start + col;
        if (kpos <= pos0) {
            tile_max = max(tile_max, scores[row * cols + col] * INV_SQRT_256);
        }
    }
    tile_max = simd_max(tile_max);
    if (lane == 0u) {
        partial_max[row * chunks + chunk] = tile_max;
    }
}

kernel void kv_softmax_prob(
    device const float* scores [[buffer(0)]],
    device float* probs [[buffer(1)]],
    device const float* partial_max [[buffer(2)]],
    device const float* row_max [[buffer(3)]],
    device float* partial_sum [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    constant uint& chunks [[buffer(6)]],
    constant uint& pos0 [[buffer(7)]],
    constant uint& tile_start [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint chunk = tgid.x;
    uint row = tgid.y;
    float tile_max = -INFINITY;
    for (uint i = lane; i < chunks; i += 32u) {
        tile_max = max(tile_max, partial_max[row * chunks + i]);
    }
    tile_max = simd_max(tile_max);
    float old_m = row_max[row];
    float new_m = max(old_m, tile_max);
    uint base = chunk * 256u;
    float psum = 0.0f;
    bool seen = !isinf(tile_max);
    for (uint step = 0u; step < 8u; ++step) {
        uint col = base + lane + step * 32u;
        float prob = 0.0f;
        if (col < cols) {
            uint kpos = tile_start + col;
            if (seen && kpos <= pos0) {
                prob = exp(scores[row * cols + col] * INV_SQRT_256 - new_m);
            }
            probs[row * cols + col] = prob;
        }
        psum += prob;
    }
    psum = simd_sum(psum);
    if (lane == 0u) {
        partial_sum[row * chunks + chunk] = psum;
    }
}

kernel void kv_softmax_max_rows(
    device const float* scores [[buffer(0)]],
    device float* partial_max [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    constant uint& chunks [[buffer(3)]],
    constant uint& pad_count [[buffer(4)]],
    constant uint& count [[buffer(5)]],
    constant uint& pos0 [[buffer(6)]],
    constant uint& tile_start [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint chunk = tgid.x;
    uint row = tgid.y;
    uint token = row % pad_count;
    float tile_max = -INFINITY;
    if (token < count) {
        uint qpos = pos0 + token;
        uint limit = pos0 + count;
        uint base = chunk * 256u;
        for (uint step = 0u; step < 8u; ++step) {
            uint col = base + lane + step * 32u;
            if (col >= cols) {
                continue;
            }
            uint kpos = tile_start + col;
            if (kpos <= qpos && kpos < limit) {
                tile_max = max(tile_max, scores[row * cols + col] * INV_SQRT_256);
            }
        }
    }
    tile_max = simd_max(tile_max);
    if (lane == 0u) {
        partial_max[row * chunks + chunk] = tile_max;
    }
}

kernel void kv_softmax_prob_rows(
    device const float* scores [[buffer(0)]],
    device float* probs [[buffer(1)]],
    device const float* partial_max [[buffer(2)]],
    device const float* row_max [[buffer(3)]],
    device float* partial_sum [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    constant uint& chunks [[buffer(6)]],
    constant uint& pad_count [[buffer(7)]],
    constant uint& count [[buffer(8)]],
    constant uint& pos0 [[buffer(9)]],
    constant uint& tile_start [[buffer(10)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint chunk = tgid.x;
    uint row = tgid.y;
    uint token = row % pad_count;
    bool live = token < count;
    float tile_max = -INFINITY;
    if (live) {
        for (uint i = lane; i < chunks; i += 32u) {
            tile_max = max(tile_max, partial_max[row * chunks + i]);
        }
    }
    tile_max = simd_max(tile_max);
    float old_m = row_max[row];
    float new_m = max(old_m, tile_max);
    uint qpos = pos0 + token;
    uint limit = pos0 + count;
    uint base = chunk * 256u;
    float psum = 0.0f;
    bool seen = live && !isinf(tile_max);
    for (uint step = 0u; step < 8u; ++step) {
        uint col = base + lane + step * 32u;
        float prob = 0.0f;
        if (col < cols) {
            uint kpos = tile_start + col;
            if (seen && kpos <= qpos && kpos < limit) {
                prob = exp(scores[row * cols + col] * INV_SQRT_256 - new_m);
            }
            probs[row * cols + col] = prob;
        }
        psum += prob;
    }
    psum = simd_sum(psum);
    if (lane == 0u) {
        partial_sum[row * chunks + chunk] = psum;
    }
}

kernel void kv_softmax_close(
    device float* acc [[buffer(0)]],
    device float* row_max [[buffer(1)]],
    device float* row_sum [[buffer(2)]],
    device const float* partial_max [[buffer(3)]],
    device const float* partial_sum [[buffer(4)]],
    constant uint& chunks [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    float tile_max = -INFINITY;
    float psum = 0.0f;
    for (uint i = lane; i < chunks; i += 32u) {
        tile_max = max(tile_max, partial_max[row * chunks + i]);
        psum += partial_sum[row * chunks + i];
    }
    tile_max = simd_max(tile_max);
    psum = simd_sum(psum);
    if (isinf(tile_max)) {
        return;
    }
    float old_m = row_max[row];
    float new_m = max(old_m, tile_max);
    float alpha = isinf(old_m) ? 0.0f : exp(old_m - new_m);
    for (uint channel = lane; channel < 256u; channel += 32u) {
        acc[row * 256u + channel] *= alpha;
    }
    if (lane == 0u) {
        row_sum[row] = row_sum[row] * alpha + psum;
        row_max[row] = new_m;
    }
}

kernel void kv_cast_probs(
    device const float* probs [[buffer(0)]],
    device half* packed [[buffer(1)]],
    device float* scales [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& src_stride [[buffer(4)]],
    constant uint& src_col [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    float peak = 0.0f;
    for (uint col = lane; col < cols; col += 32u) {
        peak = max(peak, fabs(probs[row * src_stride + src_col + col]));
    }
    peak = simd_max(peak);
    if (lane == 0u) {
        scales[row] = peak;
    }
    float inv = peak > 0.0f ? 1.0f / peak : 0.0f;
    for (uint col = lane; col < cols; col += 32u) {
        packed[row * cols + col] = half(probs[row * src_stride + src_col + col] * inv);
    }
}

kernel void kv_scale_partial(
    device float* partial [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    constant uint& channels [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint channel [[thread_position_in_threadgroup]])
{
    if (channel >= channels) {
        return;
    }
    partial[row * channels + channel] *= scales[row];
}

kernel void kv_pv_split8(
    device half* activation [[buffer(0)]],
    device uchar* weights [[buffer(1)]],
    device uchar* scales [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& channels [[buffer(5)]],
    constant uint& block [[buffer(6)]],
    constant uint& row_stride [[buffer(7)]],
    constant uint& value_stride [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    uint k0 = tgid.y * block;
    device half* act = activation + k0;
    device uchar* data = weights + k0;
    device uchar* scale = scales + (k0 >> 5);
    device float* dest = destination + (tgid.y * rows * channels);
    tensor<device half, dextents<int, 2>, tensor_inline> left(
        act,
        dextents<int, 2>(int(block), int(rows)),
        array<int, 2>{1, int(row_stride)});
    tensor<device metal_fp8_e4m3_format, dextents<int, 2>, tensor_inline, scales_plane> right(
        data,
        dextents<int, 2>(int(block), int(channels)),
        array<int, 2>{1, int(value_stride)},
        scales_plane(scale));
    tensor<device float, dextents<int, 2>, tensor_inline> out(
        dest,
        dextents<int, 2>(int(channels), int(rows)),
        array<int, 2>{1, int(channels)});
    constexpr auto desc = matmul2d_descriptor(
        8, 64, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto tile_left = left.slice(0, 0);
    auto tile_right = right.slice(0, int(tgid.x) * 64);
    auto tile_out = out.slice(int(tgid.x) * 64, 0);
    op.run(tile_left, tile_right, tile_out);
}

kernel void kv_part_sum(
    device const float* parts [[buffer(0)]],
    device float* destination [[buffer(1)]],
    constant uint& blocks [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= width) {
        return;
    }
    float sum = 0.0f;
    for (uint block = 0u; block < blocks; ++block) {
        sum += parts[block * width + tid];
    }
    destination[tid] += sum;
}

kernel void kv_acc_add(
    device float* acc [[buffer(0)]],
    device const float* partial [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    acc[tid] += partial[tid];
}

kernel void kv_attend_write(
    device const float* acc [[buffer(0)]],
    device const float* row_sum [[buffer(1)]],
    device const float* gate [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& pad_count [[buffer(4)]],
    constant uint& count [[buffer(5)]],
    constant uint& kv_head [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 lane [[thread_position_in_threadgroup]])
{
    uint channel = lane.x;
    uint token = tgid.x;
    uint local = tgid.y;
    if (token >= count || local >= GROUP) {
        return;
    }
    uint row = local * pad_count + token;
    uint global = kv_head * GROUP + local;
    uint index = (token * Q_HEADS + global) * 256u + channel;
    float sum = row_sum[row];
    float inv = sum == 0.0f ? 0.0f : 1.0f / sum;
    destination[index] = acc[row * 256u + channel] * inv * sigmoid_f(gate[index]);
}

kernel void dequant_e4m3(
    device const uchar* codes [[buffer(0)]],
    device float* destination [[buffer(1)]],
    constant uint& scale [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= 32u) {
        return;
    }
    destination[tid] = e4m3_fn(codes[tid]) * exp2(float(scale) - 127.0f);
}

kernel void argmax_partial(
    device const float* logits [[buffer(0)]],
    device float* best_value [[buffer(1)]],
    device uint* best_index [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]])
{
    threadgroup float vals[8];
    threadgroup uint ids[8];
    uint index = group * 256u + tid;
    float value = index < n ? logits[index] : -INFINITY;
    uint id = index;
    float other = simd_shuffle_xor(value, 1u);
    uint other_id = simd_shuffle_xor(id, 1u);
    if (other > value || (other == value && other_id < id)) {
        value = other;
        id = other_id;
    }
    other = simd_shuffle_xor(value, 2u);
    other_id = simd_shuffle_xor(id, 2u);
    if (other > value || (other == value && other_id < id)) {
        value = other;
        id = other_id;
    }
    other = simd_shuffle_xor(value, 4u);
    other_id = simd_shuffle_xor(id, 4u);
    if (other > value || (other == value && other_id < id)) {
        value = other;
        id = other_id;
    }
    other = simd_shuffle_xor(value, 8u);
    other_id = simd_shuffle_xor(id, 8u);
    if (other > value || (other == value && other_id < id)) {
        value = other;
        id = other_id;
    }
    other = simd_shuffle_xor(value, 16u);
    other_id = simd_shuffle_xor(id, 16u);
    if (other > value || (other == value && other_id < id)) {
        value = other;
        id = other_id;
    }
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
