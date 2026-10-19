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
constant float INV_SQRT_72 = 0.11785113019775792f;
constant float GELU_K = 0.7978845608028654f;
constant float GELU_CUBE = 0.044715f;

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

static float gelu_tanh_f(float x) {
    float inner = GELU_K * (x + GELU_CUBE * x * x * x);
    return 0.5f * x * (1.0f + tanh(inner));
}

constant float FP4_TABLE[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

static float fp4_e2m1(uint nibble) {
    return FP4_TABLE[nibble & 15u];
}

// One simdgroup, one output row. Inner is a multiple of 256. The eight
// nibbles in a word are eight consecutive activations, so the loads are
// float4 and the weight word is one uint.
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

// Pair i keeps the temporal frequency unless the interleaved sections overwrite it.
// Height owns 1, 4, ..., 31. Width owns 2, 5, ..., 29. The rest stay temporal.
static uint mrope_axis(uint pair) {
    if ((pair % 3u) == 1u && pair < 33u) {
        return 1u;
    }
    if ((pair % 3u) == 2u && pair < 30u) {
        return 2u;
    }
    return 0u;
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

kernel void prefill_fp4_expert_tile(
    device half* pack [[buffer(0)]],
    device uchar* file [[buffer(1)]],
    device float* destination [[buffer(2)]],
    device const uint* jobs [[buffer(3)]],
    device const uint* expert_of [[buffer(4)]],
    device const uint* row_base [[buffer(5)]],
    device const uint* info [[buffer(6)]],
    constant uint& inner [[buffer(7)]],
    constant uint& cols [[buffer(8)]],
    constant uint& image_data [[buffer(9)]],
    constant uint& image_scale [[buffer(10)]],
    constant uint& image_bytes [[buffer(11)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    uint n_jobs = info[1];
    if (tgid.y >= n_jobs) {
        return;
    }
    uint slot = jobs[tgid.y * 2u];
    uint row0 = jobs[tgid.y * 2u + 1u] * 64u;
    uint expert = expert_of[slot];
    uint col0 = tgid.x * 64u;
    uint base_row = row_base[slot] + row0;
    device half* act = pack + base_row * inner;
    device uchar* base = file + expert * image_bytes;
    device float* out = destination + base_row * cols;
    tensor<device half, dextents<int, 2>, tensor_inline> left(
        act, dextents<int, 2>(int(inner), 64), array<int, 2>{1, int(inner)});
    tensor<device metal_fp4_e2m1_format, dextents<int, 2>, tensor_inline, scales_plane> right(
        base + image_data,
        dextents<int, 2>(int(inner), int(cols)),
        array<int, 2>{1, int(inner)},
        scales_plane(base + image_scale));
    tensor<device float, dextents<int, 2>, tensor_inline> dest(
        out, dextents<int, 2>(int(cols), 64), array<int, 2>{1, int(cols)});
    constexpr auto desc = matmul2d_descriptor(
        64, 64, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto tile_left = left.slice(0, 0);
    auto tile_right = right.slice(0, int(col0));
    auto tile_out = dest.slice(int(col0), 0);
    op.run(tile_left, tile_right, tile_out);
}

// Same tile. ``offsets[expert]`` is the byte offset when the images are not contiguous.
kernel void prefill_fp4_expert_tile_at(
    device half* pack [[buffer(0)]],
    device uchar* file [[buffer(1)]],
    device float* destination [[buffer(2)]],
    device const uint* jobs [[buffer(3)]],
    device const uint* expert_of [[buffer(4)]],
    device const uint* row_base [[buffer(5)]],
    device const uint* info [[buffer(6)]],
    device const ulong* offsets [[buffer(7)]],
    constant uint& inner [[buffer(8)]],
    constant uint& cols [[buffer(9)]],
    constant uint& image_data [[buffer(10)]],
    constant uint& image_scale [[buffer(11)]],
    constant uint& image_bytes [[buffer(12)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    uint n_jobs = info[1];
    if (tgid.y >= n_jobs) {
        return;
    }
    uint slot = jobs[tgid.y * 2u];
    uint row0 = jobs[tgid.y * 2u + 1u] * 64u;
    uint expert = expert_of[slot];
    uint col0 = tgid.x * 64u;
    uint base_row = row_base[slot] + row0;
    device half* act = pack + base_row * inner;
    device uchar* base = file + offsets[expert];
    device float* out = destination + base_row * cols;
    tensor<device half, dextents<int, 2>, tensor_inline> left(
        act, dextents<int, 2>(int(inner), 64), array<int, 2>{1, int(inner)});
    tensor<device metal_fp4_e2m1_format, dextents<int, 2>, tensor_inline, scales_plane> right(
        base + image_data,
        dextents<int, 2>(int(inner), int(cols)),
        array<int, 2>{1, int(inner)},
        scales_plane(base + image_scale));
    tensor<device float, dextents<int, 2>, tensor_inline> dest(
        out, dextents<int, 2>(int(cols), 64), array<int, 2>{1, int(cols)});
    constexpr auto desc = matmul2d_descriptor(
        64, 64, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto tile_left = left.slice(0, 0);
    auto tile_right = right.slice(0, int(col0));
    auto tile_out = dest.slice(int(col0), 0);
    op.run(tile_left, tile_right, tile_out);
}

kernel void prefill_f16(
    device half* activation [[buffer(0)]],
    device half* weights [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    tensor<device half, dextents<int, 2>, tensor_inline> left(
        activation,
        dextents<int, 2>(int(inner), int(rows)),
        array<int, 2>{1, int(inner)});
    tensor<device half, dextents<int, 2>, tensor_inline> right(
        weights,
        dextents<int, 2>(int(inner), int(cols)),
        array<int, 2>{1, int(inner)});
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

kernel void prefill_fp4_n64_stride(
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
    tensor<device metal_fp4_e2m1_format, dextents<int, 2>, tensor_inline, scales_plane> right(
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

kernel void prefill_fp8_n64_sweep(
    device half* activation [[buffer(0)]],
    device uchar* weights [[buffer(1)]],
    device uchar* scales [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    constant uint& inner [[buffer(6)]],
    constant uint& ngroups [[buffer(7)]],
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
    constexpr auto desc = matmul2d_descriptor(64, 64, 256, false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    uint ntiles = cols >> 6;
    auto tile_left = left.slice(0, 0);
    for (uint tile = gid; tile < ntiles; tile += ngroups) {
        auto tile_right = right.slice(0, int(tile) * 64);
        auto tile_out = out.slice(int(tile) * 64, 0);
        op.run(tile_left, tile_right, tile_out);
    }
}

// Eight live query rows. One threadgroup owns a 256-key output tile.
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

kernel void prefill_fp8_e8m0(
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
        64, 32, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto tile_left = left.slice(0, int(tgid.y) * 64);
    auto tile_right = right.slice(0, int(tgid.x) * 32);
    auto tile_out = out.slice(int(tgid.x) * 32, int(tgid.y) * 64);
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

kernel void gemv_f16(
    device const half* activation [[buffer(0)]],
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
    device const half* rowp = weights + row * inner;
    float acc = 0.0f;
    for (uint i = lane; i < inner; i += 32u) {
        acc += float(rowp[i]) * float(activation[i]);
    }
    float sum = simd_sum(acc);
    if (lane == 0u) {
        destination[row] = sum;
    }
}

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

kernel void layernorm_f32(
    device const float* x [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device float* y [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[8];
    float sum = 0.0f;
    for (uint i = tid; i < width; i += 256u) {
        sum += x[i];
    }
    float part = simd_sum(sum);
    if (lane == 0u) {
        red[simd] = part;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float total = red[0] + red[1] + red[2] + red[3] + red[4] + red[5] + red[6] + red[7];
        red[0] = total / float(width);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = red[0];
    float var = 0.0f;
    for (uint i = tid; i < width; i += 256u) {
        float diff = x[i] - mean;
        var += diff * diff;
    }
    part = simd_sum(var);
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
        y[i] = (x[i] - mean) * inv * float(weight[i]) + float(bias[i]);
    }
}

kernel void silu_f32(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    y[tid] = silu_f(x[tid]);
}

kernel void sigmoid_f32(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    y[tid] = sigmoid_f(x[tid]);
}

kernel void softplus_f32(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    y[tid] = softplus_f(x[tid]);
}

kernel void gelu_tanh_f32(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    y[tid] = gelu_tanh_f(x[tid]);
}

kernel void residual_add_f32(
    device const float* left [[buffer(0)]],
    device const float* right [[buffer(1)]],
    device float* y [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    y[tid] = left[tid] + right[tid];
}

kernel void swiglu_f32(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* y [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    float value = gate[tid];
    y[tid] = silu_f(value) * up[tid];
}

kernel void l2_heads_f32(
    device float* data [[buffer(0)]],
    constant float& eps [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint head [[threadgroup_position_in_grid]])
{
    threadgroup float red[4];
    device float* row = data + head * 128u;
    float value = row[tid];
    float part = simd_sum(value * value);
    if (lane == 0u) {
        red[simd] = part;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = red[0] + red[1] + red[2] + red[3];
    row[tid] = value * rsqrt(total + eps) * scale;
}

kernel void rope_partial(
    device float* hidden [[buffer(0)]],
    constant uint* positions [[buffer(1)]],
    constant float& theta [[buffer(2)]],
    uint pair [[thread_position_in_grid]])
{
    if (pair >= 32u) {
        return;
    }
    uint axis = mrope_axis(pair);
    float angle = float(positions[axis]) * pow(theta, -float(pair) / 32.0f);
    float c = cos(angle);
    float s = sin(angle);
    uint i0 = pair;
    uint i1 = pair + 32u;
    float x0 = hidden[i0];
    float x1 = hidden[i1];
    hidden[i0] = x0 * c - x1 * s;
    hidden[i1] = x0 * s + x1 * c;
}

kernel void gdn_conv(
    device const half* history [[buffer(0)]],
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
}

kernel void gdn_state(
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
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint head [[threadgroup_position_in_grid]])
{
    threadgroup float red[4];
    threadgroup float qshare[128];
    threadgroup float kshare[128];
    uint qhead = head / 2u;
    float qv = mixed[qhead * 128u + tid];
    float kv = mixed[2048u + qhead * 128u + tid];
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

    float along = float(a_log[head]);
    float g = -exp(along) * softplus_f(raw_a[head] + float(dt_bias[head]));
    float alpha = exp(g);
    float beta = sigmoid_f(raw_b[head]);
    device float* row = state + (head * 128u + tid) * 128u;
    for (uint j = 0u; j < 128u; ++j) {
        row[j] *= alpha;
    }
    float retrieved = 0.0f;
    for (uint j = 0u; j < 128u; ++j) {
        retrieved += row[j] * kshare[j];
    }
    float delta = beta * (mixed[4096u + head * 128u + tid] - retrieved);
    for (uint j = 0u; j < 128u; ++j) {
        row[j] += delta * kshare[j];
    }
    float y = 0.0f;
    for (uint j = 0u; j < 128u; ++j) {
        y += row[j] * qshare[j];
    }
    float ypart = simd_sum(y * y);
    if (lane == 0u) {
        red[simd] = ypart;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ytot = red[0] + red[1] + red[2] + red[3];
        float inv = rsqrt(ytot / 128.0f + eps);
        float gated = y * inv * float(norm_w[tid]);
        destination[head * 128u + tid] = gated * silu_f(gate[head * 128u + tid]);
}

static float head_dot(
    uint tid,
    uint lane,
    uint simd,
    uint dim,
    threadgroup float* red,
    device const float* left,
    device const float* right)
{
    float mine = tid < dim ? left[tid] * right[tid] : 0.0f;
    float part = simd_sum(mine);
    if (lane == 0u) {
        red[simd] = part;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0u; i < 8u; ++i) {
        total += red[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

kernel void attend_gqa(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& tokens [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[8];
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    for (uint token = 0u; token < tokens; ++token) {
        float score = head_dot(tid, lane, simd, 256u, red, query, key + token * 256u) * INV_SQRT_256;
        float next = max(running_max, score);
        running_sum = running_sum * exp(running_max - next) + exp(score - next);
        running_max = next;
    }
    float acc = 0.0f;
    for (uint token = 0u; token < tokens; ++token) {
        float score = head_dot(tid, lane, simd, 256u, red, query, key + token * 256u) * INV_SQRT_256;
        float weight = exp(score - running_max) / running_sum;
        acc += weight * value[token * 256u + tid];
    }
    destination[tid] = acc;
}

kernel void attend_head72(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& tokens [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[3];
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    for (uint token = 0u; token < tokens; ++token) {
        float mine = tid < 72u ? query[tid] * key[token * 72u + tid] : 0.0f;
        float part = simd_sum(mine);
        if (lane == 0u) {
            red[simd] = part;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float score = (red[0] + red[1] + red[2]) * INV_SQRT_72;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float next = max(running_max, score);
        running_sum = running_sum * exp(running_max - next) + exp(score - next);
        running_max = next;
    }
    float acc = 0.0f;
    for (uint token = 0u; token < tokens; ++token) {
        float mine = tid < 72u ? query[tid] * key[token * 72u + tid] : 0.0f;
        float part = simd_sum(mine);
        if (lane == 0u) {
            red[simd] = part;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float score = (red[0] + red[1] + red[2]) * INV_SQRT_72;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float weight = exp(score - running_max) / running_sum;
        if (tid < 72u) {
            acc += weight * value[token * 72u + tid];
        }
    }
    if (tid < 72u) {
        destination[tid] = acc;
    }
}

kernel void topk8(
    device const float* logits [[buffer(0)]],
    device uint* index [[buffer(1)]],
    device float* weight [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float vals[256];
    threadgroup uint used[256];
    vals[tid] = logits[tid];
    used[tid] = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float chosen[8];
        for (uint k = 0u; k < 8u; ++k) {
            uint best = 0u;
            float bestv = -INFINITY;
            for (uint i = 0u; i < 256u; ++i) {
                if (used[i] != 0u) {
                    continue;
                }
                if (vals[i] > bestv) {
                    bestv = vals[i];
                    best = i;
                }
            }
            used[best] = 1u;
            index[k] = best;
            chosen[k] = bestv;
        }
        float sum = 0.0f;
        for (uint k = 0u; k < 8u; ++k) {
            sum += chosen[k];
        }
        for (uint k = 0u; k < 8u; ++k) {
            weight[k] = chosen[k] / sum;
        }
    }
}

kernel void gather_row(
    device const float* table [[buffer(0)]],
    device float* destination [[buffer(1)]],
    constant uint& inner [[buffer(2)]],
    constant uint& row [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= inner) {
        return;
    }
    destination[tid] = table[row * inner + tid];
}

kernel void cast_f32_f16(
    device const float* source [[buffer(0)]],
    device half* destination [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    destination[tid] = half(source[tid]);
}

kernel void shift_history(
    device half* history [[buffer(0)]],
    device const float* current [[buffer(1)]],
    constant uint& channels [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= channels) {
        return;
    }
    history[tid] = history[channels + tid];
    history[channels + tid] = history[2u * channels + tid];
    history[2u * channels + tid] = half(current[tid]);
}

kernel void store_row(
    device float* cache [[buffer(0)]],
    device const float* row [[buffer(1)]],
    constant uint& slot [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    cache[slot * 256u + tid] = row[tid];
}

kernel void split_q_gate(
    device const float* mixed [[buffer(0)]],
    device float* query [[buffer(1)]],
    device float* gate [[buffer(2)]],
    constant uint& per_head [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= 4096u) {
        return;
    }
    if (per_head == 0u) {
        query[tid] = mixed[tid];
        gate[tid] = mixed[4096u + tid];
        return;
    }
    uint head = tid >> 8;
    uint lane = tid & 255u;
    uint base = head * 512u + lane;
    query[tid] = mixed[base];
    gate[tid] = mixed[base + 256u];
}

kernel void axpy_f32(
    device float* destination [[buffer(0)]],
    device const float* source [[buffer(1)]],
    constant float& alpha [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    destination[tid] += alpha * source[tid];
}

kernel void zero_f32(
    device float* destination [[buffer(0)]],
    uint tid [[thread_position_in_grid]])
{
    destination[tid] = 0.0f;
}

kernel void mul_f32(
    device const float* left [[buffer(0)]],
    device const float* right [[buffer(1)]],
    device float* destination [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    destination[tid] = left[tid] * right[tid];
}

kernel void softmax256(
    device const float* logits [[buffer(0)]],
    device float* destination [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[8];
    threadgroup float vals[256];
    vals[tid] = logits[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mine = vals[tid];
    float part = simd_max(mine);
    if (lane == 0u) {
        red[simd] = part;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float peak = red[0];
    for (uint i = 1u; i < 8u; ++i) {
        peak = max(peak, red[i]);
    }
    float shifted = exp(mine - peak);
    part = simd_sum(shifted);
    if (lane == 0u) {
        red[simd] = part;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = red[0] + red[1] + red[2] + red[3] + red[4] + red[5] + red[6] + red[7];
    destination[tid] = shifted / total;
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

// One weight matrix, many activation rows. The row stays in cache across tokens.
kernel void gemv_fp4_batch(
    device const float* activation [[buffer(0)]],
    device const uchar* weights [[buffer(1)]],
    device const uchar* scales [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    constant uint& inner [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    if (row >= cols) {
        return;
    }
    uint blocks = inner >> 5;
    uint stride = inner >> 1;
    device const uchar* packed = weights + row * stride;
    device const uchar* row_scale = scales + row * blocks;
    for (uint token = 0u; token < count; ++token) {
        device const float* act = activation + token * inner;
        float acc = 0.0f;
        for (uint block = 0u; block < blocks; ++block) {
            uint element = (block << 5) + lane;
            uchar byte = packed[element >> 1];
            uint nibble = (element & 1u) ? (byte >> 4) : (byte & 0x0Fu);
            float factor = exp2(float(row_scale[block]) - 127.0f);
            acc += fp4_e2m1(nibble) * factor * act[element];
        }
        float sum = simd_sum(acc);
        if (lane == 0u) {
            destination[token * cols + row] = sum;
        }
    }
}

// Four output rows share one activation stream. The rows are independent, so the
// weight loads stay in flight together. cols is a multiple of 4.
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
    uint blocks = inner >> 5;
    uint stride = inner >> 1;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    device const uchar* w0 = weights + row0 * stride;
    device const uchar* w1 = w0 + stride;
    device const uchar* w2 = w1 + stride;
    device const uchar* w3 = w2 + stride;
    device const uchar* s0 = scales + row0 * blocks;
    device const uchar* s1 = s0 + blocks;
    device const uchar* s2 = s1 + blocks;
    device const uchar* s3 = s2 + blocks;
    for (uint block = 0u; block < blocks; ++block) {
        uint element = (block << 5) + lane;
        float a = activation[element];
        uchar b0 = w0[element >> 1];
        uchar b1 = w1[element >> 1];
        uchar b2 = w2[element >> 1];
        uchar b3 = w3[element >> 1];
        uint shift = (element & 1u) << 2;
        float f0 = exp2(float(s0[block]) - 127.0f);
        float f1 = exp2(float(s1[block]) - 127.0f);
        float f2 = exp2(float(s2[block]) - 127.0f);
        float f3 = exp2(float(s3[block]) - 127.0f);
        acc0 += fp4_e2m1((b0 >> shift) & 0xFu) * f0 * a;
        acc1 += fp4_e2m1((b1 >> shift) & 0xFu) * f1 * a;
        acc2 += fp4_e2m1((b2 >> shift) & 0xFu) * f2 * a;
        acc3 += fp4_e2m1((b3 >> shift) & 0xFu) * f3 * a;
    }
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

// Eight fp4 values per 32-bit load. Inner must be a multiple of 256 so each
// lane owns the same number of words and a word never crosses a scale block.
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

// Eight experts in one grid. Slot y reads ids[y] from the layer file.
// act_stride 0 means every slot reads the same activation.
kernel void gemv_fp4_vec_slots(
    device const float* activation [[buffer(0)]],
    device const uchar* file [[buffer(1)]],
    device const uint* ids [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    constant uint& data_off [[buffer(6)]],
    constant uint& scale_off [[buffer(7)]],
    constant uint& image [[buffer(8)]],
    constant uint& act_stride [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    uint row = tgid.x;
    uint slot = tgid.y;
    if (row >= cols || slot >= 8u) {
        return;
    }
    uint id = ids[slot];
    device const uchar* base = file + (ulong)id * (ulong)image;
    device const uchar* weights = base + data_off;
    device const uchar* scales = base + scale_off;
    device const float* act = activation + (act_stride == 0u ? 0u : slot * act_stride);
    float sum = fp4_vec_dot(act, weights, scales, row, inner, lane);
    if (lane == 0u) {
        destination[slot * cols + row] = sum;
    }
}

kernel void silu_mul_slots(
    device const float* gate_up [[buffer(0)]],
    device float* gated [[buffer(1)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    uint row = tgid.x;
    uint slot = tgid.y;
    if (row >= 512u || slot >= 8u) {
        return;
    }
    float gate = gate_up[slot * 1024u + row];
    float up = gate_up[slot * 1024u + 512u + row];
    gated[slot * 512u + row] = silu_f(gate) * up;
}

kernel void router8(
    device const float* logits [[buffer(0)]],
    device uint* index [[buffer(1)]],
    device float* weight [[buffer(2)]])
{
    float bestv[8];
    uint besti[8];
    for (uint k = 0u; k < 8u; ++k) {
        bestv[k] = -INFINITY;
        besti[k] = 0u;
    }
    for (uint i = 0u; i < 256u; ++i) {
        float value = logits[i];
        uint place = 8u;
        for (uint k = 0u; k < 8u; ++k) {
            if (value > bestv[k]) {
                place = k;
                break;
            }
        }
        if (place == 8u) {
            continue;
        }
        for (uint k = 7u; k > place; --k) {
            bestv[k] = bestv[k - 1u];
            besti[k] = besti[k - 1u];
        }
        bestv[place] = value;
        besti[place] = i;
    }
    float peak = bestv[0];
    float sum = 0.0f;
    float raised[8];
    for (uint k = 0u; k < 8u; ++k) {
        raised[k] = exp(bestv[k] - peak);
        sum += raised[k];
    }
    for (uint k = 0u; k < 8u; ++k) {
        index[k] = besti[k];
        weight[k] = raised[k] / sum;
    }
}

// Checkpoint router rows become file slots. ``order`` is 256 bytes per layer.
kernel void remap_expert_ids(
    device uint* ids [[buffer(0)]],
    device const uchar* order [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant uint& layer [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) {
        return;
    }
    ids[tid] = order[layer * 256u + ids[tid]];
}

kernel void mix_experts(
    device float* destination [[buffer(0)]],
    device const float* partial [[buffer(1)]],
    device const float* shared [[buffer(2)]],
    device const float* weights [[buffer(3)]],
    device const float* coeff [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    float acc = coeff[0] * shared[tid];
    for (uint slot = 0u; slot < 8u; ++slot) {
        acc += weights[slot] * partial[slot * 2048u + tid];
    }
    destination[tid] = acc;
}

// qkv, z, a, and b of one Gated DeltaNet layer. One launch, independent rows.
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
    float sum = 0.0f;
    if (job < 8192u) {
        sum = fp4_vec_dot(activation, qkv_w, qkv_s, job, 2048u, lane);
        if (lane == 0u) {
            qkv[job] = sum;
        }
        return;
    }
    if (job < 12288u) {
        uint row = job - 8192u;
        sum = fp4_vec_dot(activation, z_w, z_s, row, 2048u, lane);
        if (lane == 0u) {
            z[row] = sum;
        }
        return;
    }
    if (job < 12320u) {
        uint row = job - 12288u;
        sum = fp4_vec_dot(activation, a_w, a_s, row, 2048u, lane);
        if (lane == 0u) {
            a[row] = sum;
        }
        return;
    }
    if (job < 12352u) {
        uint row = job - 12320u;
        sum = fp4_vec_dot(activation, b_w, b_s, row, 2048u, lane);
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
    if (job < 8192u) {
        float sum = fp4_vec_dot(activation, q_w, q_s, job, 2048u, lane);
        if (lane == 0u) {
            query[job] = sum;
        }
        return;
    }
    if (job < 8704u) {
        uint row = job - 8192u;
        float sum = fp4_vec_dot(activation, k_w, k_s, row, 2048u, lane);
        if (lane == 0u) {
            key[row] = sum;
        }
        return;
    }
    if (job < 9216u) {
        uint row = job - 8704u;
        float sum = fp4_vec_dot(activation, v_w, v_s, row, 2048u, lane);
        if (lane == 0u) {
            value[row] = sum;
        }
    }
}

kernel void gemv_fp4_vec_acc(
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
        destination[row] += sum;
    }
}

kernel void gemv_f16_f32(
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
    device const half* rowp = weights + row * inner;
    float acc = 0.0f;
    for (uint i = lane; i < inner; i += 32u) {
        acc += float(rowp[i]) * activation[i];
    }
    float sum = simd_sum(acc);
    if (lane == 0u) {
        destination[row] = sum;
    }
}

// Router rows 0..255 and the shared coefficient at row 256, then sigmoid.
kernel void router_coeff(
    device const float* activation [[buffer(0)]],
    device const half* router_w [[buffer(1)]],
    device const half* coeff_w [[buffer(2)]],
    device float* router [[buffer(3)]],
    device float* coeff [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    device const half* rowp = row < 256u ? router_w + row * 2048u : coeff_w;
    float acc = 0.0f;
    for (uint i = lane; i < 2048u; i += 32u) {
        acc += float(rowp[i]) * activation[i];
    }
    float sum = simd_sum(acc);
    if (lane != 0u) {
        return;
    }
    if (row < 256u) {
        router[row] = sum;
    } else if (row == 256u) {
        coeff[0] = sigmoid_f(sum);
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

// Sixteen query heads and two key heads. RMSNorm, then the 64-channel RoPE.
kernel void qk_norm_rope(
    device float* query [[buffer(0)]],
    device float* key [[buffer(1)]],
    device const half* q_weight [[buffer(2)]],
    device const half* k_weight [[buffer(3)]],
    constant uint* positions [[buffer(4)]],
    constant float& theta [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint head [[threadgroup_position_in_grid]])
{
    threadgroup float red[8];
    device float* row = head < 16u ? query + head * 256u : key + (head - 16u) * 256u;
    device const half* weight = head < 16u ? q_weight : k_weight;
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
    uint axis = mrope_axis(tid);
    float angle = float(positions[axis]) * pow(theta, -float(tid) / 32.0f);
    float c = cos(angle);
    float s = sin(angle);
    uint i0 = tid;
    uint i1 = tid + 32u;
    float x0 = row[i0];
    float x1 = row[i1];
    row[i0] = x0 * c - x1 * s;
    row[i1] = x0 * s + x1 * c;
}

kernel void store_kv(
    device float* key_cache [[buffer(0)]],
    device float* value_cache [[buffer(1)]],
    device const float* key [[buffer(2)]],
    device const float* value [[buffer(3)]],
    constant uint& pos [[buffer(4)]],
    constant uint& stride [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint head [[threadgroup_position_in_grid]])
{
    key_cache[head * stride + pos * 256u + tid] = key[head * 256u + tid];
    value_cache[head * stride + pos * 256u + tid] = value[head * 256u + tid];
}

kernel void attend_gqa_heads(
    device const float* query [[buffer(0)]],
    device const float* key_cache [[buffer(1)]],
    device const float* value_cache [[buffer(2)]],
    device const float* gate [[buffer(3)]],
    device float* destination [[buffer(4)]],
    constant uint& tokens [[buffer(5)]],
    constant uint& stride [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint head [[threadgroup_position_in_grid]])
{
    threadgroup float red[8];
    uint kv = head >> 3;
    device const float* key = key_cache + kv * stride;
    device const float* value = value_cache + kv * stride;
    device const float* q = query + head * 256u;
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    for (uint token = 0u; token < tokens; ++token) {
        float score = head_dot(tid, lane, simd, 256u, red, q, key + token * 256u) * INV_SQRT_256;
        float next = max(running_max, score);
        running_sum = running_sum * exp(running_max - next) + exp(score - next);
        running_max = next;
    }
    float acc = 0.0f;
    for (uint token = 0u; token < tokens; ++token) {
        float score = head_dot(tid, lane, simd, 256u, red, q, key + token * 256u) * INV_SQRT_256;
        float weight = exp(score - running_max) / running_sum;
        acc += weight * value[token * 256u + tid];
    }
    destination[head * 256u + tid] = acc * sigmoid_f(gate[head * 256u + tid]);
}

// Eight routed gate-up rows, plus the shared expert. Each job owns one of the
// 512 gate rows and the matching up row, and writes SiLU(gate) * up.
kernel void moe_gate_silu(
    device const float* activation [[buffer(0)]],
    device const uchar* file [[buffer(1)]],
    device const uint* ids [[buffer(2)]],
    device const uchar* shared_w [[buffer(3)]],
    device const uchar* shared_s [[buffer(4)]],
    device float* gated [[buffer(5)]],
    device float* shared_gated [[buffer(6)]],
    constant uint& image [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint job [[threadgroup_position_in_grid]])
{
    if (job < 4096u) {
        uint slot = job >> 9;
        uint row = job & 511u;
        device const uchar* base = file + (ulong)ids[slot] * (ulong)image;
        float gate = fp4_vec_dot(activation, base, base + 1048576u, row, 2048u, lane);
        float up = fp4_vec_dot(activation, base, base + 1048576u, row + 512u, 2048u, lane);
        if (lane == 0u) {
            gated[slot * 512u + row] = silu_f(gate) * up;
        }
        return;
    }
    if (job < 4608u) {
        uint row = job - 4096u;
        float gate = fp4_vec_dot(activation, shared_w, shared_s, row, 2048u, lane);
        float up = fp4_vec_dot(activation, shared_w, shared_s, row + 512u, 2048u, lane);
        if (lane == 0u) {
            shared_gated[row] = silu_f(gate) * up;
        }
    }
}

// One output row accumulates eight routed downs and the shared down, then
// adds that residual into the hidden state.
kernel void moe_down_mix(
    device const float* gated [[buffer(0)]],
    device const float* shared_gated [[buffer(1)]],
    device const uchar* file [[buffer(2)]],
    device const uint* ids [[buffer(3)]],
    device const float* mix [[buffer(4)]],
    device const uchar* shared_w [[buffer(5)]],
    device const uchar* shared_s [[buffer(6)]],
    device const float* coeff [[buffer(7)]],
    device float* hidden [[buffer(8)]],
    constant uint& image [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    if (row >= 2048u) {
        return;
    }
    float acc = 0.0f;
    for (uint slot = 0u; slot < 8u; ++slot) {
        device const uchar* down = file + (ulong)ids[slot] * (ulong)image + 1114112ul;
        float dot = fp4_vec_dot(gated + slot * 512u, down, down + 524288u, row, 512u, lane);
        acc += mix[slot] * dot;
    }
    float shared = fp4_vec_dot(shared_gated, shared_w, shared_s, row, 512u, lane);
    acc += coeff[0] * shared;
    if (lane == 0u) {
        hidden[row] += acc;
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

// Causal width-4 convolution. Each thread is one token and one channel.
// The delay line is the three inputs before this token, then the current one.
kernel void gdn_conv_rows(
    device const half* history [[buffer(0)]],
    device const float* qkv [[buffer(1)]],
    device const half* weight [[buffer(2)]],
    device float* mixed [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    constant uint& channels [[buffer(5)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
    uint token = tgid.x;
    uint channel = tgid.y * 256u + tid.x;
    if (token >= count || channel >= channels) {
        return;
    }
    float tap[4];
    tap[3] = qkv[token * channels + channel];
    for (uint age = 1u; age <= 3u; ++age) {
        int index = int(token) - int(age);
        if (index >= 0) {
            tap[3u - age] = qkv[uint(index) * channels + channel];
        } else {
            tap[3u - age] = float(history[(uint(index + 3)) * channels + channel]);
        }
    }
    float acc = 0.0f;
    for (uint i = 0u; i < 4u; ++i) {
        acc += tap[i] * float(weight[channel * 4u + i]);
    }
    mixed[token * channels + channel] = silu_f(acc);
}

kernel void gdn_conv_keep(
    device half* history [[buffer(0)]],
    device const float* qkv [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant uint& channels [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    uint channel = tgid * 256u + tid;
    if (channel >= channels || count == 0u) {
        return;
    }
    float kept[3];
    for (uint age = 0u; age < 3u; ++age) {
        int index = int(count) - 3 + int(age);
        if (index >= 0) {
            kept[age] = qkv[uint(index) * channels + channel];
        } else {
            kept[age] = float(history[(uint(index + 3)) * channels + channel]);
        }
    }
    history[channel] = half(kept[0]);
    history[channels + channel] = half(kept[1]);
    history[2u * channels + channel] = half(kept[2]);
}

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
    threadgroup float red[4];
    threadgroup float qshare[128];
    threadgroup float kshare[128];
    uint qhead = head / 2u;
    for (uint token = 0u; token < count; ++token) {
        device const float* mix = mixed + token * 8192u;
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
        device const float* avec = raw_a + token * 32u;
        device const float* bvec = raw_b + token * 32u;
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
        float delta = beta * (mix[4096u + head * 128u + tid] - retrieved);
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
        device const float* z = gate + token * 4096u;
        destination[token * 4096u + head * 128u + tid] = gated * silu_f(z[head * 128u + tid]);
    }
}

// L2-normalize query and key for every token. Query is scaled by 1/sqrt(128).
kernel void gdn_l2_rows(
    device float* mixed [[buffer(0)]],
    constant float& eps [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    if (tgid.x >= count) {
        return;
    }
    threadgroup float red[4];
    device float* q = mixed + tgid.x * 8192u + tgid.y * 128u;
    device float* k = mixed + tgid.x * 8192u + 2048u + tgid.y * 128u;
    float qv = q[tid];
    float kv = k[tid];
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
    q[tid] = qv * rsqrt(qtot + eps) * INV_SQRT_128;
    k[tid] = kv * rsqrt(ktot + eps);
}

// One chunk of the gated delta rule (Yang et al., arXiv:2412.06464, §3.3).
// W is the ungated Householder factor. U carries the cumulative decay.
// The chunk state is gamma_last * S + Vbar^T K_right, with
// Vbar = U - (gamma * W) S^T.
kernel void gdn_chunk_solve(
    device const float* state [[buffer(0)]],
    device const float* mixed [[buffer(1)]],
    device const float* raw_a [[buffer(2)]],
    device const float* raw_b [[buffer(3)]],
    device const half* a_log [[buffer(4)]],
    device const half* dt_bias [[buffer(5)]],
    device float* kkt [[buffer(6)]],
    device float* wbuf [[buffer(7)]],
    device float* ubuf [[buffer(8)]],
    device float* vbar [[buffer(9)]],
    device float* logg [[buffer(10)]],
    constant uint& c0 [[buffer(11)]],
    constant uint& clen [[buffer(12)]],
    uint tid [[thread_index_in_threadgroup]],
    uint head [[threadgroup_position_in_grid]])
{
    uint qhead = head >> 1;
    device float* gram = kkt + head * 64u * 64u;
    device float* ww = wbuf + head * 64u * 128u;
    device float* uu = ubuf + head * 64u * 128u;
    device float* vv = vbar + head * 64u * 128u;
    device float* lg = logg + head * 64u;
    if (tid < clen) {
        float along = float(a_log[head]);
        float aval = raw_a[(c0 + tid) * 32u + head];
        lg[tid] = -exp(along) * softplus_f(aval + float(dt_bias[head]));
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0u) {
        float acc = 0.0f;
        for (uint r = 0u; r < clen; ++r) {
            acc += lg[r];
            lg[r] = acc;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (tid < clen) {
        for (uint r = 0u; r < clen; ++r) {
            device const float* kr = mixed + (c0 + r) * 8192u + 2048u + qhead * 128u;
            device const float* ki = mixed + (c0 + tid) * 8192u + 2048u + qhead * 128u;
            float dot = 0.0f;
            for (uint j = 0u; j < 128u; ++j) {
                dot += kr[j] * ki[j];
            }
            gram[r * 64u + tid] = dot;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    for (uint r = 0u; r < clen; ++r) {
        float beta = sigmoid_f(raw_b[(c0 + r) * 32u + head]);
        device const float* vr = mixed + (c0 + r) * 8192u + 4096u + head * 128u;
        device const float* kr = mixed + (c0 + r) * 8192u + 2048u + qhead * 128u;
        float acc_u = beta * vr[tid];
        float acc_w = beta * kr[tid];
        for (uint i = 0u; i < r; ++i) {
            float kk = gram[r * 64u + i];
            float decay = exp(lg[r] - lg[i]);
            acc_u -= beta * decay * kk * uu[i * 128u + tid];
            acc_w -= beta * kk * ww[i * 128u + tid];
        }
        uu[r * 128u + tid] = acc_u;
        ww[r * 128u + tid] = acc_w;
        threadgroup_barrier(mem_flags::mem_device);
    }
    device const float* srow = state + (head * 128u + tid) * 128u;
    for (uint r = 0u; r < clen; ++r) {
        float dot = 0.0f;
        device const float* wr = ww + r * 128u;
        for (uint j = 0u; j < 128u; ++j) {
            dot += srow[j] * wr[j];
        }
        vv[r * 128u + tid] = uu[r * 128u + tid] - exp(lg[r]) * dot;
    }
}

kernel void gdn_chunk_out(
    device const float* state [[buffer(0)]],
    device const float* mixed [[buffer(1)]],
    device const float* vbar [[buffer(2)]],
    device const float* logg [[buffer(3)]],
    device float* destination [[buffer(4)]],
    constant uint& c0 [[buffer(5)]],
    constant uint& clen [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    uint head = tgid.x;
    uint r = tgid.y;
    if (r >= clen) {
        return;
    }
    uint qhead = head >> 1;
    threadgroup float scores[64];
    if (tid < clen) {
        device const float* qq = mixed + (c0 + r) * 8192u + qhead * 128u;
        device const float* kk = mixed + (c0 + tid) * 8192u + 2048u + qhead * 128u;
        float dot = 0.0f;
        for (uint j = 0u; j < 128u; ++j) {
            dot += qq[j] * kk[j];
        }
        scores[tid] = dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const float* lg = logg + head * 64u;
    device const float* vv = vbar + head * 64u * 128u;
    device const float* srow = state + (head * 128u + tid) * 128u;
    device const float* qq = mixed + (c0 + r) * 8192u + qhead * 128u;
    float acc = 0.0f;
    for (uint j = 0u; j < 128u; ++j) {
        acc += srow[j] * qq[j];
    }
    acc *= exp(lg[r]);
    for (uint i = 0u; i <= r; ++i) {
        acc += exp(lg[r] - lg[i]) * scores[i] * vv[i * 128u + tid];
    }
    destination[(c0 + r) * 4096u + head * 128u + tid] = acc;
}

kernel void gdn_chunk_state(
    device float* state [[buffer(0)]],
    device const float* mixed [[buffer(1)]],
    device const float* vbar [[buffer(2)]],
    device const float* logg [[buffer(3)]],
    constant uint& c0 [[buffer(4)]],
    constant uint& clen [[buffer(5)]],
    uint col [[thread_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    uint head = tgid.x;
    uint row = tgid.y;
    uint qhead = head >> 1;
    device const float* lg = logg + head * 64u;
    device const float* vv = vbar + (head * 64u + 0u) * 128u + row;
    device float* srow = state + (head * 128u + row) * 128u;
    float acc = exp(lg[clen - 1u]) * srow[col];
    for (uint r = 0u; r < clen; ++r) {
        float key = mixed[(c0 + r) * 8192u + 2048u + qhead * 128u + col];
        key *= exp(lg[clen - 1u] - lg[r]);
        acc += vv[r * 128u] * key;
    }
    srow[col] = acc;
}

kernel void gdn_y_norm(
    device float* y [[buffer(0)]],
    device const float* gate [[buffer(1)]],
    device const half* norm_w [[buffer(2)]],
    constant float& eps [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    if (tgid.x >= count) {
        return;
    }
    threadgroup float red[4];
    device float* row = y + tgid.x * 4096u + tgid.y * 128u;
    float value = row[tid];
    float part = simd_sum(value * value);
    if (lane == 0u) {
        red[simd] = part;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ytot = red[0] + red[1] + red[2] + red[3];
    float inv = rsqrt(ytot / 128.0f + eps);
    float gated = value * inv * float(norm_w[tid]);
    device const float* z = gate + tgid.x * 4096u + tgid.y * 128u;
    row[tid] = gated * silu_f(z[tid]);
}

kernel void split_q_gate_rows(
    device const float* mixed [[buffer(0)]],
    device float* query [[buffer(1)]],
    device float* gate [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    uint span = count * 4096u;
    if (tid >= span) {
        return;
    }
    uint token = tid >> 12;
    uint local = tid & 4095u;
    uint head = local >> 8;
    uint lane = local & 255u;
    uint base = token * 8192u + head * 512u + lane;
    query[tid] = mixed[base];
    gate[tid] = mixed[base + 256u];
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
    device float* row = head < 16u ? query + (token * 16u + head) * 256u
                                   : key + (token * 2u + (head - 16u)) * 256u;
    device const half* weight = head < 16u ? q_weight : k_weight;
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
    // Text puts the same position on every mRoPE axis. The first 64 channels
    // rotate as two halves: channel i with channel i+32.
    float position = float(pos0 + token);
    float angle = position * pow(theta, -float(tid) / 32.0f);
    float c = cos(angle);
    float s = sin(angle);
    uint i0 = tid;
    uint i1 = tid + 32u;
    float x0 = row[i0];
    float x1 = row[i1];
    row[i0] = x0 * c - x1 * s;
    row[i1] = x0 * s + x1 * c;
}

kernel void store_kv_rows(
    device float* key_cache [[buffer(0)]],
    device float* value_cache [[buffer(1)]],
    device const float* key [[buffer(2)]],
    device const float* value [[buffer(3)]],
    constant uint& pos0 [[buffer(4)]],
    constant uint& stride [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    uint token = tgid.x;
    uint kv = tgid.y;
    uint pos = pos0 + token;
    key_cache[kv * stride + pos * 256u + tid] = key[(token * 2u + kv) * 256u + tid];
    value_cache[kv * stride + pos * 256u + tid] = value[(token * 2u + kv) * 256u + tid];
}

// One query, one head. Scores for eight keys are loaded before the softmax
// update so those loads are independent.
kernel void attend_prefill(
    device const float* query [[buffer(0)]],
    device const float* key_cache [[buffer(1)]],
    device const float* value_cache [[buffer(2)]],
    device const float* gate [[buffer(3)]],
    device float* destination [[buffer(4)]],
    constant uint& count [[buffer(5)]],
    constant uint& pos0 [[buffer(6)]],
    constant uint& stride [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    uint token = tgid.x;
    uint head = tgid.y;
    if (token >= count || head >= 16u) {
        return;
    }
    uint kv = head >> 3;
    device const float* keys = key_cache + kv * stride;
    device const float* values = value_cache + kv * stride;
    device const float* q = query + (token * 16u + head) * 256u;
    float qreg[8];
    for (uint i = 0u; i < 8u; ++i) {
        qreg[i] = q[lane * 8u + i];
    }
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float outv[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    uint tokens = pos0 + token + 1u;
    uint step = 0u;
    for (; step + 8u <= tokens; step += 8u) {
        float score[8];
        for (uint b = 0u; b < 8u; ++b) {
            device const float* ks = keys + (step + b) * 256u;
            float partial = 0.0f;
            for (uint i = 0u; i < 8u; ++i) {
                partial += qreg[i] * ks[lane * 8u + i];
            }
            score[b] = simd_sum(partial) * INV_SQRT_256;
        }
        for (uint b = 0u; b < 8u; ++b) {
            float next = max(running_max, score[b]);
            float rescale = exp(running_max - next);
            float prob = exp(score[b] - next);
            running_sum = running_sum * rescale + prob;
            device const float* vs = values + (step + b) * 256u;
            for (uint i = 0u; i < 8u; ++i) {
                outv[i] = outv[i] * rescale + prob * vs[lane * 8u + i];
            }
            running_max = next;
        }
    }
    for (; step < tokens; ++step) {
        device const float* ks = keys + step * 256u;
        float partial = 0.0f;
        for (uint i = 0u; i < 8u; ++i) {
            partial += qreg[i] * ks[lane * 8u + i];
        }
        float score = simd_sum(partial) * INV_SQRT_256;
        float next = max(running_max, score);
        float rescale = exp(running_max - next);
        float prob = exp(score - next);
        running_sum = running_sum * rescale + prob;
        device const float* vs = values + step * 256u;
        for (uint i = 0u; i < 8u; ++i) {
            outv[i] = outv[i] * rescale + prob * vs[lane * 8u + i];
        }
        running_max = next;
    }
    device const float* grow = gate + (token * 16u + head) * 256u;
    device float* dest = destination + (token * 16u + head) * 256u;
    float inv = 1.0f / running_sum;
    for (uint i = 0u; i < 8u; ++i) {
        dest[lane * 8u + i] = outv[i] * inv * sigmoid_f(grow[lane * 8u + i]);
    }
}

kernel void attend_prefill2(
    device const float* query [[buffer(0)]],
    device const float* key_cache [[buffer(1)]],
    device const float* value_cache [[buffer(2)]],
    device const float* gate [[buffer(3)]],
    device float* destination [[buffer(4)]],
    constant uint& count [[buffer(5)]],
    constant uint& pos0 [[buffer(6)]],
    constant uint& stride [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    uint token = tgid.x * 2u;
    uint head = tgid.y;
    if (token >= count || head >= 16u) {
        return;
    }
    uint token1 = token + 1u;
    bool second = token1 < count;
    uint kv = head >> 3;
    device const float* keys = key_cache + kv * stride;
    device const float* values = value_cache + kv * stride;
    device const float* q = query + (token * 16u + head) * 256u;
    float qreg[8];
    float qreg1[8];
    for (uint i = 0u; i < 8u; ++i) {
        qreg[i] = q[lane * 8u + i];
        qreg1[i] = 0.0f;
    }
    if (second) {
        device const float* q1 = query + (token1 * 16u + head) * 256u;
        for (uint i = 0u; i < 8u; ++i) {
            qreg1[i] = q1[lane * 8u + i];
        }
    }
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float running_max1 = -INFINITY;
    float running_sum1 = 0.0f;
    float outv[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float outv1[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    uint tokens = pos0 + (second ? token1 : token) + 1u;
    uint limit0 = pos0 + token + 1u;
    uint step = 0u;
    for (; step + 8u <= tokens; step += 8u) {
        float score[8];
        float score1[8];
        for (uint b = 0u; b < 8u; ++b) {
            device const float* ks = keys + (step + b) * 256u;
            float partial = 0.0f;
            float partial1 = 0.0f;
            for (uint i = 0u; i < 8u; ++i) {
                float key = ks[lane * 8u + i];
                partial += qreg[i] * key;
                partial1 += qreg1[i] * key;
            }
            score[b] = simd_sum(partial) * INV_SQRT_256;
            score1[b] = simd_sum(partial1) * INV_SQRT_256;
            if (step + b >= limit0) {
                score[b] = -INFINITY;
            }
        }
        for (uint b = 0u; b < 8u; ++b) {
            device const float* vs = values + (step + b) * 256u;
            if (score[b] != -INFINITY) {
                float next = max(running_max, score[b]);
                float rescale = exp(running_max - next);
                float prob = exp(score[b] - next);
                running_sum = running_sum * rescale + prob;
                for (uint i = 0u; i < 8u; ++i) {
                    outv[i] = outv[i] * rescale + prob * vs[lane * 8u + i];
                }
                running_max = next;
            }
            if (second) {
                float next1 = max(running_max1, score1[b]);
                float rescale1 = exp(running_max1 - next1);
                float prob1 = exp(score1[b] - next1);
                running_sum1 = running_sum1 * rescale1 + prob1;
                for (uint i = 0u; i < 8u; ++i) {
                    outv1[i] = outv1[i] * rescale1 + prob1 * vs[lane * 8u + i];
                }
                running_max1 = next1;
            }
        }
    }
    for (; step < tokens; ++step) {
        device const float* ks = keys + step * 256u;
        float partial = 0.0f;
        float partial1 = 0.0f;
        for (uint i = 0u; i < 8u; ++i) {
            float key = ks[lane * 8u + i];
            partial += qreg[i] * key;
            partial1 += qreg1[i] * key;
        }
        device const float* vs = values + step * 256u;
        if (step < limit0) {
            float value = simd_sum(partial) * INV_SQRT_256;
            float next = max(running_max, value);
            float rescale = exp(running_max - next);
            float prob = exp(value - next);
            running_sum = running_sum * rescale + prob;
            for (uint i = 0u; i < 8u; ++i) {
                outv[i] = outv[i] * rescale + prob * vs[lane * 8u + i];
            }
            running_max = next;
        }
        if (second) {
            float value1 = simd_sum(partial1) * INV_SQRT_256;
            float next1 = max(running_max1, value1);
            float rescale1 = exp(running_max1 - next1);
            float prob1 = exp(value1 - next1);
            running_sum1 = running_sum1 * rescale1 + prob1;
            for (uint i = 0u; i < 8u; ++i) {
                outv1[i] = outv1[i] * rescale1 + prob1 * vs[lane * 8u + i];
            }
            running_max1 = next1;
        }
    }
    device float* dest = destination + (token * 16u + head) * 256u;
    device const float* grow = gate + (token * 16u + head) * 256u;
    float inv = 1.0f / running_sum;
    for (uint i = 0u; i < 8u; ++i) {
        dest[lane * 8u + i] = outv[i] * inv * sigmoid_f(grow[lane * 8u + i]);
    }
    if (!second) {
        return;
    }
    device float* dest1 = destination + (token1 * 16u + head) * 256u;
    device const float* grow1 = gate + (token1 * 16u + head) * 256u;
    float inv1 = 1.0f / running_sum1;
    for (uint i = 0u; i < 8u; ++i) {
        dest1[lane * 8u + i] = outv1[i] * inv1 * sigmoid_f(grow1[lane * 8u + i]);
    }
}

kernel void gemv_f16_grid(
    device const float* activation [[buffer(0)]],
    device const half* weights [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    uint row = tgid.x;
    uint token = tgid.y;
    if (row >= cols || token >= count) {
        return;
    }
    device const half* rowp = weights + row * inner;
    device const float* act = activation + token * inner;
    float acc = 0.0f;
    for (uint i = lane; i < inner; i += 32u) {
        acc += float(rowp[i]) * act[i];
    }
    float sum = simd_sum(acc);
    if (lane == 0u) {
        destination[token * cols + row] = sum;
    }
}

kernel void gemv_f16_rows(
    device const float* activation [[buffer(0)]],
    device const half* weights [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    if (row >= cols) {
        return;
    }
    device const half* rowp = weights + row * inner;
    for (uint base = 0u; base < count; base += 8u) {
        float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        uint live = min(count - base, 8u);
        for (uint i = lane; i < inner; i += 32u) {
            float w = float(rowp[i]);
            for (uint t = 0u; t < live; ++t) {
                acc[t] += w * activation[(base + t) * inner + i];
            }
        }
        for (uint t = 0u; t < live; ++t) {
            float sum = simd_sum(acc[t]);
            if (lane == 0u) {
                destination[(base + t) * cols + row] = sum;
            }
        }
    }
}

kernel void coeff_rows(
    device const float* activation [[buffer(0)]],
    device const half* weights [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant uint& inner [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint token [[threadgroup_position_in_grid]])
{
    if (token >= count) {
        return;
    }
    device const float* act = activation + token * inner;
    float acc = 0.0f;
    for (uint i = lane; i < inner; i += 32u) {
        acc += float(weights[i]) * act[i];
    }
    float sum = simd_sum(acc);
    if (lane == 0u) {
        destination[token] = sigmoid_f(sum);
    }
}

kernel void router8_rows(
    device const float* logits [[buffer(0)]],
    device uint* index [[buffer(1)]],
    device float* weight [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint token [[thread_position_in_grid]])
{
    if (token >= count) {
        return;
    }
    logits += token * 256u;
    float bestv[8];
    uint besti[8];
    for (uint k = 0u; k < 8u; ++k) {
        bestv[k] = -INFINITY;
        besti[k] = 0u;
    }
    for (uint i = 0u; i < 256u; ++i) {
        float value = logits[i];
        uint place = 8u;
        for (uint k = 0u; k < 8u; ++k) {
            if (value > bestv[k]) {
                place = k;
                break;
            }
        }
        if (place == 8u) {
            continue;
        }
        for (uint k = 7u; k > place; --k) {
            bestv[k] = bestv[k - 1u];
            besti[k] = besti[k - 1u];
        }
        bestv[place] = value;
        besti[place] = i;
    }
    float peak = bestv[0];
    float sum = 0.0f;
    float raised[8];
    for (uint k = 0u; k < 8u; ++k) {
        raised[k] = exp(bestv[k] - peak);
        sum += raised[k];
    }
    index += token * 8u;
    weight += token * 8u;
    for (uint k = 0u; k < 8u; ++k) {
        index[k] = besti[k];
        weight[k] = raised[k] / sum;
    }
}

kernel void silu_mul_rows(
    device const float* gate_up [[buffer(0)]],
    device float* gated [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    uint span = rows * 512u;
    if (tid >= span) {
        return;
    }
    uint token = tid >> 9;
    uint row = tid & 511u;
    float gate = gate_up[token * 1024u + row];
    float up = gate_up[token * 1024u + 512u + row];
    gated[tid] = silu_f(gate) * up;
}

kernel void gather_half_rows(
    device const float* source [[buffer(0)]],
    device const uint* index [[buffer(1)]],
    device half* destination [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant uint& inner [[buffer(4)]],
    constant uint& stride [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    if (row >= count) {
        return;
    }
    device const float* src = source + index[row] * stride;
    device half* dest = destination + row * inner;
    for (uint i = tid; i < inner; i += 256u) {
        dest[i] = half(src[i]);
    }
}

kernel void gather_half_slots(
    device const float* source [[buffer(0)]],
    device const uint* index [[buffer(1)]],
    device half* pack [[buffer(2)]],
    device const uint* slot_n [[buffer(3)]],
    device const uint* slot_csr [[buffer(4)]],
    constant uint& max_pad [[buffer(5)]],
    constant uint& inner [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    uint row = tgid.x;
    uint slot = tgid.y;
    if (row >= slot_n[slot]) {
        return;
    }
    uint token = index[slot_csr[slot] + row];
    device const float* src = source + token * inner;
    device half* dest = pack + (slot * max_pad + row) * inner;
    for (uint i = tid; i < inner; i += 256u) {
        dest[i] = half(src[i]);
    }
}

kernel void silu_wave(
    device const float* gate_up [[buffer(0)]],
    device float* gated [[buffer(1)]],
    constant uint& max_pad [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    uint local = (tid >> 9) % max_pad;
    uint slot = tid / (max_pad << 9);
    uint row = tid & 511u;
    uint span_row = slot * max_pad + local;
    float gate = gate_up[span_row * 1024u + row];
    float up = gate_up[span_row * 1024u + 512u + row];
    gated[span_row * 512u + row] = silu_f(gate) * up;
}

kernel void moe_group(
    device const uint* ids [[buffer(0)]],
    device const float* mix [[buffer(1)]],
    device uint* csr_i [[buffer(2)]],
    device float* csr_w [[buffer(3)]],
    device uint* expert_of [[buffer(4)]],
    device uint* slot_n [[buffer(5)]],
    device uint* slot_csr [[buffer(6)]],
    device uint* row_base [[buffer(7)]],
    device uint* jobs [[buffer(8)]],
    device uint* info [[buffer(9)]],
    device uint* csr_lane [[buffer(10)]],
    device uint* pack_row [[buffer(11)]],
    constant uint& count [[buffer(12)]],
    uint tid [[thread_index_in_threadgroup]])
{
    // One thread keeps the stable token order. The other threads only fill the group.
    if (tid != 0u) {
        return;
    }
    uint counts[256];
    uint filled[256];
    uint offsets[256];
    for (uint expert = 0u; expert < 256u; ++expert) {
        counts[expert] = 0u;
        filled[expert] = 0u;
    }
    uint nassign = count * 8u;
    for (uint index = 0u; index < nassign; ++index) {
        uint expert = ids[index];
        if (expert >= 256u) {
            continue;
        }
        counts[expert] += 1u;
    }
    uint cursor = 0u;
    for (uint expert = 0u; expert < 256u; ++expert) {
        uint align = (4u - (cursor & 3u)) & 3u;
        cursor += align;
        offsets[expert] = cursor;
        cursor += counts[expert];
    }
    for (uint token = 0u; token < count; ++token) {
        for (uint lane = 0u; lane < 8u; ++lane) {
            uint expert = ids[token * 8u + lane];
            if (expert >= 256u) {
                pack_row[token * 8u + lane] = 0xffffffffu;
                continue;
            }
            uint at = offsets[expert] + filled[expert];
            filled[expert] += 1u;
            csr_i[at] = token;
            csr_w[at] = mix[token * 8u + lane];
            csr_lane[at] = lane;
        }
    }
    uint slot = 0u;
    uint row = 0u;
    uint job = 0u;
    for (uint expert = 0u; expert < 256u; ++expert) {
        uint n = counts[expert];
        if (n == 0u) {
            continue;
        }
        uint pad = (n + 63u) & ~63u;
        expert_of[slot] = expert;
        slot_n[slot] = n;
        slot_csr[slot] = offsets[expert];
        row_base[slot] = row;
        uint tiles = pad >> 6u;
        for (uint tile = 0u; tile < tiles; ++tile) {
            jobs[job * 2u] = slot;
            jobs[job * 2u + 1u] = tile;
            job += 1u;
        }
        for (uint local = 0u; local < n; ++local) {
            uint at = offsets[expert] + local;
            pack_row[csr_i[at] * 8u + csr_lane[at]] = row + local;
        }
        row += pad;
        slot += 1u;
    }
    info[0] = slot;
    info[1] = job;
    info[2] = row;
    info[3] = 0u;
    // Each record is 4 uints, 16 bytes. [0]=gather [1]=zero [2]=gate [3]=silu [4]=cast [5]=down.
    uint jobs_y = job == 0u ? 1u : job;
    uint rows_g = row == 0u ? 1u : row * 2u;
    uint zero_g = row == 0u ? 1u : row * 4u;
    info[4] = jobs_y;
    info[5] = 1u;
    info[6] = 1u;
    info[7] = 0u;
    info[8] = zero_g;
    info[9] = 1u;
    info[10] = 1u;
    info[11] = 0u;
    info[12] = 16u;
    info[13] = jobs_y;
    info[14] = 1u;
    info[15] = 0u;
    info[16] = rows_g;
    info[17] = 1u;
    info[18] = 1u;
    info[19] = 0u;
    info[20] = rows_g;
    info[21] = 1u;
    info[22] = 1u;
    info[23] = 0u;
    info[24] = 32u;
    info[25] = jobs_y;
    info[26] = 1u;
    info[27] = 0u;
}

kernel void gather_half_grouped(
    device const float* source [[buffer(0)]],
    device const uint* index [[buffer(1)]],
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
    uint begin = tile * 64u;
    uint end = begin + 64u;
    if (end > n) {
        end = n;
    }
    uint csr = slot_csr[slot] + begin;
    uint base = row_base[slot] + begin;
    for (uint row = begin; row < end; ++row) {
        uint token = index[csr];
        device const float* src = source + token * inner;
        device half* dest = pack + base * inner;
        for (uint i = tid; i < inner; i += 256u) {
            dest[i] = half(src[i]);
        }
        csr += 1u;
        base += 1u;
    }
}

kernel void silu_mul_grouped(
    device const float* gate_up [[buffer(0)]],
    device float* gated [[buffer(1)]],
    device const uint* info [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    uint rows = info[2];
    uint span = rows * 512u;
    if (tid >= span) {
        return;
    }
    uint token = tid >> 9u;
    uint row = tid & 511u;
    float gate = gate_up[token * 1024u + row];
    float up = gate_up[token * 1024u + 512u + row];
    gated[tid] = silu_f(gate) * up;
}

kernel void cast_f32_f16_n(
    device const float* source [[buffer(0)]],
    device half* destination [[buffer(1)]],
    device const uint* info [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    uint n = info[2] * 512u;
    if (tid >= n) {
        return;
    }
    destination[tid] = half(source[tid]);
}

kernel void scatter_add_grouped(
    device float* hidden [[buffer(0)]],
    device const float* partial [[buffer(1)]],
    device const uint* pack_row [[buffer(2)]],
    device const float* mix [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& count [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint token [[threadgroup_position_in_grid]])
{
    if (token >= count) {
        return;
    }
    device float* dest = hidden + token * cols;
    for (uint lane = 0u; lane < 8u; ++lane) {
        uint row = pack_row[token * 8u + lane];
        if (row == 0xffffffffu) {
            continue;
        }
        float weight = mix[token * 8u + lane];
        device const float* src = partial + row * cols;
        for (uint i = tid; i < cols; i += 256u) {
            dest[i] += weight * src[i];
        }
    }
}

kernel void scatter_add_rows(
    device float* hidden [[buffer(0)]],
    device const float* partial [[buffer(1)]],
    device const uint* index [[buffer(2)]],
    device const float* scale [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    if (row >= count) {
        return;
    }
    float weight = scale[row];
    device float* dest = hidden + index[row] * cols;
    device const float* src = partial + row * cols;
    for (uint i = tid; i < cols; i += 256u) {
        dest[i] += weight * src[i];
    }
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

// Key/value records. Mode 0 is E4M3, one byte and an E8M0 scale every 32
// channels. Mode 1 is fp4 E2M1, low nibble first, with the same scale.
// A head record is the data plane then the eight scale bytes.

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

static uint kv_fp4_nibble(float unit) {
    float ax = fabs(unit);
    float edges[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
    uint choice = 7u;
    for (uint i = 0u; i < 7u; ++i) {
        if (ax <= edges[i]) {
            choice = i;
            break;
        }
    }
    if (choice < 7u && ax == edges[choice] && (choice & 1u) == 1u) {
        choice += 1u;
    }
    uint sign = unit < 0.0f ? 8u : 0u;
    return choice | sign;
}

// Key plane: [head, token, channel], E8M0 every 32 channels.
// Value plane: [head, channel, token], E8M0 every 32 tokens.
// Scores and the value mix are matmul2d on those planes.

kernel void kv_plane_k(
    device uchar* data [[buffer(0)]],
    device uchar* scales [[buffer(1)]],
    device const float* key [[buffer(2)]],
    constant uint& pos0 [[buffer(3)]],
    constant uint& mode [[buffer(4)]],
    constant uint& max_pos [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 local [[thread_position_in_threadgroup]])
{
    uint channel = local.x;
    uint head = tgid.y;
    uint token = pos0 + tgid.x;
    float value = key[(tgid.x * 2u + head) * 256u + channel];
    float qmax = mode == 0u ? 448.0f : 6.0f;
    float peak = simd_max(fabs(value));
    int power = kv_e8m0_power(peak, qmax);
    float unit = value * exp2(-float(power));
    uint block = channel >> 5;
    if ((channel & 31u) == 0u) {
        scales[(head * max_pos + token) * 8u + block] = uchar(power + 127);
    }
    if (mode == 0u) {
        data[(head * max_pos + token) * 256u + channel] = uchar(kv_e4m3_code(unit));
        return;
    }
    uint nibble = kv_fp4_nibble(unit);
    uint paired = simd_shuffle_xor(nibble, 1u);
    if ((channel & 1u) == 0u) {
        data[(head * max_pos + token) * 128u + (channel >> 1)] =
            uchar(nibble | (paired << 4));
    }
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
        values[((src + tgid.x) * 2u + head) * 256u + channel];
}

kernel void kv_plane_v(
    device uchar* data [[buffer(0)]],
    device uchar* scales [[buffer(1)]],
    device const float* tail [[buffer(2)]],
    constant uint& block_start [[buffer(3)]],
    constant uint& valid [[buffer(4)]],
    constant uint& mode [[buffer(5)]],
    constant uint& max_pos [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    uint channel = tgid.x;
    uint head = tgid.y;
    float value = 0.0f;
    if (lane < valid) {
        value = tail[(head * 32u + lane) * 256u + channel];
    }
    float qmax = mode == 0u ? 448.0f : 6.0f;
    float peak = simd_max(fabs(value));
    int power = kv_e8m0_power(peak, qmax);
    uint token = block_start + lane;
    uint scale_index = (head * 256u + channel) * (max_pos >> 5) + (block_start >> 5);
    if (lane == 0u) {
        scales[scale_index] = uchar(power + 127);
    }
    if (mode == 0u) {
        uchar code = 0u;
        if (lane < valid) {
            code = uchar(kv_e4m3_code(value * exp2(-float(power))));
        }
        data[(head * 256u + channel) * max_pos + token] = code;
        return;
    }
    uint nibble = 0u;
    if (lane < valid) {
        nibble = kv_fp4_nibble(value * exp2(-float(power)));
    }
    uint paired = simd_shuffle_xor(nibble, 1u);
    if ((lane & 1u) == 0u) {
        data[(head * 256u + channel) * (max_pos >> 1) + (token >> 1)] =
            uchar(nibble | (paired << 4));
    }
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
    uint global = kv_head * 8u + local;
    destination[row * 256u + channel] = half(query[(token * 16u + global) * 256u + channel]);
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

// Decode softmax over 256-key chunks. Every row is the same query position.
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

// Prefill rows. Query position is pos0 plus the token inside the head group.
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

kernel void kv_pv_split(
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
        64, 64, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto tile_left = left.slice(0, 0);
    auto tile_right = right.slice(0, int(tgid.x) * 64);
    auto tile_out = out.slice(int(tgid.x) * 64, 0);
    op.run(tile_left, tile_right, tile_out);
}

// Same split as kv_pv_split, for the eight live query rows.
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

// One peak per 512-key block. The half values sit in span order so kv_pv_split8
// can mix every block in one dispatch.
kernel void kv_cast_blocks(
    device const float* probs [[buffer(0)]],
    device half* packed [[buffer(1)]],
    device float* peaks [[buffer(2)]],
    constant uint& block [[buffer(3)]],
    constant uint& src_stride [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 lane [[thread_position_in_threadgroup]])
{
    uint part = tgid.x;
    uint row = tgid.y;
    uint col0 = lane.x;
    device const float* src = probs + row * src_stride + part * block;
    float peak = 0.0f;
    for (uint col = col0; col < block; col += 32u) {
        peak = max(peak, fabs(src[col]));
    }
    peak = simd_max(peak);
    float inv = peak > 0.0f ? 1.0f / peak : 0.0f;
    for (uint col = col0; col < block; col += 32u) {
        packed[row * src_stride + part * block + col] = half(src[col] * inv);
    }
    if (col0 == 0u) {
        peaks[part * rows + row] = peak;
    }
}

// Scale each block's partial by that block's peak and add it to the accumulator.
kernel void kv_restore_sum(
    device const float* parts [[buffer(0)]],
    device const float* peaks [[buffer(1)]],
    device float* acc [[buffer(2)]],
    constant uint& blocks [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& channels [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    uint width = rows * channels;
    if (tid >= width) {
        return;
    }
    uint row = tid / channels;
    float sum = 0.0f;
    for (uint part = 0u; part < blocks; ++part) {
        sum += parts[part * width + tid] * peaks[part * rows + row];
    }
    acc[tid] += sum;
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
    if (token >= count) {
        return;
    }
    uint row = local * pad_count + token;
    uint global = kv_head * 8u + local;
    uint index = (token * 16u + global) * 256u + channel;
    float sum = row_sum[row];
    float inv = sum == 0.0f ? 0.0f : 1.0f / sum;
    destination[index] = acc[row * 256u + channel] * inv * sigmoid_f(gate[index]);
}
