// One-token gate, N 17408 by K 5120. matmul2d is the TensorOps op.
// The GEMV is the one-row product. Payloads are constant finite codes.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>

using namespace metal;
using namespace mpp::tensor_ops;

using scales_plane = tensor_blockwise<tensor_plane_scales,
                                      device metal_fp8_ue8m0_format,
                                      32, 1>;

using t_half = half;
using t_bf = bfloat;
using t_f32 = float;
using t_fp4 = metal_fp4_e2m1_format;
using t_e4 = metal_fp8_e4m3_format;
using t_e5 = metal_fp8_e5m2_format;
using t_i4 = int4b_format;
using t_i2 = int2b_format;
using t_i8 = int8_t;
using t_i32 = int32_t;

constant float FP4_TABLE[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

constant uint E4M3_SUB[8] = {
    0u,
    118u << 23,
    119u << 23,
    (119u << 23) | (1u << 22),
    120u << 23,
    (120u << 23) | (1u << 21),
    (120u << 23) | (1u << 22),
    (120u << 23) | (3u << 21),
};

static float e8m0_code(uint scale) {
    uint bits = select(scale << 23, 1u << 22, scale == 0u);
    return as_type<float>(bits);
}

static float e4m3_code(uint code) {
    uint exp = (code >> 3) & 0xFu;
    uint mant = code & 7u;
    uint bits = select(((exp + 120u) << 23) | (mant << 20), E4M3_SUB[mant], exp == 0u);
    bits |= (code & 0x80u) << 24;
    return as_type<float>(bits);
}

static float fp4_code(uint nibble) {
    return FP4_TABLE[nibble & 15u];
}

static float int4_code(uint nibble) {
    int code = int(nibble & 15u);
    return float(code >= 8 ? code - 16 : code);
}

static float int2_code(uint bits) {
    int code = int(bits & 3u);
    return float(code - ((code >> 1) << 2));
}

template <typename Left, typename Right, typename Out, int TM, int TN, int SG>
static void run_scaled(
    device Left* activation,
    device uchar* weights,
    device uchar* scales,
    device Out* destination,
    uint rows,
    uint cols,
    uint inner,
    uint2 tgid)
{
    tensor<device Left, dextents<int, 2>, tensor_inline> left(
        activation,
        dextents<int, 2>(int(inner), int(rows)),
        array<int, 2>{1, int(inner)});
    tensor<device Right, dextents<int, 2>, tensor_inline, scales_plane> right(
        weights,
        dextents<int, 2>(int(inner), int(cols)),
        array<int, 2>{1, int(inner)},
        scales_plane(scales));
    tensor<device Out, dextents<int, 2>, tensor_inline> out(
        destination,
        dextents<int, 2>(int(cols), int(rows)),
        array<int, 2>{1, int(cols)});
    constexpr auto desc = matmul2d_descriptor(
        TM, TN, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto tile_left = left.slice(0, int(tgid.y) * TM);
    auto tile_right = right.slice(0, int(tgid.x) * TN);
    auto tile_out = out.slice(int(tgid.x) * TN, int(tgid.y) * TM);
    op.run(tile_left, tile_right, tile_out);
}

template <typename Elem, typename Out, int TM, int TN, int SG>
static void run_both(
    device uchar* left_data,
    device uchar* right_data,
    device uchar* left_scales,
    device uchar* right_scales,
    device Out* destination,
    uint rows,
    uint cols,
    uint inner,
    uint2 tgid)
{
    tensor<device Elem, dextents<int, 2>, tensor_inline, scales_plane> left(
        left_data,
        dextents<int, 2>(int(inner), int(rows)),
        array<int, 2>{1, int(inner)},
        scales_plane(left_scales));
    tensor<device Elem, dextents<int, 2>, tensor_inline, scales_plane> right(
        right_data,
        dextents<int, 2>(int(inner), int(cols)),
        array<int, 2>{1, int(inner)},
        scales_plane(right_scales));
    tensor<device Out, dextents<int, 2>, tensor_inline> out(
        destination,
        dextents<int, 2>(int(cols), int(rows)),
        array<int, 2>{1, int(cols)});
    constexpr auto desc = matmul2d_descriptor(
        TM, TN, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto tile_left = left.slice(0, int(tgid.y) * TM);
    auto tile_right = right.slice(0, int(tgid.x) * TN);
    auto tile_out = out.slice(int(tgid.x) * TN, int(tgid.y) * TM);
    op.run(tile_left, tile_right, tile_out);
}

template <typename Left, typename Right, typename Out, int TM, int TN, int SG>
static void run_plain(
    device Left* activation,
    device Right* weights,
    device Out* destination,
    uint rows,
    uint cols,
    uint inner,
    uint2 tgid)
{
    tensor<device Left, dextents<int, 2>, tensor_inline> left(
        activation,
        dextents<int, 2>(int(inner), int(rows)),
        array<int, 2>{1, int(inner)});
    tensor<device Right, dextents<int, 2>, tensor_inline> right(
        weights,
        dextents<int, 2>(int(inner), int(cols)),
        array<int, 2>{1, int(inner)});
    tensor<device Out, dextents<int, 2>, tensor_inline> out(
        destination,
        dextents<int, 2>(int(cols), int(rows)),
        array<int, 2>{1, int(cols)});
    constexpr auto desc = matmul2d_descriptor(
        TM, TN, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto tile_left = left.slice(0, int(tgid.y) * TM);
    auto tile_right = right.slice(0, int(tgid.x) * TN);
    auto tile_out = out.slice(int(tgid.x) * TN, int(tgid.y) * TM);
    op.run(tile_left, tile_right, tile_out);
}

#define WRAP_SCALED(NAME, L, R, O, TM, TN, SG) \
kernel void NAME( \
    device L* activation [[buffer(0)]], \
    device uchar* weights [[buffer(1)]], \
    device uchar* scales [[buffer(2)]], \
    device O* destination [[buffer(3)]], \
    constant uint& rows [[buffer(4)]], \
    constant uint& cols [[buffer(5)]], \
    constant uint& inner [[buffer(6)]], \
    uint2 tgid [[threadgroup_position_in_grid]]) \
{ \
    run_scaled<L, R, O, TM, TN, SG>( \
        activation, weights, scales, destination, rows, cols, inner, tgid); \
}

#define WRAP_BOTH(NAME, E, O, TM, TN, SG) \
kernel void NAME( \
    device uchar* left_data [[buffer(0)]], \
    device uchar* right_data [[buffer(1)]], \
    device uchar* left_scales [[buffer(2)]], \
    device uchar* right_scales [[buffer(3)]], \
    device O* destination [[buffer(4)]], \
    constant uint& rows [[buffer(5)]], \
    constant uint& cols [[buffer(6)]], \
    constant uint& inner [[buffer(7)]], \
    uint2 tgid [[threadgroup_position_in_grid]]) \
{ \
    run_both<E, O, TM, TN, SG>( \
        left_data, right_data, left_scales, right_scales, destination, \
        rows, cols, inner, tgid); \
}

#define WRAP_PLAIN(NAME, L, R, O, TM, TN, SG) \
kernel void NAME( \
    device L* activation [[buffer(0)]], \
    device R* weights [[buffer(1)]], \
    device O* destination [[buffer(2)]], \
    constant uint& rows [[buffer(3)]], \
    constant uint& cols [[buffer(4)]], \
    constant uint& inner [[buffer(5)]], \
    uint2 tgid [[threadgroup_position_in_grid]]) \
{ \
    run_plain<L, R, O, TM, TN, SG>( \
        activation, weights, destination, rows, cols, inner, tgid); \
}

WRAP_SCALED(mm64_h_fp4, t_half, t_fp4, t_f32, 64, 64, 4)
WRAP_SCALED(mm64_h_fp4_h, t_half, t_fp4, t_half, 64, 64, 4)
WRAP_SCALED(mm64_h_e4, t_half, t_e4, t_f32, 64, 64, 4)
WRAP_SCALED(mm64_h_e5, t_half, t_e5, t_f32, 64, 64, 4)
WRAP_SCALED(mm64_h_i4, t_half, t_i4, t_f32, 64, 64, 4)
WRAP_SCALED(mm64_h_i2, t_half, t_i2, t_f32, 64, 64, 4)
WRAP_BOTH(mm64_fp4_fp4, t_fp4, t_f32, 64, 64, 4)
WRAP_BOTH(mm64_e4_e4, t_e4, t_f32, 64, 64, 4)
WRAP_PLAIN(mm64_h_h, t_half, t_half, t_f32, 64, 64, 4)
WRAP_PLAIN(mm64_bf_bf, t_bf, t_bf, t_f32, 64, 64, 4)
WRAP_PLAIN(mm64_i8, t_i8, t_i8, t_i32, 64, 64, 4)

WRAP_SCALED(mm32_h_fp4, t_half, t_fp4, t_f32, 32, 128, 4)
WRAP_SCALED(mm32_h_e4, t_half, t_e4, t_f32, 32, 128, 4)
WRAP_PLAIN(mm32_i8, t_i8, t_i8, t_i32, 32, 128, 4)

WRAP_SCALED(mm8_h_fp4, t_half, t_fp4, t_f32, 8, 256, 2)
WRAP_SCALED(mm8_h_e4, t_half, t_e4, t_f32, 8, 256, 2)
WRAP_PLAIN(mm8_i8, t_i8, t_i8, t_i32, 8, 256, 2)

// Unscaled quantized right. `matmul2d` rejects an M that is not a multiple of 8 or 16. Separate from run_scaled so the type has no scale plane.
template <typename Right, typename Out, int TM, int TN, int SG>
static void run_fp_plain(
    device half* activation,
    device uchar* weights,
    device Out* destination,
    uint rows,
    uint cols,
    uint inner,
    uint2 tgid)
{
    tensor<device half, dextents<int, 2>, tensor_inline> left(
        activation,
        dextents<int, 2>(int(inner), int(rows)),
        array<int, 2>{1, int(inner)});
    tensor<device Right, dextents<int, 2>, tensor_inline> right(
        weights,
        dextents<int, 2>(int(inner), int(cols)),
        array<int, 2>{1, int(inner)});
    tensor<device Out, dextents<int, 2>, tensor_inline> out(
        destination,
        dextents<int, 2>(int(cols), int(rows)),
        array<int, 2>{1, int(cols)});
    constexpr auto desc = matmul2d_descriptor(
        TM, TN, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto tile_left = left.slice(0, int(tgid.y) * TM);
    auto tile_right = right.slice(0, int(tgid.x) * TN);
    auto tile_out = out.slice(int(tgid.x) * TN, int(tgid.y) * TM);
    op.run(tile_left, tile_right, tile_out);
}

kernel void mm64_h_fp4_raw(
    device half* activation [[buffer(0)]],
    device uchar* weights [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    run_fp_plain<t_fp4, t_f32, 64, 64, 4>(
        activation, weights, destination, rows, cols, inner, tgid);
}

kernel void mm64_h_e4_raw(
    device half* activation [[buffer(0)]],
    device uchar* weights [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    run_fp_plain<t_e4, t_f32, 64, 64, 4>(
        activation, weights, destination, rows, cols, inner, tgid);
}

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
    thread float& a3,
    bool scaled)
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
        float f0 = scaled ? e8m0_code(s0[i >> 2]) : 1.0f;
        float f1 = scaled ? e8m0_code(s1[i >> 2]) : 1.0f;
        float f2 = scaled ? e8m0_code(s2[i >> 2]) : 1.0f;
        float f3 = scaled ? e8m0_code(s3[i >> 2]) : 1.0f;
        uint base = i << 3;
        for (uint n = 0u; n < 8u; ++n) {
            float act = activation[base + n];
            uint shift = n * 4u;
            acc0 += fp4_code((w0 >> shift) & 0xFu) * f0 * act;
            acc1 += fp4_code((w1 >> shift) & 0xFu) * f1 * act;
            acc2 += fp4_code((w2 >> shift) & 0xFu) * f2 * act;
            acc3 += fp4_code((w3 >> shift) & 0xFu) * f3 * act;
        }
    }
    a0 = acc0;
    a1 = acc1;
    a2 = acc2;
    a3 = acc3;
}

kernel void gemv_fp4_x4(
    device const float* activation [[buffer(0)]],
    device const uchar* weights [[buffer(1)]],
    device const uchar* scales [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    constant uint& scaled [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint group [[threadgroup_position_in_grid]])
{
    uint row0 = group << 2;
    if (row0 >= cols) {
        return;
    }
    float acc0, acc1, acc2, acc3;
    fp4_rows4(activation, weights, scales, row0, inner, lane, acc0, acc1, acc2, acc3, scaled != 0u);
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

#define STORE4(lane, simd, span, row0, destination, acc0, acc1, acc2, acc3) \
    do { \
        float _o0 = simd_sum(acc0); \
        float _o1 = simd_sum(acc1); \
        float _o2 = simd_sum(acc2); \
        float _o3 = simd_sum(acc3); \
        threadgroup float _part[8][4]; \
        if ((lane) == 0u) { \
            _part[simd][0] = _o0; \
            _part[simd][1] = _o1; \
            _part[simd][2] = _o2; \
            _part[simd][3] = _o3; \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        if ((simd) == 0u && (lane) == 0u) { \
            float _s0 = 0.0f; \
            float _s1 = 0.0f; \
            float _s2 = 0.0f; \
            float _s3 = 0.0f; \
            uint _nsimd = (span) >> 5; \
            for (uint _s = 0u; _s < _nsimd; ++_s) { \
                _s0 += _part[_s][0]; \
                _s1 += _part[_s][1]; \
                _s2 += _part[_s][2]; \
                _s3 += _part[_s][3]; \
            } \
            (destination)[row0] = _s0; \
            (destination)[row0 + 1u] = _s1; \
            (destination)[row0 + 2u] = _s2; \
            (destination)[row0 + 3u] = _s3; \
        } \
    } while (0)

static void dot_u32x4(
    device const float* activation,
    device const uint* p0,
    device const uint* p1,
    device const uint* p2,
    device const uint* p3,
    device const uchar* s0,
    device const uchar* s1,
    device const uchar* s2,
    device const uchar* s3,
    uint words,
    uint lane,
    uint codes,
    bool scaled,
    bool as_int2,
    thread float& a0,
    thread float& a1,
    thread float& a2,
    thread float& a3)
{
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    uint shift_step = 32u / codes;
    for (uint i = lane; i < words; i += 32u) {
        uint w0 = p0[i];
        uint w1 = p1[i];
        uint w2 = p2[i];
        uint w3 = p3[i];
        uint element = i * codes;
        float f0 = scaled ? e8m0_code(s0[element >> 5]) : 1.0f;
        float f1 = scaled ? e8m0_code(s1[element >> 5]) : 1.0f;
        float f2 = scaled ? e8m0_code(s2[element >> 5]) : 1.0f;
        float f3 = scaled ? e8m0_code(s3[element >> 5]) : 1.0f;
        for (uint n = 0u; n < codes; ++n) {
            float act = activation[element + n];
            uint shift = n * shift_step;
            float c0 = as_int2 ? int2_code(w0 >> shift) : int4_code(w0 >> shift);
            float c1 = as_int2 ? int2_code(w1 >> shift) : int4_code(w1 >> shift);
            float c2 = as_int2 ? int2_code(w2 >> shift) : int4_code(w2 >> shift);
            float c3 = as_int2 ? int2_code(w3 >> shift) : int4_code(w3 >> shift);
            acc0 = fma(c0 * f0, act, acc0);
            acc1 = fma(c1 * f1, act, acc1);
            acc2 = fma(c2 * f2, act, acc2);
            acc3 = fma(c3 * f3, act, acc3);
        }
    }
    a0 = acc0;
    a1 = acc1;
    a2 = acc2;
    a3 = acc3;
}

kernel void gemv_sub4(
    device const float* activation [[buffer(0)]],
    device const uchar* weights [[buffer(1)]],
    device const uchar* scales [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    constant uint& as_int2 [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint group [[threadgroup_position_in_grid]])
{
    uint row0 = group << 2;
    if (row0 >= cols) {
        return;
    }
    bool narrow = as_int2 != 0u;
    uint codes = narrow ? 16u : 8u;
    uint stride = inner / (narrow ? 4u : 2u);
    uint words = inner / codes;
    uint blocks = inner >> 5;
    device const uint* p0 = (device const uint*)(weights + row0 * stride);
    device const uint* p1 = (device const uint*)(weights + (row0 + 1u) * stride);
    device const uint* p2 = (device const uint*)(weights + (row0 + 2u) * stride);
    device const uint* p3 = (device const uint*)(weights + (row0 + 3u) * stride);
    float acc0, acc1, acc2, acc3;
    dot_u32x4(
        activation, p0, p1, p2, p3,
        scales + row0 * blocks, scales + (row0 + 1u) * blocks,
        scales + (row0 + 2u) * blocks, scales + (row0 + 3u) * blocks,
        words, lane, codes, true, narrow, acc0, acc1, acc2, acc3);
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

kernel void gemv_fp8_x4(
    device const float* activation [[buffer(0)]],
    device const uchar* weights [[buffer(1)]],
    device const uchar* scales [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint tid [[thread_index_in_threadgroup]],
    uint span [[threads_per_threadgroup]],
    uint group [[threadgroup_position_in_grid]])
{
    uint row0 = group << 2;
    if (row0 >= cols) {
        return;
    }
    uint vecs = inner >> 4;
    uint blocks = inner >> 5;
    device const uint4* p0 = (device const uint4*)(weights + row0 * inner);
    device const uint4* p1 = (device const uint4*)(weights + (row0 + 1u) * inner);
    device const uint4* p2 = (device const uint4*)(weights + (row0 + 2u) * inner);
    device const uint4* p3 = (device const uint4*)(weights + (row0 + 3u) * inner);
    device const uchar* s0 = scales + row0 * blocks;
    device const uchar* s1 = scales + (row0 + 1u) * blocks;
    device const uchar* s2 = scales + (row0 + 2u) * blocks;
    device const uchar* s3 = scales + (row0 + 3u) * blocks;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    for (uint i = tid; i < vecs; i += span) {
        uint4 w0 = p0[i];
        uint4 w1 = p1[i];
        uint4 w2 = p2[i];
        uint4 w3 = p3[i];
        float f0 = e8m0_code(s0[i >> 1]);
        float f1 = e8m0_code(s1[i >> 1]);
        float f2 = e8m0_code(s2[i >> 1]);
        float f3 = e8m0_code(s3[i >> 1]);
        uint base = i << 4;
        float4 act = *(device const float4*)(activation + base);
        acc0 = fma(e4m3_code(w0.x & 0xFFu) * f0, act.x, acc0);
        acc1 = fma(e4m3_code(w1.x & 0xFFu) * f1, act.x, acc1);
        acc2 = fma(e4m3_code(w2.x & 0xFFu) * f2, act.x, acc2);
        acc3 = fma(e4m3_code(w3.x & 0xFFu) * f3, act.x, acc3);
        acc0 = fma(e4m3_code((w0.x >> 8) & 0xFFu) * f0, act.y, acc0);
        acc1 = fma(e4m3_code((w1.x >> 8) & 0xFFu) * f1, act.y, acc1);
        acc2 = fma(e4m3_code((w2.x >> 8) & 0xFFu) * f2, act.y, acc2);
        acc3 = fma(e4m3_code((w3.x >> 8) & 0xFFu) * f3, act.y, acc3);
        acc0 = fma(e4m3_code((w0.x >> 16) & 0xFFu) * f0, act.z, acc0);
        acc1 = fma(e4m3_code((w1.x >> 16) & 0xFFu) * f1, act.z, acc1);
        acc2 = fma(e4m3_code((w2.x >> 16) & 0xFFu) * f2, act.z, acc2);
        acc3 = fma(e4m3_code((w3.x >> 16) & 0xFFu) * f3, act.z, acc3);
        acc0 = fma(e4m3_code(w0.x >> 24) * f0, act.w, acc0);
        acc1 = fma(e4m3_code(w1.x >> 24) * f1, act.w, acc1);
        acc2 = fma(e4m3_code(w2.x >> 24) * f2, act.w, acc2);
        acc3 = fma(e4m3_code(w3.x >> 24) * f3, act.w, acc3);
        act = *(device const float4*)(activation + base + 4u);
        acc0 = fma(e4m3_code(w0.y & 0xFFu) * f0, act.x, acc0);
        acc1 = fma(e4m3_code(w1.y & 0xFFu) * f1, act.x, acc1);
        acc2 = fma(e4m3_code(w2.y & 0xFFu) * f2, act.x, acc2);
        acc3 = fma(e4m3_code(w3.y & 0xFFu) * f3, act.x, acc3);
        acc0 = fma(e4m3_code((w0.y >> 8) & 0xFFu) * f0, act.y, acc0);
        acc1 = fma(e4m3_code((w1.y >> 8) & 0xFFu) * f1, act.y, acc1);
        acc2 = fma(e4m3_code((w2.y >> 8) & 0xFFu) * f2, act.y, acc2);
        acc3 = fma(e4m3_code((w3.y >> 8) & 0xFFu) * f3, act.y, acc3);
        acc0 = fma(e4m3_code((w0.y >> 16) & 0xFFu) * f0, act.z, acc0);
        acc1 = fma(e4m3_code((w1.y >> 16) & 0xFFu) * f1, act.z, acc1);
        acc2 = fma(e4m3_code((w2.y >> 16) & 0xFFu) * f2, act.z, acc2);
        acc3 = fma(e4m3_code((w3.y >> 16) & 0xFFu) * f3, act.z, acc3);
        acc0 = fma(e4m3_code(w0.y >> 24) * f0, act.w, acc0);
        acc1 = fma(e4m3_code(w1.y >> 24) * f1, act.w, acc1);
        acc2 = fma(e4m3_code(w2.y >> 24) * f2, act.w, acc2);
        acc3 = fma(e4m3_code(w3.y >> 24) * f3, act.w, acc3);
        act = *(device const float4*)(activation + base + 8u);
        acc0 = fma(e4m3_code(w0.z & 0xFFu) * f0, act.x, acc0);
        acc1 = fma(e4m3_code(w1.z & 0xFFu) * f1, act.x, acc1);
        acc2 = fma(e4m3_code(w2.z & 0xFFu) * f2, act.x, acc2);
        acc3 = fma(e4m3_code(w3.z & 0xFFu) * f3, act.x, acc3);
        acc0 = fma(e4m3_code((w0.z >> 8) & 0xFFu) * f0, act.y, acc0);
        acc1 = fma(e4m3_code((w1.z >> 8) & 0xFFu) * f1, act.y, acc1);
        acc2 = fma(e4m3_code((w2.z >> 8) & 0xFFu) * f2, act.y, acc2);
        acc3 = fma(e4m3_code((w3.z >> 8) & 0xFFu) * f3, act.y, acc3);
        acc0 = fma(e4m3_code((w0.z >> 16) & 0xFFu) * f0, act.z, acc0);
        acc1 = fma(e4m3_code((w1.z >> 16) & 0xFFu) * f1, act.z, acc1);
        acc2 = fma(e4m3_code((w2.z >> 16) & 0xFFu) * f2, act.z, acc2);
        acc3 = fma(e4m3_code((w3.z >> 16) & 0xFFu) * f3, act.z, acc3);
        acc0 = fma(e4m3_code(w0.z >> 24) * f0, act.w, acc0);
        acc1 = fma(e4m3_code(w1.z >> 24) * f1, act.w, acc1);
        acc2 = fma(e4m3_code(w2.z >> 24) * f2, act.w, acc2);
        acc3 = fma(e4m3_code(w3.z >> 24) * f3, act.w, acc3);
        act = *(device const float4*)(activation + base + 12u);
        acc0 = fma(e4m3_code(w0.w & 0xFFu) * f0, act.x, acc0);
        acc1 = fma(e4m3_code(w1.w & 0xFFu) * f1, act.x, acc1);
        acc2 = fma(e4m3_code(w2.w & 0xFFu) * f2, act.x, acc2);
        acc3 = fma(e4m3_code(w3.w & 0xFFu) * f3, act.x, acc3);
        acc0 = fma(e4m3_code((w0.w >> 8) & 0xFFu) * f0, act.y, acc0);
        acc1 = fma(e4m3_code((w1.w >> 8) & 0xFFu) * f1, act.y, acc1);
        acc2 = fma(e4m3_code((w2.w >> 8) & 0xFFu) * f2, act.y, acc2);
        acc3 = fma(e4m3_code((w3.w >> 8) & 0xFFu) * f3, act.y, acc3);
        acc0 = fma(e4m3_code((w0.w >> 16) & 0xFFu) * f0, act.z, acc0);
        acc1 = fma(e4m3_code((w1.w >> 16) & 0xFFu) * f1, act.z, acc1);
        acc2 = fma(e4m3_code((w2.w >> 16) & 0xFFu) * f2, act.z, acc2);
        acc3 = fma(e4m3_code((w3.w >> 16) & 0xFFu) * f3, act.z, acc3);
        acc0 = fma(e4m3_code(w0.w >> 24) * f0, act.w, acc0);
        acc1 = fma(e4m3_code(w1.w >> 24) * f1, act.w, acc1);
        acc2 = fma(e4m3_code(w2.w >> 24) * f2, act.w, acc2);
        acc3 = fma(e4m3_code(w3.w >> 24) * f3, act.w, acc3);
    }
    STORE4(lane, simd, span, row0, destination, acc0, acc1, acc2, acc3);
}

kernel void gemv_i8f_x4(
    device const float* activation [[buffer(0)]],
    device const int8_t* weights [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& inner [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint tid [[thread_index_in_threadgroup]],
    uint span [[threads_per_threadgroup]],
    uint group [[threadgroup_position_in_grid]])
{
    uint row0 = group << 2;
    if (row0 >= cols) {
        return;
    }
    device const int8_t* p0 = weights + row0 * inner;
    device const int8_t* p1 = weights + (row0 + 1u) * inner;
    device const int8_t* p2 = weights + (row0 + 2u) * inner;
    device const int8_t* p3 = weights + (row0 + 3u) * inner;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    uint vecs = inner >> 4;
    for (uint i = tid; i < vecs; i += span) {
        uint base = i << 4;
        float4 act = *(device const float4*)(activation + base);
        uint4 w0 = *(device const uint4*)(p0 + base);
        uint4 w1 = *(device const uint4*)(p1 + base);
        uint4 w2 = *(device const uint4*)(p2 + base);
        uint4 w3 = *(device const uint4*)(p3 + base);
        acc0 = fma(float(as_type<int8_t>(uchar(w0.x & 0xFFu))), act.x, acc0);
        acc1 = fma(float(as_type<int8_t>(uchar(w1.x & 0xFFu))), act.x, acc1);
        acc2 = fma(float(as_type<int8_t>(uchar(w2.x & 0xFFu))), act.x, acc2);
        acc3 = fma(float(as_type<int8_t>(uchar(w3.x & 0xFFu))), act.x, acc3);
        acc0 = fma(float(as_type<int8_t>(uchar((w0.x >> 8) & 0xFFu))), act.y, acc0);
        acc1 = fma(float(as_type<int8_t>(uchar((w1.x >> 8) & 0xFFu))), act.y, acc1);
        acc2 = fma(float(as_type<int8_t>(uchar((w2.x >> 8) & 0xFFu))), act.y, acc2);
        acc3 = fma(float(as_type<int8_t>(uchar((w3.x >> 8) & 0xFFu))), act.y, acc3);
        acc0 = fma(float(as_type<int8_t>(uchar((w0.x >> 16) & 0xFFu))), act.z, acc0);
        acc1 = fma(float(as_type<int8_t>(uchar((w1.x >> 16) & 0xFFu))), act.z, acc1);
        acc2 = fma(float(as_type<int8_t>(uchar((w2.x >> 16) & 0xFFu))), act.z, acc2);
        acc3 = fma(float(as_type<int8_t>(uchar((w3.x >> 16) & 0xFFu))), act.z, acc3);
        acc0 = fma(float(as_type<int8_t>(uchar(w0.x >> 24))), act.w, acc0);
        acc1 = fma(float(as_type<int8_t>(uchar(w1.x >> 24))), act.w, acc1);
        acc2 = fma(float(as_type<int8_t>(uchar(w2.x >> 24))), act.w, acc2);
        acc3 = fma(float(as_type<int8_t>(uchar(w3.x >> 24))), act.w, acc3);
        act = *(device const float4*)(activation + base + 4u);
        acc0 = fma(float(as_type<int8_t>(uchar(w0.y & 0xFFu))), act.x, acc0);
        acc1 = fma(float(as_type<int8_t>(uchar(w1.y & 0xFFu))), act.x, acc1);
        acc2 = fma(float(as_type<int8_t>(uchar(w2.y & 0xFFu))), act.x, acc2);
        acc3 = fma(float(as_type<int8_t>(uchar(w3.y & 0xFFu))), act.x, acc3);
        acc0 = fma(float(as_type<int8_t>(uchar((w0.y >> 8) & 0xFFu))), act.y, acc0);
        acc1 = fma(float(as_type<int8_t>(uchar((w1.y >> 8) & 0xFFu))), act.y, acc1);
        acc2 = fma(float(as_type<int8_t>(uchar((w2.y >> 8) & 0xFFu))), act.y, acc2);
        acc3 = fma(float(as_type<int8_t>(uchar((w3.y >> 8) & 0xFFu))), act.y, acc3);
        acc0 = fma(float(as_type<int8_t>(uchar((w0.y >> 16) & 0xFFu))), act.z, acc0);
        acc1 = fma(float(as_type<int8_t>(uchar((w1.y >> 16) & 0xFFu))), act.z, acc1);
        acc2 = fma(float(as_type<int8_t>(uchar((w2.y >> 16) & 0xFFu))), act.z, acc2);
        acc3 = fma(float(as_type<int8_t>(uchar((w3.y >> 16) & 0xFFu))), act.z, acc3);
        acc0 = fma(float(as_type<int8_t>(uchar(w0.y >> 24))), act.w, acc0);
        acc1 = fma(float(as_type<int8_t>(uchar(w1.y >> 24))), act.w, acc1);
        acc2 = fma(float(as_type<int8_t>(uchar(w2.y >> 24))), act.w, acc2);
        acc3 = fma(float(as_type<int8_t>(uchar(w3.y >> 24))), act.w, acc3);
        act = *(device const float4*)(activation + base + 8u);
        acc0 = fma(float(as_type<int8_t>(uchar(w0.z & 0xFFu))), act.x, acc0);
        acc1 = fma(float(as_type<int8_t>(uchar(w1.z & 0xFFu))), act.x, acc1);
        acc2 = fma(float(as_type<int8_t>(uchar(w2.z & 0xFFu))), act.x, acc2);
        acc3 = fma(float(as_type<int8_t>(uchar(w3.z & 0xFFu))), act.x, acc3);
        acc0 = fma(float(as_type<int8_t>(uchar((w0.z >> 8) & 0xFFu))), act.y, acc0);
        acc1 = fma(float(as_type<int8_t>(uchar((w1.z >> 8) & 0xFFu))), act.y, acc1);
        acc2 = fma(float(as_type<int8_t>(uchar((w2.z >> 8) & 0xFFu))), act.y, acc2);
        acc3 = fma(float(as_type<int8_t>(uchar((w3.z >> 8) & 0xFFu))), act.y, acc3);
        acc0 = fma(float(as_type<int8_t>(uchar((w0.z >> 16) & 0xFFu))), act.z, acc0);
        acc1 = fma(float(as_type<int8_t>(uchar((w1.z >> 16) & 0xFFu))), act.z, acc1);
        acc2 = fma(float(as_type<int8_t>(uchar((w2.z >> 16) & 0xFFu))), act.z, acc2);
        acc3 = fma(float(as_type<int8_t>(uchar((w3.z >> 16) & 0xFFu))), act.z, acc3);
        acc0 = fma(float(as_type<int8_t>(uchar(w0.z >> 24))), act.w, acc0);
        acc1 = fma(float(as_type<int8_t>(uchar(w1.z >> 24))), act.w, acc1);
        acc2 = fma(float(as_type<int8_t>(uchar(w2.z >> 24))), act.w, acc2);
        acc3 = fma(float(as_type<int8_t>(uchar(w3.z >> 24))), act.w, acc3);
        act = *(device const float4*)(activation + base + 12u);
        acc0 = fma(float(as_type<int8_t>(uchar(w0.w & 0xFFu))), act.x, acc0);
        acc1 = fma(float(as_type<int8_t>(uchar(w1.w & 0xFFu))), act.x, acc1);
        acc2 = fma(float(as_type<int8_t>(uchar(w2.w & 0xFFu))), act.x, acc2);
        acc3 = fma(float(as_type<int8_t>(uchar(w3.w & 0xFFu))), act.x, acc3);
        acc0 = fma(float(as_type<int8_t>(uchar((w0.w >> 8) & 0xFFu))), act.y, acc0);
        acc1 = fma(float(as_type<int8_t>(uchar((w1.w >> 8) & 0xFFu))), act.y, acc1);
        acc2 = fma(float(as_type<int8_t>(uchar((w2.w >> 8) & 0xFFu))), act.y, acc2);
        acc3 = fma(float(as_type<int8_t>(uchar((w3.w >> 8) & 0xFFu))), act.y, acc3);
        acc0 = fma(float(as_type<int8_t>(uchar((w0.w >> 16) & 0xFFu))), act.z, acc0);
        acc1 = fma(float(as_type<int8_t>(uchar((w1.w >> 16) & 0xFFu))), act.z, acc1);
        acc2 = fma(float(as_type<int8_t>(uchar((w2.w >> 16) & 0xFFu))), act.z, acc2);
        acc3 = fma(float(as_type<int8_t>(uchar((w3.w >> 16) & 0xFFu))), act.z, acc3);
        acc0 = fma(float(as_type<int8_t>(uchar(w0.w >> 24))), act.w, acc0);
        acc1 = fma(float(as_type<int8_t>(uchar(w1.w >> 24))), act.w, acc1);
        acc2 = fma(float(as_type<int8_t>(uchar(w2.w >> 24))), act.w, acc2);
        acc3 = fma(float(as_type<int8_t>(uchar(w3.w >> 24))), act.w, acc3);
    }
    STORE4(lane, simd, span, row0, destination, acc0, acc1, acc2, acc3);
}

kernel void gemv_half_x4(
    device const float* activation [[buffer(0)]],
    device const half* weights [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& inner [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint tid [[thread_index_in_threadgroup]],
    uint span [[threads_per_threadgroup]],
    uint group [[threadgroup_position_in_grid]])
{
    uint row0 = group << 2;
    if (row0 >= cols) {
        return;
    }
    device const half* p0 = weights + row0 * inner;
    device const half* p1 = weights + (row0 + 1u) * inner;
    device const half* p2 = weights + (row0 + 2u) * inner;
    device const half* p3 = weights + (row0 + 3u) * inner;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    uint vecs = inner >> 2;
    for (uint i = tid; i < vecs; i += span) {
        uint base = i << 2;
        float4 act = *(device const float4*)(activation + base);
        half4 h0 = *(device const half4*)(p0 + base);
        half4 h1 = *(device const half4*)(p1 + base);
        half4 h2 = *(device const half4*)(p2 + base);
        half4 h3 = *(device const half4*)(p3 + base);
        acc0 = fma(float(h0.x), act.x, acc0);
        acc1 = fma(float(h1.x), act.x, acc1);
        acc2 = fma(float(h2.x), act.x, acc2);
        acc3 = fma(float(h3.x), act.x, acc3);
        acc0 = fma(float(h0.y), act.y, acc0);
        acc1 = fma(float(h1.y), act.y, acc1);
        acc2 = fma(float(h2.y), act.y, acc2);
        acc3 = fma(float(h3.y), act.y, acc3);
        acc0 = fma(float(h0.z), act.z, acc0);
        acc1 = fma(float(h1.z), act.z, acc1);
        acc2 = fma(float(h2.z), act.z, acc2);
        acc3 = fma(float(h3.z), act.z, acc3);
        acc0 = fma(float(h0.w), act.w, acc0);
        acc1 = fma(float(h1.w), act.w, acc1);
        acc2 = fma(float(h2.w), act.w, acc2);
        acc3 = fma(float(h3.w), act.w, acc3);
    }
    float o0 = simd_sum(acc0);
    float o1 = simd_sum(acc1);
    float o2 = simd_sum(acc2);
    float o3 = simd_sum(acc3);
    threadgroup float part[8][4];
    if (lane == 0u) {
        part[simd][0] = o0;
        part[simd][1] = o1;
        part[simd][2] = o2;
        part[simd][3] = o3;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0u && lane == 0u) {
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        uint nsimd = span >> 5;
        for (uint s = 0u; s < nsimd; ++s) {
            s0 += part[s][0];
            s1 += part[s][1];
            s2 += part[s][2];
            s3 += part[s][3];
        }
        destination[row0] = s0;
        destination[row0 + 1u] = s1;
        destination[row0 + 2u] = s2;
        destination[row0 + 3u] = s3;
    }
}

kernel void gemv_bf16_x4(
    device const float* activation [[buffer(0)]],
    device const bfloat* weights [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& inner [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint tid [[thread_index_in_threadgroup]],
    uint span [[threads_per_threadgroup]],
    uint group [[threadgroup_position_in_grid]])
{
    uint row0 = group << 2;
    if (row0 >= cols) {
        return;
    }
    device const bfloat* p0 = weights + row0 * inner;
    device const bfloat* p1 = weights + (row0 + 1u) * inner;
    device const bfloat* p2 = weights + (row0 + 2u) * inner;
    device const bfloat* p3 = weights + (row0 + 3u) * inner;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    for (uint i = tid; i < inner; i += span) {
        float act = activation[i];
        acc0 = fma(float(p0[i]), act, acc0);
        acc1 = fma(float(p1[i]), act, acc1);
        acc2 = fma(float(p2[i]), act, acc2);
        acc3 = fma(float(p3[i]), act, acc3);
    }
    float o0 = simd_sum(acc0);
    float o1 = simd_sum(acc1);
    float o2 = simd_sum(acc2);
    float o3 = simd_sum(acc3);
    threadgroup float part[8][4];
    if (lane == 0u) {
        part[simd][0] = o0;
        part[simd][1] = o1;
        part[simd][2] = o2;
        part[simd][3] = o3;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0u && lane == 0u) {
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        uint nsimd = span >> 5;
        for (uint s = 0u; s < nsimd; ++s) {
            s0 += part[s][0];
            s1 += part[s][1];
            s2 += part[s][2];
            s3 += part[s][3];
        }
        destination[row0] = s0;
        destination[row0 + 1u] = s1;
        destination[row0 + 2u] = s2;
        destination[row0 + 3u] = s3;
    }
}

kernel void gemv_i8_x4(
    device const int8_t* activation [[buffer(0)]],
    device const int8_t* weights [[buffer(1)]],
    device int32_t* destination [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& inner [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint tid [[thread_index_in_threadgroup]],
    uint span [[threads_per_threadgroup]],
    uint group [[threadgroup_position_in_grid]])
{
    uint row0 = group << 2;
    if (row0 >= cols) {
        return;
    }
    device const int8_t* p0 = weights + row0 * inner;
    device const int8_t* p1 = weights + (row0 + 1u) * inner;
    device const int8_t* p2 = weights + (row0 + 2u) * inner;
    device const int8_t* p3 = weights + (row0 + 3u) * inner;
    int acc0 = 0;
    int acc1 = 0;
    int acc2 = 0;
    int acc3 = 0;
    for (uint i = tid; i < inner; i += span) {
        int act = int(activation[i]);
        acc0 += int(p0[i]) * act;
        acc1 += int(p1[i]) * act;
        acc2 += int(p2[i]) * act;
        acc3 += int(p3[i]) * act;
    }
    int o0 = int(simd_sum(acc0));
    int o1 = int(simd_sum(acc1));
    int o2 = int(simd_sum(acc2));
    int o3 = int(simd_sum(acc3));
    threadgroup int part[8][4];
    if (lane == 0u) {
        part[simd][0] = o0;
        part[simd][1] = o1;
        part[simd][2] = o2;
        part[simd][3] = o3;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0u && lane == 0u) {
        int s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        uint nsimd = span >> 5;
        for (uint s = 0u; s < nsimd; ++s) {
            s0 += part[s][0];
            s1 += part[s][1];
            s2 += part[s][2];
            s3 += part[s][3];
        }
        destination[row0] = s0;
        destination[row0 + 1u] = s1;
        destination[row0 + 2u] = s2;
        destination[row0 + 3u] = s3;
    }
}

kernel void copy_uint4(
    device const uint4* src [[buffer(0)]],
    device uint4* dst [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) {
        dst[gid] = src[gid];
    }
}
