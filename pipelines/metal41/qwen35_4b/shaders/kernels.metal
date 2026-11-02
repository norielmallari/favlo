// Qwen3.5-4B Metal kernels. fp8 E4M3 + E8M0 per the OCP MX specification.
// Gated DeltaNet follows Yang, Kautz, and Hatamizadeh, arXiv:2412.06464.
// Prefill uses matmul2d (WWDC26 session 330) over a 64-token tile.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>
using namespace metal;
using namespace mpp::tensor_ops;

using scales_plane = tensor_blockwise<tensor_plane_scales,
                                      device metal_fp8_ue8m0_format,
                                      32, 1>;

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

// Round-to-nearest-even E4M3FN. |x| is already scaled into [0, 448].
static uint e4m3_encode(float x) {
    uint sign = x < 0.0f ? 0x80u : 0u;
    float mag = min(fabs(x), 448.0f);
    if (mag < 0.015625f) {
        uint sub = uint(rint(mag * 512.0f));
        return sign | min(sub, 8u);
    }
    int e;
    float frac = frexp(mag, e);
    float mant = frac * 2.0f - 1.0f;
    int ex = e - 1;
    uint m = uint(rint(mant * 8.0f));
    if (m == 8u) {
        m = 0u;
        ex += 1;
    }
    int biased = ex + 7;
    if (biased > 15 || (biased == 15 && m > 6u)) {
        return sign | 0x7Eu;
    }
    return sign | (uint(biased) << 3) | m;
}

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
    256.0f, 288.0f, 320.0f, 352.0f, 384.0f, 416.0f, 448.0f, 480.0f,
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
    -256.0f, -288.0f, -320.0f, -352.0f, -384.0f, -416.0f, -448.0f, -480.0f,
};

static float silu(float x) {
    return x / (1.0f + exp(-x));
}

static float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

kernel void zero_f32(
    device float* destination [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) {
        destination[gid] = 0.0f;
    }
}

kernel void zero_u32(
    device uint* destination [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) {
        destination[gid] = 0u;
    }
}

// One row of the fp8 table. The token id is read from the device.
kernel void embed_row(
    device const uchar* table [[buffer(0)]],
    device const uchar* scales [[buffer(1)]],
    device const uint* tokens [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& index [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= inner) {
        return;
    }
    uint row = tokens[index];
    device const uchar* weights = table + ulong(row) * inner;
    device const uchar* scale_row = scales + ulong(row) * (inner >> 5);
    destination[gid] = e4m3_code(weights[gid]) * e8m0_code(scale_row[gid >> 5]);
}

static float block_sum(float value, threadgroup float* shared, uint lane, uint simd, uint simds) {
    value = simd_sum(value);
    if (lane == 0u) {
        shared[simd] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0u; i < simds; ++i) {
        total += shared[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

// y = x * rsqrt(mean(x^2) + eps) * (gain + shift). One threadgroup of 256.
kernel void rmsnorm(
    device const float* input [[buffer(0)]],
    device const half* gain [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    constant float& shift [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint simds [[simdgroups_per_threadgroup]],
    uint tid [[thread_index_in_threadgroup]],
    uint count [[threads_per_threadgroup]])
{
    threadgroup float shared[32];
    float acc = 0.0f;
    for (uint i = tid; i < width; i += count) {
        float v = input[i];
        acc = fma(v, v, acc);
    }
    float total = block_sum(acc, shared, lane, simd, simds);
    float inv = rsqrt(total / float(width) + eps);
    for (uint i = tid; i < width; i += count) {
        destination[i] = input[i] * inv * (float(gain[i]) + shift);
    }
}

// In-place RMSNorm of each head. One threadgroup per head, one thread per element.
kernel void head_rmsnorm(
    device float* data [[buffer(0)]],
    device const half* gain [[buffer(1)]],
    constant uint& stride [[buffer(2)]],
    constant float& eps [[buffer(3)]],
    constant float& shift [[buffer(4)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint dim [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint simds [[simdgroups_per_threadgroup]])
{
    threadgroup float shared[32];
    device float* row = data + head * stride;
    float v = row[tid];
    float total = block_sum(v * v, shared, lane, simd, simds);
    float inv = rsqrt(total / float(dim) + eps);
    row[tid] = v * inv * (float(gain[tid]) + shift);
}

// GDN output norm: o = RMSNorm(o) * gain * silu(z), per value head.
kernel void gated_rmsnorm(
    device float* data [[buffer(0)]],
    device const float* gate [[buffer(1)]],
    device const half* gain [[buffer(2)]],
    constant float& eps [[buffer(3)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint dim [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint simds [[simdgroups_per_threadgroup]])
{
    threadgroup float shared[32];
    uint at = head * dim + tid;
    float v = data[at];
    float total = block_sum(v * v, shared, lane, simd, simds);
    float inv = rsqrt(total / float(dim) + eps);
    data[at] = v * inv * float(gain[tid]) * silu(gate[at]);
}

kernel void silu_mul(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= width) {
        return;
    }
    destination[gid] = silu(gate[gid]) * up[gid];
}

kernel void residual_add(
    device float* stream [[buffer(0)]],
    device const float* delta [[buffer(1)]],
    constant uint& width [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < width) {
        stream[gid] += delta[gid];
    }
}

// One E8M0 block of 32 codes against float activation. Two uint4 loads cover the block.
static float fp8_block(
    uint4 packed0,
    uint4 packed1,
    device const float* activation,
    uint block,
    float factor)
{
    device const float4* act = (device const float4*)(activation + (block << 5u));
    float sum = 0.0f;
    uint4 words[2] = {packed0, packed1};
    for (uint chunk = 0u; chunk < 2u; ++chunk) {
        uint word0 = words[chunk].x;
        uint word1 = words[chunk].y;
        uint word2 = words[chunk].z;
        uint word3 = words[chunk].w;
        float4 a0 = act[chunk * 4u];
        float4 a1 = act[chunk * 4u + 1u];
        float4 a2 = act[chunk * 4u + 2u];
        float4 a3 = act[chunk * 4u + 3u];
        sum = fma(e4m3_code(word0 & 255u), a0.x, sum);
        sum = fma(e4m3_code((word0 >> 8u) & 255u), a0.y, sum);
        sum = fma(e4m3_code((word0 >> 16u) & 255u), a0.z, sum);
        sum = fma(e4m3_code((word0 >> 24u) & 255u), a0.w, sum);
        sum = fma(e4m3_code(word1 & 255u), a1.x, sum);
        sum = fma(e4m3_code((word1 >> 8u) & 255u), a1.y, sum);
        sum = fma(e4m3_code((word1 >> 16u) & 255u), a1.z, sum);
        sum = fma(e4m3_code((word1 >> 24u) & 255u), a1.w, sum);
        sum = fma(e4m3_code(word2 & 255u), a2.x, sum);
        sum = fma(e4m3_code((word2 >> 8u) & 255u), a2.y, sum);
        sum = fma(e4m3_code((word2 >> 16u) & 255u), a2.z, sum);
        sum = fma(e4m3_code((word2 >> 24u) & 255u), a2.w, sum);
        sum = fma(e4m3_code(word3 & 255u), a3.x, sum);
        sum = fma(e4m3_code((word3 >> 8u) & 255u), a3.y, sum);
        sum = fma(e4m3_code((word3 >> 16u) & 255u), a3.z, sum);
        sum = fma(e4m3_code((word3 >> 24u) & 255u), a3.w, sum);
    }
    return sum * factor;
}

// One simdgroup owns one output row. Lanes walk contiguous 32-byte scale blocks.
static float dot_fp8_vec(
    device const float* activation,
    device const uchar* weights,
    device const uchar* scales,
    uint row,
    uint inner,
    uint lane)
{
    uint blocks = inner >> 5u;
    device const uchar* row_w = weights + ulong(row) * inner;
    device const uchar* row_s = scales + ulong(row) * blocks;
    float acc = 0.0f;
    for (uint block = lane; block < blocks; block += 32u) {
        device const uint4* words = (device const uint4*)(row_w + (block << 5u));
        uint4 p0 = words[0];
        uint4 p1 = words[1];
        float factor = e8m0_code(row_s[block]);
        acc += fp8_block(p0, p1, activation, block, factor);
    }
    return simd_sum(acc);
}

// One fp8 row dotted with a float activation. ``inner`` is a multiple of 512.
// Each lane reads 16 codes per step; one scale byte covers 32 codes.
static float dot_fp8_row(
    device const float* x,
    device const uchar* w,
    device const uchar* s,
    uint inner,
    uint lane)
{
    device const uint4* w16 = (device const uint4*)w;
    device const float4* x4 = (device const float4*)x;
    float acc = 0.0f;
    for (uint base = lane * 16u; base < inner; base += 512u) {
        uint4 packed = w16[base >> 4];
        float scale = e8m0_code(s[base >> 5]);
        uint words[4] = {packed.x, packed.y, packed.z, packed.w};
        float part = 0.0f;
        for (uint j = 0u; j < 4u; ++j) {
            uint word = words[j];
            float4 xv = x4[(base >> 2) + j];
            part = fma(E4M3_TABLE[word & 0xFFu], xv.x, part);
            part = fma(E4M3_TABLE[(word >> 8) & 0xFFu], xv.y, part);
            part = fma(E4M3_TABLE[(word >> 16) & 0xFFu], xv.z, part);
            part = fma(E4M3_TABLE[word >> 24], xv.w, part);
        }
        acc = fma(part, scale, acc);
    }
    return acc;
}

// y[rows] = W[rows, inner] x. Eight rows per threadgroup of 256, one simdgroup per row.
kernel void gemv_fp8(
    device const float* activation [[buffer(0)]],
    device const uchar* weights [[buffer(1)]],
    device const uchar* scales [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]])
{
    uint row = group * 8u + simd;
    if (row >= rows) {
        return;
    }
    float acc = dot_fp8_row(
        activation,
        weights + ulong(row) * inner,
        scales + ulong(row) * (inner >> 5),
        inner,
        lane);
    acc = simd_sum(acc);
    if (lane == 0u) {
        destination[row] = acc;
    }
}

kernel void gemv_fp8_add(
    device const float* activation [[buffer(0)]],
    device const uchar* weights [[buffer(1)]],
    device const uchar* scales [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]])
{
    uint row = group * 8u + simd;
    if (row >= rows) {
        return;
    }
    float acc = dot_fp8_row(
        activation,
        weights + ulong(row) * inner,
        scales + ulong(row) * (inner >> 5),
        inner,
        lane);
    acc = simd_sum(acc);
    if (lane == 0u) {
        destination[row] += acc;
    }
}


// Four output rows share one activation stream. cols is a multiple of 4.
kernel void gemv_fp8_x4(
    device const float* activation [[buffer(0)]],
    device const uchar* weights [[buffer(1)]],
    device const uchar* scales [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint group [[threadgroup_position_in_grid]])
{
    uint row0 = group << 2;
    if (row0 >= rows) {
        return;
    }
    uint blocks = inner >> 5;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    device const uchar* w0 = weights + ulong(row0) * inner;
    device const uchar* w1 = w0 + inner;
    device const uchar* w2 = w1 + inner;
    device const uchar* w3 = w2 + inner;
    device const uchar* s0 = scales + ulong(row0) * blocks;
    device const uchar* s1 = s0 + blocks;
    device const uchar* s2 = s1 + blocks;
    device const uchar* s3 = s2 + blocks;
    device const float4* x4 = (device const float4*)activation;
    for (uint base = lane * 16u; base < inner; base += 512u) {
        float4 xv0 = x4[(base >> 2) + 0u];
        float4 xv1 = x4[(base >> 2) + 1u];
        float4 xv2 = x4[(base >> 2) + 2u];
        float4 xv3 = x4[(base >> 2) + 3u];
        float scale0 = e8m0_code(s0[base >> 5]);
        float scale1 = e8m0_code(s1[base >> 5]);
        float scale2 = e8m0_code(s2[base >> 5]);
        float scale3 = e8m0_code(s3[base >> 5]);
        device const uint4* p0 = (device const uint4*)(w0 + base);
        device const uint4* p1 = (device const uint4*)(w1 + base);
        device const uint4* p2 = (device const uint4*)(w2 + base);
        device const uint4* p3 = (device const uint4*)(w3 + base);
        uint4 a = p0[0];
        uint4 b = p1[0];
        uint4 c = p2[0];
        uint4 d = p3[0];
        float4 xs[4] = {xv0, xv1, xv2, xv3};
        uint words0[4] = {a.x, a.y, a.z, a.w};
        uint words1[4] = {b.x, b.y, b.z, b.w};
        uint words2[4] = {c.x, c.y, c.z, c.w};
        uint words3[4] = {d.x, d.y, d.z, d.w};
        float part0 = 0.0f;
        float part1 = 0.0f;
        float part2 = 0.0f;
        float part3 = 0.0f;
        for (uint j = 0u; j < 4u; ++j) {
            float4 xv = xs[j];
            uint w;
            w = words0[j];
            part0 = fma(E4M3_TABLE[w & 0xFFu], xv.x, part0);
            part0 = fma(E4M3_TABLE[(w >> 8) & 0xFFu], xv.y, part0);
            part0 = fma(E4M3_TABLE[(w >> 16) & 0xFFu], xv.z, part0);
            part0 = fma(E4M3_TABLE[w >> 24], xv.w, part0);
            w = words1[j];
            part1 = fma(E4M3_TABLE[w & 0xFFu], xv.x, part1);
            part1 = fma(E4M3_TABLE[(w >> 8) & 0xFFu], xv.y, part1);
            part1 = fma(E4M3_TABLE[(w >> 16) & 0xFFu], xv.z, part1);
            part1 = fma(E4M3_TABLE[w >> 24], xv.w, part1);
            w = words2[j];
            part2 = fma(E4M3_TABLE[w & 0xFFu], xv.x, part2);
            part2 = fma(E4M3_TABLE[(w >> 8) & 0xFFu], xv.y, part2);
            part2 = fma(E4M3_TABLE[(w >> 16) & 0xFFu], xv.z, part2);
            part2 = fma(E4M3_TABLE[w >> 24], xv.w, part2);
            w = words3[j];
            part3 = fma(E4M3_TABLE[w & 0xFFu], xv.x, part3);
            part3 = fma(E4M3_TABLE[(w >> 8) & 0xFFu], xv.y, part3);
            part3 = fma(E4M3_TABLE[(w >> 16) & 0xFFu], xv.z, part3);
            part3 = fma(E4M3_TABLE[w >> 24], xv.w, part3);
        }
        acc0 = fma(part0, scale0, acc0);
        acc1 = fma(part1, scale1, acc1);
        acc2 = fma(part2, scale2, acc2);
        acc3 = fma(part3, scale3, acc3);
    }
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);
    if (lane == 0u) {
        destination[row0] = acc0;
        destination[row0 + 1u] = acc1;
        destination[row0 + 2u] = acc2;
        destination[row0 + 3u] = acc3;
    }
}

// One simdgroup per row. Kept for checks; decode prefers gemv_fp8.
kernel void gemv_fp8_vec(
    device const float* activation [[buffer(0)]],
    device const uchar* weights [[buffer(1)]],
    device const uchar* scales [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    if (row >= rows) {
        return;
    }
    float sum = dot_fp8_vec(activation, weights, scales, row, inner, lane);
    if (lane == 0u) {
        destination[row] = sum;
    }
}

// Causal depthwise convolution, kernel 4, then SiLU. ``history`` keeps the three
// prior inputs per channel. Taps are [channels, 4]; tap 3 multiplies the current input.
kernel void conv_silu(
    device const half* taps [[buffer(0)]],
    device float* mixed [[buffer(1)]],
    device float* history [[buffer(2)]],
    constant uint& channels [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= channels) {
        return;
    }
    device float* h = history + gid * 3u;
    float x0 = h[0];
    float x1 = h[1];
    float x2 = h[2];
    float x3 = mixed[gid];
    device const half* w = taps + gid * 4u;
    float y = float(w[0]) * x0 + float(w[1]) * x1 + float(w[2]) * x2 + float(w[3]) * x3;
    mixed[gid] = silu(y);
    h[0] = x1;
    h[1] = x2;
    h[2] = x3;
}

// One Gated DeltaNet step for one value head (Yang et al. §3):
//   S_t = alpha_t S_{t-1} (I - beta_t k k^T) + beta_t v k^T,  o_t = S_t q_t.
// alpha = exp(-exp(A_log) * softplus(a + dt_bias)), beta = sigmoid(b).
// q and k are L2-normalized; q is scaled by 1/sqrt(d). Key heads repeat x2 onto
// value heads. State is stored key-major, S[c * 128 + r], so thread r reads a
// contiguous row across the threadgroup.
kernel void gdn_step(
    device float* state [[buffer(0)]],
    device const float* mixed [[buffer(1)]],
    device const float* proj_a [[buffer(2)]],
    device const float* proj_b [[buffer(3)]],
    device const half* a_log [[buffer(4)]],
    device const half* dt_bias [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant uint& key_heads [[buffer(7)]],
    constant uint& value_heads [[buffer(8)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint simds [[simdgroups_per_threadgroup]])
{
    constexpr uint D = 128u;
    threadgroup float shared[32];
    threadgroup float qs[D];
    threadgroup float ks[D];
    uint kh = head / (value_heads / key_heads);
    uint key_dim = key_heads * D;
    float qv = mixed[kh * D + tid];
    float kv = mixed[key_dim + kh * D + tid];
    float vv = mixed[2u * key_dim + head * D + tid];
    float nq = block_sum(qv * qv, shared, lane, simd, simds);
    float nk = block_sum(kv * kv, shared, lane, simd, simds);
    qs[tid] = qv * rsqrt(nq + 1e-6f) * rsqrt(float(D));
    ks[tid] = kv * rsqrt(nk + 1e-6f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float beta = sigmoid(proj_b[head]);
    float x = proj_a[head] + float(dt_bias[head]);
    float soft = x > 20.0f ? x : log(1.0f + exp(x));
    float alpha = exp(-exp(float(a_log[head])) * soft);
    device float* S = state + ulong(head) * D * D;
    uint r = tid;
    float sk = 0.0f;
    for (uint c = 0u; c < D; ++c) {
        sk = fma(S[c * D + r], ks[c], sk);
    }
    float delta = beta * (vv - alpha * sk);
    float o = 0.0f;
    for (uint c = 0u; c < D; ++c) {
        float s = fma(alpha, S[c * D + r], ks[c] * delta);
        S[c * D + r] = s;
        o = fma(s, qs[c], o);
    }
    output[head * D + r] = o;
}

// Rotate-half RoPE on the first ``rope_dim`` channels of each head, in place.
// inv_freq_i = theta^(-2i / rope_dim). One thread per (head, pair).
kernel void rope_heads(
    device float* data [[buffer(0)]],
    constant uint& heads [[buffer(1)]],
    constant uint& stride [[buffer(2)]],
    constant uint& rope_dim [[buffer(3)]],
    constant uint& position [[buffer(4)]],
    constant float& theta [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint pairs = rope_dim >> 1;
    if (gid >= heads * pairs) {
        return;
    }
    uint head = gid / pairs;
    uint i = gid % pairs;
    float freq = pow(theta, -float(2u * i) / float(rope_dim));
    float angle = float(position) * freq;
    float c = cos(angle);
    float s = sin(angle);
    device float* row = data + head * stride;
    float x0 = row[i];
    float x1 = row[i + pairs];
    row[i] = x0 * c - x1 * s;
    row[i + pairs] = x1 * c + x0 * s;
}

// Quantize one token's keys and values into the fp8 cache with one E8M0 scale per
// 32 channels. Grid (kv_heads, 2): plane 0 is the key, plane 1 the value.
// Cache rows are [position][kv_head][head_dim]; scales are [position][kv_head][head_dim/32].
kernel void store_kv(
    device const float* key [[buffer(0)]],
    device const float* value [[buffer(1)]],
    device uchar* key_data [[buffer(2)]],
    device uchar* key_scale [[buffer(3)]],
    device uchar* value_data [[buffer(4)]],
    device uchar* value_scale [[buffer(5)]],
    constant uint& position [[buffer(6)]],
    constant uint& kv_heads [[buffer(7)]],
    constant uint& plane [[buffer(8)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint dim [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]])
{
    bool is_value = plane == 1u;
    device const float* source = is_value ? value : key;
    device uchar* data = is_value ? value_data : key_data;
    device uchar* scale = is_value ? value_scale : key_scale;
    float x = source[head * dim + tid];
    float amax = simd_max(fabs(x));
    uint code = 0u;
    if (amax > 0.0f) {
        float power = ceil(log2(amax / 448.0f));
        code = uint(clamp(power + 127.0f, 0.0f, 254.0f));
    }
    float inv = code == 0u ? 0.0f : 1.0f / e8m0_code(code);
    ulong row = ulong(position) * kv_heads + head;
    data[row * dim + tid] = uchar(e4m3_encode(x * inv));
    if (lane == 0u) {
        scale[row * (dim >> 5) + simd] = uchar(code);
    }
}

// Decode attention for one query head over positions [0, length). Query rows are
// [q(256) | gate(256)] per head; the gate is applied as sigmoid after the softmax
// mix. ``scores`` is [heads][context] float scratch. One threadgroup of 256 per head.
kernel void attend_decode(
    device const float* query_gate [[buffer(0)]],
    device const uchar* key_data [[buffer(1)]],
    device const uchar* key_scale [[buffer(2)]],
    device const uchar* value_data [[buffer(3)]],
    device const uchar* value_scale [[buffer(4)]],
    device float* scores [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant uint& length [[buffer(7)]],
    constant uint& context [[buffer(8)]],
    constant uint& kv_heads [[buffer(9)]],
    constant uint& group_size [[buffer(10)]],
    constant float& scale [[buffer(11)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint simds [[simdgroups_per_threadgroup]])
{
    constexpr uint D = 256u;
    threadgroup float shared[32];
    uint kvh = head / group_size;
    device const float* q = query_gate + head * (2u * D);
    float qreg[8];
    for (uint j = 0u; j < 8u; ++j) {
        qreg[j] = q[lane * 8u + j];
    }
    device float* row = scores + ulong(head) * context;
    float local_max = -INFINITY;
    for (uint p = simd; p < length; p += simds) {
        ulong slot = ulong(p) * kv_heads + kvh;
        device const uchar* k = key_data + slot * D + lane * 8u;
        float ks = e8m0_code(key_scale[slot * (D >> 5) + (lane >> 2)]);
        uint2 packed = *((device const uint2*)k);
        float part = 0.0f;
        for (uint j = 0u; j < 4u; ++j) {
            part = fma(e4m3_code((packed.x >> (8u * j)) & 0xFFu), qreg[j], part);
            part = fma(e4m3_code((packed.y >> (8u * j)) & 0xFFu), qreg[4u + j], part);
        }
        float dot = simd_sum(part * ks) * scale;
        if (lane == 0u) {
            row[p] = dot;
        }
        local_max = max(local_max, dot);
    }
    if (lane == 0u) {
        shared[simd] = local_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    float peak = -INFINITY;
    for (uint i = 0u; i < simds; ++i) {
        peak = max(peak, shared[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_sum = 0.0f;
    for (uint p = tid; p < length; p += D) {
        float e = exp(row[p] - peak);
        row[p] = e;
        local_sum += e;
    }
    float total = block_sum(local_sum, shared, lane, simd, simds);
    threadgroup_barrier(mem_flags::mem_device);
    float acc = 0.0f;
    for (uint p = 0u; p < length; ++p) {
        ulong slot = ulong(p) * kv_heads + kvh;
        float vs = e8m0_code(value_scale[slot * (D >> 5) + (tid >> 5)]);
        acc = fma(row[p], e4m3_code(value_data[slot * D + tid]) * vs, acc);
    }
    float gate = q[D + tid];
    output[head * D + tid] = acc / total * sigmoid(gate);
}

// logits -= presence * [count > 0] + frequency * count.
// Prefill: embed many token ids into consecutive rows of width ``inner``.
kernel void embed_rows(
    device const uchar* table [[buffer(0)]],
    device const uchar* scales [[buffer(1)]],
    device const uint* tokens [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;
    if (row >= count || col >= inner) {
        return;
    }
    uint token = tokens[row];
    device const uchar* weights = table + ulong(token) * inner;
    device const uchar* scale_row = scales + ulong(token) * (inner >> 5);
    destination[ulong(row) * inner + col] = e4m3_code(weights[col]) * e8m0_code(scale_row[col >> 5]);
}

kernel void cast_f32_f16(
    device const float* source [[buffer(0)]],
    device half* destination [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) {
        destination[gid] = half(source[gid]);
    }
}

kernel void copy_row_f32(
    device const float* source [[buffer(0)]],
    device float* destination [[buffer(1)]],
    constant uint& width [[buffer(2)]],
    constant uint& row [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < width) {
        destination[gid] = source[ulong(row) * width + gid];
    }
}

// y[rows, cols] = x[rows, inner] @ W[cols, inner]^T with fp8 W and E8M0 scales.
// ``rows`` and ``cols`` are padded up to the 64-wide matmul2d tile.
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
        weights, dextents<int, 2>(int(inner), int(cols)), array<int, 2>{1, int(inner)}, scales_plane(scales));
    tensor<device float, dextents<int, 2>, tensor_inline> out(
        destination, dextents<int, 2>(int(cols), int(rows)), array<int, 2>{1, int(cols)});
    constexpr auto desc = matmul2d_descriptor(64, 64, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto tile_left = left.slice(0, int(tgid.y) * 64);
    auto tile_right = right.slice(0, int(tgid.x) * 64);
    auto tile_out = out.slice(int(tgid.x) * 64, int(tgid.y) * 64);
    op.run(tile_left, tile_right, tile_out);
}

// Narrow N=32 tile for GDN a/b (LINEAR_VALUE_HEADS). Rows stay 64.
kernel void matmul_fp8_n32(
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
        weights, dextents<int, 2>(int(inner), int(cols)), array<int, 2>{1, int(inner)}, scales_plane(scales));
    tensor<device float, dextents<int, 2>, tensor_inline> out(
        destination, dextents<int, 2>(int(cols), int(rows)), array<int, 2>{1, int(cols)});
    constexpr auto desc = matmul2d_descriptor(64, 32, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto tile_left = left.slice(0, int(tgid.y) * 64);
    auto tile_right = right.slice(0, int(tgid.x) * 32);
    auto tile_out = out.slice(int(tgid.x) * 32, int(tgid.y) * 64);
    op.run(tile_left, tile_right, tile_out);
}

kernel void rmsnorm_rows(
    device const float* input [[buffer(0)]],
    device const half* gain [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    constant float& shift [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint simds [[simdgroups_per_threadgroup]],
    uint tid [[thread_index_in_threadgroup]],
    uint count [[threads_per_threadgroup]])
{
    if (row >= rows) {
        return;
    }
    threadgroup float shared[32];
    device const float* in = input + ulong(row) * width;
    device float* out = destination + ulong(row) * width;
    float acc = 0.0f;
    for (uint i = tid; i < width; i += count) {
        float v = in[i];
        acc = fma(v, v, acc);
    }
    float total = block_sum(acc, shared, lane, simd, simds);
    float inv = rsqrt(total / float(width) + eps);
    for (uint i = tid; i < width; i += count) {
        out[i] = in[i] * inv * (float(gain[i]) + shift);
    }
}

kernel void silu_mul_rows(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;
    if (row >= rows || col >= width) {
        return;
    }
    ulong at = ulong(row) * width + col;
    destination[at] = silu(gate[at]) * up[at];
}

kernel void residual_add_rows(
    device float* stream [[buffer(0)]],
    device const float* delta [[buffer(1)]],
    constant uint& width [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;
    if (row >= rows || col >= width) {
        return;
    }
    ulong at = ulong(row) * width + col;
    stream[at] += delta[at];
}

// Causal depthwise conv over a token chunk. ``mixed`` is [count][channels].
kernel void conv_silu_chunk(
    device const half* taps [[buffer(0)]],
    device float* mixed [[buffer(1)]],
    device float* history [[buffer(2)]],
    constant uint& channels [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= channels) {
        return;
    }
    device float* h = history + gid * 3u;
    float x0 = h[0];
    float x1 = h[1];
    float x2 = h[2];
    device const half* w = taps + gid * 4u;
    for (uint t = 0u; t < count; ++t) {
        float x3 = mixed[ulong(t) * channels + gid];
        float y = float(w[0]) * x0 + float(w[1]) * x1 + float(w[2]) * x2 + float(w[3]) * x3;
        mixed[ulong(t) * channels + gid] = silu(y);
        x0 = x1;
        x1 = x2;
        x2 = x3;
    }
    h[0] = x0;
    h[1] = x1;
    h[2] = x2;
}

// Prefill Gated DeltaNet over a chunk (Yang et al. §3 recurrence, sequential in t).
// ``mixed`` is [count][q|k|v], ``proj_a``/``proj_b`` are [count][value_heads],
// ``output`` is [count][value_heads * D].
kernel void gdn_chunk(
    device float* state [[buffer(0)]],
    device const float* mixed [[buffer(1)]],
    device const float* proj_a [[buffer(2)]],
    device const float* proj_b [[buffer(3)]],
    device const half* a_log [[buffer(4)]],
    device const half* dt_bias [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant uint& key_heads [[buffer(7)]],
    constant uint& value_heads [[buffer(8)]],
    constant uint& count [[buffer(9)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint simds [[simdgroups_per_threadgroup]])
{
    constexpr uint D = 128u;
    threadgroup float shared[32];
    threadgroup float qs[D];
    threadgroup float ks[D];
    uint kh = head / (value_heads / key_heads);
    uint key_dim = key_heads * D;
    uint width = 2u * key_dim + value_heads * D;
    device float* S = state + ulong(head) * D * D;
    for (uint t = 0u; t < count; ++t) {
        device const float* row = mixed + ulong(t) * width;
        float qv = row[kh * D + tid];
        float kv = row[key_dim + kh * D + tid];
        float vv = row[2u * key_dim + head * D + tid];
        float nq = block_sum(qv * qv, shared, lane, simd, simds);
        float nk = block_sum(kv * kv, shared, lane, simd, simds);
        qs[tid] = qv * rsqrt(nq + 1e-6f) * rsqrt(float(D));
        ks[tid] = kv * rsqrt(nk + 1e-6f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float beta = sigmoid(proj_b[ulong(t) * value_heads + head]);
        float x = proj_a[ulong(t) * value_heads + head] + float(dt_bias[head]);
        float soft = x > 20.0f ? x : log(1.0f + exp(x));
        float alpha = exp(-exp(float(a_log[head])) * soft);
        uint r = tid;
        float sk = 0.0f;
        for (uint c = 0u; c < D; ++c) {
            sk = fma(S[c * D + r], ks[c], sk);
        }
        float delta = beta * (vv - alpha * sk);
        float o = 0.0f;
        for (uint c = 0u; c < D; ++c) {
            float s = fma(alpha, S[c * D + r], ks[c] * delta);
            S[c * D + r] = s;
            o = fma(s, qs[c], o);
        }
        output[ulong(t) * value_heads * D + head * D + r] = o;
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }
}

kernel void gated_rmsnorm_rows(
    device float* data [[buffer(0)]],
    device const float* gate [[buffer(1)]],
    device const half* gain [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& heads [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint simds [[simdgroups_per_threadgroup]])
{
    constexpr uint D = 128u;
    uint row = tgp.y;
    uint head = tgp.x;
    if (row >= rows || head >= heads || tid >= D) {
        return;
    }
    threadgroup float shared[32];
    ulong at = (ulong(row) * heads + head) * D + tid;
    float v = data[at];
    float total = block_sum(v * v, shared, lane, simd, simds);
    float inv = rsqrt(total / float(D) + eps);
    data[at] = v * inv * float(gain[tid]) * silu(gate[at]);
}

kernel void head_rmsnorm_rows(
    device float* data [[buffer(0)]],
    device const half* gain [[buffer(1)]],
    constant uint& stride [[buffer(2)]],
    constant uint& heads [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    constant float& shift [[buffer(6)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint simds [[simdgroups_per_threadgroup]])
{
    constexpr uint D = 256u;
    uint row = tgp.y;
    uint head = tgp.x;
    if (row >= rows || head >= heads || tid >= D) {
        return;
    }
    threadgroup float shared[32];
    device float* rowp = data + (ulong(row) * heads + head) * stride;
    float v = rowp[tid];
    float total = block_sum(v * v, shared, lane, simd, simds);
    float inv = rsqrt(total / float(D) + eps);
    rowp[tid] = v * inv * (float(gain[tid]) + shift);
}

kernel void rope_heads_rows(
    device float* data [[buffer(0)]],
    constant uint& heads [[buffer(1)]],
    constant uint& stride [[buffer(2)]],
    constant uint& rope_dim [[buffer(3)]],
    constant uint& position0 [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant float& theta [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint pairs = rope_dim >> 1;
    uint row = gid.y;
    uint at = gid.x;
    if (row >= rows || at >= heads * pairs) {
        return;
    }
    uint head = at / pairs;
    uint i = at % pairs;
    float freq = pow(theta, -float(2u * i) / float(rope_dim));
    float angle = float(position0 + row) * freq;
    float c = cos(angle);
    float s = sin(angle);
    device float* rowp = data + (ulong(row) * heads + head) * stride;
    float x0 = rowp[i];
    float x1 = rowp[i + pairs];
    rowp[i] = x0 * c - x1 * s;
    rowp[i + pairs] = x1 * c + x0 * s;
}

kernel void store_kv_rows(
    device const float* key [[buffer(0)]],
    device const float* value [[buffer(1)]],
    device uchar* key_data [[buffer(2)]],
    device uchar* key_scale [[buffer(3)]],
    device uchar* value_data [[buffer(4)]],
    device uchar* value_scale [[buffer(5)]],
    constant uint& position0 [[buffer(6)]],
    constant uint& kv_heads [[buffer(7)]],
    constant uint& rows [[buffer(8)]],
    constant uint& plane [[buffer(9)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]])
{
    constexpr uint D = 256u;
    uint row = tgp.y;
    uint head = tgp.x;
    if (row >= rows || head >= kv_heads || tid >= D) {
        return;
    }
    bool is_value = plane == 1u;
    device const float* source = is_value ? value : key;
    device uchar* data = is_value ? value_data : key_data;
    device uchar* scale = is_value ? value_scale : key_scale;
    float x = source[(ulong(row) * kv_heads + head) * D + tid];
    float amax = simd_max(fabs(x));
    uint code = 0u;
    if (amax > 0.0f) {
        float power = ceil(log2(amax / 448.0f));
        code = uint(clamp(power + 127.0f, 0.0f, 254.0f));
    }
    float inv = code == 0u ? 0.0f : 1.0f / e8m0_code(code);
    ulong slot = ulong(position0 + row) * kv_heads + head;
    data[slot * D + tid] = uchar(e4m3_encode(x * inv));
    if (lane == 0u) {
        scale[slot * (D >> 5) + simd] = uchar(code);
    }
}

// Prefill attention for ``rows`` queries. Query rows are [q|gate] per head.
// Causal: query row r attends to positions [0, position0 + r].
kernel void attend_prefill(
    device const float* query_gate [[buffer(0)]],
    device const uchar* key_data [[buffer(1)]],
    device const uchar* key_scale [[buffer(2)]],
    device const uchar* value_data [[buffer(3)]],
    device const uchar* value_scale [[buffer(4)]],
    device float* scores [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant uint& position0 [[buffer(7)]],
    constant uint& rows [[buffer(8)]],
    constant uint& context [[buffer(9)]],
    constant uint& attn_heads [[buffer(10)]],
    constant uint& kv_heads [[buffer(11)]],
    constant uint& group_size [[buffer(12)]],
    constant float& scale [[buffer(13)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint simds [[simdgroups_per_threadgroup]])
{
    constexpr uint D = 256u;
    uint row = tgp.y;
    uint head = tgp.x;
    if (row >= rows || head >= attn_heads || tid >= D) {
        return;
    }
    threadgroup float shared[32];
    uint kvh = head / group_size;
    uint length = position0 + row + 1u;
    device const float* q = query_gate + (ulong(row) * attn_heads + head) * (2u * D);
    float qreg[8];
    for (uint j = 0u; j < 8u; ++j) {
        qreg[j] = q[lane * 8u + j];
    }
    device float* srow = scores + (ulong(row) * attn_heads + head) * context;
    float local_max = -INFINITY;
    for (uint p = simd; p < length; p += simds) {
        ulong slot = ulong(p) * kv_heads + kvh;
        device const uchar* k = key_data + slot * D + lane * 8u;
        float ks = e8m0_code(key_scale[slot * (D >> 5) + (lane >> 2)]);
        uint2 packed = *((device const uint2*)k);
        float part = 0.0f;
        for (uint j = 0u; j < 4u; ++j) {
            part = fma(e4m3_code((packed.x >> (8u * j)) & 0xFFu), qreg[j], part);
            part = fma(e4m3_code((packed.y >> (8u * j)) & 0xFFu), qreg[4u + j], part);
        }
        float dot = simd_sum(part * ks) * scale;
        if (lane == 0u) {
            srow[p] = dot;
        }
        local_max = max(local_max, dot);
    }
    if (lane == 0u) {
        shared[simd] = local_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    float peak = -INFINITY;
    for (uint i = 0u; i < simds; ++i) {
        peak = max(peak, shared[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_sum = 0.0f;
    for (uint p = tid; p < length; p += D) {
        float e = exp(srow[p] - peak);
        srow[p] = e;
        local_sum += e;
    }
    float total = block_sum(local_sum, shared, lane, simd, simds);
    threadgroup_barrier(mem_flags::mem_device);
    float acc = 0.0f;
    uint p = 0u;
    // Unrolled value mix; each thread owns one output dim and walks the keys.
    for (; p + 3u < length; p += 4u) {
        ulong slot0 = ulong(p) * kv_heads + kvh;
        ulong slot1 = slot0 + kv_heads;
        ulong slot2 = slot1 + kv_heads;
        ulong slot3 = slot2 + kv_heads;
        uint scale_i = tid >> 5;
        float v0 = e4m3_code(value_data[slot0 * D + tid]) * e8m0_code(value_scale[slot0 * (D >> 5) + scale_i]);
        float v1 = e4m3_code(value_data[slot1 * D + tid]) * e8m0_code(value_scale[slot1 * (D >> 5) + scale_i]);
        float v2 = e4m3_code(value_data[slot2 * D + tid]) * e8m0_code(value_scale[slot2 * (D >> 5) + scale_i]);
        float v3 = e4m3_code(value_data[slot3 * D + tid]) * e8m0_code(value_scale[slot3 * (D >> 5) + scale_i]);
        acc = fma(srow[p], v0, acc);
        acc = fma(srow[p + 1u], v1, acc);
        acc = fma(srow[p + 2u], v2, acc);
        acc = fma(srow[p + 3u], v3, acc);
    }
    for (; p < length; ++p) {
        ulong slot = ulong(p) * kv_heads + kvh;
        float vs = e8m0_code(value_scale[slot * (D >> 5) + (tid >> 5)]);
        acc = fma(srow[p], e4m3_code(value_data[slot * D + tid]) * vs, acc);
    }
    float gate = q[D + tid];
    output[(ulong(row) * attn_heads + head) * D + tid] = acc / total * sigmoid(gate);
}

// Float16 key/value cache. Same layouts as the e4m3 store, two bytes per element,
// no scale plane.
kernel void store_kv_f16(
    device const float* key [[buffer(0)]],
    device const float* value [[buffer(1)]],
    device half* key_data [[buffer(2)]],
    device half* value_data [[buffer(3)]],
    constant uint& position [[buffer(4)]],
    constant uint& kv_heads [[buffer(5)]],
    constant uint& plane [[buffer(6)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint dim [[threads_per_threadgroup]])
{
    bool is_value = plane == 1u;
    device const float* source = is_value ? value : key;
    device half* data = is_value ? value_data : key_data;
    ulong row = ulong(position) * kv_heads + head;
    data[row * dim + tid] = half(source[head * dim + tid]);
}

kernel void store_kv_rows_f16(
    device const float* key [[buffer(0)]],
    device const float* value [[buffer(1)]],
    device half* key_data [[buffer(2)]],
    device half* value_data [[buffer(3)]],
    constant uint& position0 [[buffer(4)]],
    constant uint& kv_heads [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& plane [[buffer(7)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    constexpr uint D = 256u;
    uint row = tgp.y;
    uint head = tgp.x;
    if (row >= rows || head >= kv_heads || tid >= D) {
        return;
    }
    bool is_value = plane == 1u;
    device const float* source = is_value ? value : key;
    device half* data = is_value ? value_data : key_data;
    ulong slot = ulong(position0 + row) * kv_heads + head;
    data[slot * D + tid] = half(source[(ulong(row) * kv_heads + head) * D + tid]);
}

kernel void attend_decode_f16(
    device const float* query_gate [[buffer(0)]],
    device const half* key_data [[buffer(1)]],
    device const half* value_data [[buffer(2)]],
    device float* scores [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& length [[buffer(5)]],
    constant uint& context [[buffer(6)]],
    constant uint& kv_heads [[buffer(7)]],
    constant uint& group_size [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint simds [[simdgroups_per_threadgroup]])
{
    constexpr uint D = 256u;
    threadgroup float shared[32];
    uint kvh = head / group_size;
    device const float* q = query_gate + head * (2u * D);
    float qreg[8];
    for (uint j = 0u; j < 8u; ++j) {
        qreg[j] = q[lane * 8u + j];
    }
    device float* row = scores + ulong(head) * context;
    float local_max = -INFINITY;
    for (uint p = simd; p < length; p += simds) {
        ulong slot = ulong(p) * kv_heads + kvh;
        device const half* k = key_data + slot * D + lane * 8u;
        float part = 0.0f;
        for (uint j = 0u; j < 8u; ++j) {
            part = fma(float(k[j]), qreg[j], part);
        }
        float dot = simd_sum(part) * scale;
        if (lane == 0u) {
            row[p] = dot;
        }
        local_max = max(local_max, dot);
    }
    if (lane == 0u) {
        shared[simd] = local_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    float peak = -INFINITY;
    for (uint i = 0u; i < simds; ++i) {
        peak = max(peak, shared[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_sum = 0.0f;
    for (uint p = tid; p < length; p += D) {
        float e = exp(row[p] - peak);
        row[p] = e;
        local_sum += e;
    }
    float total = block_sum(local_sum, shared, lane, simd, simds);
    threadgroup_barrier(mem_flags::mem_device);
    float acc = 0.0f;
    for (uint p = 0u; p < length; ++p) {
        ulong slot = ulong(p) * kv_heads + kvh;
        acc = fma(row[p], float(value_data[slot * D + tid]), acc);
    }
    float gate = q[D + tid];
    output[head * D + tid] = acc / total * sigmoid(gate);
}

kernel void attend_prefill_f16(
    device const float* query_gate [[buffer(0)]],
    device const half* key_data [[buffer(1)]],
    device const half* value_data [[buffer(2)]],
    device float* scores [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& position0 [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& context [[buffer(7)]],
    constant uint& attn_heads [[buffer(8)]],
    constant uint& kv_heads [[buffer(9)]],
    constant uint& group_size [[buffer(10)]],
    constant float& scale [[buffer(11)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint simds [[simdgroups_per_threadgroup]])
{
    constexpr uint D = 256u;
    uint row = tgp.y;
    uint head = tgp.x;
    if (row >= rows || head >= attn_heads || tid >= D) {
        return;
    }
    threadgroup float shared[32];
    uint kvh = head / group_size;
    uint length = position0 + row + 1u;
    device const float* q = query_gate + (ulong(row) * attn_heads + head) * (2u * D);
    float qreg[8];
    for (uint j = 0u; j < 8u; ++j) {
        qreg[j] = q[lane * 8u + j];
    }
    device float* srow = scores + (ulong(row) * attn_heads + head) * context;
    float local_max = -INFINITY;
    for (uint p = simd; p < length; p += simds) {
        ulong slot = ulong(p) * kv_heads + kvh;
        device const half* k = key_data + slot * D + lane * 8u;
        float part = 0.0f;
        for (uint j = 0u; j < 8u; ++j) {
            part = fma(float(k[j]), qreg[j], part);
        }
        float dot = simd_sum(part) * scale;
        if (lane == 0u) {
            srow[p] = dot;
        }
        local_max = max(local_max, dot);
    }
    if (lane == 0u) {
        shared[simd] = local_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    float peak = -INFINITY;
    for (uint i = 0u; i < simds; ++i) {
        peak = max(peak, shared[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_sum = 0.0f;
    for (uint p = tid; p < length; p += D) {
        float e = exp(srow[p] - peak);
        srow[p] = e;
        local_sum += e;
    }
    float total = block_sum(local_sum, shared, lane, simd, simds);
    threadgroup_barrier(mem_flags::mem_device);
    float acc = 0.0f;
    uint p = 0u;
    for (; p + 3u < length; p += 4u) {
        ulong slot0 = ulong(p) * kv_heads + kvh;
        float v0 = float(value_data[slot0 * D + tid]);
        float v1 = float(value_data[(slot0 + kv_heads) * D + tid]);
        float v2 = float(value_data[(slot0 + 2u * kv_heads) * D + tid]);
        float v3 = float(value_data[(slot0 + 3u * kv_heads) * D + tid]);
        acc = fma(srow[p], v0, acc);
        acc = fma(srow[p + 1u], v1, acc);
        acc = fma(srow[p + 2u], v2, acc);
        acc = fma(srow[p + 3u], v3, acc);
    }
    for (; p < length; ++p) {
        ulong slot = ulong(p) * kv_heads + kvh;
        acc = fma(srow[p], float(value_data[slot * D + tid]), acc);
    }
    float gate = q[D + tid];
    output[(ulong(row) * attn_heads + head) * D + tid] = acc / total * sigmoid(gate);
}

kernel void penalize(
    device float* logits [[buffer(0)]],
    device const uint* counts [[buffer(1)]],
    constant uint& vocab [[buffer(2)]],
    constant float& presence [[buffer(3)]],
    constant float& frequency [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= vocab) {
        return;
    }
    uint n = counts[gid];
    if (n > 0u) {
        logits[gid] -= presence + frequency * float(n);
    }
}

kernel void bump_count(
    device uint* counts [[buffer(0)]],
    device const uint* tokens [[buffer(1)]],
    constant uint& index [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid == 0u) {
        counts[tokens[index]] += 1u;
    }
}

static void better(thread float& best, thread uint& best_i, float other, uint other_i) {
    if (other > best || (other == best && other_i < best_i)) {
        best = other;
        best_i = other_i;
    }
}

kernel void argmax_partial(
    device const float* logits [[buffer(0)]],
    device float* values [[buffer(1)]],
    device uint* indices [[buffer(2)]],
    constant uint& vocab [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    uint start = group * 256u;
    float best = -INFINITY;
    uint best_i = 0xFFFFFFFFu;
    for (uint i = tid; i < 256u; i += 32u) {
        uint idx = start + i;
        if (idx < vocab) {
            better(best, best_i, logits[idx], idx);
        }
    }
    for (uint offset = 16u; offset > 0u; offset >>= 1u) {
        float other = simd_shuffle_xor(best, offset);
        uint other_i = simd_shuffle_xor(best_i, offset);
        better(best, best_i, other, other_i);
    }
    if (tid == 0u) {
        values[group] = best;
        indices[group] = best_i;
    }
}

// Writes the winner into tokens[index].
kernel void argmax_final(
    device const float* values [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device uint* tokens [[buffer(2)]],
    constant uint& groups [[buffer(3)]],
    constant uint& index [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]])
{
    float best = -INFINITY;
    uint best_i = 0xFFFFFFFFu;
    for (uint i = tid; i < groups; i += 32u) {
        better(best, best_i, values[i], indices[i]);
    }
    for (uint offset = 16u; offset > 0u; offset >>= 1u) {
        float other = simd_shuffle_xor(best, offset);
        uint other_i = simd_shuffle_xor(best_i, offset);
        better(best, best_i, other, other_i);
    }
    if (tid == 0u) {
        tokens[index] = best_i;
    }
}

// Threadgroup argmax over 256 threads. Every thread receives the winner.
static void group_best(
    thread float& best,
    thread uint& best_i,
    threadgroup float* sv,
    threadgroup uint* si,
    uint lane,
    uint simd,
    uint simds)
{
    for (uint offset = 16u; offset > 0u; offset >>= 1u) {
        float other = simd_shuffle_xor(best, offset);
        uint other_i = simd_shuffle_xor(best_i, offset);
        better(best, best_i, other, other_i);
    }
    if (lane == 0u) {
        sv[simd] = best;
        si[simd] = best_i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    best = -INFINITY;
    best_i = 0xFFFFFFFFu;
    for (uint i = 0u; i < simds; ++i) {
        better(best, best_i, sv[i], si[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

constant uint TOPK_SPAN = 64u;

// Top ``k`` of one chunk of 256 * 64 logits, in descending order. Each thread keeps
// the best of its 64 entries; only the thread that won a round rescans.
// Chosen entries are set to -inf in ``logits``.
kernel void topk_partial(
    device float* logits [[buffer(0)]],
    device float* values [[buffer(1)]],
    device uint* indices [[buffer(2)]],
    constant uint& vocab [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint simds [[simdgroups_per_threadgroup]])
{
    threadgroup float sv[32];
    threadgroup uint si[32];
    uint start = group * 256u * TOPK_SPAN + tid;
    float mine = -INFINITY;
    uint mine_i = 0xFFFFFFFFu;
    for (uint j = 0u; j < TOPK_SPAN; ++j) {
        uint idx = start + j * 256u;
        if (idx < vocab) {
            better(mine, mine_i, logits[idx], idx);
        }
    }
    for (uint round = 0u; round < k; ++round) {
        float best = mine;
        uint best_i = mine_i;
        group_best(best, best_i, sv, si, lane, simd, simds);
        if (tid == 0u) {
            values[group * k + round] = best;
            indices[group * k + round] = best_i;
        }
        if (best_i == mine_i && best_i != 0xFFFFFFFFu) {
            logits[best_i] = -INFINITY;
            mine = -INFINITY;
            mine_i = 0xFFFFFFFFu;
            for (uint j = 0u; j < TOPK_SPAN; ++j) {
                uint idx = start + j * 256u;
                if (idx < vocab) {
                    better(mine, mine_i, logits[idx], idx);
                }
            }
        }
    }
}

// Merge the partial candidates into the global top ``k``, apply temperature and
// nucleus ``top_p``, and draw with the host's uniform ``draw`` in [0, 1).
// Writes the token into tokens[index]. One threadgroup of 256; count <= 4096.
kernel void topk_sample(
    device const float* values [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device uint* tokens [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    constant float& temperature [[buffer(5)]],
    constant float& top_p [[buffer(6)]],
    constant float& draw [[buffer(7)]],
    constant uint& index [[buffer(8)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint simds [[simdgroups_per_threadgroup]])
{
    threadgroup float cand_v[4096];
    threadgroup uint cand_i[4096];
    threadgroup float top_v[64];
    threadgroup uint top_i[64];
    threadgroup float sv[32];
    threadgroup uint si[32];
    threadgroup uint slot_won[1];
    for (uint i = tid; i < count; i += 256u) {
        cand_v[i] = values[i];
        cand_i[i] = indices[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint round = 0u; round < k; ++round) {
        float best = -INFINITY;
        uint best_i = 0xFFFFFFFFu;
        uint best_slot = 0xFFFFFFFFu;
        for (uint i = tid; i < count; i += 256u) {
            float v = cand_v[i];
            uint id = cand_i[i];
            if (v > best || (v == best && id < best_i)) {
                best = v;
                best_i = id;
                best_slot = i;
            }
        }
        float group_v = best;
        uint group_i = best_i;
        group_best(group_v, group_i, sv, si, lane, simd, simds);
        if (best_i == group_i && best_slot != 0xFFFFFFFFu && group_v == best) {
            slot_won[0] = best_slot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            top_v[round] = group_v;
            top_i[round] = group_i;
            if (group_i != 0xFFFFFFFFu) {
                cand_v[slot_won[0]] = -INFINITY;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid != 0u) {
        return;
    }
    float peak = top_v[0];
    float inv_t = 1.0f / max(temperature, 1e-5f);
    float total = 0.0f;
    uint live = 0u;
    for (uint i = 0u; i < k; ++i) {
        if (top_i[i] == 0xFFFFFFFFu || !isfinite(top_v[i])) {
            break;
        }
        float e = exp((top_v[i] - peak) * inv_t);
        top_v[i] = e;
        total += e;
        live = i + 1u;
    }
    if (live == 0u) {
        tokens[index] = top_i[0];
        return;
    }
    float cumulative = 0.0f;
    uint keep = live;
    for (uint i = 0u; i < live; ++i) {
        cumulative += top_v[i] / total;
        if (cumulative >= top_p) {
            keep = i + 1u;
            break;
        }
    }
    float kept = 0.0f;
    for (uint i = 0u; i < keep; ++i) {
        kept += top_v[i];
    }
    float target = draw * kept;
    float run = 0.0f;
    uint chosen = top_i[keep - 1u];
    for (uint i = 0u; i < keep; ++i) {
        run += top_v[i];
        if (target < run) {
            chosen = top_i[i];
            break;
        }
    }
    tokens[index] = chosen;
}

// Interleaved M-RoPE axes for sections (11, 11, 10): t, h, w, repeating, then t, h.
constant uint MROPE_AXIS[32] = {
    0u, 1u, 2u, 0u, 1u, 2u, 0u, 1u, 2u, 0u, 1u, 2u, 0u, 1u, 2u, 0u,
    1u, 2u, 0u, 1u, 2u, 0u, 1u, 2u, 0u, 1u, 2u, 0u, 1u, 2u, 0u, 1u,
};

// ``positions`` is axis-major ``[3, prompt_len]``. Text rows store one index on every axis.
kernel void rope_mrope_rows(
    device float* data [[buffer(0)]],
    device const uint* positions [[buffer(1)]],
    constant uint& heads [[buffer(2)]],
    constant uint& stride [[buffer(3)]],
    constant uint& rope_dim [[buffer(4)]],
    constant uint& position0 [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& prompt_len [[buffer(7)]],
    constant float& theta [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint pairs = rope_dim >> 1;
    uint row = gid.y;
    uint at = gid.x;
    if (row >= rows || at >= heads * pairs) {
        return;
    }
    uint head = at / pairs;
    uint i = at % pairs;
    uint axis = MROPE_AXIS[i];
    uint pos = positions[axis * prompt_len + position0 + row];
    float freq = pow(theta, -float(2u * i) / float(rope_dim));
    float angle = float(pos) * freq;
    float c = cos(angle);
    float s = sin(angle);
    device float* rowp = data + (ulong(row) * heads + head) * stride;
    float x0 = rowp[i];
    float x1 = rowp[i + pairs];
    rowp[i] = x0 * c - x1 * s;
    rowp[i + pairs] = x1 * c + x0 * s;
}

// Overwrite hidden rows whose index is not 0xFFFFFFFF. ``rows`` is [slots, width].
kernel void splice_rows(
    device const float* rows [[buffer(0)]],
    device const uint* index [[buffer(1)]],
    device float* hidden [[buffer(2)]],
    constant uint& position0 [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint col = gid.x;
    uint row = gid.y;
    if (row >= count || col >= width) {
        return;
    }
    uint slot = index[position0 + row];
    if (slot == 0xFFFFFFFFu) {
        return;
    }
    hidden[ulong(row) * width + col] = rows[ulong(slot) * width + col];
}

kernel void cast_pad_f16(
    device const float* source [[buffer(0)]],
    device half* destination [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& dst_rows [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint col = gid.x;
    uint row = gid.y;
    if (col >= cols || row >= dst_rows) {
        return;
    }
    float value = 0.0f;
    if (row < rows) {
        value = source[ulong(row) * cols + col];
    }
    destination[ulong(row) * cols + col] = half(value);
}

kernel void copy_bias_f32(
    device const float* wide [[buffer(0)]],
    device const half* bias [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint col = gid.x;
    uint row = gid.y;
    if (col >= cols || row >= rows) {
        return;
    }
    destination[ulong(row) * cols + col] = wide[ulong(row) * cols + col] + float(bias[col]);
}

kernel void layernorm_bias_rows(
    device const float* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint simds [[simdgroups_per_threadgroup]],
    uint tid [[thread_index_in_threadgroup]])
{
    if (row >= rows) {
        return;
    }
    threadgroup float shared[32];
    device const float* in = input + ulong(row) * width;
    device float* out = destination + ulong(row) * width;
    float sum = 0.0f;
    for (uint i = tid; i < width; i += 256u) {
        sum += in[i];
    }
    float mean = block_sum(sum, shared, lane, simd, simds) / float(width);
    float var = 0.0f;
    for (uint i = tid; i < width; i += 256u) {
        float delta = in[i] - mean;
        var = fma(delta, delta, var);
    }
    float inv = rsqrt(block_sum(var, shared, lane, simd, simds) / float(width) + eps);
    for (uint i = tid; i < width; i += 256u) {
        out[i] = (in[i] - mean) * inv * float(weight[i]) + float(bias[i]);
    }
}

kernel void gelu_tanh_n(
    device float* values [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) {
        return;
    }
    float x = values[tid];
    // tanh(z) via exp overflows once |x| is past about 10. The cubic is saturated by |x| == 8.
    if (fabs(x) >= 8.0f) {
        values[tid] = x > 0.0f ? x : 0.0f;
        return;
    }
    float inner = 0.7978845608028654f * fma(0.044715f * x * x, x, x);
    values[tid] = 0.5f * x * (1.0f + tanh(inner));
}

static float erf_f(float x) {
    float sign = x < 0.0f ? -1.0f : 1.0f;
    float ax = fabs(x);
    float t = 1.0f / (1.0f + 0.3275911f * ax);
    float poly = 1.061405429f;
    poly = poly * t - 1.453152027f;
    poly = poly * t + 1.421413741f;
    poly = poly * t - 0.284496736f;
    poly = poly * t + 0.254829592f;
    poly = poly * t;
    return sign * (1.0f - poly * exp(-ax * ax));
}

kernel void gelu_erf_n(
    device float* values [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) {
        return;
    }
    float x = values[tid];
    values[tid] = 0.5f * x * (1.0f + erf_f(x * 0.7071067811865476f));
}

kernel void split_qkv1024(
    device const float* qkv [[buffer(0)]],
    device float* q [[buffer(1)]],
    device float* k [[buffer(2)]],
    device float* v [[buffer(3)]],
    constant uint& tokens [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint channel = gid.x;
    uint token = gid.y;
    if (token >= tokens || channel >= 1024u) {
        return;
    }
    uint base = token * 3072u + channel;
    uint out = token * 1024u + channel;
    q[out] = qkv[base];
    k[out] = qkv[base + 1024u];
    v[out] = qkv[base + 2048u];
}

// Duplicate 32 frequencies are already in the table. Rotate the two halves of the head.
kernel void vision_rope64(
    device float* values [[buffer(0)]],
    device const float* cosv [[buffer(1)]],
    device const float* sinv [[buffer(2)]],
    constant uint& tokens [[buffer(3)]],
    constant uint& heads [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint token [[threadgroup_position_in_grid]])
{
    if (token >= tokens) {
        return;
    }
    uint pairs = heads * 32u;
    for (uint index = tid; index < pairs; index += 256u) {
        uint head = index / 32u;
        uint pair = index - head * 32u;
        device float* row = values + (token * heads + head) * 64u;
        float c = cosv[token * 64u + pair];
        float s = sinv[token * 64u + pair];
        float x0 = row[pair];
        float x1 = row[32u + pair];
        row[pair] = x0 * c - x1 * s;
        row[32u + pair] = x1 * c + x0 * s;
    }
}

kernel void vision_attend64(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& tokens [[buffer(4)]],
    constant uint& heads [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
    uint head = tgid.x;
    uint qindex = tgid.y;
    if (head >= heads || qindex >= tokens) {
        return;
    }
    device const float* qrow = query + (qindex * heads + head) * 64u;
    threadgroup float red[2];
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    for (uint token = 0u; token < tokens; ++token) {
        device const float* krow = key + (token * heads + head) * 64u;
        float mine = qrow[tid] * krow[tid];
        float part = simd_sum(mine);
        if (lane == 0u) {
            red[simd] = part;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float score = (red[0] + red[1]) * 0.125f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float next = max(running_max, score);
        running_sum = running_sum * exp(running_max - next) + exp(score - next);
        running_max = next;
    }
    float acc = 0.0f;
    for (uint token = 0u; token < tokens; ++token) {
        device const float* krow = key + (token * heads + head) * 64u;
        device const float* vrow = value + (token * heads + head) * 64u;
        float mine = qrow[tid] * krow[tid];
        float part = simd_sum(mine);
        if (lane == 0u) {
            red[simd] = part;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float score = (red[0] + red[1]) * 0.125f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float weight = exp(score - running_max) / running_sum;
        acc += weight * vrow[tid];
    }
    destination[(qindex * heads + head) * 64u + tid] = acc;
}

kernel void dequant_fp8_rows(
    device const uchar* weights [[buffer(0)]],
    device const uchar* scales [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& logical [[buffer(3)]],
    constant uint& inner [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    if (row >= rows) {
        return;
    }
    uint blocks = logical >> 5;
    device const uchar* packed = weights + ulong(row) * inner;
    device const uchar* row_scale = scales + ulong(row) * (inner >> 5);
    device float* dest = destination + ulong(row) * logical;
    for (uint block = 0u; block < blocks; ++block) {
        float factor = exp2(float(row_scale[block]) - 127.0f);
        uint element = (block << 5) + lane;
        dest[element] = e4m3_code(packed[element]) * factor;
    }
}

// int8 prefill. Activation and weight rows each collapse to one channel scale
// (peak / 127), the matmul2d runs int8 x int8 -> int32, and scale_i8_f32 turns
// the int32 accumulators back into floats in place.

// Reduce a per-thread peak over a 256-thread group. ``shared`` holds 8 floats.
static float group_peak(float value, threadgroup float* shared, uint lane, uint simd, uint tid) {
    value = simd_max(value);
    if (lane == 0u) {
        shared[simd] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float peak = shared[0];
    for (uint i = 1u; i < 8u; ++i) {
        peak = max(peak, shared[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return peak;
}

// One threadgroup per weight row: peak = max |e4m3(code) * e8m0(scale)| over K,
// codes = clamp(rint(value / (peak / 127)), -127, 127). ``channel[row]`` = peak / 127.
kernel void restore_fp8_i8(
    device const uchar* weights [[buffer(0)]],
    device const uchar* scales [[buffer(1)]],
    device char* codes [[buffer(2)]],
    device float* channel [[buffer(3)]],
    constant uint& inner [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    threadgroup float shared[8];
    device const uchar4* packed = (device const uchar4*)(weights + ulong(row) * inner);
    device const uchar* row_scale = scales + ulong(row) * (inner >> 5u);
    uint quads = inner >> 2u;
    float peak = 0.0f;
    for (uint q = tid; q < quads; q += 256u) {
        uchar4 word = packed[q];
        float factor = e8m0_code(row_scale[q >> 3u]);
        peak = max(peak, fabs(e4m3_code(word.x)) * factor);
        peak = max(peak, fabs(e4m3_code(word.y)) * factor);
        peak = max(peak, fabs(e4m3_code(word.z)) * factor);
        peak = max(peak, fabs(e4m3_code(word.w)) * factor);
    }
    peak = group_peak(peak, shared, lane, simd, tid);
    float step = peak / 127.0f;
    float inverse = step > 0.0f ? 1.0f / step : 0.0f;
    device char4* out = (device char4*)(codes + ulong(row) * inner);
    for (uint q = tid; q < quads; q += 256u) {
        uchar4 word = packed[q];
        float factor = e8m0_code(row_scale[q >> 3u]) * inverse;
        float4 scaled = float4(
            e4m3_code(word.x), e4m3_code(word.y), e4m3_code(word.z), e4m3_code(word.w)) * factor;
        out[q] = char4(clamp(rint(scaled), -127.0f, 127.0f));
    }
    if (tid == 0u) {
        channel[row] = step;
    }
}

// One threadgroup per activation row, same peak / 127 rule over a half row.
kernel void quantize_half_i8(
    device const half* activation [[buffer(0)]],
    device char* codes [[buffer(1)]],
    device float* channel [[buffer(2)]],
    constant uint& inner [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    threadgroup float shared[8];
    device const half4* source = (device const half4*)(activation + ulong(row) * inner);
    uint quads = inner >> 2u;
    float peak = 0.0f;
    for (uint q = tid; q < quads; q += 256u) {
        float4 v = fabs(float4(source[q]));
        peak = max(peak, max(max(v.x, v.y), max(v.z, v.w)));
    }
    peak = group_peak(peak, shared, lane, simd, tid);
    float step = peak / 127.0f;
    float inverse = step > 0.0f ? 1.0f / step : 0.0f;
    device char4* out = (device char4*)(codes + ulong(row) * inner);
    for (uint q = tid; q < quads; q += 256u) {
        out[q] = char4(clamp(rint(float4(source[q]) * inverse), -127.0f, 127.0f));
    }
    if (tid == 0u) {
        channel[row] = step;
    }
}

// y[rows, cols] (int32) = x[rows, inner] (int8) @ W[cols, inner]^T (int8).
// ``rows`` and ``cols`` are multiples of 64.
kernel void prefill_i8_n64(
    device int8_t* activation [[buffer(0)]],
    device int8_t* weights [[buffer(1)]],
    device int* destination [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    tensor<device int8_t, dextents<int, 2>, tensor_inline> left(
        activation, dextents<int, 2>(int(inner), int(rows)), array<int, 2>{1, int(inner)});
    tensor<device int8_t, dextents<int, 2>, tensor_inline> right(
        weights, dextents<int, 2>(int(inner), int(cols)), array<int, 2>{1, int(inner)});
    tensor<device int, dextents<int, 2>, tensor_inline> out(
        destination, dextents<int, 2>(int(cols), int(rows)), array<int, 2>{1, int(cols)});
    constexpr auto desc = matmul2d_descriptor(64, 64, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto tile_left = left.slice(0, int(tgid.y) * 64);
    auto tile_right = right.slice(0, int(tgid.x) * 64);
    auto tile_out = out.slice(int(tgid.x) * 64, int(tgid.y) * 64);
    // int8 x int8 -> int32 accumulates in a cooperative tensor, then stores.
    auto acc = op.get_destination_cooperative_tensor<decltype(tile_left), decltype(tile_right), int>();
    op.run(tile_left, tile_right, acc);
    acc.store(tile_out);
}

// 32 rows x 128 columns per group. ``rows`` is a multiple of 32, ``cols`` of 128.
kernel void prefill_i8_m32(
    device int8_t* activation [[buffer(0)]],
    device int8_t* weights [[buffer(1)]],
    device int* destination [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    tensor<device int8_t, dextents<int, 2>, tensor_inline> left(
        activation, dextents<int, 2>(int(inner), int(rows)), array<int, 2>{1, int(inner)});
    tensor<device int8_t, dextents<int, 2>, tensor_inline> right(
        weights, dextents<int, 2>(int(inner), int(cols)), array<int, 2>{1, int(inner)});
    tensor<device int, dextents<int, 2>, tensor_inline> out(
        destination, dextents<int, 2>(int(cols), int(rows)), array<int, 2>{1, int(cols)});
    constexpr auto desc = matmul2d_descriptor(32, 128, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto tile_left = left.slice(0, int(tgid.y) * 32);
    auto tile_right = right.slice(0, int(tgid.x) * 128);
    auto tile_out = out.slice(int(tgid.x) * 128, int(tgid.y) * 32);
    // int8 x int8 -> int32 accumulates in a cooperative tensor, then stores.
    auto acc = op.get_destination_cooperative_tensor<decltype(tile_left), decltype(tile_right), int>();
    op.run(tile_left, tile_right, acc);
    acc.store(tile_out);
}

// In place over the int32 accumulators: float(acc) * row_scale[row] * col_scale[col].
// Four columns per thread; ``cols`` is a multiple of 4.
kernel void scale_i8_f32(
    device uint4* data [[buffer(0)]],
    device const float* row_scale [[buffer(1)]],
    device const float* col_scale [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint quads = cols >> 2u;
    if (gid >= rows * quads) {
        return;
    }
    uint row = gid / quads;
    uint col = (gid - row * quads) << 2u;
    float4 acc = float4(as_type<int4>(data[gid]));
    float4 cs = *((device const float4*)(col_scale + col));
    data[gid] = as_type<uint4>(acc * cs * row_scale[row]);
}
