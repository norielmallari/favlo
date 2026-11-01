#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>

using namespace metal;
using namespace mpp::tensor_ops;

using scales_plane = tensor_blockwise<tensor_plane_scales,
                                      device metal_fp8_ue8m0_format,
                                      32, 1>;

constant float FP4_MAG[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
constant float INV_SQRT2 = 0.7071067811865476f;
constant float ATTN_SCALE = 0.0625f;
constant uint REDUCE_GROUPS = 256u;

struct Pair {
    float p;
    uint id;
};

static float dec_e4m3(uint code) {
    uint sign = code & 0x80u;
    uint exp = (code >> 3) & 0xFu;
    uint mant = code & 0x7u;
    float mag;
    if (exp == 0u) {
        mag = (float(mant) / 8.0f) * exp2(-6.0f);
    } else if (exp == 15u && mant == 7u) {
        return NAN;
    } else {
        mag = (1.0f + float(mant) / 8.0f) * exp2(float(exp) - 7.0f);
    }
    return sign != 0u ? -mag : mag;
}

static uint enc_e4m3(float value) {
    uint bits = as_type<uint>(value);
    uint sign = (bits >> 24) & 0x80u;
    uint exp = (bits >> 23) & 0xffu;
    uint mant = bits & 0x7fffffu;
    float mag = fabs(value);
    if (exp == 0u || exp == 255u || mag < 0.015625f || mag >= 448.0f) {
        if (!(mag > 0.0f)) {
            return sign;
        }
        if (mag >= 448.0f) {
            return sign | 0x7Eu;
        }
        uint sub = uint(round(mag * 512.0f));
        return sign | min(sub, 7u);
    }
    int unbiased = int(exp) - 127;
    uint top = mant >> 20;
    uint rem = mant & 0xfffffu;
    if (rem > 0x80000u || (rem == 0x80000u && (top & 1u) != 0u)) {
        top += 1u;
    }
    int biased = unbiased + 7;
    if (top >= 8u) {
        top = 0u;
        biased += 1;
    }
    // Exponent 15 is finite through mantissa 6 (256..448). Only the next
    // bin, and mantissa 7, saturate. A carry into exponent 15 is 256.
    if (biased > 15 || (biased == 15 && top > 6u)) {
        return sign | 0x7Eu;
    }
    if (biased <= 0) {
        uint sub = uint(round(mag * 512.0f));
        return sign | min(sub, 7u);
    }
    return sign | (uint(biased) << 3) | (top & 7u);
}

static uint e8m0_byte(float amax, float qmax) {
    if (!(amax > 0.0f)) {
        return 0u;
    }
    int biased = int(ceil(log2(amax / qmax))) + 127;
    if (biased < 0) {
        biased = 0;
    }
    if (biased > 254) {
        biased = 254;
    }
    return uint(biased);
}

static float gelu_erf(float value) {
    float sign = value < 0.0f ? -1.0f : 1.0f;
    float ax = fabs(value);
    float t = 1.0f / (1.0f + 0.3275911f * ax);
    float poly = (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t;
    return sign * (1.0f - poly * exp(-ax * ax));
}

static float dequant_fp4(device uchar* data, device uchar* scales, uint row, uint k, uint inner) {
    uint packed = data[row * (inner >> 1) + (k >> 1)];
    uint nibble = (k & 1u) ? (packed >> 4) : (packed & 0xFu);
    float mag = FP4_MAG[nibble & 7u];
    float sign = (nibble & 8u) ? -1.0f : 1.0f;
    uint scale = scales[row * (inner >> 5) + (k >> 5)];
    return sign * mag * exp2(float(scale) - 127.0f);
}

static float fp4_signed(uint nibble) {
    float mag = FP4_MAG[nibble & 7u];
    return (nibble & 8u) ? -mag : mag;
}

template <bool NORM, bool ACCUM>
static void gemv_group_body(
    threadgroup float* act,
    threadgroup float* part,
    device float* activation,
    device half* norm_w,
    device uchar* weights,
    device uchar* scales,
    device float* destination,
    device float* normed,
    uint inner,
    float eps,
    uint tgid,
    uint tid)
{
    uint simd = tid >> 5;
    uint lane = tid & 31u;
    float local = 0.0f;
    for (uint k = tid; k < inner; k += 256u) {
        float value = activation[k];
        act[k] = value;
        if (NORM) {
            local += value * value;
        }
    }
    if (NORM) {
        local = simd_sum(local);
        if (lane == 0u) {
            part[simd] = local;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (NORM) {
        if (tid == 0u) {
            float sum = part[0] + part[1] + part[2] + part[3] + part[4] + part[5] + part[6] + part[7];
            part[0] = rsqrt(sum / float(inner) + eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv = part[0];
        for (uint k = tid; k < inner; k += 256u) {
            act[k] = act[k] * float(norm_w[k]) * inv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tgid == 0u) {
            for (uint k = tid; k < inner; k += 256u) {
                normed[k] = act[k];
            }
        }
    }
    uint row0 = tgid * 32u + simd * 4u;
    uint bytes = inner >> 1;
    uint scale_stride = inner >> 5;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    for (uint base = 0u; base < inner; base += 64u) {
        uint k = base + lane * 2u;
        float a0 = act[k];
        float a1 = act[k + 1u];
        uint byte_index = k >> 1;
        uint scale_index = k >> 5;
        uchar p0 = weights[row0 * bytes + byte_index];
        uchar p1 = weights[(row0 + 1u) * bytes + byte_index];
        uchar p2 = weights[(row0 + 2u) * bytes + byte_index];
        uchar p3 = weights[(row0 + 3u) * bytes + byte_index];
        float s0 = exp2(float(scales[row0 * scale_stride + scale_index]) - 127.0f);
        float s1 = exp2(float(scales[(row0 + 1u) * scale_stride + scale_index]) - 127.0f);
        float s2 = exp2(float(scales[(row0 + 2u) * scale_stride + scale_index]) - 127.0f);
        float s3 = exp2(float(scales[(row0 + 3u) * scale_stride + scale_index]) - 127.0f);
        acc0 += (a0 * fp4_signed(p0 & 0xFu) + a1 * fp4_signed(p0 >> 4)) * s0;
        acc1 += (a0 * fp4_signed(p1 & 0xFu) + a1 * fp4_signed(p1 >> 4)) * s1;
        acc2 += (a0 * fp4_signed(p2 & 0xFu) + a1 * fp4_signed(p2 >> 4)) * s2;
        acc3 += (a0 * fp4_signed(p3 & 0xFu) + a1 * fp4_signed(p3 >> 4)) * s3;
    }
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);
    if (lane == 0u) {
        if (ACCUM) {
            destination[row0] += acc0;
            destination[row0 + 1u] += acc1;
            destination[row0 + 2u] += acc2;
            destination[row0 + 3u] += acc3;
        } else {
            destination[row0] = acc0;
            destination[row0 + 1u] = acc1;
            destination[row0 + 2u] = acc2;
            destination[row0 + 3u] = acc3;
        }
    }
}

static void dot_rows(
    device float* activation,
    device uchar* weights,
    device uchar* scales,
    uint row0,
    uint inner,
    uint lane,
    thread float& acc0,
    thread float& acc1,
    thread float& acc2,
    thread float& acc3)
{
    uint bytes = inner >> 1;
    uint scale_stride = inner >> 5;
    for (uint base = 0u; base < inner; base += 64u) {
        uint k = base + lane * 2u;
        float a0 = activation[k];
        float a1 = activation[k + 1u];
        uint byte_index = k >> 1;
        uint scale_index = k >> 5;
        uchar p0 = weights[row0 * bytes + byte_index];
        uchar p1 = weights[(row0 + 1u) * bytes + byte_index];
        uchar p2 = weights[(row0 + 2u) * bytes + byte_index];
        uchar p3 = weights[(row0 + 3u) * bytes + byte_index];
        float s0 = exp2(float(scales[row0 * scale_stride + scale_index]) - 127.0f);
        float s1 = exp2(float(scales[(row0 + 1u) * scale_stride + scale_index]) - 127.0f);
        float s2 = exp2(float(scales[(row0 + 2u) * scale_stride + scale_index]) - 127.0f);
        float s3 = exp2(float(scales[(row0 + 3u) * scale_stride + scale_index]) - 127.0f);
        acc0 += (a0 * fp4_signed(p0 & 0xFu) + a1 * fp4_signed(p0 >> 4)) * s0;
        acc1 += (a0 * fp4_signed(p1 & 0xFu) + a1 * fp4_signed(p1 >> 4)) * s1;
        acc2 += (a0 * fp4_signed(p2 & 0xFu) + a1 * fp4_signed(p2 >> 4)) * s2;
        acc3 += (a0 * fp4_signed(p3 & 0xFu) + a1 * fp4_signed(p3 >> 4)) * s3;
    }
}

static float key_at(device uchar* data, device uchar* scale, uint kv, uint slot, uint channel, uint capacity) {
    uint code = data[(kv * capacity + slot) * 256u + channel];
    uint sc = scale[(kv * capacity + slot) * 8u + (channel >> 5)];
    return dec_e4m3(code) * exp2(float(sc) - 127.0f);
}

static float value_at(device uchar* data, device uchar* scale, uint kv, uint slot, uint channel, uint capacity) {
    uint code = data[(kv * 256u + channel) * capacity + slot];
    uint blocks = capacity >> 5;
    uint sc = scale[(kv * 256u + channel) * blocks + (slot >> 5)];
    return dec_e4m3(code) * exp2(float(sc) - 127.0f);
}

kernel void gemv_fp4(
    device float* activation [[buffer(0)]],
    device uchar* weights [[buffer(1)]],
    device uchar* scales [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& inner [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    uint row0 = tgid * 4u;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    dot_rows(activation, weights, scales, row0, inner, lane, acc0, acc1, acc2, acc3);
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

static float dot8(float a0, float a1, float a2, float a3, float a4, float a5, float a6, float a7, uint packed) {
    return a0 * fp4_signed(packed & 0xFu)
        + a1 * fp4_signed((packed >> 4) & 0xFu)
        + a2 * fp4_signed((packed >> 8) & 0xFu)
        + a3 * fp4_signed((packed >> 12) & 0xFu)
        + a4 * fp4_signed((packed >> 16) & 0xFu)
        + a5 * fp4_signed((packed >> 20) & 0xFu)
        + a6 * fp4_signed((packed >> 24) & 0xFu)
        + a7 * fp4_signed(packed >> 28);
}

kernel void gemv_fp4_vec(
    device float* activation [[buffer(0)]],
    device uchar* weights [[buffer(1)]],
    device uchar* scales [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& inner [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    uint row0 = tgid * 4u;
    uint bytes = inner >> 1;
    uint scale_stride = inner >> 5;
    device uchar* w0 = weights + row0 * bytes;
    device uchar* w1 = w0 + bytes;
    device uchar* w2 = w1 + bytes;
    device uchar* w3 = w2 + bytes;
    device uchar* c0 = scales + row0 * scale_stride;
    device uchar* c1 = c0 + scale_stride;
    device uchar* c2 = c1 + scale_stride;
    device uchar* c3 = c2 + scale_stride;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    for (uint base = 0u; base < inner; base += 256u) {
        uint k = base + lane * 8u;
        uint scale_index = k >> 5;
        float s0 = exp2(float(c0[scale_index]) - 127.0f);
        float s1 = exp2(float(c1[scale_index]) - 127.0f);
        float s2 = exp2(float(c2[scale_index]) - 127.0f);
        float s3 = exp2(float(c3[scale_index]) - 127.0f);
        float a0 = activation[k];
        float a1 = activation[k + 1u];
        float a2 = activation[k + 2u];
        float a3 = activation[k + 3u];
        float a4 = activation[k + 4u];
        float a5 = activation[k + 5u];
        float a6 = activation[k + 6u];
        float a7 = activation[k + 7u];
        uint byte_index = k >> 1;
        uint p0 = *reinterpret_cast<device uint*>(w0 + byte_index);
        uint p1 = *reinterpret_cast<device uint*>(w1 + byte_index);
        uint p2 = *reinterpret_cast<device uint*>(w2 + byte_index);
        uint p3 = *reinterpret_cast<device uint*>(w3 + byte_index);
        acc0 += s0 * dot8(a0, a1, a2, a3, a4, a5, a6, a7, p0);
        acc1 += s1 * dot8(a0, a1, a2, a3, a4, a5, a6, a7, p1);
        acc2 += s2 * dot8(a0, a1, a2, a3, a4, a5, a6, a7, p2);
        acc3 += s3 * dot8(a0, a1, a2, a3, a4, a5, a6, a7, p3);
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

kernel void gemv_fp4_vec_rms(
    device float* activation [[buffer(0)]],
    device half* norm_w [[buffer(1)]],
    device uchar* weights [[buffer(2)]],
    device uchar* scales [[buffer(3)]],
    device float* destination [[buffer(4)]],
    device float* normed [[buffer(5)]],
    constant uint& inner [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    float sumsq = 0.0f;
    for (uint k = lane; k < inner; k += 32u) {
        float value = activation[k];
        sumsq += value * value;
    }
    sumsq = simd_sum(sumsq);
    float inv = rsqrt(sumsq / float(inner) + eps);
    if (tgid == 0u) {
        for (uint k = lane; k < inner; k += 32u) {
            normed[k] = activation[k] * float(norm_w[k]) * inv;
        }
    }
    uint row0 = tgid * 4u;
    uint bytes = inner >> 1;
    uint scale_stride = inner >> 5;
    device uchar* w0 = weights + row0 * bytes;
    device uchar* w1 = w0 + bytes;
    device uchar* w2 = w1 + bytes;
    device uchar* w3 = w2 + bytes;
    device uchar* c0 = scales + row0 * scale_stride;
    device uchar* c1 = c0 + scale_stride;
    device uchar* c2 = c1 + scale_stride;
    device uchar* c3 = c2 + scale_stride;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    for (uint base = 0u; base < inner; base += 256u) {
        uint k = base + lane * 8u;
        uint scale_index = k >> 5;
        float s0 = exp2(float(c0[scale_index]) - 127.0f);
        float s1 = exp2(float(c1[scale_index]) - 127.0f);
        float s2 = exp2(float(c2[scale_index]) - 127.0f);
        float s3 = exp2(float(c3[scale_index]) - 127.0f);
        float a0 = activation[k] * float(norm_w[k]) * inv;
        float a1 = activation[k + 1u] * float(norm_w[k + 1u]) * inv;
        float a2 = activation[k + 2u] * float(norm_w[k + 2u]) * inv;
        float a3 = activation[k + 3u] * float(norm_w[k + 3u]) * inv;
        float a4 = activation[k + 4u] * float(norm_w[k + 4u]) * inv;
        float a5 = activation[k + 5u] * float(norm_w[k + 5u]) * inv;
        float a6 = activation[k + 6u] * float(norm_w[k + 6u]) * inv;
        float a7 = activation[k + 7u] * float(norm_w[k + 7u]) * inv;
        uint byte_index = k >> 1;
        uint p0 = *reinterpret_cast<device uint*>(w0 + byte_index);
        uint p1 = *reinterpret_cast<device uint*>(w1 + byte_index);
        uint p2 = *reinterpret_cast<device uint*>(w2 + byte_index);
        uint p3 = *reinterpret_cast<device uint*>(w3 + byte_index);
        acc0 += s0 * dot8(a0, a1, a2, a3, a4, a5, a6, a7, p0);
        acc1 += s1 * dot8(a0, a1, a2, a3, a4, a5, a6, a7, p1);
        acc2 += s2 * dot8(a0, a1, a2, a3, a4, a5, a6, a7, p2);
        acc3 += s3 * dot8(a0, a1, a2, a3, a4, a5, a6, a7, p3);
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

kernel void gemv_fp4_vec_add(
    device float* activation [[buffer(0)]],
    device uchar* weights [[buffer(1)]],
    device uchar* scales [[buffer(2)]],
    device float* hidden [[buffer(3)]],
    constant uint& inner [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    uint row0 = tgid * 4u;
    uint bytes = inner >> 1;
    uint scale_stride = inner >> 5;
    device uchar* w0 = weights + row0 * bytes;
    device uchar* w1 = w0 + bytes;
    device uchar* w2 = w1 + bytes;
    device uchar* w3 = w2 + bytes;
    device uchar* c0 = scales + row0 * scale_stride;
    device uchar* c1 = c0 + scale_stride;
    device uchar* c2 = c1 + scale_stride;
    device uchar* c3 = c2 + scale_stride;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    for (uint base = 0u; base < inner; base += 256u) {
        uint k = base + lane * 8u;
        uint scale_index = k >> 5;
        float s0 = exp2(float(c0[scale_index]) - 127.0f);
        float s1 = exp2(float(c1[scale_index]) - 127.0f);
        float s2 = exp2(float(c2[scale_index]) - 127.0f);
        float s3 = exp2(float(c3[scale_index]) - 127.0f);
        float a0 = activation[k];
        float a1 = activation[k + 1u];
        float a2 = activation[k + 2u];
        float a3 = activation[k + 3u];
        float a4 = activation[k + 4u];
        float a5 = activation[k + 5u];
        float a6 = activation[k + 6u];
        float a7 = activation[k + 7u];
        uint byte_index = k >> 1;
        uint p0 = *reinterpret_cast<device uint*>(w0 + byte_index);
        uint p1 = *reinterpret_cast<device uint*>(w1 + byte_index);
        uint p2 = *reinterpret_cast<device uint*>(w2 + byte_index);
        uint p3 = *reinterpret_cast<device uint*>(w3 + byte_index);
        acc0 += s0 * dot8(a0, a1, a2, a3, a4, a5, a6, a7, p0);
        acc1 += s1 * dot8(a0, a1, a2, a3, a4, a5, a6, a7, p1);
        acc2 += s2 * dot8(a0, a1, a2, a3, a4, a5, a6, a7, p2);
        acc3 += s3 * dot8(a0, a1, a2, a3, a4, a5, a6, a7, p3);
    }
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);
    if (lane == 0u) {
        hidden[row0] += acc0;
        hidden[row0 + 1u] += acc1;
        hidden[row0 + 2u] += acc2;
        hidden[row0 + 3u] += acc3;
    }
}

kernel void gemv_group(
    device float* activation [[buffer(0)]],
    device uchar* weights [[buffer(1)]],
    device uchar* scales [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& inner [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float act[2048];
    threadgroup float part[8];
    gemv_group_body<false, false>(
        act, part, activation, nullptr, weights, scales, destination, nullptr, inner, 0.0f, tgid, tid);
}

kernel void gemv_group_rms(
    device float* activation [[buffer(0)]],
    device half* norm_w [[buffer(1)]],
    device uchar* weights [[buffer(2)]],
    device uchar* scales [[buffer(3)]],
    device float* destination [[buffer(4)]],
    device float* normed [[buffer(5)]],
    constant uint& inner [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float act[2048];
    threadgroup float part[8];
    gemv_group_body<true, false>(
        act, part, activation, norm_w, weights, scales, destination, normed, inner, eps, tgid, tid);
}

kernel void gemv_group_add(
    device float* activation [[buffer(0)]],
    device uchar* weights [[buffer(1)]],
    device uchar* scales [[buffer(2)]],
    device float* hidden [[buffer(3)]],
    constant uint& inner [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float act[2048];
    threadgroup float part[8];
    gemv_group_body<false, true>(
        act, part, activation, nullptr, weights, scales, hidden, nullptr, inner, 0.0f, tgid, tid);
}

kernel void gemv_group_add_wide(
    device float* activation [[buffer(0)]],
    device uchar* weights [[buffer(1)]],
    device uchar* scales [[buffer(2)]],
    device float* hidden [[buffer(3)]],
    constant uint& inner [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float act[6656];
    threadgroup float part[8];
    gemv_group_body<false, true>(
        act, part, activation, nullptr, weights, scales, hidden, nullptr, inner, 0.0f, tgid, tid);
}

kernel void gemv_f16(
    device float* activation [[buffer(0)]],
    device half* weights [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& inner [[buffer(3)]],
    constant uint& dest_stride [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    uint row = tgid.x;
    uint token = tgid.y;
    float acc = 0.0f;
    for (uint k = lane; k < inner; k += 32u) {
        acc += activation[token * inner + k] * float(weights[row * inner + k]);
    }
    acc = simd_sum(acc);
    if (lane == 0u) {
        destination[token * dest_stride + row] = acc;
    }
}

kernel void embed_row(
    device uchar* weights [[buffer(0)]],
    device uchar* scales [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& inner [[buffer(3)]],
    uint lane [[thread_index_in_threadgroup]])
{
    for (uint k = lane; k < inner; k += 32u) {
        destination[k] = dequant_fp4(weights, scales, 0u, k, inner);
    }
}

kernel void rmsnorm(
    device float* source [[buffer(0)]],
    device half* weight [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    device float* in = source + row * n;
    device float* out = destination + row * n;
    float sum = 0.0f;
    for (uint i = lane; i < n; i += 32u) {
        float value = in[i];
        sum += value * value;
    }
    sum = simd_sum(sum);
    float inv = rsqrt(sum / float(n) + eps);
    for (uint i = lane; i < n; i += 32u) {
        out[i] = float(weight[i]) * in[i] * inv;
    }
}

kernel void residual(
    device float* hidden [[buffer(0)]],
    device float* update [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    device float* base = hidden + row * n;
    device float* add = update + row * n;
    for (uint i = lane; i < n; i += 32u) {
        base[i] += add[i];
    }
}

kernel void cast_f16(
    device float* source [[buffer(0)]],
    device half* destination [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    if (id < n) {
        destination[id] = half(source[id]);
    }
}

kernel void geglu(
    device float* gate [[buffer(0)]],
    device float* up [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= n) {
        return;
    }
    float g = gate[id];
    destination[id] = 0.5f * g * (1.0f + gelu_erf(g * INV_SQRT2)) * up[id];
}

struct RopeCfg {
    uint pos;
    uint width;
    float theta;
};

kernel void rope_qk(
    device float* rows [[buffer(0)]],
    constant RopeCfg& cfg [[buffer(1)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint pair [[thread_index_in_threadgroup]])
{
    uint vec = tgid.x;
    device float* row = rows + tgid.y * 3072u;
    uint span = cfg.width >> 1;
    if (pair >= span) {
        return;
    }
    device float* base = vec < 8u ? row + vec * 256u : row + 2048u + (vec - 8u) * 256u;
    float inv = pow(cfg.theta, -float(pair) / float(span));
    float angle = float(cfg.pos + tgid.y) * inv;
    float c = cos(angle);
    float s = sin(angle);
    float x1 = base[pair];
    float x2 = base[span + pair];
    base[pair] = x1 * c - x2 * s;
    base[span + pair] = x2 * c + x1 * s;
}

struct StoreCfg {
    uint pos;
    uint capacity;
};

kernel void store_full(
    device float* rows [[buffer(0)]],
    device uchar* key_data [[buffer(1)]],
    device uchar* key_scale [[buffer(2)]],
    device uchar* value_data [[buffer(3)]],
    device uchar* value_scale [[buffer(4)]],
    device float* scratch [[buffer(5)]],
    constant StoreCfg& cfg [[buffer(6)]],
    uint token [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    device float* row = rows + token * 3072u;
    uint pos = cfg.pos + token;
    uint slot = pos & 31u;
    uint base = pos - slot;
    device float* key = row + 2048u;
    device float* value = row + 2560u;
    for (uint head = 0u; head < 2u; ++head) {
        for (uint group = lane; group < 8u; group += 32u) {
            float amax = 0.0f;
            for (uint j = 0u; j < 32u; ++j) {
                amax = max(amax, fabs(key[head * 256u + group * 32u + j]));
            }
            uint sc = e8m0_byte(amax, 448.0f);
            key_scale[(head * cfg.capacity + pos) * 8u + group] = uchar(sc);
            float scale = exp2(float(sc) - 127.0f);
            for (uint j = 0u; j < 32u; ++j) {
                float q = scale > 0.0f ? key[head * 256u + group * 32u + j] / scale : 0.0f;
                key_data[(head * cfg.capacity + pos) * 256u + group * 32u + j] = uchar(enc_e4m3(q));
            }
        }
        for (uint channel = lane; channel < 256u; channel += 32u) {
            scratch[(head * 256u + channel) * 32u + slot] = value[head * 256u + channel];
            float amax = 0.0f;
            for (uint s = 0u; s <= slot; ++s) {
                amax = max(amax, fabs(scratch[(head * 256u + channel) * 32u + s]));
            }
            uint sc = e8m0_byte(amax, 448.0f);
            uint blocks = cfg.capacity >> 5;
            value_scale[(head * 256u + channel) * blocks + (base >> 5)] = uchar(sc);
            float scale = exp2(float(sc) - 127.0f);
            for (uint s = 0u; s <= slot; ++s) {
                float raw = scratch[(head * 256u + channel) * 32u + s];
                float q = scale > 0.0f ? raw / scale : 0.0f;
                value_data[(head * 256u + channel) * cfg.capacity + base + s] = uchar(enc_e4m3(q));
            }
        }
    }
}

kernel void store_slide(
    device float* row [[buffer(0)]],
    device uchar* key_data [[buffer(1)]],
    device uchar* key_scale [[buffer(2)]],
    device uchar* value_data [[buffer(3)]],
    device uchar* value_scale [[buffer(4)]],
    device float* scratch [[buffer(5)]],
    constant StoreCfg& cfg [[buffer(6)]],
    uint lane [[thread_index_in_threadgroup]])
{
    uint slot = cfg.pos % 512u;
    uint block = slot >> 5;
    device float* key = row + 2048u;
    device float* value = row + 2560u;
    for (uint head = 0u; head < 2u; ++head) {
        for (uint group = lane; group < 8u; group += 32u) {
            float amax = 0.0f;
            for (uint j = 0u; j < 32u; ++j) {
                amax = max(amax, fabs(key[head * 256u + group * 32u + j]));
            }
            uint sc = e8m0_byte(amax, 448.0f);
            key_scale[(head * 512u + slot) * 8u + group] = uchar(sc);
            float scale = exp2(float(sc) - 127.0f);
            for (uint j = 0u; j < 32u; ++j) {
                float q = scale > 0.0f ? key[head * 256u + group * 32u + j] / scale : 0.0f;
                key_data[(head * 512u + slot) * 256u + group * 32u + j] = uchar(enc_e4m3(q));
            }
        }
        for (uint channel = lane; channel < 256u; channel += 32u) {
            scratch[(head * 256u + channel) * 512u + slot] = value[head * 256u + channel];
            float amax = 0.0f;
            for (uint s = 0u; s < 32u; ++s) {
                uint ring = block * 32u + s;
                bool occupied = cfg.pos >= 511u || ring <= cfg.pos;
                if (!occupied) {
                    continue;
                }
                amax = max(amax, fabs(scratch[(head * 256u + channel) * 512u + ring]));
            }
            uint sc = e8m0_byte(amax, 448.0f);
            value_scale[(head * 256u + channel) * 16u + block] = uchar(sc);
            float scale = exp2(float(sc) - 127.0f);
            for (uint s = 0u; s < 32u; ++s) {
                uint ring = block * 32u + s;
                bool occupied = cfg.pos >= 511u || ring <= cfg.pos;
                if (!occupied) {
                    continue;
                }
                float raw = scratch[(head * 256u + channel) * 512u + ring];
                float q = scale > 0.0f ? raw / scale : 0.0f;
                value_data[(head * 256u + channel) * 512u + ring] = uchar(enc_e4m3(q));
            }
        }
    }
}

struct RebuildCfg {
    uint length;
    uint capacity;
};

kernel void rebuild_full(
    device uchar* value_data [[buffer(0)]],
    device uchar* value_scale [[buffer(1)]],
    device float* scratch [[buffer(2)]],
    constant RebuildCfg& cfg [[buffer(3)]],
    uint lane [[thread_index_in_threadgroup]])
{
    uint filled = cfg.length & 31u;
    if (filled == 0u) {
        return;
    }
    uint base = cfg.length - filled;
    uint blocks = cfg.capacity >> 5;
    for (uint head = 0u; head < 2u; ++head) {
        for (uint channel = lane; channel < 256u; channel += 32u) {
            uint sc = value_scale[(head * 256u + channel) * blocks + (base >> 5)];
            float scale = exp2(float(sc) - 127.0f);
            for (uint s = 0u; s < filled; ++s) {
                uint code = value_data[(head * 256u + channel) * cfg.capacity + base + s];
                scratch[(head * 256u + channel) * 32u + s] = dec_e4m3(code) * scale;
            }
        }
    }
}

kernel void rebuild_slide(
    device uchar* value_data [[buffer(0)]],
    device uchar* value_scale [[buffer(1)]],
    device float* scratch [[buffer(2)]],
    constant RebuildCfg& cfg [[buffer(3)]],
    uint lane [[thread_index_in_threadgroup]])
{
    for (uint head = 0u; head < 2u; ++head) {
        for (uint channel = lane; channel < 256u; channel += 32u) {
            for (uint ring = 0u; ring < 512u; ++ring) {
                bool occupied = cfg.length >= 512u || ring < cfg.length;
                if (!occupied) {
                    scratch[(head * 256u + channel) * 512u + ring] = 0.0f;
                    continue;
                }
                uint sc = value_scale[(head * 256u + channel) * 16u + (ring >> 5)];
                float scale = exp2(float(sc) - 127.0f);
                uint code = value_data[(head * 256u + channel) * 512u + ring];
                scratch[(head * 256u + channel) * 512u + ring] = dec_e4m3(code) * scale;
            }
        }
    }
}

struct AttendCfg {
    uint pos;
    uint capacity;
    uint window;
    uint score_stride;
};

kernel void score_keys(
    device float* queries [[buffer(0)]],
    device uchar* key_data [[buffer(1)]],
    device uchar* key_scale [[buffer(2)]],
    device float* scores [[buffer(3)]],
    constant AttendCfg& cfg [[buffer(4)]],
    uint qh [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    uint start = 0u;
    uint count = cfg.pos + 1u;
    if (cfg.window != 0u && count > cfg.window) {
        start = count - cfg.window;
        count = cfg.window;
    }
    uint kv = qh >> 2;
    for (uint i = 0u; i < count; ++i) {
        uint slot = cfg.window == 0u ? (start + i) : ((start + i) % cfg.window);
        float partial = 0.0f;
        for (uint c = lane; c < 256u; c += 32u) {
            partial += queries[qh * 256u + c] * key_at(key_data, key_scale, kv, slot, c, cfg.capacity);
        }
        float score = simd_sum(partial) * ATTN_SCALE;
        if (lane == 0u) {
            scores[qh * cfg.score_stride + i] = score;
        }
    }
}

kernel void mix_values(
    device float* scores [[buffer(0)]],
    device uchar* value_data [[buffer(1)]],
    device uchar* value_scale [[buffer(2)]],
    device float* mixed [[buffer(3)]],
    constant AttendCfg& cfg [[buffer(4)]],
    uint qh [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    uint start = 0u;
    uint count = cfg.pos + 1u;
    if (cfg.window != 0u && count > cfg.window) {
        start = count - cfg.window;
        count = cfg.window;
    }
    uint kv = qh >> 2;
    float peak = -1.0e30f;
    for (uint i = lane; i < count; i += 32u) {
        peak = max(peak, scores[qh * cfg.score_stride + i]);
    }
    peak = simd_max(peak);
    float sum = 0.0f;
    for (uint i = lane; i < count; i += 32u) {
        sum += exp(scores[qh * cfg.score_stride + i] - peak);
    }
    sum = simd_sum(sum);
    if (sum == 0.0f) {
        sum = 1.0f;
    }
    for (uint j = 0u; j < 8u; ++j) {
        uint channel = lane + j * 32u;
        float acc = 0.0f;
        for (uint i = 0u; i < count; ++i) {
            uint slot = cfg.window == 0u ? (start + i) : ((start + i) % cfg.window);
            float weight = exp(scores[qh * cfg.score_stride + i] - peak) / sum;
            acc += weight * value_at(value_data, value_scale, kv, slot, channel, cfg.capacity);
        }
        mixed[qh * 256u + channel] = acc;
    }
}

struct SpanStore {
    uint pos;
    uint capacity;
    uint count;
};

kernel void store_full_keys(
    device float* rows [[buffer(0)]],
    device uchar* key_data [[buffer(1)]],
    device uchar* key_scale [[buffer(2)]],
    constant SpanStore& cfg [[buffer(3)]],
    uint token [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    if (token >= cfg.count) {
        return;
    }
    uint pos = cfg.pos + token;
    device float* key = rows + token * 3072u + 2048u;
    for (uint head = 0u; head < 2u; ++head) {
        for (uint group = 0u; group < 8u; ++group) {
            float amax = simd_max(fabs(key[head * 256u + group * 32u + lane]));
            uint sc = e8m0_byte(amax, 448.0f);
            float scale = exp2(float(sc) - 127.0f);
            float raw = key[head * 256u + group * 32u + lane];
            float q = scale > 0.0f ? raw / scale : 0.0f;
            key_data[(head * cfg.capacity + pos) * 256u + group * 32u + lane] = uchar(enc_e4m3(q));
            if (lane == 0u) {
                key_scale[(head * cfg.capacity + pos) * 8u + group] = uchar(sc);
            }
        }
    }
}

kernel void store_slide_keys(
    device float* rows [[buffer(0)]],
    device uchar* key_data [[buffer(1)]],
    device uchar* key_scale [[buffer(2)]],
    constant SpanStore& cfg [[buffer(3)]],
    uint token [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    if (token >= cfg.count) {
        return;
    }
    uint pos = cfg.pos + token;
    uint slot = pos % 512u;
    device float* key = rows + token * 3072u + 2048u;
    for (uint head = 0u; head < 2u; ++head) {
        for (uint group = 0u; group < 8u; ++group) {
            float amax = simd_max(fabs(key[head * 256u + group * 32u + lane]));
            uint sc = e8m0_byte(amax, 448.0f);
            float scale = exp2(float(sc) - 127.0f);
            float raw = key[head * 256u + group * 32u + lane];
            float q = scale > 0.0f ? raw / scale : 0.0f;
            key_data[(head * 512u + slot) * 256u + group * 32u + lane] = uchar(enc_e4m3(q));
            if (lane == 0u) {
                key_scale[(head * 512u + slot) * 8u + group] = uchar(sc);
            }
        }
    }
}

kernel void store_full_span(
    device float* rows [[buffer(0)]],
    device uchar* key_data [[buffer(1)]],
    device uchar* key_scale [[buffer(2)]],
    device uchar* value_data [[buffer(3)]],
    device uchar* value_scale [[buffer(4)]],
    device float* scratch [[buffer(5)]],
    constant SpanStore& cfg [[buffer(6)]],
    uint lane [[thread_index_in_threadgroup]])
{
    if (cfg.count == 1u) {
        device float* key = rows + 2048u;
        for (uint head = 0u; head < 2u; ++head) {
            for (uint group = 0u; group < 8u; ++group) {
                float amax = simd_max(fabs(key[head * 256u + group * 32u + lane]));
                uint sc = e8m0_byte(amax, 448.0f);
                float scale = exp2(float(sc) - 127.0f);
                float raw = key[head * 256u + group * 32u + lane];
                float q = scale > 0.0f ? raw / scale : 0.0f;
                key_data[(head * cfg.capacity + cfg.pos) * 256u + group * 32u + lane] = uchar(enc_e4m3(q));
                if (lane == 0u) {
                    key_scale[(head * cfg.capacity + cfg.pos) * 8u + group] = uchar(sc);
                }
            }
        }
    }
    uint span_end = cfg.pos + cfg.count;
    uint first = cfg.pos >> 5;
    uint last = (span_end - 1u) >> 5;
    uint blocks = cfg.capacity >> 5;
    for (uint block = first; block <= last; ++block) {
        uint base = block << 5;
        uint block_end = min(base + 32u, span_end);
        for (uint pos = max(cfg.pos, base); pos < block_end; ++pos) {
            device float* row = rows + (pos - cfg.pos) * 3072u;
            uint slot = pos & 31u;
            device float* value = row + 2560u;
            for (uint head = 0u; head < 2u; ++head) {
                for (uint channel = lane; channel < 256u; channel += 32u) {
                    scratch[(head * 256u + channel) * 32u + slot] = value[head * 256u + channel];
                }
            }
        }
        uint slot_limit = block_end - base;
        for (uint head = 0u; head < 2u; ++head) {
            for (uint channel = lane; channel < 256u; channel += 32u) {
                float amax = 0.0f;
                for (uint s = 0u; s < slot_limit; ++s) {
                    amax = max(amax, fabs(scratch[(head * 256u + channel) * 32u + s]));
                }
                uint sc = e8m0_byte(amax, 448.0f);
                value_scale[(head * 256u + channel) * blocks + block] = uchar(sc);
                float scale = exp2(float(sc) - 127.0f);
                for (uint s = 0u; s < slot_limit; ++s) {
                    float raw = scratch[(head * 256u + channel) * 32u + s];
                    float q = scale > 0.0f ? raw / scale : 0.0f;
                    value_data[(head * 256u + channel) * cfg.capacity + base + s] = uchar(enc_e4m3(q));
                }
            }
        }
    }
}

kernel void store_slide_span(
    device float* rows [[buffer(0)]],
    device uchar* key_data [[buffer(1)]],
    device uchar* key_scale [[buffer(2)]],
    device uchar* value_data [[buffer(3)]],
    device uchar* value_scale [[buffer(4)]],
    device float* scratch [[buffer(5)]],
    constant SpanStore& cfg [[buffer(6)]],
    uint lane [[thread_index_in_threadgroup]])
{
    if (cfg.count == 1u) {
        uint slot0 = cfg.pos % 512u;
        device float* key = rows + 2048u;
        for (uint head = 0u; head < 2u; ++head) {
            for (uint group = 0u; group < 8u; ++group) {
                float amax = simd_max(fabs(key[head * 256u + group * 32u + lane]));
                uint sc = e8m0_byte(amax, 448.0f);
                float scale = exp2(float(sc) - 127.0f);
                float raw = key[head * 256u + group * 32u + lane];
                float q = scale > 0.0f ? raw / scale : 0.0f;
                key_data[(head * 512u + slot0) * 256u + group * 32u + lane] = uchar(enc_e4m3(q));
                if (lane == 0u) {
                    key_scale[(head * 512u + slot0) * 8u + group] = uchar(sc);
                }
            }
        }
    }
    uint span_end = cfg.pos + cfg.count;
    uint latest = span_end - 1u;
    for (uint token = 0u; token < cfg.count; ++token) {
        uint pos = cfg.pos + token;
        uint slot = pos % 512u;
        device float* row = rows + token * 3072u;
        device float* value = row + 2560u;
        for (uint head = 0u; head < 2u; ++head) {
            for (uint channel = lane; channel < 256u; channel += 32u) {
                scratch[(head * 256u + channel) * 512u + slot] = value[head * 256u + channel];
            }
        }
    }
    for (uint block = 0u; block < 16u; ++block) {
        bool touched = false;
        for (uint token = 0u; token < cfg.count; ++token) {
            uint slot = (cfg.pos + token) % 512u;
            if ((slot >> 5) == block) {
                touched = true;
            }
        }
        if (!touched) {
            continue;
        }
        for (uint head = 0u; head < 2u; ++head) {
            for (uint channel = lane; channel < 256u; channel += 32u) {
                float amax = 0.0f;
                for (uint s = 0u; s < 32u; ++s) {
                    uint ring = block * 32u + s;
                    bool occupied = latest >= 511u || ring <= latest;
                    if (!occupied) {
                        continue;
                    }
                    amax = max(amax, fabs(scratch[(head * 256u + channel) * 512u + ring]));
                }
                uint sc = e8m0_byte(amax, 448.0f);
                value_scale[(head * 256u + channel) * 16u + block] = uchar(sc);
                float scale = exp2(float(sc) - 127.0f);
                for (uint s = 0u; s < 32u; ++s) {
                    uint ring = block * 32u + s;
                    bool occupied = latest >= 511u || ring <= latest;
                    if (!occupied) {
                        continue;
                    }
                    float raw = scratch[(head * 256u + channel) * 512u + ring];
                    float q = scale > 0.0f ? raw / scale : 0.0f;
                    value_data[(head * 256u + channel) * 512u + ring] = uchar(enc_e4m3(q));
                }
            }
        }
    }
}

struct AttendSpan {
    uint pos;
    uint capacity;
    uint window;
    uint score_stride;
    uint count;
};

static void attend_range(
    device float* query,
    device uchar* key_data,
    device uchar* key_scale,
    device uchar* value_data,
    device uchar* value_scale,
    uint kv,
    uint start,
    uint begin,
    uint end,
    uint window,
    uint capacity,
    uint lane,
    thread float& peak,
    thread float& sum,
    thread float acc[8])
{
    for (uint i = begin; i < end; ++i) {
        uint slot = window == 0u ? (start + i) : ((start + i) % window);
        float partial = 0.0f;
        for (uint group = 0u; group < 8u; ++group) {
            uint channel = group * 32u + lane;
            uint sc = key_scale[(kv * capacity + slot) * 8u + group];
            float scale = exp2(float(sc) - 127.0f);
            uint code = key_data[(kv * capacity + slot) * 256u + channel];
            partial += query[channel] * dec_e4m3(code) * scale;
        }
        float score = simd_sum(partial) * ATTN_SCALE;
        float next = max(peak, score);
        float decay = exp(peak - next);
        float weight = exp(score - next);
        sum = sum * decay + weight;
        for (uint j = 0u; j < 8u; ++j) {
            uint channel = lane + j * 32u;
            acc[j] = acc[j] * decay + weight * value_at(value_data, value_scale, kv, slot, channel, capacity);
        }
        peak = next;
    }
}

kernel void attend_span(
    device float* queries [[buffer(0)]],
    device uchar* key_data [[buffer(1)]],
    device uchar* key_scale [[buffer(2)]],
    device uchar* value_data [[buffer(3)]],
    device uchar* value_scale [[buffer(4)]],
    device float* gates [[buffer(5)]],
    device float* mixed [[buffer(6)]],
    constant AttendSpan& cfg [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    uint token = tgid.y;
    if (token >= cfg.count) {
        return;
    }
    uint qh = tgid.x;
    uint kv = qh >> 2;
    uint pos = cfg.pos + token;
    uint start = 0u;
    uint count = pos + 1u;
    if (cfg.window != 0u && count > cfg.window) {
        start = count - cfg.window;
        count = cfg.window;
    }
    float acc[8];
    for (uint j = 0u; j < 8u; ++j) {
        acc[j] = 0.0f;
    }
    float peak = -1.0e30f;
    float sum = 0.0f;
    attend_range(
        queries + token * 3072u + qh * 256u,
        key_data, key_scale, value_data, value_scale,
        kv, start, 0u, count, cfg.window, cfg.capacity, lane, peak, sum, acc);
    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    float gate = 1.0f / (1.0f + exp(-gates[token * 8u + qh]));
    for (uint j = 0u; j < 8u; ++j) {
        uint channel = lane + j * 32u;
        mixed[(token * 8u + qh) * 256u + channel] = acc[j] * inv * gate;
    }
}

kernel void attend_tile(
    device float* queries [[buffer(0)]],
    device uchar* key_data [[buffer(1)]],
    device uchar* key_scale [[buffer(2)]],
    device uchar* value_data [[buffer(3)]],
    device uchar* value_scale [[buffer(4)]],
    device float* partial [[buffer(5)]],
    device float* state [[buffer(6)]],
    constant AttendSpan& cfg [[buffer(7)]],
    constant uint& tile_stride [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    uint qh = tgid.x;
    uint tile = tgid.y;
    uint kv = qh >> 2;
    uint start = 0u;
    uint count = cfg.pos + 1u;
    if (cfg.window != 0u && count > cfg.window) {
        start = count - cfg.window;
        count = cfg.window;
    }
    uint begin = tile * 64u;
    device float* row = partial + (qh * tile_stride + tile) * 256u;
    if (begin >= count) {
        for (uint j = 0u; j < 8u; ++j) {
            row[lane + j * 32u] = 0.0f;
        }
        if (lane == 0u) {
            state[(qh * tile_stride + tile) * 2u] = -1.0e30f;
            state[(qh * tile_stride + tile) * 2u + 1u] = 0.0f;
        }
        return;
    }
    float acc[8];
    for (uint j = 0u; j < 8u; ++j) {
        acc[j] = 0.0f;
    }
    float peak = -1.0e30f;
    float sum = 0.0f;
    uint end = min(begin + 64u, count);
    attend_range(
        queries + qh * 256u,
        key_data, key_scale, value_data, value_scale,
        kv, start, begin, end, cfg.window, cfg.capacity, lane, peak, sum, acc);
    for (uint j = 0u; j < 8u; ++j) {
        row[lane + j * 32u] = acc[j];
    }
    if (lane == 0u) {
        state[(qh * tile_stride + tile) * 2u] = peak;
        state[(qh * tile_stride + tile) * 2u + 1u] = sum;
    }
}

kernel void attend_merge(
    device float* partial [[buffer(0)]],
    device float* state [[buffer(1)]],
    device float* gates [[buffer(2)]],
    device float* mixed [[buffer(3)]],
    constant uint& tile_stride [[buffer(4)]],
    constant uint& used [[buffer(5)]],
    uint qh [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    float peak = -1.0e30f;
    for (uint tile = 0u; tile < used; ++tile) {
        peak = max(peak, state[(qh * tile_stride + tile) * 2u]);
    }
    float acc[8];
    for (uint j = 0u; j < 8u; ++j) {
        acc[j] = 0.0f;
    }
    float sum = 0.0f;
    for (uint tile = 0u; tile < used; ++tile) {
        float tile_peak = state[(qh * tile_stride + tile) * 2u];
        float tile_sum = state[(qh * tile_stride + tile) * 2u + 1u];
        float scale = exp(tile_peak - peak);
        sum += tile_sum * scale;
        device float* row = partial + (qh * tile_stride + tile) * 256u;
        for (uint j = 0u; j < 8u; ++j) {
            acc[j] += row[lane + j * 32u] * scale;
        }
    }
    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    float gate = 1.0f / (1.0f + exp(-gates[qh]));
    for (uint j = 0u; j < 8u; ++j) {
        mixed[qh * 256u + lane + j * 32u] = acc[j] * inv * gate;
    }
}

kernel void gemv_fp4_rms(
    device float* activation [[buffer(0)]],
    device half* norm_w [[buffer(1)]],
    device uchar* weights [[buffer(2)]],
    device uchar* scales [[buffer(3)]],
    device float* destination [[buffer(4)]],
    device float* normed [[buffer(5)]],
    constant uint& inner [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    float sumsq = 0.0f;
    for (uint k = lane; k < inner; k += 32u) {
        float value = activation[k];
        sumsq += value * value;
    }
    sumsq = simd_sum(sumsq);
    float inv = rsqrt(sumsq / float(inner) + eps);
    if (tgid == 0u) {
        for (uint k = lane; k < inner; k += 32u) {
            normed[k] = activation[k] * float(norm_w[k]) * inv;
        }
    }
    uint row0 = tgid * 4u;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    uint bytes = inner >> 1;
    uint scale_stride = inner >> 5;
    for (uint base = 0u; base < inner; base += 64u) {
        uint k = base + lane * 2u;
        float a0 = activation[k] * float(norm_w[k]) * inv;
        float a1 = activation[k + 1u] * float(norm_w[k + 1u]) * inv;
        uint byte_index = k >> 1;
        uint scale_index = k >> 5;
        uchar p0 = weights[row0 * bytes + byte_index];
        uchar p1 = weights[(row0 + 1u) * bytes + byte_index];
        uchar p2 = weights[(row0 + 2u) * bytes + byte_index];
        uchar p3 = weights[(row0 + 3u) * bytes + byte_index];
        float s0 = exp2(float(scales[row0 * scale_stride + scale_index]) - 127.0f);
        float s1 = exp2(float(scales[(row0 + 1u) * scale_stride + scale_index]) - 127.0f);
        float s2 = exp2(float(scales[(row0 + 2u) * scale_stride + scale_index]) - 127.0f);
        float s3 = exp2(float(scales[(row0 + 3u) * scale_stride + scale_index]) - 127.0f);
        acc0 += (a0 * fp4_signed(p0 & 0xFu) + a1 * fp4_signed(p0 >> 4)) * s0;
        acc1 += (a0 * fp4_signed(p1 & 0xFu) + a1 * fp4_signed(p1 >> 4)) * s1;
        acc2 += (a0 * fp4_signed(p2 & 0xFu) + a1 * fp4_signed(p2 >> 4)) * s2;
        acc3 += (a0 * fp4_signed(p3 & 0xFu) + a1 * fp4_signed(p3 >> 4)) * s3;
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

kernel void gemv_fp4_add(
    device float* activation [[buffer(0)]],
    device uchar* weights [[buffer(1)]],
    device uchar* scales [[buffer(2)]],
    device float* hidden [[buffer(3)]],
    constant uint& inner [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    uint row0 = tgid * 4u;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    dot_rows(activation, weights, scales, row0, inner, lane, acc0, acc1, acc2, acc3);
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);
    if (lane == 0u) {
        hidden[row0] += acc0;
        hidden[row0 + 1u] += acc1;
        hidden[row0 + 2u] += acc2;
        hidden[row0 + 3u] += acc3;
    }
}

kernel void gate_rows(
    device float* mixed [[buffer(0)]],
    device float* gates [[buffer(1)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    uint row = tgid.x;
    uint head = tgid.y;
    float gate = 1.0f / (1.0f + exp(-gates[row * 8u + head]));
    device float* base = mixed + (row * 8u + head) * 256u;
    for (uint c = lane; c < 256u; c += 32u) {
        base[c] *= gate;
    }
}

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

kernel void matmul_fp4_32(
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
        weights, dextents<int, 2>(int(inner), int(cols)), array<int, 2>{1, int(inner)}, scales_plane(scales));
    tensor<device float, dextents<int, 2>, tensor_inline> out(
        destination, dextents<int, 2>(int(cols), int(rows)), array<int, 2>{1, int(cols)});
    constexpr auto desc = matmul2d_descriptor(32, 128, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto tile_left = left.slice(0, int(tgid.y) * 32);
    auto tile_right = right.slice(0, int(tgid.x) * 128);
    auto tile_out = out.slice(int(tgid.x) * 128, int(tgid.y) * 32);
    op.run(tile_left, tile_right, tile_out);
}

kernel void zero_f32(
    device float* values [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    if (id < n) {
        values[id] = 0.0f;
    }
}

kernel void zero_u32(
    device uint* values [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    if (id < n) {
        values[id] = 0u;
    }
}

kernel void partial_max(
    device float* values [[buffer(0)]],
    device float* partial [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    uint slice = (n + REDUCE_GROUPS - 1u) / REDUCE_GROUPS;
    uint begin = tgid * slice;
    uint end = min(begin + slice, n);
    float best = -1.0e30f;
    for (uint i = begin + lane; i < end; i += 256u) {
        best = max(best, values[i]);
    }
    threadgroup float shared[256];
    shared[lane] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 128u; offset > 0u; offset >>= 1) {
        if (lane < offset) {
            shared[lane] = max(shared[lane], shared[lane + offset]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) {
        partial[tgid] = shared[0];
    }
}

kernel void final_max(
    device float* partial [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint lane [[thread_index_in_threadgroup]])
{
    threadgroup float shared[256];
    shared[lane] = partial[lane];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 128u; offset > 0u; offset >>= 1) {
        if (lane < offset) {
            shared[lane] = max(shared[lane], shared[lane + offset]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) {
        out[0] = shared[0];
    }
}

kernel void partial_exp(
    device float* logits [[buffer(0)]],
    device float* partial [[buffer(1)]],
    device float* peak [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant float& temperature [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    uint slice = (n + REDUCE_GROUPS - 1u) / REDUCE_GROUPS;
    uint begin = tgid * slice;
    uint end = min(begin + slice, n);
    float sum = 0.0f;
    float inv = 1.0f / temperature;
    for (uint i = begin + lane; i < end; i += 256u) {
        sum += exp((logits[i] - peak[0]) * inv);
    }
    threadgroup float shared[256];
    shared[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 128u; offset > 0u; offset >>= 1) {
        if (lane < offset) {
            shared[lane] += shared[lane + offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) {
        partial[tgid] = shared[0];
    }
}

kernel void final_sum(
    device float* partial [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint lane [[thread_index_in_threadgroup]])
{
    threadgroup float shared[256];
    shared[lane] = partial[lane];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 128u; offset > 0u; offset >>= 1) {
        if (lane < offset) {
            shared[lane] += shared[lane + offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) {
        out[0] = shared[0];
    }
}

kernel void write_probs(
    device float* logits [[buffer(0)]],
    device float* probs [[buffer(1)]],
    device float* peak [[buffer(2)]],
    device float* sum [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant float& temperature [[buffer(5)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= n) {
        return;
    }
    float mass = sum[0] == 0.0f ? 1.0f : sum[0];
    probs[id] = exp((logits[id] - peak[0]) / temperature) / mass;
}

kernel void penalize(
    device float* logits [[buffer(0)]],
    device uint* counts [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant float& presence [[buffer(3)]],
    constant float& frequency [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= n) {
        return;
    }
    float seen = float(counts[id]);
    if (seen > 0.0f) {
        logits[id] -= presence;
        logits[id] -= frequency * seen;
    }
}

kernel void bump_count(
    device uint* counts [[buffer(0)]],
    device uint* token [[buffer(1)]])
{
    counts[token[0]] += 1u;
}

kernel void argmax_partial(
    device float* values [[buffer(0)]],
    device float* peaks [[buffer(1)]],
    device uint* indexes [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    uint slice = (n + REDUCE_GROUPS - 1u) / REDUCE_GROUPS;
    uint begin = tgid * slice;
    uint end = min(begin + slice, n);
    float best = -1.0e30f;
    uint index = 0u;
    for (uint i = begin + lane; i < end; i += 256u) {
        float value = values[i];
        if (value > best || (value == best && i < index)) {
            best = value;
            index = i;
        }
    }
    threadgroup float shared_v[256];
    threadgroup uint shared_i[256];
    shared_v[lane] = best;
    shared_i[lane] = index;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 128u; offset > 0u; offset >>= 1) {
        if (lane < offset) {
            float other = shared_v[lane + offset];
            uint other_i = shared_i[lane + offset];
            if (other > shared_v[lane] || (other == shared_v[lane] && other_i < shared_i[lane])) {
                shared_v[lane] = other;
                shared_i[lane] = other_i;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) {
        peaks[tgid] = shared_v[0];
        indexes[tgid] = shared_i[0];
    }
}

kernel void argmax_final(
    device float* peaks [[buffer(0)]],
    device uint* indexes [[buffer(1)]],
    device uint* out [[buffer(2)]],
    uint lane [[thread_index_in_threadgroup]])
{
    threadgroup float shared_v[256];
    threadgroup uint shared_i[256];
    shared_v[lane] = peaks[lane];
    shared_i[lane] = indexes[lane];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 128u; offset > 0u; offset >>= 1) {
        if (lane < offset) {
            float other = shared_v[lane + offset];
            uint other_i = shared_i[lane + offset];
            if (other > shared_v[lane] || (other == shared_v[lane] && other_i < shared_i[lane])) {
                shared_v[lane] = other;
                shared_i[lane] = other_i;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) {
        out[0] = shared_i[0];
    }
}

kernel void fill_pairs(
    device float* probs [[buffer(0)]],
    device Pair* pairs [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    if (id < n) {
        pairs[id].p = probs[id];
        pairs[id].id = id;
    }
}

kernel void bitonic(
    device Pair* pairs [[buffer(0)]],
    constant uint& stage [[buffer(1)]],
    constant uint& stride [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    uint partner = i ^ stride;
    if (partner <= i) {
        return;
    }
    Pair left = pairs[i];
    Pair right = pairs[partner];
    bool left_first = (left.p > right.p) || (left.p == right.p && left.id < right.id);
    bool descending = (i & stage) == 0u;
    if (descending) {
        if (!left_first) {
            pairs[i] = right;
            pairs[partner] = left;
        }
    } else if (left_first) {
        pairs[i] = right;
        pairs[partner] = left;
    }
}

kernel void export_probs(
    device Pair* pairs [[buffer(0)]],
    device float* values [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    values[id] = pairs[id].p;
}

kernel void block_scan(
    device float* values [[buffer(0)]],
    device float* totals [[buffer(1)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    uint index = tgid * 256u + lane;
    threadgroup float shared[256];
    shared[lane] = values[index];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 1u; offset < 256u; offset <<= 1) {
        float extra = 0.0f;
        if (lane >= offset) {
            extra = shared[lane - offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        shared[lane] += extra;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    values[index] = shared[lane];
    if (lane == 255u) {
        totals[tgid] = shared[255];
    }
}

kernel void scan_blocks(
    device float* totals [[buffer(0)]],
    device float* exclusive [[buffer(1)]],
    uint lane [[thread_index_in_threadgroup]])
{
    threadgroup float shared[512];
    shared[lane] = totals[lane];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 1u; offset < 512u; offset <<= 1) {
        float extra = 0.0f;
        if (lane >= offset) {
            extra = shared[lane - offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane >= offset) {
            shared[lane] += extra;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    exclusive[lane] = lane == 0u ? 0.0f : shared[lane - 1u];
}

kernel void add_block_prefix(
    device float* values [[buffer(0)]],
    device float* exclusive [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    values[id] += exclusive[id >> 8];
}

struct PickCfg {
    uint n;
    float top_p;
    float u;
};

kernel void pick_nucleus(
    device float* cumulative [[buffer(0)]],
    device Pair* pairs [[buffer(1)]],
    device uint* out [[buffer(2)]],
    constant PickCfg& cfg [[buffer(3)]])
{
    float total = cumulative[cfg.n - 1u];
    if (!(total > 0.0f)) {
        out[0] = 0u;
        return;
    }
    float mass = total;
    uint cutoff = cfg.n - 1u;
    if (cfg.top_p < 1.0f) {
        float limit = total * cfg.top_p;
        uint lo = 0u;
        uint hi = cfg.n;
        while (lo < hi) {
            uint mid = (lo + hi) >> 1;
            if (cumulative[mid] >= limit) {
                hi = mid;
            } else {
                lo = mid + 1u;
            }
        }
        cutoff = lo >= cfg.n ? cfg.n - 1u : lo;
        mass = cumulative[cutoff];
    }
    float target = cfg.u * mass;
    uint lo = 0u;
    uint hi = cutoff + 1u;
    while (lo < hi) {
        uint mid = (lo + hi) >> 1;
        if (cumulative[mid] >= target) {
            hi = mid;
        } else {
            lo = mid + 1u;
        }
    }
    if (lo > cutoff) {
        lo = cutoff;
    }
    out[0] = pairs[lo].id;
}

kernel void partial_count(
    device float* values [[buffer(0)]],
    device uint* partial [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant float& threshold [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    uint slice = (n + REDUCE_GROUPS - 1u) / REDUCE_GROUPS;
    uint begin = tgid * slice;
    uint end = min(begin + slice, n);
    uint count = 0u;
    for (uint i = begin + lane; i < end; i += 256u) {
        if (values[i] >= threshold) {
            count += 1u;
        }
    }
    threadgroup uint shared[256];
    shared[lane] = count;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 128u; offset > 0u; offset >>= 1) {
        if (lane < offset) {
            shared[lane] += shared[lane + offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) {
        partial[tgid] = shared[0];
    }
}

kernel void final_count(
    device uint* partial [[buffer(0)]],
    device uint* out [[buffer(1)]],
    uint lane [[thread_index_in_threadgroup]])
{
    threadgroup uint shared[256];
    shared[lane] = partial[lane];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 128u; offset > 0u; offset >>= 1) {
        if (lane < offset) {
            shared[lane] += shared[lane + offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) {
        out[0] = shared[0];
    }
}

kernel void partial_min(
    device float* values [[buffer(0)]],
    device float* partial [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    uint slice = (n + REDUCE_GROUPS - 1u) / REDUCE_GROUPS;
    uint begin = tgid * slice;
    uint end = min(begin + slice, n);
    float best = 1.0e30f;
    for (uint i = begin + lane; i < end; i += 256u) {
        best = min(best, values[i]);
    }
    threadgroup float shared[256];
    shared[lane] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 128u; offset > 0u; offset >>= 1) {
        if (lane < offset) {
            shared[lane] = min(shared[lane], shared[lane + offset]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) {
        partial[tgid] = shared[0];
    }
}

kernel void final_min(
    device float* partial [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint lane [[thread_index_in_threadgroup]])
{
    threadgroup float shared[256];
    shared[lane] = partial[lane];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 128u; offset > 0u; offset >>= 1) {
        if (lane < offset) {
            shared[lane] = min(shared[lane], shared[lane + offset]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) {
        out[0] = shared[0];
    }
}

kernel void mask_below(
    device float* values [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    constant float& threshold [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    if (id < n && values[id] < threshold) {
        values[id] = -1.0e30f;
    }
}

kernel void copy_token_major(
    device uchar* source [[buffer(0)]],
    device uchar* destination [[buffer(1)]],
    constant uint& heads [[buffer(2)]],
    constant uint& channels [[buffer(3)]],
    constant uint& length [[buffer(4)]],
    constant uint& old_cap [[buffer(5)]],
    constant uint& new_cap [[buffer(6)]],
    uint id [[thread_position_in_grid]])
{
    uint total = heads * length;
    if (id >= total) {
        return;
    }
    uint head = id / length;
    uint token = id - head * length;
    uint from = (head * old_cap + token) * channels;
    uint to = (head * new_cap + token) * channels;
    for (uint c = 0u; c < channels; ++c) {
        destination[to + c] = source[from + c];
    }
}

kernel void copy_channel_major(
    device uchar* source [[buffer(0)]],
    device uchar* destination [[buffer(1)]],
    constant uint& heads [[buffer(2)]],
    constant uint& channels [[buffer(3)]],
    constant uint& length [[buffer(4)]],
    constant uint& old_cap [[buffer(5)]],
    constant uint& new_cap [[buffer(6)]],
    uint id [[thread_position_in_grid]])
{
    uint total = heads * channels;
    if (id >= total) {
        return;
    }
    uint head = id / channels;
    uint channel = id - head * channels;
    uint from = (head * channels + channel) * old_cap;
    uint to = (head * channels + channel) * new_cap;
    for (uint token = 0u; token < length; ++token) {
        destination[to + token] = source[from + token];
    }
}
