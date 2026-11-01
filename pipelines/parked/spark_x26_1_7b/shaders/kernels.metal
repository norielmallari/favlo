#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>

using namespace metal;
using namespace mpp::tensor_ops;

using scales_plane = tensor_blockwise<tensor_plane_scales,
                                      device metal_fp8_ue8m0_format,
                                      32, 1>;

constant float INV_SQRT2 = 0.7071067811865476f;
constant float ATTN_SCALE = 0.0625f;
constant float E4M3[256] = {
    0.00000000e+00f, 1.95312500e-03f, 3.90625000e-03f, 5.85937500e-03f, 7.81250000e-03f, 9.76562500e-03f, 1.17187500e-02f, 1.36718750e-02f,
    1.56250000e-02f, 1.75781250e-02f, 1.95312500e-02f, 2.14843750e-02f, 2.34375000e-02f, 2.53906250e-02f, 2.73437500e-02f, 2.92968750e-02f,
    3.12500000e-02f, 3.51562500e-02f, 3.90625000e-02f, 4.29687500e-02f, 4.68750000e-02f, 5.07812500e-02f, 5.46875000e-02f, 5.85937500e-02f,
    6.25000000e-02f, 7.03125000e-02f, 7.81250000e-02f, 8.59375000e-02f, 9.37500000e-02f, 1.01562500e-01f, 1.09375000e-01f, 1.17187500e-01f,
    1.25000000e-01f, 1.40625000e-01f, 1.56250000e-01f, 1.71875000e-01f, 1.87500000e-01f, 2.03125000e-01f, 2.18750000e-01f, 2.34375000e-01f,
    2.50000000e-01f, 2.81250000e-01f, 3.12500000e-01f, 3.43750000e-01f, 3.75000000e-01f, 4.06250000e-01f, 4.37500000e-01f, 4.68750000e-01f,
    5.00000000e-01f, 5.62500000e-01f, 6.25000000e-01f, 6.87500000e-01f, 7.50000000e-01f, 8.12500000e-01f, 8.75000000e-01f, 9.37500000e-01f,
    1.00000000e+00f, 1.12500000e+00f, 1.25000000e+00f, 1.37500000e+00f, 1.50000000e+00f, 1.62500000e+00f, 1.75000000e+00f, 1.87500000e+00f,
    2.00000000e+00f, 2.25000000e+00f, 2.50000000e+00f, 2.75000000e+00f, 3.00000000e+00f, 3.25000000e+00f, 3.50000000e+00f, 3.75000000e+00f,
    4.00000000e+00f, 4.50000000e+00f, 5.00000000e+00f, 5.50000000e+00f, 6.00000000e+00f, 6.50000000e+00f, 7.00000000e+00f, 7.50000000e+00f,
    8.00000000e+00f, 9.00000000e+00f, 1.00000000e+01f, 1.10000000e+01f, 1.20000000e+01f, 1.30000000e+01f, 1.40000000e+01f, 1.50000000e+01f,
    1.60000000e+01f, 1.80000000e+01f, 2.00000000e+01f, 2.20000000e+01f, 2.40000000e+01f, 2.60000000e+01f, 2.80000000e+01f, 3.00000000e+01f,
    3.20000000e+01f, 3.60000000e+01f, 4.00000000e+01f, 4.40000000e+01f, 4.80000000e+01f, 5.20000000e+01f, 5.60000000e+01f, 6.00000000e+01f,
    6.40000000e+01f, 7.20000000e+01f, 8.00000000e+01f, 8.80000000e+01f, 9.60000000e+01f, 1.04000000e+02f, 1.12000000e+02f, 1.20000000e+02f,
    1.28000000e+02f, 1.44000000e+02f, 1.60000000e+02f, 1.76000000e+02f, 1.92000000e+02f, 2.08000000e+02f, 2.24000000e+02f, 2.40000000e+02f,
    2.56000000e+02f, 2.88000000e+02f, 3.20000000e+02f, 3.52000000e+02f, 3.84000000e+02f, 4.16000000e+02f, 4.48000000e+02f, NAN,
    -0.00000000e+00f, -1.95312500e-03f, -3.90625000e-03f, -5.85937500e-03f, -7.81250000e-03f, -9.76562500e-03f, -1.17187500e-02f, -1.36718750e-02f,
    -1.56250000e-02f, -1.75781250e-02f, -1.95312500e-02f, -2.14843750e-02f, -2.34375000e-02f, -2.53906250e-02f, -2.73437500e-02f, -2.92968750e-02f,
    -3.12500000e-02f, -3.51562500e-02f, -3.90625000e-02f, -4.29687500e-02f, -4.68750000e-02f, -5.07812500e-02f, -5.46875000e-02f, -5.85937500e-02f,
    -6.25000000e-02f, -7.03125000e-02f, -7.81250000e-02f, -8.59375000e-02f, -9.37500000e-02f, -1.01562500e-01f, -1.09375000e-01f, -1.17187500e-01f,
    -1.25000000e-01f, -1.40625000e-01f, -1.56250000e-01f, -1.71875000e-01f, -1.87500000e-01f, -2.03125000e-01f, -2.18750000e-01f, -2.34375000e-01f,
    -2.50000000e-01f, -2.81250000e-01f, -3.12500000e-01f, -3.43750000e-01f, -3.75000000e-01f, -4.06250000e-01f, -4.37500000e-01f, -4.68750000e-01f,
    -5.00000000e-01f, -5.62500000e-01f, -6.25000000e-01f, -6.87500000e-01f, -7.50000000e-01f, -8.12500000e-01f, -8.75000000e-01f, -9.37500000e-01f,
    -1.00000000e+00f, -1.12500000e+00f, -1.25000000e+00f, -1.37500000e+00f, -1.50000000e+00f, -1.62500000e+00f, -1.75000000e+00f, -1.87500000e+00f,
    -2.00000000e+00f, -2.25000000e+00f, -2.50000000e+00f, -2.75000000e+00f, -3.00000000e+00f, -3.25000000e+00f, -3.50000000e+00f, -3.75000000e+00f,
    -4.00000000e+00f, -4.50000000e+00f, -5.00000000e+00f, -5.50000000e+00f, -6.00000000e+00f, -6.50000000e+00f, -7.00000000e+00f, -7.50000000e+00f,
    -8.00000000e+00f, -9.00000000e+00f, -1.00000000e+01f, -1.10000000e+01f, -1.20000000e+01f, -1.30000000e+01f, -1.40000000e+01f, -1.50000000e+01f,
    -1.60000000e+01f, -1.80000000e+01f, -2.00000000e+01f, -2.20000000e+01f, -2.40000000e+01f, -2.60000000e+01f, -2.80000000e+01f, -3.00000000e+01f,
    -3.20000000e+01f, -3.60000000e+01f, -4.00000000e+01f, -4.40000000e+01f, -4.80000000e+01f, -5.20000000e+01f, -5.60000000e+01f, -6.00000000e+01f,
    -6.40000000e+01f, -7.20000000e+01f, -8.00000000e+01f, -8.80000000e+01f, -9.60000000e+01f, -1.04000000e+02f, -1.12000000e+02f, -1.20000000e+02f,
    -1.28000000e+02f, -1.44000000e+02f, -1.60000000e+02f, -1.76000000e+02f, -1.92000000e+02f, -2.08000000e+02f, -2.24000000e+02f, -2.40000000e+02f,
    -2.56000000e+02f, -2.88000000e+02f, -3.20000000e+02f, -3.52000000e+02f, -3.84000000e+02f, -4.16000000e+02f, -4.48000000e+02f, NAN,
};
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

static float dequant_fp8(device uchar* data, device uchar* scales, uint row, uint k, uint inner) {
    uint code = data[row * inner + k];
    uint scale = scales[row * (inner >> 5) + (k >> 5)];
    return E4M3[code] * exp2(float(scale) - 127.0f);
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
    uint bytes = inner;
    uint scale_stride = inner >> 5;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    for (uint base = 0u; base < inner; base += 64u) {
        uint k = base + lane * 2u;
        float a0 = act[k];
        float a1 = act[k + 1u];
        uint scale_index = k >> 5;
        uchar p0 = weights[row0 * bytes + k];
        uchar n0 = weights[row0 * bytes + k + 1u];
        uchar p1 = weights[(row0 + 1u) * bytes + k];
        uchar n1 = weights[(row0 + 1u) * bytes + k + 1u];
        uchar p2 = weights[(row0 + 2u) * bytes + k];
        uchar n2 = weights[(row0 + 2u) * bytes + k + 1u];
        uchar p3 = weights[(row0 + 3u) * bytes + k];
        uchar n3 = weights[(row0 + 3u) * bytes + k + 1u];
        float s0 = exp2(float(scales[row0 * scale_stride + scale_index]) - 127.0f);
        float s1 = exp2(float(scales[(row0 + 1u) * scale_stride + scale_index]) - 127.0f);
        float s2 = exp2(float(scales[(row0 + 2u) * scale_stride + scale_index]) - 127.0f);
        float s3 = exp2(float(scales[(row0 + 3u) * scale_stride + scale_index]) - 127.0f);
        acc0 += (a0 * E4M3[p0] + a1 * E4M3[n0]) * s0;
        acc1 += (a0 * E4M3[p1] + a1 * E4M3[n1]) * s1;
        acc2 += (a0 * E4M3[p2] + a1 * E4M3[n2]) * s2;
        acc3 += (a0 * E4M3[p3] + a1 * E4M3[n3]) * s3;
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
    uint bytes = inner;
    uint scale_stride = inner >> 5;
    for (uint base = 0u; base < inner; base += 64u) {
        uint k = base + lane * 2u;
        float a0 = activation[k];
        float a1 = activation[k + 1u];
        uint scale_index = k >> 5;
        uchar p0 = weights[row0 * bytes + k];
        uchar n0 = weights[row0 * bytes + k + 1u];
        uchar p1 = weights[(row0 + 1u) * bytes + k];
        uchar n1 = weights[(row0 + 1u) * bytes + k + 1u];
        uchar p2 = weights[(row0 + 2u) * bytes + k];
        uchar n2 = weights[(row0 + 2u) * bytes + k + 1u];
        uchar p3 = weights[(row0 + 3u) * bytes + k];
        uchar n3 = weights[(row0 + 3u) * bytes + k + 1u];
        float s0 = exp2(float(scales[row0 * scale_stride + scale_index]) - 127.0f);
        float s1 = exp2(float(scales[(row0 + 1u) * scale_stride + scale_index]) - 127.0f);
        float s2 = exp2(float(scales[(row0 + 2u) * scale_stride + scale_index]) - 127.0f);
        float s3 = exp2(float(scales[(row0 + 3u) * scale_stride + scale_index]) - 127.0f);
        acc0 += (a0 * E4M3[p0] + a1 * E4M3[n0]) * s0;
        acc1 += (a0 * E4M3[p1] + a1 * E4M3[n1]) * s1;
        acc2 += (a0 * E4M3[p2] + a1 * E4M3[n2]) * s2;
        acc3 += (a0 * E4M3[p3] + a1 * E4M3[n3]) * s3;
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

kernel void gemv_fp8(
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

static float dot8(float a0, float a1, float a2, float a3, float a4, float a5, float a6, float a7, uint2 packed) {
    return a0 * E4M3[packed.x & 0xFFu]
        + a1 * E4M3[(packed.x >> 8) & 0xFFu]
        + a2 * E4M3[(packed.x >> 16) & 0xFFu]
        + a3 * E4M3[packed.x >> 24]
        + a4 * E4M3[packed.y & 0xFFu]
        + a5 * E4M3[(packed.y >> 8) & 0xFFu]
        + a6 * E4M3[(packed.y >> 16) & 0xFFu]
        + a7 * E4M3[packed.y >> 24];
}

kernel void gemv_fp8_row(
    device float* activation [[buffer(0)]],
    device uchar* weights [[buffer(1)]],
    device uchar* scales [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& inner [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    uint bytes = inner;
    device uchar* weight = weights + row * bytes;
    device uchar* scale_row = scales + row * (inner >> 5);
    float acc = 0.0f;
    for (uint base = 0u; base < inner; base += 256u) {
        uint k = base + lane * 8u;
        float scale = exp2(float(scale_row[k >> 5]) - 127.0f);
        uint2 packed = *reinterpret_cast<device uint2*>(weight + k);
        acc += scale * dot8(
            activation[k], activation[k + 1u], activation[k + 2u], activation[k + 3u],
            activation[k + 4u], activation[k + 5u], activation[k + 6u], activation[k + 7u],
            packed);
    }
    acc = simd_sum(acc);
    if (lane == 0u) {
        destination[row] = acc;
    }
}

kernel void gemv_fp8_row_add(
    device float* activation [[buffer(0)]],
    device uchar* weights [[buffer(1)]],
    device uchar* scales [[buffer(2)]],
    device float* hidden [[buffer(3)]],
    constant uint& inner [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    uint bytes = inner;
    device uchar* weight = weights + row * bytes;
    device uchar* scale_row = scales + row * (inner >> 5);
    float acc = 0.0f;
    for (uint base = 0u; base < inner; base += 256u) {
        uint k = base + lane * 8u;
        float scale = exp2(float(scale_row[k >> 5]) - 127.0f);
        uint2 packed = *reinterpret_cast<device uint2*>(weight + k);
        acc += scale * dot8(
            activation[k], activation[k + 1u], activation[k + 2u], activation[k + 3u],
            activation[k + 4u], activation[k + 5u], activation[k + 6u], activation[k + 7u],
            packed);
    }
    acc = simd_sum(acc);
    if (lane == 0u) {
        hidden[row] += acc;
    }
}

kernel void gemv_fp8_vec(
    device float* activation [[buffer(0)]],
    device uchar* weights [[buffer(1)]],
    device uchar* scales [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& inner [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    uint row0 = tgid * 4u;
    uint bytes = inner;
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
        uint2 p0 = *reinterpret_cast<device uint2*>(w0 + k);
        uint2 p1 = *reinterpret_cast<device uint2*>(w1 + k);
        uint2 p2 = *reinterpret_cast<device uint2*>(w2 + k);
        uint2 p3 = *reinterpret_cast<device uint2*>(w3 + k);
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

template <bool NORM, bool ACCUM>
static void gemv_cached(
    threadgroup float* act,
    threadgroup float* red,
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
            red[simd] = local;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (NORM) {
        if (tid == 0u) {
            float sum = red[0] + red[1] + red[2] + red[3] + red[4] + red[5] + red[6] + red[7];
            red[0] = rsqrt(sum / float(inner) + eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv = red[0];
        for (uint k = tid; k < inner; k += 256u) {
            act[k] = act[k] * float(norm_w[k]) * inv;
            if (tgid == 0u) {
                normed[k] = act[k];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint row0 = tgid * 32u + simd * 4u;
    uint bytes = inner;
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
        float a0 = act[k];
        float a1 = act[k + 1u];
        float a2 = act[k + 2u];
        float a3 = act[k + 3u];
        float a4 = act[k + 4u];
        float a5 = act[k + 5u];
        float a6 = act[k + 6u];
        float a7 = act[k + 7u];
        uint2 p0 = *reinterpret_cast<device uint2*>(w0 + k);
        uint2 p1 = *reinterpret_cast<device uint2*>(w1 + k);
        uint2 p2 = *reinterpret_cast<device uint2*>(w2 + k);
        uint2 p3 = *reinterpret_cast<device uint2*>(w3 + k);
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

kernel void gemv_fp8_vec_rms(
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
    threadgroup float red[8];
    gemv_cached<true, false>(
        act, red, activation, norm_w, weights, scales, destination, normed, inner, eps, tgid, tid);
}

kernel void gemv_fp8_vec_add(
    device float* activation [[buffer(0)]],
    device uchar* weights [[buffer(1)]],
    device uchar* scales [[buffer(2)]],
    device float* hidden [[buffer(3)]],
    constant uint& inner [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    uint row0 = tgid * 4u;
    uint bytes = inner;
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
        uint2 p0 = *reinterpret_cast<device uint2*>(w0 + k);
        uint2 p1 = *reinterpret_cast<device uint2*>(w1 + k);
        uint2 p2 = *reinterpret_cast<device uint2*>(w2 + k);
        uint2 p3 = *reinterpret_cast<device uint2*>(w3 + k);
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

kernel void gemv_fp8_vec_add_cached(
    device float* activation [[buffer(0)]],
    device uchar* weights [[buffer(1)]],
    device uchar* scales [[buffer(2)]],
    device float* hidden [[buffer(3)]],
    constant uint& inner [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float act[2048];
    threadgroup float red[8];
    gemv_cached<false, true>(
        act, red, activation, nullptr, weights, scales, hidden, nullptr, inner, 0.0f, tgid, tid);
}

kernel void gemv_fp8_vec_add_wide(
    device float* activation [[buffer(0)]],
    device uchar* weights [[buffer(1)]],
    device uchar* scales [[buffer(2)]],
    device float* hidden [[buffer(3)]],
    constant uint& inner [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float act[6656];
    threadgroup float red[8];
    gemv_cached<false, true>(
        act, red, activation, nullptr, weights, scales, hidden, nullptr, inner, 0.0f, tgid, tid);
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
        destination[k] = dequant_fp8(weights, scales, 0u, k, inner);
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
    uint tid [[thread_position_in_grid]])
{
    uint lane = tid & 31u;
    uint simd = tid >> 5;
    if (cfg.count == 1u && simd < 16u) {
        uint key_head = simd >> 3;
        uint group = simd & 7u;
        device float* key = rows + 2048u;
        float amax = simd_max(fabs(key[key_head * 256u + group * 32u + lane]));
        uint sc = e8m0_byte(amax, 448.0f);
        float scale = exp2(float(sc) - 127.0f);
        float raw = key[key_head * 256u + group * 32u + lane];
        float q = scale > 0.0f ? raw / scale : 0.0f;
        key_data[(key_head * cfg.capacity + cfg.pos) * 256u + group * 32u + lane] = uchar(enc_e4m3(q));
        if (lane == 0u) {
            key_scale[(key_head * cfg.capacity + cfg.pos) * 8u + group] = uchar(sc);
        }
    }
    uint head = tid >> 8;
    uint channel = tid & 255u;
    if (head >= 2u) {
        return;
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
            scratch[(head * 256u + channel) * 32u + slot] = value[head * 256u + channel];
        }
        uint slot_limit = block_end - base;
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

kernel void store_slide_span(
    device float* rows [[buffer(0)]],
    device uchar* key_data [[buffer(1)]],
    device uchar* key_scale [[buffer(2)]],
    device uchar* value_data [[buffer(3)]],
    device uchar* value_scale [[buffer(4)]],
    device float* scratch [[buffer(5)]],
    constant SpanStore& cfg [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    uint lane = tid & 31u;
    uint simd = tid >> 5;
    if (cfg.count == 1u && simd < 16u) {
        uint slot0 = cfg.pos % 512u;
        uint key_head = simd >> 3;
        uint group = simd & 7u;
        device float* key = rows + 2048u;
        float amax = simd_max(fabs(key[key_head * 256u + group * 32u + lane]));
        uint sc = e8m0_byte(amax, 448.0f);
        float scale = exp2(float(sc) - 127.0f);
        float raw = key[key_head * 256u + group * 32u + lane];
        float q = scale > 0.0f ? raw / scale : 0.0f;
        key_data[(key_head * 512u + slot0) * 256u + group * 32u + lane] = uchar(enc_e4m3(q));
        if (lane == 0u) {
            key_scale[(key_head * 512u + slot0) * 8u + group] = uchar(sc);
        }
    }
    uint head = tid >> 8;
    uint channel = tid & 255u;
    if (head >= 2u) {
        return;
    }
    uint span_end = cfg.pos + cfg.count;
    uint latest = span_end - 1u;
    for (uint token = 0u; token < cfg.count; ++token) {
        uint pos = cfg.pos + token;
        uint slot = pos % 512u;
        device float* row = rows + token * 3072u;
        device float* value = row + 2560u;
        scratch[(head * 256u + channel) * 512u + slot] = value[head * 256u + channel];
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
    uint blocks = capacity >> 5;
    uint base = lane * 8u;
    for (uint i = begin; i < end; ++i) {
        uint slot = window == 0u ? (start + i) : ((start + i) % window);
        device uchar* key_row = key_data + (kv * capacity + slot) * 256u;
        uint2 packed = *reinterpret_cast<device uint2*>(key_row + base);
        uint sc = key_scale[(kv * capacity + slot) * 8u + (base >> 5)];
        float scale = exp2(float(sc) - 127.0f);
        float partial = query[base] * E4M3[packed.x & 0xFFu]
            + query[base + 1u] * E4M3[(packed.x >> 8) & 0xFFu]
            + query[base + 2u] * E4M3[(packed.x >> 16) & 0xFFu]
            + query[base + 3u] * E4M3[packed.x >> 24]
            + query[base + 4u] * E4M3[packed.y & 0xFFu]
            + query[base + 5u] * E4M3[(packed.y >> 8) & 0xFFu]
            + query[base + 6u] * E4M3[(packed.y >> 16) & 0xFFu]
            + query[base + 7u] * E4M3[packed.y >> 24];
        float score = simd_sum(partial * scale) * ATTN_SCALE;
        float next = max(peak, score);
        float decay = exp(peak - next);
        float weight = exp(score - next);
        sum = sum * decay + weight;
        uint block = slot >> 5;
        for (uint j = 0u; j < 8u; ++j) {
            uint channel = base + j;
            uint code = value_data[(kv * 256u + channel) * capacity + slot];
            uint vc = value_scale[(kv * 256u + channel) * blocks + block];
            float value = E4M3[code] * exp2(float(vc) - 127.0f);
            acc[j] = acc[j] * decay + weight * value;
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
        uint channel = lane * 8u + j;
        mixed[(token * 8u + qh) * 256u + channel] = acc[j] * inv * gate;
    }
}

kernel void attend_part(
    device float* queries [[buffer(0)]],
    device uchar* key_data [[buffer(1)]],
    device uchar* key_scale [[buffer(2)]],
    device uchar* value_data [[buffer(3)]],
    device uchar* value_scale [[buffer(4)]],
    device float* partial [[buffer(5)]],
    device float* state [[buffer(6)]],
    constant AttendSpan& cfg [[buffer(7)]],
    constant uint& tile_stride [[buffer(8)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    uint qh = tgid.x;
    uint token = tgid.y;
    uint part = tgid.z;
    if (token >= cfg.count) {
        return;
    }
    uint kv = qh >> 2;
    uint pos = cfg.pos + token;
    uint start = 0u;
    uint count = pos + 1u;
    if (cfg.window != 0u && count > cfg.window) {
        start = count - cfg.window;
        count = cfg.window;
    }
    uint begin = part * 64u;
    uint slot_index = (token * 8u + qh) * tile_stride + part;
    device float* row = partial + slot_index * 256u;
    if (begin >= count) {
        for (uint j = 0u; j < 8u; ++j) {
            row[lane * 8u + j] = 0.0f;
        }
        if (lane == 0u) {
            state[slot_index * 2u] = -1.0e30f;
            state[slot_index * 2u + 1u] = 0.0f;
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
        queries + token * 3072u + qh * 256u,
        key_data, key_scale, value_data, value_scale,
        kv, start, begin, end, cfg.window, cfg.capacity, lane, peak, sum, acc);
    for (uint j = 0u; j < 8u; ++j) {
        row[lane * 8u + j] = acc[j];
    }
    if (lane == 0u) {
        state[slot_index * 2u] = peak;
        state[slot_index * 2u + 1u] = sum;
    }
}

kernel void attend_chunk_merge(
    device float* partial [[buffer(0)]],
    device float* state [[buffer(1)]],
    device float* gates [[buffer(2)]],
    device float* mixed [[buffer(3)]],
    constant uint& tile_stride [[buffer(4)]],
    constant uint& parts [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    uint qh = tgid.x;
    uint token = tgid.y;
    uint base = (token * 8u + qh) * tile_stride;
    float peak = -1.0e30f;
    for (uint part = 0u; part < parts; ++part) {
        peak = max(peak, state[(base + part) * 2u]);
    }
    float acc[8];
    for (uint j = 0u; j < 8u; ++j) {
        acc[j] = 0.0f;
    }
    float sum = 0.0f;
    for (uint part = 0u; part < parts; ++part) {
        float tile_peak = state[(base + part) * 2u];
        float tile_sum = state[(base + part) * 2u + 1u];
        float scale = exp(tile_peak - peak);
        sum += tile_sum * scale;
        device float* row = partial + (base + part) * 256u;
        for (uint j = 0u; j < 8u; ++j) {
            acc[j] += row[lane * 8u + j] * scale;
        }
    }
    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    float gate = 1.0f / (1.0f + exp(-gates[token * 8u + qh]));
    for (uint j = 0u; j < 8u; ++j) {
        mixed[(token * 8u + qh) * 256u + lane * 8u + j] = acc[j] * inv * gate;
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
            row[lane * 8u + j] = 0.0f;
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
        row[lane * 8u + j] = acc[j];
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
            acc[j] += row[lane * 8u + j] * scale;
        }
    }
    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    float gate = 1.0f / (1.0f + exp(-gates[qh]));
    for (uint j = 0u; j < 8u; ++j) {
        mixed[qh * 256u + lane * 8u + j] = acc[j] * inv * gate;
    }
}

static void attend_range_f16(
    device float* query,
    device half* key_data,
    device half* value_data,
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
    uint base = lane * 8u;
    for (uint i = begin; i < end; ++i) {
        uint slot = window == 0u ? (start + i) : ((start + i) % window);
        device half* key_row = key_data + (kv * capacity + slot) * 256u;
        float partial = query[base] * float(key_row[base])
            + query[base + 1u] * float(key_row[base + 1u])
            + query[base + 2u] * float(key_row[base + 2u])
            + query[base + 3u] * float(key_row[base + 3u])
            + query[base + 4u] * float(key_row[base + 4u])
            + query[base + 5u] * float(key_row[base + 5u])
            + query[base + 6u] * float(key_row[base + 6u])
            + query[base + 7u] * float(key_row[base + 7u]);
        float score = simd_sum(partial) * ATTN_SCALE;
        float next = max(peak, score);
        float decay = exp(peak - next);
        float weight = exp(score - next);
        sum = sum * decay + weight;
        for (uint j = 0u; j < 8u; ++j) {
            uint channel = base + j;
            float value = float(value_data[(kv * 256u + channel) * capacity + slot]);
            acc[j] = acc[j] * decay + weight * value;
        }
        peak = next;
    }
}

kernel void store_full_keys_f16(
    device float* rows [[buffer(0)]],
    device half* key_data [[buffer(1)]],
    constant SpanStore& cfg [[buffer(2)]],
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
            key_data[(head * cfg.capacity + pos) * 256u + group * 32u + lane] =
                half(key[head * 256u + group * 32u + lane]);
        }
    }
}

kernel void store_slide_keys_f16(
    device float* rows [[buffer(0)]],
    device half* key_data [[buffer(1)]],
    constant SpanStore& cfg [[buffer(2)]],
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
            key_data[(head * 512u + slot) * 256u + group * 32u + lane] =
                half(key[head * 256u + group * 32u + lane]);
        }
    }
}

kernel void store_full_span_f16(
    device float* rows [[buffer(0)]],
    device half* key_data [[buffer(1)]],
    device half* value_data [[buffer(2)]],
    constant SpanStore& cfg [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    uint lane = tid & 31u;
    uint simd = tid >> 5;
    if (cfg.count == 1u && simd < 16u) {
        uint key_head = simd >> 3;
        uint group = simd & 7u;
        device float* key = rows + 2048u;
        key_data[(key_head * cfg.capacity + cfg.pos) * 256u + group * 32u + lane] =
            half(key[key_head * 256u + group * 32u + lane]);
    }
    uint head = tid >> 8;
    uint channel = tid & 255u;
    if (head >= 2u) {
        return;
    }
    for (uint token = 0u; token < cfg.count; ++token) {
        uint pos = cfg.pos + token;
        device float* value = rows + token * 3072u + 2560u;
        value_data[(head * 256u + channel) * cfg.capacity + pos] = half(value[head * 256u + channel]);
    }
}

kernel void store_slide_span_f16(
    device float* rows [[buffer(0)]],
    device half* key_data [[buffer(1)]],
    device half* value_data [[buffer(2)]],
    constant SpanStore& cfg [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    uint lane = tid & 31u;
    uint simd = tid >> 5;
    if (cfg.count == 1u && simd < 16u) {
        uint slot0 = cfg.pos % 512u;
        uint key_head = simd >> 3;
        uint group = simd & 7u;
        device float* key = rows + 2048u;
        key_data[(key_head * 512u + slot0) * 256u + group * 32u + lane] =
            half(key[key_head * 256u + group * 32u + lane]);
    }
    uint head = tid >> 8;
    uint channel = tid & 255u;
    if (head >= 2u) {
        return;
    }
    for (uint token = 0u; token < cfg.count; ++token) {
        uint pos = cfg.pos + token;
        uint slot = pos % 512u;
        device float* value = rows + token * 3072u + 2560u;
        value_data[(head * 256u + channel) * 512u + slot] = half(value[head * 256u + channel]);
    }
}

kernel void attend_span_f16(
    device float* queries [[buffer(0)]],
    device half* key_data [[buffer(1)]],
    device half* value_data [[buffer(2)]],
    device float* gates [[buffer(3)]],
    device float* mixed [[buffer(4)]],
    constant AttendSpan& cfg [[buffer(5)]],
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
    attend_range_f16(
        queries + token * 3072u + qh * 256u,
        key_data,
        value_data,
        kv,
        start,
        0u,
        count,
        cfg.window,
        cfg.capacity,
        lane,
        peak,
        sum,
        acc);
    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    float gate = 1.0f / (1.0f + exp(-gates[token * 8u + qh]));
    for (uint j = 0u; j < 8u; ++j) {
        uint channel = lane * 8u + j;
        mixed[(token * 8u + qh) * 256u + channel] = acc[j] * inv * gate;
    }
}

kernel void attend_part_f16(
    device float* queries [[buffer(0)]],
    device half* key_data [[buffer(1)]],
    device half* value_data [[buffer(2)]],
    device float* partial [[buffer(3)]],
    device float* state [[buffer(4)]],
    constant AttendSpan& cfg [[buffer(5)]],
    constant uint& tile_stride [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    uint qh = tgid.x;
    uint token = tgid.y;
    uint part = tgid.z;
    if (token >= cfg.count) {
        return;
    }
    uint kv = qh >> 2;
    uint pos = cfg.pos + token;
    uint start = 0u;
    uint count = pos + 1u;
    if (cfg.window != 0u && count > cfg.window) {
        start = count - cfg.window;
        count = cfg.window;
    }
    uint begin = part * 64u;
    uint slot_index = (token * 8u + qh) * tile_stride + part;
    device float* row = partial + slot_index * 256u;
    if (begin >= count) {
        for (uint j = 0u; j < 8u; ++j) {
            row[lane * 8u + j] = 0.0f;
        }
        if (lane == 0u) {
            state[slot_index * 2u] = -1.0e30f;
            state[slot_index * 2u + 1u] = 0.0f;
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
    attend_range_f16(
        queries + token * 3072u + qh * 256u,
        key_data,
        value_data,
        kv,
        start,
        begin,
        end,
        cfg.window,
        cfg.capacity,
        lane,
        peak,
        sum,
        acc);
    for (uint j = 0u; j < 8u; ++j) {
        row[lane * 8u + j] = acc[j];
    }
    if (lane == 0u) {
        state[slot_index * 2u] = peak;
        state[slot_index * 2u + 1u] = sum;
    }
}

kernel void attend_tile_f16(
    device float* queries [[buffer(0)]],
    device half* key_data [[buffer(1)]],
    device half* value_data [[buffer(2)]],
    device float* partial [[buffer(3)]],
    device float* state [[buffer(4)]],
    constant AttendSpan& cfg [[buffer(5)]],
    constant uint& tile_stride [[buffer(6)]],
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
            row[lane * 8u + j] = 0.0f;
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
    attend_range_f16(
        queries + qh * 256u,
        key_data,
        value_data,
        kv,
        start,
        begin,
        end,
        cfg.window,
        cfg.capacity,
        lane,
        peak,
        sum,
        acc);
    for (uint j = 0u; j < 8u; ++j) {
        row[lane * 8u + j] = acc[j];
    }
    if (lane == 0u) {
        state[(qh * tile_stride + tile) * 2u] = peak;
        state[(qh * tile_stride + tile) * 2u + 1u] = sum;
    }
}

kernel void copy_token_major_u16(
    device ushort* source [[buffer(0)]],
    device ushort* destination [[buffer(1)]],
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

kernel void copy_channel_major_u16(
    device ushort* source [[buffer(0)]],
    device ushort* destination [[buffer(1)]],
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

kernel void gemv_fp8_rms(
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
    uint bytes = inner;
    uint scale_stride = inner >> 5;
    for (uint base = 0u; base < inner; base += 64u) {
        uint k = base + lane * 2u;
        float a0 = activation[k] * float(norm_w[k]) * inv;
        float a1 = activation[k + 1u] * float(norm_w[k + 1u]) * inv;
        uint scale_index = k >> 5;
        uchar p0 = weights[row0 * bytes + k];
        uchar n0 = weights[row0 * bytes + k + 1u];
        uchar p1 = weights[(row0 + 1u) * bytes + k];
        uchar n1 = weights[(row0 + 1u) * bytes + k + 1u];
        uchar p2 = weights[(row0 + 2u) * bytes + k];
        uchar n2 = weights[(row0 + 2u) * bytes + k + 1u];
        uchar p3 = weights[(row0 + 3u) * bytes + k];
        uchar n3 = weights[(row0 + 3u) * bytes + k + 1u];
        float s0 = exp2(float(scales[row0 * scale_stride + scale_index]) - 127.0f);
        float s1 = exp2(float(scales[(row0 + 1u) * scale_stride + scale_index]) - 127.0f);
        float s2 = exp2(float(scales[(row0 + 2u) * scale_stride + scale_index]) - 127.0f);
        float s3 = exp2(float(scales[(row0 + 3u) * scale_stride + scale_index]) - 127.0f);
        acc0 += (a0 * E4M3[p0] + a1 * E4M3[n0]) * s0;
        acc1 += (a0 * E4M3[p1] + a1 * E4M3[n1]) * s1;
        acc2 += (a0 * E4M3[p2] + a1 * E4M3[n2]) * s2;
        acc3 += (a0 * E4M3[p3] + a1 * E4M3[n3]) * s3;
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

kernel void gemv_fp8_add(
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

kernel void matmul_fp8_64(
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

kernel void matmul_fp8_32(
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
    constexpr auto desc = matmul2d_descriptor(32, 128, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto tile_left = left.slice(0, int(tgid.y) * 32);
    auto tile_right = right.slice(0, int(tgid.x) * 128);
    auto tile_out = out.slice(int(tgid.x) * 128, int(tgid.y) * 32);
    op.run(tile_left, tile_right, tile_out);
}

kernel void gemv_fp8_split(
    device float* activation [[buffer(0)]],
    device uchar* weights [[buffer(1)]],
    device uchar* scales [[buffer(2)]],
    device float* partial [[buffer(3)]],
    constant uint& inner [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& splits [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    uint span = inner / splits;
    uint row0 = tgid.x * 4u;
    uint k0 = tgid.y * span;
    uint bytes = inner;
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
    for (uint base = k0; base < k0 + span; base += 256u) {
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
        uint2 p0 = *reinterpret_cast<device uint2*>(w0 + k);
        uint2 p1 = *reinterpret_cast<device uint2*>(w1 + k);
        uint2 p2 = *reinterpret_cast<device uint2*>(w2 + k);
        uint2 p3 = *reinterpret_cast<device uint2*>(w3 + k);
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
        device float* dest = partial + tgid.y * rows + row0;
        dest[0] = acc0;
        dest[1] = acc1;
        dest[2] = acc2;
        dest[3] = acc3;
    }
}

kernel void gemv_split_sum(
    device float* partial [[buffer(0)]],
    device float* destination [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& splits [[buffer(3)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= rows) {
        return;
    }
    float sum = 0.0f;
    for (uint part = 0u; part < splits; part++) {
        sum += partial[part * rows + row];
    }
    destination[row] = sum;
}

kernel void mm_m8_n128(
    device half* activation [[buffer(0)]],
    device uchar* weights [[buffer(1)]],
    device uchar* scales [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]])
{
    tensor<device half, dextents<int, 2>, tensor_inline> left(
        activation, dextents<int, 2>(int(inner), 8), array<int, 2>{1, int(inner)});
    tensor<device metal_fp8_e4m3_format, dextents<int, 2>, tensor_inline, scales_plane> right(
        weights, dextents<int, 2>(int(inner), int(cols)), array<int, 2>{1, int(inner)}, scales_plane(scales));
    tensor<device float, dextents<int, 2>, tensor_inline> out(
        destination, dextents<int, 2>(int(cols), 8), array<int, 2>{1, int(cols)});
    constexpr auto desc = matmul2d_descriptor(8, 128, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto tile_left = left.slice(0, 0);
    auto tile_right = right.slice(0, int(tgid) * 128);
    auto tile_out = out.slice(int(tgid) * 128, 0);
    op.run(tile_left, tile_right, tile_out);
}

kernel void mm_m8_n32(
    device half* activation [[buffer(0)]],
    device uchar* weights [[buffer(1)]],
    device uchar* scales [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]])
{
    tensor<device half, dextents<int, 2>, tensor_inline> left(
        activation, dextents<int, 2>(int(inner), 8), array<int, 2>{1, int(inner)});
    tensor<device metal_fp8_e4m3_format, dextents<int, 2>, tensor_inline, scales_plane> right(
        weights, dextents<int, 2>(int(inner), int(cols)), array<int, 2>{1, int(inner)}, scales_plane(scales));
    tensor<device float, dextents<int, 2>, tensor_inline> out(
        destination, dextents<int, 2>(int(cols), 8), array<int, 2>{1, int(cols)});
    constexpr auto desc = matmul2d_descriptor(8, 32, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<1>> op;
    auto tile_left = left.slice(0, 0);
    auto tile_right = right.slice(0, int(tgid) * 32);
    auto tile_out = out.slice(int(tgid) * 32, 0);
    op.run(tile_left, tile_right, tile_out);
}

kernel void mm_m16_n128(
    device half* activation [[buffer(0)]],
    device uchar* weights [[buffer(1)]],
    device uchar* scales [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]])
{
    tensor<device half, dextents<int, 2>, tensor_inline> left(
        activation, dextents<int, 2>(int(inner), 16), array<int, 2>{1, int(inner)});
    tensor<device metal_fp8_e4m3_format, dextents<int, 2>, tensor_inline, scales_plane> right(
        weights, dextents<int, 2>(int(inner), int(cols)), array<int, 2>{1, int(inner)}, scales_plane(scales));
    tensor<device float, dextents<int, 2>, tensor_inline> out(
        destination, dextents<int, 2>(int(cols), 16), array<int, 2>{1, int(cols)});
    constexpr auto desc = matmul2d_descriptor(16, 128, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto tile_left = left.slice(0, 0);
    auto tile_right = right.slice(0, int(tgid) * 128);
    auto tile_out = out.slice(int(tgid) * 128, 0);
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
    // An empty or non-finite distribution is the end-of-sentence id.
    // Token 0 is the start-of-sentence marker, so it must not be the fallback.
    if (!(total > 0.0f)) {
        out[0] = 1u;
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
