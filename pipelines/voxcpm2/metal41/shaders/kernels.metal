#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>

using namespace metal;
using namespace mpp::tensor_ops;

constant int TILE_M = 64;
constant int TILE_N = 32;

// One thread per row. Variance is accumulated in float, then stored as half.
kernel void rmsnorm_row(
    device const half *hidden [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device half *out [[buffer(2)]],
    constant uint &width [[buffer(3)]],
    constant float &eps [[buffer(4)]],
    uint row [[thread_position_in_grid]])
{
    float sum = 0.0f;
    for (uint i = 0; i < width; ++i) {
        float value = float(hidden[row * width + i]);
        sum += value * value;
    }
    float scale = rsqrt(sum / float(width) + eps);
    for (uint i = 0; i < width; ++i) {
        uint index = row * width + i;
        out[index] = half(float(hidden[index]) * scale * float(weight[i]));
    }
}

// Half activations, a Linear weight stored [out, in], float destination.
// Rows are a multiple of 64, columns a multiple of 32, and the inner extent a multiple of 256.
kernel void matmul_f16(
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
    auto tile_left = left.slice(0, int(tgid.y) * TILE_M);
    auto tile_right = right.slice(0, int(tgid.x) * TILE_N);
    auto tile_out = out.slice(int(tgid.x) * TILE_N, int(tgid.y) * TILE_M);
    op.run(tile_left, tile_right, tile_out);
}

// Rotate-half RoPE. ``cos`` and ``sin`` are duplicated across the head, so the first half is enough.
kernel void rope_half(
    device const float* src [[buffer(0)]],
    device const float* cos_tab [[buffer(1)]],
    device const float* sin_tab [[buffer(2)]],
    device float* dst [[buffer(3)]],
    constant uint& heads [[buffer(4)]],
    constant uint& dim [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint pair = gid.x;
    uint head = gid.y;
    uint seq = gid.z;
    uint pairs = dim / 2;
    if (pair >= pairs || head >= heads) {
        return;
    }
    uint base = (seq * heads + head) * dim;
    float left = src[base + pair];
    float right = src[base + pairs + pair];
    float cos_value = cos_tab[seq * dim + pair];
    float sin_value = sin_tab[seq * dim + pair];
    dst[base + pair] = left * cos_value - right * sin_value;
    dst[base + pairs + pair] = right * cos_value + left * sin_value;
}

// Grouped-query attention. A causal query sees keys up through its own position.
kernel void gqa_attention(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device float* out [[buffer(3)]],
    device float* scratch [[buffer(4)]],
    constant uint& q_len [[buffer(5)]],
    constant uint& kv_len [[buffer(6)]],
    constant uint& heads [[buffer(7)]],
    constant uint& kv_heads [[buffer(8)]],
    constant uint& dim [[buffer(9)]],
    constant uint& causal [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint head = gid.x;
    uint qi = gid.y;
    if (head >= heads || qi >= q_len) {
        return;
    }
    uint group = heads / kv_heads;
    uint kv = head / group;
    float scale = rsqrt(float(dim));
    uint limit = kv_len;
    if (causal != 0) {
        limit = (kv_len - q_len) + qi + 1;
    }
    const device float* q = query + (qi * heads + head) * dim;
    device float* weights = scratch + (qi * heads + head) * kv_len;
    float max_score = -INFINITY;
    for (uint t = 0; t < limit; ++t) {
        const device float* k = key + (t * kv_heads + kv) * dim;
        float dot = 0.0f;
        for (uint d = 0; d < dim; ++d) {
            dot += q[d] * k[d];
        }
        float score = dot * scale;
        weights[t] = score;
        max_score = max(max_score, score);
    }
    float sum = 0.0f;
    for (uint t = 0; t < limit; ++t) {
        float weight = exp(weights[t] - max_score);
        weights[t] = weight;
        sum += weight;
    }
    float inv = 1.0f / max(sum, 1.0e-20f);
    for (uint d = 0; d < dim; ++d) {
        float acc = 0.0f;
        for (uint t = 0; t < limit; ++t) {
            float v = value[(t * kv_heads + kv) * dim + d];
            acc += weights[t] * inv * v;
        }
        out[(qi * heads + head) * dim + d] = acc;
    }
}

// Causal conv1d. The host left-pads by passing ``left``; the kernel reads zeros outside the input.
kernel void causal_conv1d(
    device const float* source [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& cin [[buffer(4)]],
    constant uint& cout [[buffer(5)]],
    constant uint& time [[buffer(6)]],
    constant uint& taps [[buffer(7)]],
    constant uint& dilation [[buffer(8)]],
    constant uint& left [[buffer(9)]],
    constant uint& groups [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint oc = gid.x;
    uint t = gid.y;
    if (oc >= cout || t >= time) {
        return;
    }
    uint cin_g = cin / groups;
    uint cout_g = cout / groups;
    uint group = oc / cout_g;
    float acc = bias[oc];
    for (uint ic = 0; ic < cin_g; ++ic) {
        uint in_c = group * cin_g + ic;
        for (uint tap = 0; tap < taps; ++tap) {
            int index = int(t) + int(tap) * int(dilation) - int(left);
            float value = 0.0f;
            if (index >= 0 && index < int(time)) {
                value = source[in_c * time + uint(index)];
            }
            acc += value * weight[(oc * cin_g + ic) * taps + tap];
        }
    }
    destination[oc * time + t] = acc;
}

// Causal transposed conv. Weight is [in, out, kernel]. The right crop is applied by writing only ``tout`` samples.
kernel void causal_transpose1d(
    device const float* source [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& cin [[buffer(4)]],
    constant uint& cout [[buffer(5)]],
    constant uint& tin [[buffer(6)]],
    constant uint& tout [[buffer(7)]],
    constant uint& taps [[buffer(8)]],
    constant uint& stride [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint oc = gid.x;
    uint t = gid.y;
    if (oc >= cout || t >= tout) {
        return;
    }
    float acc = bias[oc];
    for (uint ic = 0; ic < cin; ++ic) {
        for (uint tap = 0; tap < taps; ++tap) {
            if (t < tap || (t - tap) % stride != 0) {
                continue;
            }
            uint i = (t - tap) / stride;
            if (i >= tin) {
                continue;
            }
            acc += source[ic * tin + i] * weight[(ic * cout + oc) * taps + tap];
        }
    }
    destination[oc * tout + t] = acc;
}
