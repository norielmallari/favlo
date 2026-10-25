#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>

using namespace metal;
using namespace mpp::tensor_ops;

constant int TILE_M = 64;
constant int TILE_N = 128;

kernel void quantize_i8(
    device const float* src [[buffer(0)]],
    device int8_t* dst [[buffer(1)]],
    device float* factor [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& logical_k [[buffer(4)]],
    constant uint& stored_k [[buffer(5)]],
    constant uint& src_cols [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]])
{
    if (row >= rows) {
        return;
    }
    float peak = 0.0f;
    for (uint col = lane; col < logical_k; col += 128u) {
        peak = max(peak, abs(src[row * src_cols + col]));
    }
    float grouped = simd_max(peak);
    threadgroup float parts[4];
    if ((lane & 31u) == 0u) {
        parts[lane >> 5] = grouped;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = max(max(parts[0], parts[1]), max(parts[2], parts[3]));
    float scale = total > 0.0f ? total / 127.0f : 0.0f;
    float inv = total > 0.0f ? 127.0f / total : 0.0f;
    if (lane == 0u) {
        factor[row] = scale;
    }
    for (uint col = lane; col < stored_k; col += 128u) {
        float value = 0.0f;
        if (col < logical_k) {
            value = src[row * src_cols + col] * inv;
        }
        int code = int(rint(value));
        code = clamp(code, -127, 127);
        dst[row * stored_k + col] = int8_t(code);
    }
}

kernel void matmul_i8(
    device int8_t* activation [[buffer(0)]],
    device int8_t* weights [[buffer(1)]],
    device int32_t* destination [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    tensor<device int8_t, dextents<int, 2>, tensor_inline> left(
        activation,
        dextents<int, 2>(int(inner), int(rows)),
        array<int, 2>{1, int(inner)});
    tensor<device int8_t, dextents<int, 2>, tensor_inline> right(
        weights,
        dextents<int, 2>(int(inner), int(cols)),
        array<int, 2>{1, int(inner)});
    tensor<device int32_t, dextents<int, 2>, tensor_inline> out(
        destination,
        dextents<int, 2>(int(cols), int(rows)),
        array<int, 2>{1, int(cols)});
    constexpr auto desc = matmul2d_descriptor(
        64, 128, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto tile_left = left.slice(0, int(tgid.y) * TILE_M);
    auto tile_right = right.slice(0, int(tgid.x) * TILE_N);
    auto tile_out = out.slice(int(tgid.x) * TILE_N, int(tgid.y) * TILE_M);
    op.run(tile_left, tile_right, tile_out);
}

kernel void scale_i8(
    device const int* acc [[buffer(0)]],
    device const float* row_scale [[buffer(1)]],
    device const float* col_scale [[buffer(2)]],
    device const half* bias [[buffer(3)]],
    device float* dest [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& cols [[buffer(6)]],
    constant uint& logical [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    uint n = rows * cols;
    if (gid >= n) {
        return;
    }
    uint col = gid % cols;
    uint row = gid / cols;
    float value = float(acc[gid]) * row_scale[row] * col_scale[col];
    if (col < logical) {
        value += float(bias[col]);
    } else {
        value = 0.0f;
    }
    dest[gid] = value;
}

kernel void embed_f32(
    device const half* word [[buffer(0)]],
    device const half* position [[buffer(1)]],
    device const half* type_row [[buffer(2)]],
    device const int* ids [[buffer(3)]],
    device const int* positions [[buffer(4)]],
    device float* hidden [[buffer(5)]],
    constant uint& length [[buffer(6)]],
    constant uint& width [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    uint n = length * width;
    if (gid >= n) {
        return;
    }
    uint row = gid / width;
    uint col = gid - row * width;
    int id = ids[row];
    int pos = positions[row];
    float value = float(word[id * width + col]);
    value += float(position[pos * width + col]);
    value += float(type_row[col]);
    hidden[gid] = value;
}

kernel void layernorm_affine(
    device const float* src [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device float* dest [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]])
{
    if (row >= rows) {
        return;
    }
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (uint col = lane; col < width; col += 128u) {
        float value = src[row * width + col];
        sum += value;
        sum_sq += value * value;
    }
    float grouped = simd_sum(sum);
    float grouped_sq = simd_sum(sum_sq);
    threadgroup float parts[4];
    threadgroup float parts_sq[4];
    if ((lane & 31u) == 0u) {
        parts[lane >> 5] = grouped;
        parts_sq[lane >> 5] = grouped_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = parts[0] + parts[1] + parts[2] + parts[3];
    float total_sq = parts_sq[0] + parts_sq[1] + parts_sq[2] + parts_sq[3];
    float mean = total / float(width);
    float var = total_sq / float(width) - mean * mean;
    float inv = rsqrt(max(var, 0.0f) + eps);
    for (uint col = lane; col < width; col += 128u) {
        float value = src[row * width + col];
        dest[row * width + col] = (value - mean) * inv * float(weight[col]) + float(bias[col]);
    }
}

kernel void residual_norm(
    device const float* left [[buffer(0)]],
    device const float* right [[buffer(1)]],
    device const half* weight [[buffer(2)]],
    device const half* bias [[buffer(3)]],
    device float* dest [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& width [[buffer(6)]],
    constant uint& right_cols [[buffer(7)]],
    constant float& eps [[buffer(8)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]])
{
    if (row >= rows) {
        return;
    }
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (uint col = lane; col < width; col += 128u) {
        float value = left[row * width + col] + right[row * right_cols + col];
        sum += value;
        sum_sq += value * value;
    }
    float grouped = simd_sum(sum);
    float grouped_sq = simd_sum(sum_sq);
    threadgroup float parts[4];
    threadgroup float parts_sq[4];
    if ((lane & 31u) == 0u) {
        parts[lane >> 5] = grouped;
        parts_sq[lane >> 5] = grouped_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = parts[0] + parts[1] + parts[2] + parts[3];
    float total_sq = parts_sq[0] + parts_sq[1] + parts_sq[2] + parts_sq[3];
    float mean = total / float(width);
    float var = total_sq / float(width) - mean * mean;
    float inv = rsqrt(max(var, 0.0f) + eps);
    for (uint col = lane; col < width; col += 128u) {
        float value = left[row * width + col] + right[row * right_cols + col];
        dest[row * width + col] = (value - mean) * inv * float(weight[col]) + float(bias[col]);
    }
}

static float erf_as(float x) {
    float ax = fabs(x);
    float t = 1.0f / (1.0f + 0.3275911f * ax);
    float poly = 0.254829592f + t * (-0.284496736f + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f)));
    float value = 1.0f - t * poly * exp(-ax * ax);
    return copysign(value, x);
}

kernel void gelu_erf(
    device float* values [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) {
        return;
    }
    float value = values[gid];
    values[gid] = 0.5f * value * (1.0f + erf_as(value * 0.7071067811865476f));
}

kernel void attend_head(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device const float* mask [[buffer(3)]],
    device float* dest [[buffer(4)]],
    constant uint& length [[buffer(5)]],
    constant uint& hidden [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 lane_id [[thread_position_in_threadgroup]])
{
    uint row = tgid.x;
    uint head = tgid.y;
    uint lane = lane_id.x;
    if (row >= length || lane >= 32u) {
        return;
    }
    uint base = head * 64u;
    float q0 = query[row * hidden + base + lane];
    float q1 = query[row * hidden + base + 32u + lane];
    float peak = -INFINITY;
    for (uint key_index = 0u; key_index < length; ++key_index) {
        float part = q0 * key[key_index * hidden + base + lane];
        part += q1 * key[key_index * hidden + base + 32u + lane];
        float dot = simd_sum(part) * 0.125f + mask[key_index];
        peak = max(peak, dot);
    }
    float ceiling = simd_max(peak);
    float sum = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint key_index = 0u; key_index < length; ++key_index) {
        float part = q0 * key[key_index * hidden + base + lane];
        part += q1 * key[key_index * hidden + base + 32u + lane];
        float dot = simd_sum(part) * 0.125f + mask[key_index];
        float weight = exp(dot - ceiling);
        sum += weight;
        acc0 += weight * value[key_index * hidden + base + lane];
        acc1 += weight * value[key_index * hidden + base + 32u + lane];
    }
    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    dest[row * hidden + base + lane] = acc0 * inv;
    dest[row * hidden + base + 32u + lane] = acc1 * inv;
}

kernel void copy_f32(
    device const float* src [[buffer(0)]],
    device float* dest [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) {
        dest[gid] = src[gid];
    }
}
