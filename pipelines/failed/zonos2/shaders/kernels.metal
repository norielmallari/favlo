#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

kernel void rms_norm(
    device const float *values [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    constant float &eps [[buffer(4)]],
    uint lane [[thread_position_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]])
{
    float sum = 0.0f;
    for (uint index = lane; index < count; index += lanes) {
        float value = values[index];
        sum += value * value;
    }
    threadgroup float partial[32];
    partial[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        float total = 0.0f;
        for (uint index = 0; index < lanes; ++index) {
            total += partial[index];
        }
        float scale = rsqrt(total / float(count) + eps);
        for (uint index = 0; index < count; ++index) {
            out[index] = values[index] * scale * weight[index];
        }
    }
}

kernel void rope_pairs(
    device const float *values [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    constant uint &position [[buffer(3)]],
    constant float &theta [[buffer(4)]],
    uint lane [[thread_position_in_grid]])
{
    if (lane >= count / 2) {
        return;
    }
    float freq = float(position) / pow(theta, (2.0f * float(lane)) / float(count));
    float cosine = cos(freq);
    float sine = sin(freq);
    float left = values[2 * lane];
    float right = values[2 * lane + 1];
    out[2 * lane] = left * cosine - right * sine;
    out[2 * lane + 1] = left * sine + right * cosine;
}

kernel void swiglu(
    device const float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint lane [[thread_position_in_grid]])
{
    float value = gate[lane];
    float activated = value / (1.0f + exp(-value));
    out[lane] = activated * up[lane];
}

kernel void softcap(
    device const float *values [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant float &tau [[buffer(2)]],
    uint lane [[thread_position_in_grid]])
{
    out[lane] = tau * tanh(values[lane] / tau);
}

kernel void gemm_f16(
    tensor<device half, dextents<int, 2>> A [[buffer(0)]],
    tensor<device half, dextents<int, 2>> B [[buffer(1)]],
    tensor<device float, dextents<int, 2>> C [[buffer(2)]],
    uint2 group [[threadgroup_position_in_grid]])
{
    constexpr auto desc = matmul2d_descriptor(
        64, 64, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto left = A.slice<dynamic_extent, 64>(0, int(group.y) * 64);
    auto right = B.slice<dynamic_extent, 64>(0, int(group.x) * 64);
    auto dest = C.slice<64, 64>(int(group.x) * 64, int(group.y) * 64);
    op.run(left, right, dest);
}

kernel void gemm_fp4(
    tensor<device half, dextents<int, 2>> A [[buffer(0)]],
    tensor<device metal::metal_fp4_e2m1_format, dextents<int, 2>, tensor_handle,
           tensor_blockwise<tensor_plane_scales, device metal::metal_fp8_ue8m0_format, 32, 1>> B [[buffer(1)]],
    tensor<device float, dextents<int, 2>> C [[buffer(2)]],
    uint2 group [[threadgroup_position_in_grid]])
{
    constexpr auto desc = matmul2d_descriptor(
        64, 64, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto left = A.slice<dynamic_extent, 64>(0, int(group.y) * 64);
    auto right = B.slice<dynamic_extent, 64>(0, int(group.x) * 64);
    auto dest = C.slice<64, 64>(int(group.x) * 64, int(group.y) * 64);
    op.run(left, right, dest);
}

kernel void gemm_fp8(
    tensor<device half, dextents<int, 2>> A [[buffer(0)]],
    tensor<device metal::metal_fp8_e4m3_format, dextents<int, 2>, tensor_handle,
           tensor_blockwise<tensor_plane_scales, device metal::metal_fp8_ue8m0_format, 32, 1>> B [[buffer(1)]],
    tensor<device float, dextents<int, 2>> C [[buffer(2)]],
    uint2 group [[threadgroup_position_in_grid]])
{
    constexpr auto desc = matmul2d_descriptor(
        64, 64, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto left = A.slice<dynamic_extent, 64>(0, int(group.y) * 64);
    auto right = B.slice<dynamic_extent, 64>(0, int(group.x) * 64);
    auto dest = C.slice<64, 64>(int(group.x) * 64, int(group.y) * 64);
    op.run(left, right, dest);
}

kernel void attend(
    device const float *query [[buffer(0)]],
    device const float *key [[buffer(1)]],
    device const float *value [[buffer(2)]],
    device float *out [[buffer(3)]],
    device float *scores [[buffer(4)]],
    constant uint &steps [[buffer(5)]],
    uint head [[thread_position_in_grid]])
{
    if (head >= 16) {
        return;
    }
    constexpr uint dim = 128;
    constexpr uint kv_heads = 4;
    uint kv = head / 4;
    float scale = rsqrt(float(dim));
    float peak = -INFINITY;
    for (uint step = 0; step < steps; ++step) {
        float dot = 0.0f;
        for (uint lane = 0; lane < dim; ++lane) {
            dot += query[head * dim + lane] * key[(step * kv_heads + kv) * dim + lane];
        }
        float score = dot * scale;
        scores[head * steps + step] = score;
        peak = max(peak, score);
    }
    float sum = 0.0f;
    for (uint step = 0; step < steps; ++step) {
        float weight = exp(scores[head * steps + step] - peak);
        scores[head * steps + step] = weight;
        sum += weight;
    }
    for (uint lane = 0; lane < dim; ++lane) {
        float acc = 0.0f;
        for (uint step = 0; step < steps; ++step) {
            acc += scores[head * steps + step] / sum * value[(step * kv_heads + kv) * dim + lane];
        }
        out[head * dim + lane] = acc;
    }
}
