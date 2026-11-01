
#include <metal_stdlib>
using namespace metal;

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

kernel void gemv_f32(
    device const float* activation [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& inner [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    if (row >= cols) {
        return;
    }
    device const float* rowp = weights + row * inner;
    float acc = 0.0f;
    for (uint i = lane; i < inner; i += 32u) {
        acc += rowp[i] * activation[i];
    }
    float sum = simd_sum(acc);
    if (lane == 0u) {
        destination[row] = sum;
    }
}

kernel void gdn_delta(
    device float* state [[buffer(0)]],
    device const float* query [[buffer(1)]],
    device const float* key [[buffer(2)]],
    device const float* value [[buffer(3)]],
    device const float* alpha [[buffer(4)]],
    device const float* beta [[buffer(5)]],
    device float* output [[buffer(6)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float key_h[128];
    threadgroup float query_h[128];
    if (tid < 128u) {
        key_h[tid] = key[head * 128u + tid];
        query_h[tid] = query[head * 128u + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float* shard = state + head * 128u * 128u;
    float dot = 0.0f;
    for (uint dk = 0u; dk < 128u; ++dk) {
        dot += shard[dk * 128u + tid] * key_h[dk];
    }
    float decay = alpha[head];
    float error = value[head * 128u + tid] - decay * dot;
    float write = beta[head];
    for (uint dk = 0u; dk < 128u; ++dk) {
        uint index = dk * 128u + tid;
        shard[index] = decay * shard[index] + write * key_h[dk] * error;
    }
    float acc = 0.0f;
    for (uint dk = 0u; dk < 128u; ++dk) {
        acc += shard[dk * 128u + tid] * query_h[dk];
    }
    output[head * 128u + tid] = acc;
}

kernel void attend_gqa(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device float* scores [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& tokens [[buffer(5)]],
    constant uint& kv_heads [[buffer(6)]],
    constant uint& dim [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint kv = head / (24u / kv_heads);
    device const float* qh = query + head * dim;
    for (uint t = 0u; t < tokens; ++t) {
        device const float* kt = key + (t * kv_heads + kv) * dim;
        float acc = 0.0f;
        for (uint d = lane; d < dim; d += 32u) {
            acc += qh[d] * kt[d];
        }
        float score = simd_sum(acc) * scale;
        if (lane == 0u) {
            scores[head * tokens + t] = score;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    float maxs = -INFINITY;
    for (uint t = 0u; t < tokens; ++t) {
        maxs = max(maxs, scores[head * tokens + t]);
    }
    if (lane == 0u) {
        for (uint t = 0u; t < tokens; ++t) {
            scores[head * tokens + t] = exp(scores[head * tokens + t] - maxs);
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    float denom = 0.0f;
    for (uint t = 0u; t < tokens; ++t) {
        denom += scores[head * tokens + t];
    }
    for (uint d = lane; d < dim; d += 32u) {
        float acc = 0.0f;
        for (uint t = 0u; t < tokens; ++t) {
            device const float* vt = value + (t * kv_heads + kv) * dim;
            acc += scores[head * tokens + t] / denom * vt[d];
        }
        output[head * dim + d] = acc;
    }
}
