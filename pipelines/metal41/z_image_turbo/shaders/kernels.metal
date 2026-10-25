#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>

using namespace metal;
using namespace mpp::tensor_ops;

using scales_plane = tensor_blockwise<tensor_plane_scales,
                                      device metal_fp8_ue8m0_format,
                                      32, 1>;

constant float FP4_MAG[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

// tile_m 64 divides the 4096 image tokens. tile_n 32 divides the linear rows
// except conv_out, which is padded up to 32.
constant int TILE_M = 64;
constant int TILE_N = 32;

kernel void matmul_fp4(
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
    auto tile_left = left.slice(0, int(tgid.y) * TILE_M);
    auto tile_right = right.slice(0, int(tgid.x) * TILE_N);
    auto tile_out = out.slice(int(tgid.x) * TILE_N, int(tgid.y) * TILE_M);
    op.run(tile_left, tile_right, tile_out);
}

constant int TILE_N_I8 = 128;

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

kernel void quantize_half_i8(
    device const half* src [[buffer(0)]],
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
        peak = max(peak, abs(float(src[row * src_cols + col])));
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
            value = float(src[row * src_cols + col]) * inv;
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
    auto tile_right = right.slice(0, int(tgid.x) * TILE_N_I8);
    auto tile_out = out.slice(int(tgid.x) * TILE_N_I8, int(tgid.y) * TILE_M);
    op.run(tile_left, tile_right, tile_out);
}

kernel void scale_i8(
    device const int* acc [[buffer(0)]],
    device const float* row_scale [[buffer(1)]],
    device const float* col_scale [[buffer(2)]],
    device const half* bias [[buffer(3)]],
    device float* destination [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& padded [[buffer(6)]],
    constant uint& logical [[buffer(7)]],
    constant uint& dest_stride [[buffer(8)]],
    constant uint& has_bias [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{
    uint n = rows * padded;
    if (gid >= n) {
        return;
    }
    uint col = gid % padded;
    uint row = gid / padded;
    if (col >= logical) {
        return;
    }
    float value = float(acc[gid]) * row_scale[row] * col_scale[col];
    if (has_bias != 0u) {
        value += float(bias[col]);
    }
    destination[row * dest_stride + col] = value;
}

kernel void gather_i8(
    device const uint* ids [[buffer(0)]],
    device const int8_t* table [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& seq [[buffer(4)]],
    constant uint& limit [[buffer(5)]],
    constant uint& hidden [[buffer(6)]],
    constant uint& stored [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= hidden || gid.y >= seq) {
        return;
    }
    if (gid.y >= limit) {
        destination[gid.y * hidden + gid.x] = 0.0f;
        return;
    }
    uint row = ids[gid.y];
    destination[gid.y * hidden + gid.x] = float(table[row * stored + gid.x]) * scales[row];
}

kernel void zero_bytes(
    device uchar* data [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) {
        data[gid] = 0;
    }
}

kernel void rmsnorm(
    device const float* x [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]])
{
    if (row >= rows) {
        return;
    }
    float sum = 0.0f;
    for (uint i = lane; i < width; i += 128u) {
        float v = float(x[row * width + i]);
        sum += v * v;
    }
    float grouped = simd_sum(sum);
    threadgroup float parts[4];
    if ((lane & 31u) == 0u) {
        parts[lane >> 5] = grouped;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = parts[0] + parts[1] + parts[2] + parts[3];
    float inv = rsqrt(total / float(width) + eps);
    for (uint i = lane; i < width; i += 128u) {
        y[row * width + i] = x[row * width + i] * inv * float(weight[i]);
    }
}

kernel void layernorm(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]])
{
    if (row >= rows) {
        return;
    }
    float sum = 0.0f;
    for (uint i = lane; i < width; i += 128u) {
        sum += x[row * width + i];
    }
    float grouped = simd_sum(sum);
    threadgroup float parts[4];
    if ((lane & 31u) == 0u) {
        parts[lane >> 5] = grouped;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = (parts[0] + parts[1] + parts[2] + parts[3]) / float(width);
    float var = 0.0f;
    for (uint i = lane; i < width; i += 128u) {
        float delta = x[row * width + i] - mean;
        var += delta * delta;
    }
    float grouped_var = simd_sum(var);
    threadgroup float parts2[4];
    if ((lane & 31u) == 0u) {
        parts2[lane >> 5] = grouped_var;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = parts2[0] + parts2[1] + parts2[2] + parts2[3];
    float inv = rsqrt(total / float(width) + eps);
    for (uint i = lane; i < width; i += 128u) {
        y[row * width + i] = (x[row * width + i] - mean) * inv;
    }
}

kernel void rope_axes(
    device float* values [[buffer(0)]],
    device const int* pos [[buffer(1)]],
    constant uint& seq [[buffer(2)]],
    constant uint& heads [[buffer(3)]],
    constant uint& dim [[buffer(4)]],
    constant float& theta [[buffer(5)]],
    constant uint& axis0 [[buffer(6)]],
    constant uint& axis1 [[buffer(7)]],
    constant uint& axis2 [[buffer(8)]],
    constant uint& adjacent [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{
    uint span = seq * heads;
    if (gid >= span) {
        return;
    }
    uint token = gid / heads;
    device float* row = values + gid * dim;
    uint axis_dim[3] = {axis0, axis1, axis2};
    uint base = 0;
    for (uint axis = 0; axis < 3u; ++axis) {
        uint width = axis_dim[axis];
        if (width == 0u) {
            continue;
        }
        uint half_n = width >> 1;
        float position = float(pos[token * 3u + axis]);
        for (uint i = 0; i < half_n; ++i) {
            float freq = 1.0f / pow(theta, (2.0f * float(i)) / float(width));
            float angle = position * freq;
            float c = cos(angle);
            float s = sin(angle);
            if (adjacent != 0u) {
                uint i0 = base + (i << 1);
                float x0 = row[i0];
                float x1 = row[i0 + 1u];
                row[i0] = x0 * c - x1 * s;
                row[i0 + 1u] = x0 * s + x1 * c;
            } else {
                float x0 = row[base + i];
                float x1 = row[base + half_n + i];
                row[base + i] = x0 * c - x1 * s;
                row[base + half_n + i] = x1 * c + x0 * s;
            }
        }
        base += width;
    }
}

kernel void attend(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& seq [[buffer(4)]],
    constant uint& limit [[buffer(5)]],
    constant uint& q_heads [[buffer(6)]],
    constant uint& kv_heads [[buffer(7)]],
    constant uint& head_dim [[buffer(8)]],
    constant uint& causal [[buffer(9)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 lane_id [[thread_position_in_threadgroup]])
{
    uint lane = lane_id.x;
    uint head = tgid.x;
    uint token = tgid.y;
    uint span = head_dim >> 5;
    device float* out = destination + (token * q_heads + head) * head_dim;
    if (token >= limit || head >= q_heads) {
        if (token < seq && head < q_heads) {
            for (uint i = 0; i < span; ++i) {
                out[lane * span + i] = 0.0f;
            }
        }
        return;
    }
    uint group = q_heads / kv_heads;
    uint kv = head / group;
    float qv[16];
    for (uint i = 0; i < span; ++i) {
        qv[i] = query[(token * q_heads + head) * head_dim + lane * span + i];
    }
    float acc[16];
    for (uint i = 0; i < span; ++i) {
        acc[i] = 0.0f;
    }
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float scale = rsqrt(float(head_dim));
    for (uint key_index = 0; key_index < limit; ++key_index) {
        if (causal != 0u && key_index > token) {
            break;
        }
        float dot = 0.0f;
        uint key_base = (key_index * kv_heads + kv) * head_dim + lane * span;
        for (uint i = 0; i < span; ++i) {
            dot += qv[i] * key[key_base + i];
        }
        dot = simd_sum(dot) * scale;
        float next_max = max(running_max, dot);
        float alpha = exp(running_max - next_max);
        float weight = exp(dot - next_max);
        running_sum = running_sum * alpha + weight;
        uint value_base = (key_index * kv_heads + kv) * head_dim + lane * span;
        for (uint i = 0; i < span; ++i) {
            acc[i] = acc[i] * alpha + weight * value[value_base + i];
        }
        running_max = next_max;
    }
    float inv = 1.0f / running_sum;
    for (uint i = 0; i < span; ++i) {
        out[lane * span + i] = acc[i] * inv;
    }
}

// Eight queries share four keys. The four dots land in registers, then the
// softmax rescales once. Head dim is 128. The decoder head is 512 and stays
// on `attend`.
constant uint Q_BLOCK = 8u;

static float4 _lane4(
    device const float* src, uint token, uint heads, uint kv, uint lane)
{
    uint base = (token * heads + kv) * 128u + lane * 4u;
    return float4(src[base], src[base + 1u], src[base + 2u], src[base + 3u]);
}

static float _dot4(thread float* q, float4 k)
{
    return q[0] * k.x + q[1] * k.y + q[2] * k.z + q[3] * k.w;
}

static void _mix4(thread float* acc, thread float& sum, float dot, float next_max, float4 v)
{
    float weight = exp(dot - next_max);
    sum += weight;
    acc[0] += weight * v.x;
    acc[1] += weight * v.y;
    acc[2] += weight * v.z;
    acc[3] += weight * v.w;
}

// One query tile of 32 and one key tile of 32. Q, K, and V stay float.
// Scores stay in the threadgroup. Head dim 512 stays on `attend`.
constant int FQ = 32;
constant int FK = 32;
constant int FD = 128;

kernel void attend_flash(
    device float* query [[buffer(0)]],
    device float* key [[buffer(1)]],
    device float* value [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& seq [[buffer(4)]],
    constant uint& limit [[buffer(5)]],
    constant uint& q_heads [[buffer(6)]],
    constant uint& kv_heads [[buffer(7)]],
    constant uint& head_dim [[buffer(8)]],
    constant uint& causal [[buffer(9)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    uint head = tgid.x;
    uint q0 = tgid.y * uint(FQ);
    if (head >= q_heads || head_dim != uint(FD) || q0 >= seq) {
        return;
    }
    if (q0 >= limit) {
        for (uint i = tid; i < uint(FQ) * uint(FD); i += 128u) {
            uint r = i / uint(FD);
            uint c = i - r * uint(FD);
            uint q = q0 + r;
            if (q < seq) {
                destination[(q * q_heads + head) * uint(FD) + c] = 0.0f;
            }
        }
        return;
    }
    uint group = q_heads / kv_heads;
    uint kv = head / group;
    float scale = rsqrt(float(FD));
    threadgroup float scores[FQ * FK];
    threadgroup float prob[FQ * FK];
    threadgroup float mrun[FQ];
    threadgroup float lrun[FQ];
    threadgroup float alpha_row[FQ];
    if (tid < uint(FQ)) {
        mrun[tid] = -INFINITY;
        lrun[tid] = 0.0f;
    }
    device float* qbase = query + head * uint(FD);
    device float* kbase = key + kv * uint(FD);
    device float* vbase = value + kv * uint(FD);
    int qstride = int(q_heads * uint(FD));
    int kstride = int(kv_heads * uint(FD));
    tensor<device float, dextents<int, 2>, tensor_inline> queries(
        qbase,
        dextents<int, 2>(FD, int(seq)),
        array<int, 2>{1, qstride});
    tensor<device float, dextents<int, 2>, tensor_inline> keys(
        kbase,
        dextents<int, 2>(FD, int(seq)),
        array<int, 2>{1, kstride});
    tensor<device float, dextents<int, 2>, tensor_inline> values(
        vbase,
        dextents<int, 2>(FD, int(seq)),
        array<int, 2>{1, kstride});
    auto tile_q = queries.slice(0, int(q0));
    constexpr auto qk_desc = matmul2d_descriptor(FQ, FK, FD, false, true, false);
    matmul2d<qk_desc, execution_simdgroups<4>> qk_op;
    constexpr auto pv_desc = matmul2d_descriptor(FQ, FD, FK, false, false, false);
    matmul2d<pv_desc, execution_simdgroups<4>> pv_op;
    tensor<threadgroup float, dextents<int, 2>, tensor_inline> tile_p(
        prob, dextents<int, 2>(FK, FQ));
    float oreg[32];
    for (int i = 0; i < 32; ++i) {
        oreg[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k0 = 0u; k0 < limit; k0 += uint(FK)) {
        auto tile_k = keys.slice(0, int(k0));
        auto score_tile = qk_op.get_destination_cooperative_tensor<decltype(tile_q), decltype(tile_k), float>();
        qk_op.run(tile_q, tile_k, score_tile);
        for (uint16_t i = 0u; i < score_tile.get_capacity(); ++i) {
            if (!score_tile.is_valid_element(i)) {
                continue;
            }
            auto ix = score_tile.get_multidimensional_index(i);
            uint c = uint(ix[0]);
            uint r = uint(ix[1]);
            uint q = q0 + r;
            uint key_index = k0 + c;
            float dot = -INFINITY;
            bool live = r < uint(FQ) && c < uint(FK) && q < limit && key_index < limit;
            if (causal != 0u && key_index > q) {
                live = false;
            }
            if (live) {
                dot = score_tile[i] * scale;
            }
            scores[r * uint(FK) + c] = dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < uint(FQ)) {
            float row_max = -INFINITY;
            for (uint c = 0u; c < uint(FK); ++c) {
                row_max = max(row_max, scores[tid * uint(FK) + c]);
            }
            float next = max(mrun[tid], row_max);
            float rescale = 0.0f;
            if (mrun[tid] != -INFINITY) {
                rescale = exp(mrun[tid] - next);
            }
            alpha_row[tid] = rescale;
            float sum = 0.0f;
            for (uint c = 0u; c < uint(FK); ++c) {
                float weight = 0.0f;
                float dot = scores[tid * uint(FK) + c];
                if (dot != -INFINITY) {
                    weight = exp(dot - next);
                }
                prob[tid * uint(FK) + c] = weight;
                sum += weight;
            }
            lrun[tid] = lrun[tid] * rescale + sum;
            mrun[tid] = next;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto tile_v = values.slice(0, int(k0));
        auto mixed = pv_op.get_destination_cooperative_tensor<decltype(tile_p), decltype(tile_v), float>();
        pv_op.run(tile_p, tile_v, mixed);
        for (uint16_t i = 0u; i < mixed.get_capacity(); ++i) {
            if (!mixed.is_valid_element(i) || i >= 32u) {
                continue;
            }
            auto ix = mixed.get_multidimensional_index(i);
            uint r = uint(ix[1]);
            oreg[i] = oreg[i] * alpha_row[r] + mixed[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto tile_v0 = values.slice(0, 0);
    auto layout = pv_op.get_destination_cooperative_tensor<decltype(tile_p), decltype(tile_v0), float>();
    for (uint16_t i = 0u; i < layout.get_capacity(); ++i) {
        if (!layout.is_valid_element(i) || i >= 32u) {
            continue;
        }
        auto ix = layout.get_multidimensional_index(i);
        uint c = uint(ix[0]);
        uint r = uint(ix[1]);
        uint q = q0 + r;
        if (q >= seq || c >= uint(FD)) {
            continue;
        }
        device float* out = destination + (q * q_heads + head) * uint(FD) + c;
        if (q >= limit || lrun[r] == 0.0f) {
            *out = 0.0f;
        } else {
            *out = oreg[i] / lrun[r];
        }
    }
}

kernel void attend_q4(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& seq [[buffer(4)]],
    constant uint& limit [[buffer(5)]],
    constant uint& q_heads [[buffer(6)]],
    constant uint& kv_heads [[buffer(7)]],
    constant uint& head_dim [[buffer(8)]],
    constant uint& causal [[buffer(9)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 lane_id [[thread_position_in_threadgroup]])
{
    uint lane = lane_id.x;
    uint head = tgid.x;
    uint q0 = tgid.y * Q_BLOCK;
    if (head >= q_heads || head_dim != 128u) {
        return;
    }
    uint group = q_heads / kv_heads;
    uint kv = head / group;
    float scale = rsqrt(float(head_dim));
    float qv[Q_BLOCK][4];
    float acc[Q_BLOCK][4];
    float running_max[Q_BLOCK];
    float running_sum[Q_BLOCK];
    uint any_live = 0u;
    for (uint block = 0u; block < Q_BLOCK; ++block) {
        uint token = q0 + block;
        running_max[block] = -INFINITY;
        running_sum[block] = 0.0f;
        for (uint i = 0u; i < 4u; ++i) {
            acc[block][i] = 0.0f;
            qv[block][i] = 0.0f;
        }
        if (token < limit) {
            any_live = 1u;
            uint base = (token * q_heads + head) * head_dim + lane * 4u;
            for (uint i = 0u; i < 4u; ++i) {
                qv[block][i] = query[base + i];
            }
        }
    }
    if (any_live != 0u) {
        uint key_index = 0u;
        for (; key_index + 4u <= limit; key_index += 4u) {
            float4 k0 = _lane4(key, key_index, kv_heads, kv, lane);
            float4 k1 = _lane4(key, key_index + 1u, kv_heads, kv, lane);
            float4 k2 = _lane4(key, key_index + 2u, kv_heads, kv, lane);
            float4 k3 = _lane4(key, key_index + 3u, kv_heads, kv, lane);
            float4 v0 = _lane4(value, key_index, kv_heads, kv, lane);
            float4 v1 = _lane4(value, key_index + 1u, kv_heads, kv, lane);
            float4 v2 = _lane4(value, key_index + 2u, kv_heads, kv, lane);
            float4 v3 = _lane4(value, key_index + 3u, kv_heads, kv, lane);
            for (uint block = 0u; block < Q_BLOCK; ++block) {
                uint token = q0 + block;
                if (token >= limit || (causal != 0u && key_index > token)) {
                    continue;
                }
                float d0 = simd_sum(_dot4(qv[block], k0)) * scale;
                float d1 = simd_sum(_dot4(qv[block], k1)) * scale;
                float d2 = simd_sum(_dot4(qv[block], k2)) * scale;
                float d3 = simd_sum(_dot4(qv[block], k3)) * scale;
                if (causal != 0u) {
                    if (key_index > token) {
                        d0 = -INFINITY;
                    }
                    if (key_index + 1u > token) {
                        d1 = -INFINITY;
                    }
                    if (key_index + 2u > token) {
                        d2 = -INFINITY;
                    }
                    if (key_index + 3u > token) {
                        d3 = -INFINITY;
                    }
                }
                float tile_max = max(max(d0, d1), max(d2, d3));
                float next_max = max(running_max[block], tile_max);
                float alpha = exp(running_max[block] - next_max);
                running_sum[block] *= alpha;
                for (uint i = 0u; i < 4u; ++i) {
                    acc[block][i] *= alpha;
                }
                _mix4(acc[block], running_sum[block], d0, next_max, v0);
                _mix4(acc[block], running_sum[block], d1, next_max, v1);
                _mix4(acc[block], running_sum[block], d2, next_max, v2);
                _mix4(acc[block], running_sum[block], d3, next_max, v3);
                running_max[block] = next_max;
            }
        }
        for (; key_index < limit; ++key_index) {
            float4 kk = _lane4(key, key_index, kv_heads, kv, lane);
            float4 vv = _lane4(value, key_index, kv_heads, kv, lane);
            for (uint block = 0u; block < Q_BLOCK; ++block) {
                uint token = q0 + block;
                if (token >= limit || (causal != 0u && key_index > token)) {
                    continue;
                }
                float dot = simd_sum(_dot4(qv[block], kk)) * scale;
                float next_max = max(running_max[block], dot);
                float alpha = exp(running_max[block] - next_max);
                running_sum[block] *= alpha;
                for (uint i = 0u; i < 4u; ++i) {
                    acc[block][i] *= alpha;
                }
                _mix4(acc[block], running_sum[block], dot, next_max, vv);
                running_max[block] = next_max;
            }
        }
    }
    for (uint block = 0u; block < Q_BLOCK; ++block) {
        uint token = q0 + block;
        if (token >= seq) {
            continue;
        }
        device float* out = destination + (token * q_heads + head) * head_dim;
        if (token >= limit || running_sum[block] == 0.0f) {
            for (uint i = 0u; i < 4u; ++i) {
                out[lane * 4u + i] = 0.0f;
            }
            continue;
        }
        float inv = 1.0f / running_sum[block];
        for (uint i = 0u; i < 4u; ++i) {
            out[lane * 4u + i] = acc[block][i] * inv;
        }
    }
}

kernel void scale_to_half(
    device const float* source [[buffer(0)]],
    device half* destination [[buffer(1)]],
    device float* inv_scale [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& inner [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]])
{
    if (row >= rows) {
        return;
    }
    float peak = 0.0f;
    for (uint i = lane; i < inner; i += 128u) {
        peak = max(peak, fabs(source[row * inner + i]));
    }
    float grouped = simd_max(peak);
    threadgroup float parts[4];
    if ((lane & 31u) == 0u) {
        parts[lane >> 5] = grouped;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m = max(max(parts[0], parts[1]), max(parts[2], parts[3]));
    float s = m > 60000.0f ? 60000.0f / m : 1.0f;
    if (lane == 0u) {
        inv_scale[row] = 1.0f / s;
    }
    for (uint i = lane; i < inner; i += 128u) {
        destination[row * inner + i] = half(source[row * inner + i] * s);
    }
}

kernel void bias_cast(
    device const float* source [[buffer(0)]],
    device const half* bias [[buffer(1)]],
    device float* destination [[buffer(2)]],
    device const float* inv_scale [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    constant uint& cols_padded [[buffer(6)]],
    constant uint& has_bias [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= cols || gid.y >= rows) {
        return;
    }
    float value = source[gid.y * cols_padded + gid.x] * inv_scale[gid.y];
    if (has_bias != 0u) {
        value += float(bias[gid.x]);
    }
    destination[gid.y * cols + gid.x] = value;
}

kernel void channel_scale(
    device const float* source [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= width || gid.y >= rows) {
        return;
    }
    float value = source[gid.y * width + gid.x];
    destination[gid.y * width + gid.x] = value * (1.0f + scale[gid.x]);
}

kernel void channel_gate(
    device const float* residual [[buffer(0)]],
    device const float* branch [[buffer(1)]],
    device const float* gate [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= width || gid.y >= rows) {
        return;
    }
    uint index = gid.y * width + gid.x;
    float value = residual[index] + tanh(gate[gid.x]) * branch[index];
    destination[index] = value;
}

kernel void add_rows(
    device const float* left [[buffer(0)]],
    device const float* right [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) {
        destination[gid] = left[gid] + right[gid];
    }
}

kernel void silu_mul(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) {
        return;
    }
    float g = gate[gid];
    float activated = g / (1.0f + exp(-g));
    destination[gid] = activated * up[gid];
}

kernel void gather_fp4(
    device const uint* ids [[buffer(0)]],
    device const uchar* table [[buffer(1)]],
    device const uchar* scales [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& seq [[buffer(4)]],
    constant uint& limit [[buffer(5)]],
    constant uint& hidden [[buffer(6)]],
    constant uint& stored [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= hidden || gid.y >= seq) {
        return;
    }
    if (gid.y >= limit) {
        destination[gid.y * hidden + gid.x] = 0.0f;
        return;
    }
    uint row = ids[gid.y];
    uint col = gid.x;
    uint byte = table[row * (stored >> 1) + (col >> 1)];
    uint nibble = (col & 1u) ? (byte >> 4) : (byte & 0x0fu);
    float mag = FP4_MAG[nibble & 7u];
    float sign = (nibble & 8u) ? -1.0f : 1.0f;
    float scale = exp2(float(scales[row * (stored >> 5) + (col >> 5)]) - 127.0f);
    destination[gid.y * hidden + gid.x] = sign * mag * scale;
}

kernel void group_norm(
    device const float* source [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& channels [[buffer(4)]],
    constant uint& height [[buffer(5)]],
    constant uint& width [[buffer(6)]],
    constant uint& groups [[buffer(7)]],
    constant float& eps [[buffer(8)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]])
{
    uint per = channels / groups;
    uint spatial = height * width;
    uint count = per * spatial;
    float sum = 0.0f;
    float sumsq = 0.0f;
    for (uint i = lane; i < count; i += 128u) {
        uint c = group * per + i / spatial;
        uint p = i - (i / spatial) * spatial;
        float v = source[(c * height + p / width) * width + (p - (p / width) * width)];
        sum += v;
        sumsq += v * v;
    }
    float gsum = simd_sum(sum);
    float gsq = simd_sum(sumsq);
    threadgroup float parts[4];
    threadgroup float parts2[4];
    if ((lane & 31u) == 0u) {
        parts[lane >> 5] = gsum;
        parts2[lane >> 5] = gsq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = parts[0] + parts[1] + parts[2] + parts[3];
    float total2 = parts2[0] + parts2[1] + parts2[2] + parts2[3];
    float mean = total / float(count);
    float var = total2 / float(count) - mean * mean;
    float inv = rsqrt(var + eps);
    for (uint i = lane; i < count; i += 128u) {
        uint local = i / spatial;
        uint c = group * per + local;
        uint p = i - local * spatial;
        uint y = p / width;
        uint x = p - y * width;
        uint index = (c * height + y) * width + x;
        float v = (source[index] - mean) * inv;
        destination[index] = v * float(weight[c]) + float(bias[c]);
    }
}

kernel void im2col(
    device const float* source [[buffer(0)]],
    device half* destination [[buffer(1)]],
    constant uint& channels [[buffer(2)]],
    constant uint& height [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    constant uint& ksize [[buffer(5)]],
    constant uint& pad [[buffer(6)]],
    constant uint& stored [[buffer(7)]],
    constant uint& pixel0 [[buffer(8)]],
    constant uint& pixels [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= pixels) {
        return;
    }
    uint pixel = pixel0 + gid;
    uint y = pixel / width;
    uint x = pixel - y * width;
    device half* row = destination + gid * stored;
    uint logical = channels * ksize * ksize;
    uint cursor = 0;
    for (uint c = 0; c < channels; ++c) {
        for (uint ky = 0; ky < ksize; ++ky) {
            for (uint kx = 0; kx < ksize; ++kx) {
                int iy = int(y) + int(ky) - int(pad);
                int ix = int(x) + int(kx) - int(pad);
                float v = 0.0f;
                if (iy >= 0 && ix >= 0 && iy < int(height) && ix < int(width)) {
                    v = source[(c * height + uint(iy)) * width + uint(ix)];
                }
                row[cursor] = half(v);
                cursor += 1u;
            }
        }
    }
    for (uint k = logical; k < stored; ++k) {
        row[k] = half(0.0f);
    }
}

kernel void scatter_bias(
    device const float* source [[buffer(0)]],
    device const half* bias [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& pixels [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& cols_padded [[buffer(5)]],
    constant uint& height [[buffer(6)]],
    constant uint& width [[buffer(7)]],
    constant uint& pixel0 [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= cols || gid.y >= pixels) {
        return;
    }
    uint pixel = pixel0 + gid.y;
    uint y = pixel / width;
    uint x = pixel - y * width;
    float value = source[gid.y * cols_padded + gid.x] + float(bias[gid.x]);
    destination[(gid.x * height + y) * width + x] = value;
}

kernel void nearest_up(
    device const float* source [[buffer(0)]],
    device float* destination [[buffer(1)]],
    constant uint& channels [[buffer(2)]],
    constant uint& height [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint out_h = height << 1;
    uint out_w = width << 1;
    uint plane = out_h * out_w;
    uint n = channels * plane;
    if (gid >= n) {
        return;
    }
    uint c = gid / plane;
    uint p = gid - c * plane;
    uint y = p / out_w;
    uint x = p - y * out_w;
    destination[gid] = source[(c * height + (y >> 1)) * width + (x >> 1)];
}

kernel void add_nchw(
    device const float* left [[buffer(0)]],
    device const float* right [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) {
        destination[gid] = left[gid] + right[gid];
    }
}

kernel void silu_f32(
    device const float* source [[buffer(0)]],
    device float* destination [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) {
        return;
    }
    float value = source[gid];
    destination[gid] = value / (1.0f + exp(-value));
}

kernel void nchw_to_f32(
    device const float* source [[buffer(0)]],
    device float* destination [[buffer(1)]],
    constant uint& channels [[buffer(2)]],
    constant uint& height [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint pixels = height * width;
    if (gid.x >= channels || gid.y >= rows) {
        return;
    }
    if (gid.y >= pixels) {
        destination[gid.y * channels + gid.x] = 0.0f;
        return;
    }
    uint y = gid.y / width;
    uint x = gid.y - y * width;
    destination[gid.y * channels + gid.x] = source[(gid.x * height + y) * width + x];
}

kernel void rows_add_nchw(
    device const float* rows [[buffer(0)]],
    device const float* residual [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& channels [[buffer(3)]],
    constant uint& height [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint pixels = height * width;
    if (gid.x >= channels || gid.y >= pixels) {
        return;
    }
    uint y = gid.y / width;
    uint x = gid.y - y * width;
    uint index = (gid.x * height + y) * width + x;
    destination[index] = residual[index] + rows[gid.y * channels + gid.x];
}

kernel void euler(
    device float* latent [[buffer(0)]],
    device const float* velocity [[buffer(1)]],
    constant float& delta [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) {
        latent[gid] = latent[gid] + delta * velocity[gid];
    }
}

// (C, patch_h, 2, patch_w, 2) → token rows, channel fastest, stride `stride`.
kernel void patchify(
    device const float* latent [[buffer(0)]],
    device float* tokens [[buffer(1)]],
    constant uint& channels [[buffer(2)]],
    constant uint& patch_h [[buffer(3)]],
    constant uint& patch_w [[buffer(4)]],
    constant uint& stride [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint patches = patch_h * patch_w;
    uint span = channels * 4u;
    if (gid.y >= patches || gid.x >= span) {
        return;
    }
    uint py = gid.y / patch_w;
    uint px = gid.y - py * patch_w;
    uint inner = gid.x / channels;
    uint c = gid.x - inner * channels;
    uint ih = inner / 2u;
    uint iw = inner - ih * 2u;
    uint height = patch_h * 2u;
    uint width = patch_w * 2u;
    uint y = py * 2u + ih;
    uint x = px * 2u + iw;
    tokens[gid.y * stride + gid.x] = latent[(c * height + y) * width + x];
}

kernel void unpatchify(
    device const float* tokens [[buffer(0)]],
    device float* latent [[buffer(1)]],
    constant uint& channels [[buffer(2)]],
    constant uint& height [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    constant uint& stride [[buffer(5)]],
    constant uint& n [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) {
        return;
    }
    uint plane = height * width;
    uint c = gid / plane;
    uint rem = gid - c * plane;
    uint y = rem / width;
    uint x = rem - y * width;
    uint patch_w = width / 2u;
    uint py = y / 2u;
    uint px = x / 2u;
    uint ih = y - py * 2u;
    uint iw = x - px * 2u;
    uint patch = py * patch_w + px;
    uint elem = ((ih * 2u + iw) * channels) + c;
    latent[gid] = tokens[patch * stride + elem];
}

kernel void copy_rows(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& src_stride [[buffer(2)]],
    constant uint& dst_stride [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    constant uint& src_row [[buffer(5)]],
    constant uint& dst_row [[buffer(6)]],
    constant uint& count [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.y >= count || gid.x >= width) {
        return;
    }
    dst[(dst_row + gid.y) * dst_stride + gid.x] = src[(src_row + gid.y) * src_stride + gid.x];
}

kernel void fill_half_rows(
    device const half* token [[buffer(0)]],
    device float* rows [[buffer(1)]],
    constant uint& width [[buffer(2)]],
    constant uint& start [[buffer(3)]],
    constant uint& stop [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = start + gid.y;
    if (row >= stop || gid.x >= width) {
        return;
    }
    rows[row * width + gid.x] = float(token[gid.x]);
}

kernel void sinusoid(
    device float* dest [[buffer(0)]],
    constant float& value [[buffer(1)]],
    constant uint& dim [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= dim) {
        return;
    }
    uint half_n = dim / 2u;
    float index = float(gid < half_n ? gid : gid - half_n);
    float freq = exp(-log(10000.0f) * index / float(half_n));
    float angle = value * freq;
    dest[gid] = gid < half_n ? cos(angle) : sin(angle);
}

kernel void affine(
    device float* data [[buffer(0)]],
    constant float& scale [[buffer(1)]],
    constant float& shift [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) {
        data[gid] = data[gid] / scale + shift;
    }
}

kernel void text_positions(
    device int* pos [[buffer(0)]],
    constant uint& rows [[buffer(1)]],
    constant uint& limit [[buffer(2)]],
    constant int& start [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows) {
        return;
    }
    int temporal = gid < limit ? start + int(gid) : 0;
    pos[gid * 3u] = temporal;
    pos[gid * 3u + 1u] = 0;
    pos[gid * 3u + 2u] = 0;
}

kernel void image_positions(
    device int* pos [[buffer(0)]],
    constant uint& rows [[buffer(1)]],
    constant uint& patch_h [[buffer(2)]],
    constant uint& patch_w [[buffer(3)]],
    constant int& origin [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows) {
        return;
    }
    int temporal = 0;
    int y = 0;
    int x = 0;
    uint count = patch_h * patch_w;
    if (gid < count) {
        uint yy = gid / patch_w;
        uint xx = gid - yy * patch_w;
        temporal = origin;
        y = int(yy);
        x = int(xx);
    }
    pos[gid * 3u] = temporal;
    pos[gid * 3u + 1u] = y;
    pos[gid * 3u + 2u] = x;
}

kernel void joined_positions(
    device int* pos [[buffer(0)]],
    constant uint& rows [[buffer(1)]],
    constant uint& image_count [[buffer(2)]],
    constant uint& image_attend [[buffer(3)]],
    constant uint& patch_w [[buffer(4)]],
    constant int& origin [[buffer(5)]],
    constant uint& cap_len [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows) {
        return;
    }
    int temporal = 0;
    int y = 0;
    int x = 0;
    if (gid < image_count) {
        uint yy = gid / patch_w;
        uint xx = gid - yy * patch_w;
        temporal = origin;
        y = int(yy);
        x = int(xx);
    } else if (gid >= image_attend && gid < image_attend + cap_len) {
        temporal = int(gid - image_attend) + 1;
    }
    pos[gid * 3u] = temporal;
    pos[gid * 3u + 1u] = y;
    pos[gid * 3u + 2u] = x;
}

kernel void clip_u8_hwc(
    device const float* nchw [[buffer(0)]],
    device uchar* hwc [[buffer(1)]],
    constant uint& height [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= width || gid.y >= height) {
        return;
    }
    uint plane = height * width;
    uint pixel = gid.y * width + gid.x;
    for (uint c = 0u; c < 3u; ++c) {
        float value = nchw[c * plane + pixel];
        value = clamp(value, -1.0f, 1.0f);
        int rounded = int(rint((value + 1.0f) * 127.5f));
        hwc[pixel * 3u + c] = uchar(rounded);
    }
}
