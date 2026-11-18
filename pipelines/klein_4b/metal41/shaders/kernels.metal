#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>

using namespace metal;
using namespace mpp::tensor_ops;

constant int TILE_M = 64;
constant int TILE_N = 128;
constant int I8_GROUP = 128;

// One scale per 128 K values. A zero group stores scale 1 and zero codes.
template <typename T>
void quantize_group(
    device const T* src,
    device int8_t* codes,
    device float* scales,
    uint row,
    uint group,
    uint logical_k,
    uint stored_k,
    uint src_cols)
{
    uint k0 = group * uint(I8_GROUP);
    uint groups = stored_k / uint(I8_GROUP);
    float peak = 0.0f;
    for (uint i = 0u; i < uint(I8_GROUP); ++i) {
        uint k = k0 + i;
        if (k >= logical_k || k >= src_cols) {
            break;
        }
        peak = max(peak, fabs(float(src[row * src_cols + k])));
    }
    float inverse = peak > 0.0f ? 127.0f / peak : 0.0f;
    scales[row * groups + group] = peak > 0.0f ? peak / 127.0f : 1.0f;
    for (uint i = 0u; i < uint(I8_GROUP); ++i) {
        uint k = k0 + i;
        float value = 0.0f;
        if (k < logical_k && k < src_cols) {
            value = float(src[row * src_cols + k]);
        }
        int q = int(rint(value * inverse));
        codes[row * stored_k + k] = int8_t(clamp(q, -127, 127));
    }
}

kernel void quantize_f16_i8(
    device const half* src [[buffer(0)]],
    device int8_t* codes [[buffer(1)]],
    device float* scales [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& logical_k [[buffer(4)]],
    constant uint& stored_k [[buffer(5)]],
    constant uint& src_cols [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint groups = stored_k / uint(I8_GROUP);
    if (gid.y >= rows || gid.x >= groups) {
        return;
    }
    quantize_group(src, codes, scales, gid.y, gid.x, logical_k, stored_k, src_cols);
}

kernel void quantize_f32_i8(
    device const float* src [[buffer(0)]],
    device int8_t* codes [[buffer(1)]],
    device float* scales [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& logical_k [[buffer(4)]],
    constant uint& stored_k [[buffer(5)]],
    constant uint& src_cols [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint groups = stored_k / uint(I8_GROUP);
    if (gid.y >= rows || gid.x >= groups) {
        return;
    }
    quantize_group(src, codes, scales, gid.y, gid.x, logical_k, stored_k, src_cols);
}

// Gather one K group from a volume and quantize it. The patch is not stored.
// ``local_row`` is the row inside this span. The int8 product index stays in range.
template <typename T>
void gather_quantize_body(
    device const T* src,
    device int8_t* codes,
    device float* scales,
    uint channels,
    uint height,
    uint width,
    uint kh,
    uint kw,
    uint stride,
    uint pad_top,
    uint pad_left,
    uint out_w,
    uint logical_k,
    uint stored_k,
    uint row0,
    uint local_row,
    uint group)
{
    uint groups = stored_k / uint(I8_GROUP);
    uint k_total = channels * kh * kw;
    uint k0 = group * uint(I8_GROUP);
    uint spatial_row = row0 + local_row;
    uint y = spatial_row / out_w;
    uint x = spatial_row - y * out_w;
    uint patch = kh * kw;
    float peak = 0.0f;
    float vals[128];
    for (uint i = 0u; i < uint(I8_GROUP); ++i) {
        uint k = k0 + i;
        float value = 0.0f;
        if (k < logical_k && k < k_total) {
            uint c = k / patch;
            uint rem = k - c * patch;
            uint ky = rem / kw;
            uint kx = rem - ky * kw;
            int ih = int(y * stride + ky) - int(pad_top);
            int iw = int(x * stride + kx) - int(pad_left);
            if (ih >= 0 && iw >= 0 && ih < int(height) && iw < int(width)) {
                uint64_t index = (uint64_t(c) * uint64_t(height) + uint64_t(ih)) * uint64_t(width) + uint64_t(iw);
                value = float(src[index]);
            }
        }
        vals[i] = value;
        peak = max(peak, fabs(value));
    }
    float inverse = peak > 0.0f ? 127.0f / peak : 0.0f;
    scales[uint64_t(local_row) * uint64_t(groups) + uint64_t(group)] = peak > 0.0f ? peak / 127.0f : 1.0f;
    uint64_t code_base = uint64_t(local_row) * uint64_t(stored_k) + uint64_t(k0);
    for (uint i = 0u; i < uint(I8_GROUP); ++i) {
        int q = int(rint(vals[i] * inverse));
        codes[code_base + uint64_t(i)] = int8_t(clamp(q, -127, 127));
    }
}

kernel void gather_quantize_f16(
    device const half* src [[buffer(0)]],
    device int8_t* codes [[buffer(1)]],
    device float* scales [[buffer(2)]],
    constant uint& channels [[buffer(3)]],
    constant uint& height [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    constant uint& kh [[buffer(6)]],
    constant uint& kw [[buffer(7)]],
    constant uint& stride [[buffer(8)]],
    constant uint& pad_top [[buffer(9)]],
    constant uint& pad_left [[buffer(10)]],
    constant uint& out_w [[buffer(11)]],
    constant uint& logical_k [[buffer(12)]],
    constant uint& stored_k [[buffer(13)]],
    constant uint& row0 [[buffer(14)]],
    constant uint& taken [[buffer(15)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint groups = stored_k / uint(I8_GROUP);
    if (gid.y >= taken || gid.x >= groups) {
        return;
    }
    gather_quantize_body(
        src, codes, scales, channels, height, width, kh, kw, stride, pad_top, pad_left,
        out_w, logical_k, stored_k, row0, gid.y, gid.x);
}

kernel void gather_quantize_f32(
    device const float* src [[buffer(0)]],
    device int8_t* codes [[buffer(1)]],
    device float* scales [[buffer(2)]],
    constant uint& channels [[buffer(3)]],
    constant uint& height [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    constant uint& kh [[buffer(6)]],
    constant uint& kw [[buffer(7)]],
    constant uint& stride [[buffer(8)]],
    constant uint& pad_top [[buffer(9)]],
    constant uint& pad_left [[buffer(10)]],
    constant uint& out_w [[buffer(11)]],
    constant uint& logical_k [[buffer(12)]],
    constant uint& stored_k [[buffer(13)]],
    constant uint& row0 [[buffer(14)]],
    constant uint& taken [[buffer(15)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint groups = stored_k / uint(I8_GROUP);
    if (gid.y >= taken || gid.x >= groups) {
        return;
    }
    gather_quantize_body(
        src, codes, scales, channels, height, width, kh, kw, stride, pad_top, pad_left,
        out_w, logical_k, stored_k, row0, gid.y, gid.x);
}

// int8 activation times int8 weight. The epilogue is float32.
kernel void matmul_i8(
    device int8_t* activation [[buffer(0)]],
    device const float* row_scale [[buffer(1)]],
    device int8_t* weights [[buffer(2)]],
    device const float* col_scale [[buffer(3)]],
    device const half* bias [[buffer(4)]],
    device float* destination [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& cols [[buffer(7)]],
    constant uint& inner [[buffer(8)]],
    constant uint& logical_n [[buffer(9)]],
    constant uint& dest_stride [[buffer(10)]],
    constant uint& has_bias [[buffer(11)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    int row0 = int(tgid.y) * TILE_M;
    int col0 = int(tgid.x) * TILE_N;
    int groups = int(inner) / I8_GROUP;
    tensor<device int8_t, dextents<int, 2>, tensor_inline> left(
        activation,
        dextents<int, 2>(int(inner), int(rows)),
        array<int, 2>{1, int(inner)});
    tensor<device int8_t, dextents<int, 2>, tensor_inline> right(
        weights,
        dextents<int, 2>(int(inner), int(cols)),
        array<int, 2>{1, int(inner)});
    constexpr auto desc = matmul2d_descriptor(64, 128, I8_GROUP, false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto origin_left = left.slice(0, row0);
    auto origin_right = right.slice(0, col0);
    auto dots = op.get_destination_cooperative_tensor<
        decltype(origin_left), decltype(origin_right), int>();
    float partial[128];
    for (int i = 0; i < 128; ++i) {
        partial[i] = 0.0f;
    }
    for (int g = 0; g < groups; ++g) {
        int k0 = g * I8_GROUP;
        auto tile_left = left.slice(k0, row0);
        auto tile_right = right.slice(k0, col0);
        for (uint16_t i = 0u; i < dots.get_capacity(); ++i) {
            if (dots.is_valid_element(i)) {
                dots[i] = 0;
            }
        }
        op.run(tile_left, tile_right, dots);
        for (uint16_t i = 0u; i < dots.get_capacity(); ++i) {
            if (!dots.is_valid_element(i) || i >= 128u) {
                continue;
            }
            auto ids = dots.get_multidimensional_index(i);
            int row = row0 + int(ids[1]);
            partial[i] += float(dots[i]) * row_scale[row * groups + g];
        }
    }
    for (uint16_t i = 0u; i < dots.get_capacity(); ++i) {
        if (!dots.is_valid_element(i) || i >= 128u) {
            continue;
        }
        auto ids = dots.get_multidimensional_index(i);
        int col = col0 + int(ids[0]);
        int row = row0 + int(ids[1]);
        if (col >= int(logical_n)) {
            continue;
        }
        float value = partial[i] * col_scale[col];
        if (has_bias != 0u) {
            value += float(bias[col]);
        }
        destination[row * int(dest_stride) + col] = value;
    }
}

// One threadgroup per query and head. The score is not stored.
kernel void attend_f32(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& seq [[buffer(4)]],
    constant uint& heads [[buffer(5)]],
    constant uint& dim [[buffer(6)]],
    constant uint& valid [[buffer(7)]],
    constant uint& causal [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float parts[32];
    threadgroup float shared_score;
    uint qrow = tgid.y;
    uint head = tgid.x;
    float qv = query[((qrow * heads) + head) * dim + tid];
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float acc = 0.0f;
    float scale = rsqrt(float(dim));
    uint nsimd = (dim + 31u) / 32u;
    for (uint col = 0u; col < seq; ++col) {
        bool live = col < valid;
        if (causal != 0u && col > qrow) {
            live = false;
        }
        float part = live ? qv * key[((col * heads) + head) * dim + tid] : 0.0f;
        float sum = simd_sum(part);
        if (lane == 0u) {
            parts[simd] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float total = 0.0f;
            for (uint s = 0u; s < nsimd; ++s) {
                total += parts[s];
            }
            shared_score = live ? total * scale : -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float score = shared_score;
        if (score > -1.0e30f) {
            float next_max = max(running_max, score);
            float alpha = running_max > -1.0e30f ? exp(running_max - next_max) : 0.0f;
            float weight = exp(score - next_max);
            float sample = value[((col * heads) + head) * dim + tid];
            running_sum = running_sum * alpha + weight;
            acc = acc * alpha + weight * sample;
            running_max = next_max;
        }
    }
    destination[((qrow * heads) + head) * dim + tid] =
        running_sum > 0.0f ? acc / running_sum : 0.0f;
}

// One query tile of 32 and one key tile of 32. Head dim 128 only.
// Scores stay in the threadgroup. Head dim 512 stays on attend_f32.
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
            bool live = r < uint(FQ) && c < uint(FK) && q < seq && key_index < limit;
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
        if (lrun[r] == 0.0f) {
            *out = 0.0f;
        } else {
            *out = oreg[i] / lrun[r];
        }
    }
}

constant uint ROW_THREADS = 256u;

float reduce_row(float local, uint simd, uint lane, uint tid, threadgroup float* parts) {
    float sum = simd_sum(local);
    if (lane == 0u) {
        parts[simd] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float total = 0.0f;
        for (uint s = 0u; s < 8u; ++s) {
            total += parts[s];
        }
        parts[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return parts[0];
}

// One threadgroup per row. The weight is per element. The unused destination is not written.
kernel void rmsnorm_row(
    device const half* src [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device half* dst_half [[buffer(2)]],
    device float* dst_float [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& dim [[buffer(5)]],
    constant uint& store_float [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    if (tg >= rows) {
        return;
    }
    threadgroup float parts[8];
    float local = 0.0f;
    for (uint i = tid; i < dim; i += ROW_THREADS) {
        float value = float(src[tg * dim + i]);
        local += value * value;
    }
    float inv = rsqrt(reduce_row(local, simd, lane, tid, parts) / float(dim) + eps);
    for (uint i = tid; i < dim; i += ROW_THREADS) {
        float value = float(src[tg * dim + i]) * inv * float(weight[i]);
        uint index = tg * dim + i;
        if (store_float != 0u) {
            dst_float[index] = value;
        } else {
            dst_half[index] = half(clamp(value, -65504.0f, 65504.0f));
        }
    }
}

// LayerNorm, then (1 + scale) * norm + shift. Scale and shift are one vector.
kernel void modulate_row(
    device const half* src [[buffer(0)]],
    device const half* shift [[buffer(1)]],
    device const half* scale [[buffer(2)]],
    device half* dst [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& dim [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    if (tg >= rows) {
        return;
    }
    threadgroup float parts[8];
    float local = 0.0f;
    for (uint i = tid; i < dim; i += ROW_THREADS) {
        local += float(src[tg * dim + i]);
    }
    float mean = reduce_row(local, simd, lane, tid, parts) / float(dim);
    local = 0.0f;
    for (uint i = tid; i < dim; i += ROW_THREADS) {
        float delta = float(src[tg * dim + i]) - mean;
        local += delta * delta;
    }
    float inv = rsqrt(reduce_row(local, simd, lane, tid, parts) / float(dim) + eps);
    for (uint i = tid; i < dim; i += ROW_THREADS) {
        float norm = (float(src[tg * dim + i]) - mean) * inv;
        float value = (1.0f + float(scale[i])) * norm + float(shift[i]);
        dst[tg * dim + i] = half(clamp(value, -65504.0f, 65504.0f));
    }
}

kernel void store_f16(
    device const float* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& src_stride [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint count = rows * cols;
    if (gid >= count) {
        return;
    }
    uint row = gid / cols;
    uint col = gid - row * cols;
    float value = src[row * src_stride + col];
    dst[gid] = half(clamp(value, -65504.0f, 65504.0f));
}

kernel void store_f32(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& src_stride [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint count = rows * cols;
    if (gid >= count) {
        return;
    }
    uint row = gid / cols;
    uint col = gid - row * cols;
    dst[gid] = src[row * src_stride + col];
}

kernel void silu_f16(
    device const half* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) {
        return;
    }
    float value = float(src[gid]);
    dst[gid] = half(value / (1.0f + exp(-value)));
}

kernel void silu_mul_f16(
    device const half* gate [[buffer(0)]],
    device const half* up [[buffer(1)]],
    device half* dst [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) {
        return;
    }
    float value = float(gate[gid]);
    dst[gid] = half((value / (1.0f + exp(-value))) * float(up[gid]));
}

kernel void swiglu_f16(
    device const half* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint half_cols = cols >> 1u;
    uint count = rows * half_cols;
    if (gid >= count) {
        return;
    }
    uint row = gid / half_cols;
    uint col = gid - row * half_cols;
    float left = float(src[row * cols + col]);
    float right = float(src[row * cols + half_cols + col]);
    dst[gid] = half((left / (1.0f + exp(-left))) * right);
}

kernel void gate_f16(
    device const half* residual [[buffer(0)]],
    device const half* gate [[buffer(1)]],
    device const half* branch [[buffer(2)]],
    device half* dst [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    constant uint& use_gate [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) {
        return;
    }
    float scale = 1.0f;
    if (use_gate != 0u) {
        scale = float(gate[gid % cols]);
    }
    float value = float(residual[gid]) + scale * float(branch[gid]);
    dst[gid] = half(clamp(value, -65504.0f, 65504.0f));
}

kernel void rope_text(
    device const float* src [[buffer(0)]],
    device const float* cos [[buffer(1)]],
    device const float* sin [[buffer(2)]],
    device float* dst [[buffer(3)]],
    constant uint& seq [[buffer(4)]],
    constant uint& heads [[buffer(5)]],
    constant uint& dim [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    uint count = seq * heads * dim;
    if (gid >= count) {
        return;
    }
    uint half_dim = dim >> 1u;
    uint d = gid % dim;
    uint head = (gid / dim) % heads;
    uint row = gid / (dim * heads);
    uint base = (row * heads + head) * dim;
    float rotated = d < half_dim ? -src[base + d + half_dim] : src[base + d - half_dim];
    uint angle = row * dim + d;
    dst[gid] = src[base + d] * cos[angle] + rotated * sin[angle];
}

kernel void rope_dit(
    device const float* src [[buffer(0)]],
    device const float* cos [[buffer(1)]],
    device const float* sin [[buffer(2)]],
    device float* dst [[buffer(3)]],
    constant uint& seq [[buffer(4)]],
    constant uint& heads [[buffer(5)]],
    constant uint& dim [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    uint count = seq * heads * dim;
    if (gid >= count) {
        return;
    }
    uint d = gid % dim;
    uint head = (gid / dim) % heads;
    uint row = gid / (dim * heads);
    uint base = (row * heads + head) * dim;
    uint even = d & ~1u;
    float first = src[base + even];
    float second = src[base + even + 1u];
    float angle_c = cos[row * dim + even];
    float angle_s = sin[row * dim + even];
    if ((d & 1u) == 0u) {
        dst[gid] = first * angle_c - second * angle_s;
    } else {
        dst[gid] = first * angle_s + second * angle_c;
    }
}

kernel void repeat_heads(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& seq [[buffer(2)]],
    constant uint& kv_heads [[buffer(3)]],
    constant uint& times [[buffer(4)]],
    constant uint& dim [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint heads = kv_heads * times;
    uint count = seq * heads * dim;
    if (gid >= count) {
        return;
    }
    uint d = gid % dim;
    uint head = (gid / dim) % heads;
    uint row = gid / (dim * heads);
    uint src_index = ((row * kv_heads + head / times) * dim) + d;
    dst[gid] = src[src_index];
}

kernel void copy_f16(
    device const half* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) {
        dst[gid] = src[gid];
    }
}

kernel void copy_f32(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) {
        dst[gid] = src[gid];
    }
}

kernel void cast_f16_f32(
    device const half* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) {
        dst[gid] = float(src[gid]);
    }
}

kernel void cast_f32_f16(
    device const float* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) {
        dst[gid] = half(clamp(src[gid], -65504.0f, 65504.0f));
    }
}

kernel void place_cols_f16(
    device const half* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& dst_stride [[buffer(4)]],
    constant uint& col0 [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint count = rows * cols;
    if (gid >= count) {
        return;
    }
    uint row = gid / cols;
    uint col = gid - row * cols;
    dst[row * dst_stride + col0 + col] = src[gid];
}

kernel void slice_cols_f16(
    device const half* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& stride [[buffer(3)]],
    constant uint& col0 [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint count = rows * cols;
    if (gid >= count) {
        return;
    }
    uint row = gid / cols;
    uint col = gid - row * cols;
    dst[gid] = src[row * stride + col0 + col];
}

kernel void pack_attn_mlp(
    device const float* attn [[buffer(0)]],
    device const half* mlp [[buffer(1)]],
    device half* dst [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& attn_cols [[buffer(4)]],
    constant uint& mlp_cols [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint cols = attn_cols + mlp_cols;
    uint count = rows * cols;
    if (gid >= count) {
        return;
    }
    uint row = gid / cols;
    uint col = gid - row * cols;
    if (col < attn_cols) {
        float value = attn[row * attn_cols + col];
        dst[gid] = half(clamp(value, -65504.0f, 65504.0f));
    } else {
        dst[gid] = mlp[row * mlp_cols + (col - attn_cols)];
    }
}

kernel void groupnorm_f16(
    device const half* src [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* dst [[buffer(3)]],
    constant uint& channels [[buffer(4)]],
    constant uint& height [[buffer(5)]],
    constant uint& width [[buffer(6)]],
    constant uint& groups [[buffer(7)]],
    constant uint& silu [[buffer(8)]],
    constant float& eps [[buffer(9)]],
    uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    if (tg >= groups) {
        return;
    }
    uint spatial = height * width;
    uint group_size = channels / groups;
    uint count = group_size * spatial;
    uint channel0 = tg * group_size;
    threadgroup float parts[8];
    float local = 0.0f;
    for (uint i = tid; i < count; i += ROW_THREADS) {
        uint channel = channel0 + i / spatial;
        uint place = i - (i / spatial) * spatial;
        local += float(src[channel * spatial + place]);
    }
    float mean = reduce_row(local, simd, lane, tid, parts) / float(count);
    local = 0.0f;
    for (uint i = tid; i < count; i += ROW_THREADS) {
        uint channel = channel0 + i / spatial;
        uint place = i - (i / spatial) * spatial;
        float delta = float(src[channel * spatial + place]) - mean;
        local += delta * delta;
    }
    float inv = rsqrt(reduce_row(local, simd, lane, tid, parts) / float(count) + eps);
    for (uint i = tid; i < count; i += ROW_THREADS) {
        uint channel = channel0 + i / spatial;
        uint place = i - (i / spatial) * spatial;
        uint index = channel * spatial + place;
        float value = (float(src[index]) - mean) * inv * float(weight[channel]) + float(bias[channel]);
        if (silu != 0u) {
            value = value / (1.0f + exp(-value));
        }
        dst[index] = half(clamp(value, -65504.0f, 65504.0f));
    }
}

kernel void groupnorm_f32(
    device const float* src [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device float* dst [[buffer(3)]],
    constant uint& channels [[buffer(4)]],
    constant uint& height [[buffer(5)]],
    constant uint& width [[buffer(6)]],
    constant uint& groups [[buffer(7)]],
    constant uint& silu [[buffer(8)]],
    constant float& eps [[buffer(9)]],
    uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    if (tg >= groups) {
        return;
    }
    uint spatial = height * width;
    uint group_size = channels / groups;
    uint count = group_size * spatial;
    uint channel0 = tg * group_size;
    threadgroup float parts[8];
    float local = 0.0f;
    for (uint i = tid; i < count; i += ROW_THREADS) {
        uint channel = channel0 + i / spatial;
        uint place = i - (i / spatial) * spatial;
        local += src[channel * spatial + place];
    }
    float mean = reduce_row(local, simd, lane, tid, parts) / float(count);
    local = 0.0f;
    for (uint i = tid; i < count; i += ROW_THREADS) {
        uint channel = channel0 + i / spatial;
        uint place = i - (i / spatial) * spatial;
        float delta = src[channel * spatial + place] - mean;
        local += delta * delta;
    }
    float inv = rsqrt(reduce_row(local, simd, lane, tid, parts) / float(count) + eps);
    for (uint i = tid; i < count; i += ROW_THREADS) {
        uint channel = channel0 + i / spatial;
        uint place = i - (i / spatial) * spatial;
        uint index = channel * spatial + place;
        float value = (src[index] - mean) * inv * float(weight[channel]) + float(bias[channel]);
        if (silu != 0u) {
            value = value / (1.0f + exp(-value));
        }
        dst[index] = value;
    }
}

kernel void im2col_f16(
    device const half* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    constant uint& channels [[buffer(2)]],
    constant uint& height [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    constant uint& kh [[buffer(5)]],
    constant uint& kw [[buffer(6)]],
    constant uint& stride [[buffer(7)]],
    constant uint& pad_top [[buffer(8)]],
    constant uint& pad_left [[buffer(9)]],
    constant uint& out_h [[buffer(10)]],
    constant uint& out_w [[buffer(11)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint k_total = channels * kh * kw;
    uint spatial_n = out_h * out_w;
    if (gid.x >= k_total || gid.y >= spatial_n) {
        return;
    }
    uint y = gid.y / out_w;
    uint x = gid.y - y * out_w;
    uint patch = kh * kw;
    uint c = gid.x / patch;
    uint rem = gid.x - c * patch;
    uint ky = rem / kw;
    uint kx = rem - ky * kw;
    int ih = int(y * stride + ky) - int(pad_top);
    int iw = int(x * stride + kx) - int(pad_left);
    float value = 0.0f;
    if (ih >= 0 && iw >= 0 && ih < int(height) && iw < int(width)) {
        value = float(src[(c * height + uint(ih)) * width + uint(iw)]);
    }
    dst[gid.y * k_total + gid.x] = half(value);
}

template <typename T>
void chw_from_rows_body(
    device const T* src,
    device T* dst,
    uint channels,
    uint spatial,
    uint place0,
    uint taken,
    uint gid)
{
    uint count = channels * taken;
    if (gid >= count) {
        return;
    }
    uint c = gid / taken;
    uint local = gid - c * taken;
    uint64_t place = uint64_t(place0) + uint64_t(local);
    dst[uint64_t(c) * uint64_t(spatial) + place] = src[uint64_t(local) * uint64_t(channels) + uint64_t(c)];
}

kernel void chw_from_rows(
    device const half* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    constant uint& channels [[buffer(2)]],
    constant uint& height [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    constant uint& place0 [[buffer(5)]],
    constant uint& taken [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    chw_from_rows_body(src, dst, channels, height * width, place0, taken, gid);
}

kernel void chw_from_rows_f32(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& channels [[buffer(2)]],
    constant uint& height [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    constant uint& place0 [[buffer(5)]],
    constant uint& taken [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    chw_from_rows_body(src, dst, channels, height * width, place0, taken, gid);
}

kernel void seq_from_chw_f32(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& channels [[buffer(2)]],
    constant uint& height [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint spatial = height * width;
    uint count = channels * spatial;
    if (gid >= count) {
        return;
    }
    uint c = gid / spatial;
    uint place = gid - c * spatial;
    dst[place * channels + c] = src[gid];
}

kernel void seq_from_chw(
    device const half* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    constant uint& channels [[buffer(2)]],
    constant uint& height [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint spatial = height * width;
    uint count = channels * spatial;
    if (gid >= count) {
        return;
    }
    uint c = gid / spatial;
    uint place = gid - c * spatial;
    dst[place * channels + c] = src[gid];
}

kernel void nearest2_f16(
    device const half* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    constant uint& channels [[buffer(2)]],
    constant uint& height [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint height2 = height << 1u;
    uint width2 = width << 1u;
    uint count = channels * height2 * width2;
    if (gid >= count) {
        return;
    }
    uint spatial = height2 * width2;
    uint c = gid / spatial;
    uint place = gid - c * spatial;
    uint y = place / width2;
    uint x = place - y * width2;
    dst[gid] = src[(c * height + (y >> 1u)) * width + (x >> 1u)];
}

kernel void nearest2_f32(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& channels [[buffer(2)]],
    constant uint& height [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint height2 = height << 1u;
    uint width2 = width << 1u;
    uint count = channels * height2 * width2;
    if (gid >= count) {
        return;
    }
    uint spatial = height2 * width2;
    uint c = gid / spatial;
    uint place = gid - c * spatial;
    uint y = place / width2;
    uint x = place - y * width2;
    dst[gid] = src[(c * height + (y >> 1u)) * width + (x >> 1u)];
}

kernel void add_f32(
    device const float* left [[buffer(0)]],
    device const float* right [[buffer(1)]],
    device float* dst [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) {
        return;
    }
    dst[gid] = left[gid] + right[gid];
}

kernel void add_f16(
    device const half* left [[buffer(0)]],
    device const half* right [[buffer(1)]],
    device half* dst [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) {
        return;
    }
    float value = float(left[gid]) + float(right[gid]);
    dst[gid] = half(clamp(value, -65504.0f, 65504.0f));
}

// Patchify 2x2, whiten with the packed BatchNorm, and store tokens.
kernel void patch_pack(
    device const half* src [[buffer(0)]],
    device const float* mean [[buffer(1)]],
    device const float* var [[buffer(2)]],
    device half* dst [[buffer(3)]],
    constant uint& height [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    constant uint& grid_w [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint channel = gid.x;
    uint token = gid.y;
    if (channel >= 128u) {
        return;
    }
    uint grid_h = height >> 1u;
    if (token >= grid_h * grid_w) {
        return;
    }
    uint x = token % grid_w;
    uint y = token / grid_w;
    uint source_c = channel >> 2u;
    uint rem = channel & 3u;
    uint dy = rem >> 1u;
    uint dx = rem & 1u;
    uint sy = y * 2u + dy;
    uint sx = x * 2u + dx;
    float value = float(src[(source_c * height + sy) * width + sx]);
    float whitened = (value - mean[channel]) * rsqrt(var[channel] + eps);
    dst[token * 128u + channel] = half(whitened);
}

kernel void unpack_latent(
    device const half* src [[buffer(0)]],
    device const float* mean [[buffer(1)]],
    device const float* var [[buffer(2)]],
    device float* dst [[buffer(3)]],
    constant uint& height [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    constant uint& grid_w [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint channel = gid.x;
    uint token = gid.y;
    if (channel >= 128u) {
        return;
    }
    uint grid_h = height >> 1u;
    if (token >= grid_h * grid_w) {
        return;
    }
    uint x = token % grid_w;
    uint y = token / grid_w;
    uint source_c = channel >> 2u;
    uint rem = channel & 3u;
    uint dy = rem >> 1u;
    uint dx = rem & 1u;
    float value = float(src[token * 128u + channel]);
    float restored = value * sqrt(var[channel] + eps) + mean[channel];
    uint sy = y * 2u + dy;
    uint sx = x * 2u + dx;
    dst[(source_c * height + sy) * width + sx] = restored;
}

kernel void euler_f16(
    device const half* state [[buffer(0)]],
    device const half* velocity [[buffer(1)]],
    device half* dst [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant float& delta [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) {
        return;
    }
    float value = float(state[gid]) + delta * float(velocity[gid]);
    dst[gid] = half(clamp(value, -65504.0f, 65504.0f));
}
