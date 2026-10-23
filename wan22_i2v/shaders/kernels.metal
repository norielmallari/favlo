#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>

using namespace metal;
using namespace mpp::tensor_ops;

using scales_plane = tensor_blockwise<tensor_plane_scales,
                                      device metal_fp8_ue8m0_format,
                                      32, 1>;

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
        activation,
        dextents<int, 2>(int(inner), int(rows)),
        array<int, 2>{1, int(inner)});
    tensor<device metal_fp8_e4m3_format, dextents<int, 2>, tensor_inline, scales_plane> right(
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

kernel void zero_bytes(
    device uchar* data [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) {
        data[gid] = 0;
    }
}

kernel void gather_half(
    device const float* src [[buffer(0)]],
    device const uint* perm [[buffer(1)]],
    device half* dst [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& logical_k [[buffer(4)]],
    constant uint& stored_k [[buffer(5)]],
    constant uint& src_cols [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    uint n = rows * stored_k;
    if (gid >= n) {
        return;
    }
    uint row = gid / stored_k;
    uint col = gid - row * stored_k;
    uint src_col = perm[col];
    float value = 0.0f;
    if (src_col < logical_k) {
        value = src[row * src_cols + src_col];
    }
    dst[gid] = half(value);
}

kernel void scale_gather(
    device const float* src [[buffer(0)]],
    device const uint* perm [[buffer(1)]],
    device half* dst [[buffer(2)]],
    device float* factor [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& logical_k [[buffer(5)]],
    constant uint& stored_k [[buffer(6)]],
    constant uint& src_cols [[buffer(7)]],
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
    float scale = max(total / 60000.0f, 1.0f);
    if (lane == 0u) {
        factor[row] = scale;
    }
    for (uint col = lane; col < stored_k; col += 128u) {
        uint src_col = perm[col];
        float value = 0.0f;
        if (src_col < logical_k) {
            value = src[row * src_cols + src_col] / scale;
        }
        dst[row * stored_k + col] = half(value);
    }
}

kernel void rescale_rows(
    device float* data [[buffer(0)]],
    device const float* factor [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint n = rows * cols;
    if (gid >= n) {
        return;
    }
    data[gid] *= factor[gid / cols];
}

kernel void add_bias(
    device float* data [[buffer(0)]],
    device const half* bias [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& logical [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint n = rows * cols;
    if (gid >= n) {
        return;
    }
    uint col = gid % cols;
    if (col < logical) {
        data[gid] += float(bias[col]);
    }
}

kernel void gelu_tanh(
    device float* data [[buffer(0)]],
    constant uint& rows [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    constant uint& logical [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint n = rows * cols;
    if (gid >= n) {
        return;
    }
    uint col = gid % cols;
    if (col >= logical) {
        return;
    }
    float value = data[gid];
    float cube = value * value * value;
    float inner = 0.7978845608f * (value + 0.044715f * cube);
    // tanh(x) is implemented with exp(2x). Past about 44 that quotient is NaN.
    inner = clamp(inner, -15.0f, 15.0f);
    data[gid] = 0.5f * value * (1.0f + tanh(inner));
}

kernel void silu(
    device float* data [[buffer(0)]],
    constant uint& rows [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    constant uint& logical [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint n = rows * cols;
    if (gid >= n) {
        return;
    }
    uint col = gid % cols;
    if (col >= logical) {
        return;
    }
    float value = data[gid];
    data[gid] = value / (1.0f + exp(-value));
}

kernel void modulate(
    device float* data [[buffer(0)]],
    device const float* shift [[buffer(1)]],
    device const float* scale [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& logical [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint n = rows * cols;
    if (gid >= n) {
        return;
    }
    uint col = gid % cols;
    if (col >= logical) {
        data[gid] = 0.0f;
        return;
    }
    data[gid] = data[gid] * (1.0f + scale[col]) + shift[col];
}

kernel void gate_residual(
    device float* left [[buffer(0)]],
    device const float* right [[buffer(1)]],
    device const float* gate [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& logical [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint n = rows * cols;
    if (gid >= n) {
        return;
    }
    uint col = gid % cols;
    if (col < logical) {
        left[gid] = left[gid] + right[gid] * gate[col];
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
        float value = x[row * width + i];
        sum += value * value;
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
    float sum_sq = 0.0f;
    for (uint i = lane; i < width; i += 128u) {
        float value = x[row * width + i];
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
    for (uint i = lane; i < width; i += 128u) {
        y[row * width + i] = (x[row * width + i] - mean) * inv;
    }
}

kernel void layernorm_affine(
    device const float* x [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device float* y [[buffer(3)]],
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
    for (uint i = lane; i < width; i += 128u) {
        float value = x[row * width + i];
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
    for (uint i = lane; i < width; i += 128u) {
        float norm = (x[row * width + i] - mean) * inv;
        y[row * width + i] = norm * float(weight[i]) + float(bias[i]);
    }
}

kernel void rope3d(
    device float* values [[buffer(0)]],
    constant uint& seq [[buffer(1)]],
    constant uint& heads [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    constant uint& grid_t [[buffer(4)]],
    constant uint& grid_h [[buffer(5)]],
    constant uint& grid_w [[buffer(6)]],
    constant uint& axis_t [[buffer(7)]],
    constant uint& axis_h [[buffer(8)]],
    constant uint& axis_w [[buffer(9)]],
    constant float& theta [[buffer(10)]],
    uint token [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]])
{
    if (token >= seq) {
        return;
    }
    uint width = heads * dim;
    uint spatial = grid_h * grid_w;
    uint t = token / spatial;
    uint rem = token - t * spatial;
    uint h = rem / grid_w;
    uint w = rem - h * grid_w;
    for (uint index = lane; index < width; index += 128u) {
        uint d = index % dim;
        if (d & 1u) {
            continue;
        }
        uint head = index / dim;
        float pos;
        uint axis;
        uint local;
        if (d < axis_t) {
            pos = float(t);
            axis = axis_t;
            local = d;
        } else if (d < axis_t + axis_h) {
            pos = float(h);
            axis = axis_h;
            local = d - axis_t;
        } else {
            pos = float(w);
            axis = axis_w;
            local = d - axis_t - axis_h;
        }
        float freq = pos * pow(theta, -float(local) / float(axis));
        float c = cos(freq);
        float s = sin(freq);
        uint base = token * width + head * dim + d;
        float a = values[base];
        float b = values[base + 1];
        values[base] = a * c - b * s;
        values[base + 1] = a * s + b * c;
    }
}

kernel void attend(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& q_len [[buffer(4)]],
    constant uint& kv_len [[buffer(5)]],
    constant uint& heads [[buffer(6)]],
    constant uint& dim [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    uint3 lane [[thread_position_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]])
{
    uint head = group.x;
    uint query = group.y;
    uint lid = lane.x;
    if (head >= heads || query >= q_len || kv_len == 0u) {
        return;
    }
    device const float* qq = q + (query * heads + head) * dim;
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    for (uint t = 0u; t < kv_len; t++) {
        device const float* kk = k + (t * heads + head) * dim;
        float partial = 0.0f;
        for (uint d = lid; d < dim; d += 32u) {
            partial += qq[d] * kk[d];
        }
        float dot = simd_sum(partial) * scale;
        float next = max(running_max, dot);
        running_sum = running_sum * exp(running_max - next) + exp(dot - next);
        running_max = next;
    }
    device float* dst = out + (query * heads + head) * dim;
    for (uint d = lid; d < dim; d += 32u) {
        dst[d] = 0.0f;
    }
    for (uint t = 0u; t < kv_len; t++) {
        device const float* kk = k + (t * heads + head) * dim;
        device const float* vv = v + (t * heads + head) * dim;
        float partial = 0.0f;
        for (uint d = lid; d < dim; d += 32u) {
            partial += qq[d] * kk[d];
        }
        float dot = simd_sum(partial) * scale;
        float weight = exp(dot - running_max) / running_sum;
        for (uint d = lid; d < dim; d += 32u) {
            dst[d] += weight * vv[d];
        }
    }
}

kernel void patch_embed(
    device const float* volume [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device float* tokens [[buffer(3)]],
    constant uint& channels [[buffer(4)]],
    constant uint& time [[buffer(5)]],
    constant uint& height [[buffer(6)]],
    constant uint& width [[buffer(7)]],
    constant uint& dim [[buffer(8)]],
    uint token [[thread_position_in_grid]])
{
    uint grid_h = height / 2u;
    uint grid_w = width / 2u;
    uint spatial = grid_h * grid_w;
    if (token >= time * spatial) {
        return;
    }
    uint t = token / spatial;
    uint rem = token - t * spatial;
    uint ph = rem / grid_w;
    uint pw = rem - ph * grid_w;
    for (uint o = 0u; o < dim; o++) {
        float acc = float(bias[o]);
        for (uint c = 0u; c < channels; c++) {
            for (uint kh = 0u; kh < 2u; kh++) {
                for (uint kw = 0u; kw < 2u; kw++) {
                    uint ih = ph * 2u + kh;
                    uint iw = pw * 2u + kw;
                    float value = volume[((c * time + t) * height + ih) * width + iw];
                    uint index = (((o * channels + c) * 2u + kh) * 2u + kw);
                    acc += value * float(weight[index]);
                }
            }
        }
        tokens[token * dim + o] = acc;
    }
}

kernel void conv3d(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& cin [[buffer(4)]],
    constant uint& cout [[buffer(5)]],
    constant uint& tin [[buffer(6)]],
    constant uint& hin [[buffer(7)]],
    constant uint& win [[buffer(8)]],
    constant uint& tout [[buffer(9)]],
    constant uint& hout [[buffer(10)]],
    constant uint& wout [[buffer(11)]],
    constant uint& kt [[buffer(12)]],
    constant uint& kh [[buffer(13)]],
    constant uint& kw [[buffer(14)]],
    constant uint& stride_t [[buffer(15)]],
    constant uint& stride_h [[buffer(16)]],
    constant uint& stride_w [[buffer(17)]],
    constant int& pad_t [[buffer(18)]],
    constant int& pad_h [[buffer(19)]],
    constant int& pad_w [[buffer(20)]],
    uint gid [[thread_position_in_grid]])
{
    uint count = cout * tout * hout * wout;
    if (gid >= count) {
        return;
    }
    uint rem = gid;
    uint ow = rem % wout;
    rem /= wout;
    uint oh = rem % hout;
    rem /= hout;
    uint ot = rem % tout;
    uint oc = rem / tout;
    float acc = bias[oc];
    int t0 = int(ot) * int(stride_t) - pad_t;
    int h0 = int(oh) * int(stride_h) - pad_h;
    int w0 = int(ow) * int(stride_w) - pad_w;
    for (uint ic = 0u; ic < cin; ic++) {
        for (uint z = 0u; z < kt; z++) {
            int t = t0 + int(z);
            if (t < 0 || t >= int(tin)) {
                continue;
            }
            for (uint y = 0u; y < kh; y++) {
                int h = h0 + int(y);
                if (h < 0 || h >= int(hin)) {
                    continue;
                }
                for (uint x = 0u; x < kw; x++) {
                    int w = w0 + int(x);
                    if (w < 0 || w >= int(win)) {
                        continue;
                    }
                    uint in_index = (((ic * tin + uint(t)) * hin + uint(h)) * win + uint(w));
                    uint weight_index = ((((oc * cin + ic) * kt + z) * kh + y) * kw + x);
                    acc += input[in_index] * weight[weight_index];
                }
            }
        }
    }
    output[(((oc * tout + ot) * hout + oh) * wout + ow)] = acc;
}

kernel void conv_pack(
    device const float* weight [[buffer(0)]],
    device half* packed [[buffer(1)]],
    constant uint& cout [[buffer(2)]],
    constant uint& cin [[buffer(3)]],
    constant uint& kt [[buffer(4)]],
    constant uint& kh [[buffer(5)]],
    constant uint& kw [[buffer(6)]],
    constant uint& k_raw [[buffer(7)]],
    constant uint& k_pad [[buffer(8)]],
    constant uint& n_pad [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint k = gid.x;
    uint n = gid.y;
    if (k >= k_pad || n >= n_pad) {
        return;
    }
    if (k >= k_raw || n >= cout) {
        packed[n * k_pad + k] = half(0.0f);
        return;
    }
    uint x = k % kw;
    uint tmp = k / kw;
    uint y = tmp % kh;
    tmp /= kh;
    uint z = tmp % kt;
    uint ic = tmp / kt;
    uint index = (((n * cin + ic) * kt + z) * kh + y) * kw + x;
    packed[n * k_pad + k] = half(weight[index]);
}

kernel void conv_im2col(
    device const float* input [[buffer(0)]],
    device half* columns [[buffer(1)]],
    constant uint& tin [[buffer(2)]],
    constant uint& hin [[buffer(3)]],
    constant uint& win [[buffer(4)]],
    constant uint& kt [[buffer(5)]],
    constant uint& kh [[buffer(6)]],
    constant uint& kw [[buffer(7)]],
    constant uint& stride_t [[buffer(8)]],
    constant uint& stride_h [[buffer(9)]],
    constant uint& stride_w [[buffer(10)]],
    constant int& pad_t [[buffer(11)]],
    constant int& pad_h [[buffer(12)]],
    constant int& pad_w [[buffer(13)]],
    constant uint& hout [[buffer(14)]],
    constant uint& wout [[buffer(15)]],
    constant uint& k_raw [[buffer(16)]],
    constant uint& k_pad [[buffer(17)]],
    constant uint& row0 [[buffer(18)]],
    constant uint& rows_valid [[buffer(19)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint k = gid.x;
    uint row = gid.y;
    if (k >= k_pad) {
        return;
    }
    if (row >= rows_valid || k >= k_raw) {
        columns[row * k_pad + k] = half(0.0f);
        return;
    }
    uint spatial = hout * wout;
    uint global = row0 + row;
    uint ot = global / spatial;
    uint rem = global - ot * spatial;
    uint oh = rem / wout;
    uint ow = rem - oh * wout;
    uint x = k % kw;
    uint tmp = k / kw;
    uint y = tmp % kh;
    tmp /= kh;
    uint z = tmp % kt;
    uint ic = tmp / kt;
    int t = int(ot) * int(stride_t) - pad_t + int(z);
    int h = int(oh) * int(stride_h) - pad_h + int(y);
    int w = int(ow) * int(stride_w) - pad_w + int(x);
    float value = 0.0f;
    if (t >= 0 && h >= 0 && w >= 0 && t < int(tin) && h < int(hin) && w < int(win)) {
        value = input[((ic * tin + uint(t)) * hin + uint(h)) * win + uint(w)];
    }
    columns[row * k_pad + k] = half(value);
}

kernel void conv_scatter(
    device const float* tile [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& cout [[buffer(3)]],
    constant uint& tout [[buffer(4)]],
    constant uint& hout [[buffer(5)]],
    constant uint& wout [[buffer(6)]],
    constant uint& n_pad [[buffer(7)]],
    constant uint& rows [[buffer(8)]],
    constant uint& row0 [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint oc = gid.x;
    uint row = gid.y;
    if (oc >= cout || row >= rows) {
        return;
    }
    uint spatial = hout * wout;
    uint global = row0 + row;
    uint ot = global / spatial;
    uint rem = global - ot * spatial;
    uint oh = rem / wout;
    uint ow = rem - oh * wout;
    float value = tile[row * n_pad + oc] + bias[oc];
    output[(((oc * tout + ot) * hout + oh) * wout + ow)] = value;
}

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

kernel void cast_half(
    device const float* source [[buffer(0)]],
    device half* destination [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) {
        return;
    }
    destination[gid] = half(source[gid]);
}

kernel void add_f32(
    device float* left [[buffer(0)]],
    device const float* right [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) {
        return;
    }
    left[gid] += right[gid];
}

kernel void mul_f32(
    device float* left [[buffer(0)]],
    device const float* right [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) {
        return;
    }
    left[gid] *= right[gid];
}

kernel void add_half(
    device const float* left [[buffer(0)]],
    device const half* right [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) {
        return;
    }
    destination[gid] = left[gid] + float(right[gid]);
}

kernel void add_embed_half(
    device const half* modulation [[buffer(0)]],
    device const float* embed [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& slices [[buffer(3)]],
    constant uint& dim [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint count = slices * dim;
    if (gid >= count) {
        return;
    }
    destination[gid] = float(modulation[gid]) + embed[gid % dim];
}

kernel void copy_f32(
    device const float* source [[buffer(0)]],
    device float* destination [[buffer(1)]],
    constant uint& offset [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) {
        return;
    }
    destination[gid] = source[offset + gid];
}

kernel void axpy(
    device float* left [[buffer(0)]],
    device const float* right [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) {
        return;
    }
    left[gid] += scale * right[gid];
}

kernel void sinusoid(
    device float* destination [[buffer(0)]],
    constant uint& freq [[buffer(1)]],
    constant float& position [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= freq) {
        return;
    }
    uint mid = freq / 2u;
    float unit = float(gid % mid) / float(mid);
    float angle = position * pow(10000.0f, -unit);
    destination[gid] = (gid < mid) ? cos(angle) : sin(angle);
}

kernel void unpatchify(
    device const float* tokens [[buffer(0)]],
    device float* latent [[buffer(1)]],
    constant uint& time [[buffer(2)]],
    constant uint& height [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint hout = height * 2u;
    uint wout = width * 2u;
    uint count = 16u * time * hout * wout;
    if (gid >= count) {
        return;
    }
    uint w = gid % wout;
    uint rem = gid / wout;
    uint h = rem % hout;
    rem /= hout;
    uint t = rem % time;
    uint c = rem / time;
    uint ph = h / 2u;
    uint pw = w / 2u;
    uint kh = h - ph * 2u;
    uint kw = w - pw * 2u;
    uint token = (t * height + ph) * width + pw;
    uint inner = (kh * 2u + kw) * 16u + c;
    latent[gid] = tokens[token * 64u + inner];
}

kernel void condition36(
    device const float* noise [[buffer(0)]],
    device const float* mask [[buffer(1)]],
    device const float* latent [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& time [[buffer(4)]],
    constant uint& height [[buffer(5)]],
    constant uint& width [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    uint spatial = height * width;
    uint plane = time * spatial;
    uint count = 36u * plane;
    if (gid >= count) {
        return;
    }
    uint c = gid / plane;
    uint rest = gid - c * plane;
    uint t = rest / spatial;
    if (c < 16u) {
        destination[gid] = noise[gid];
    } else if (c < 20u) {
        destination[gid] = mask[(c - 16u) * time + t];
    } else {
        destination[gid] = latent[(c - 20u) * plane + rest];
    }
}

kernel void affine_mu(
    device const float* source [[buffer(0)]],
    device const float* mean [[buffer(1)]],
    device const float* std [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& plane [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint count = 16u * plane;
    if (gid >= count) {
        return;
    }
    uint c = gid / plane;
    destination[gid] = (source[gid] - mean[c]) / std[c];
}

kernel void affine_inv(
    device const float* source [[buffer(0)]],
    device const float* mean [[buffer(1)]],
    device const float* std [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& plane [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint count = 16u * plane;
    if (gid >= count) {
        return;
    }
    uint c = gid / plane;
    destination[gid] = source[gid] * std[c] + mean[c];
}

kernel void channel_rms(
    device const float* source [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& channels [[buffer(3)]],
    constant uint& plane [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= plane) {
        return;
    }
    float sum = 0.0f;
    for (uint c = 0u; c < channels; c++) {
        float value = source[c * plane + gid];
        sum += value * value;
    }
    float norm = max(sqrt(sum), eps);
    float scale = sqrt(float(channels));
    for (uint c = 0u; c < channels; c++) {
        destination[c * plane + gid] = source[c * plane + gid] / norm * scale * gamma[c];
    }
}

kernel void silu_clip(
    device const float* source [[buffer(0)]],
    device float* destination [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) {
        return;
    }
    float value = clamp(source[gid], -60.0f, 60.0f);
    destination[gid] = value / (1.0f + exp(-value));
}

kernel void spatial_attend(
    device const float* qkv [[buffer(0)]],
    device float* destination [[buffer(1)]],
    constant uint& channels [[buffer(2)]],
    constant uint& time [[buffer(3)]],
    constant uint& height [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint spatial = height * width;
    uint count = time * spatial;
    if (gid >= count || spatial == 0u) {
        return;
    }
    uint t = gid / spatial;
    uint pix = gid - t * spatial;
    uint plane = time * spatial;
    float scale = rsqrt(float(channels));
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    for (uint key = 0u; key < spatial; key++) {
        float dot = 0.0f;
        for (uint c = 0u; c < channels; c++) {
            float q = qkv[c * plane + t * spatial + pix];
            float k = qkv[(channels + c) * plane + t * spatial + key];
            dot += q * k;
        }
        dot *= scale;
        float next = max(running_max, dot);
        running_sum = running_sum * exp(running_max - next) + exp(dot - next);
        running_max = next;
    }
    for (uint c = 0u; c < channels; c++) {
        destination[c * plane + gid] = 0.0f;
    }
    for (uint key = 0u; key < spatial; key++) {
        float dot = 0.0f;
        for (uint c = 0u; c < channels; c++) {
            float q = qkv[c * plane + t * spatial + pix];
            float k = qkv[(channels + c) * plane + t * spatial + key];
            dot += q * k;
        }
        float weight = exp(dot * scale - running_max) / running_sum;
        for (uint c = 0u; c < channels; c++) {
            float v = qkv[(2u * channels + c) * plane + t * spatial + key];
            destination[c * plane + gid] += weight * v;
        }
    }
}

kernel void end_pad(
    device const float* source [[buffer(0)]],
    device float* destination [[buffer(1)]],
    constant uint& channels [[buffer(2)]],
    constant uint& time [[buffer(3)]],
    constant uint& height [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint hout = height + 1u;
    uint wout = width + 1u;
    uint count = channels * time * hout * wout;
    if (gid >= count) {
        return;
    }
    uint w = gid % wout;
    uint rem = gid / wout;
    uint h = rem % hout;
    rem /= hout;
    uint t = rem % time;
    uint c = rem / time;
    if (h >= height || w >= width) {
        destination[gid] = 0.0f;
        return;
    }
    uint in_index = ((c * time + t) * height + h) * width + w;
    destination[gid] = source[in_index];
}

kernel void nearest2(
    device const float* source [[buffer(0)]],
    device float* destination [[buffer(1)]],
    constant uint& channels [[buffer(2)]],
    constant uint& time [[buffer(3)]],
    constant uint& height [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint hout = height * 2u;
    uint wout = width * 2u;
    uint count = channels * time * hout * wout;
    if (gid >= count) {
        return;
    }
    uint w = gid % wout;
    uint rem = gid / wout;
    uint h = rem % hout;
    rem /= hout;
    uint t = rem % time;
    uint c = rem / time;
    uint in_index = ((c * time + t) * height + (h / 2u)) * width + (w / 2u);
    destination[gid] = source[in_index];
}

kernel void concat_time(
    device const float* left [[buffer(0)]],
    device const float* right [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& channels [[buffer(3)]],
    constant uint& left_time [[buffer(4)]],
    constant uint& right_time [[buffer(5)]],
    constant uint& height [[buffer(6)]],
    constant uint& width [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    uint spatial = height * width;
    uint total = left_time + right_time;
    uint count = channels * total * spatial;
    if (gid >= count) {
        return;
    }
    uint rest = gid % (total * spatial);
    uint c = gid / (total * spatial);
    uint t = rest / spatial;
    uint pix = rest - t * spatial;
    if (t < left_time) {
        destination[gid] = left[(c * left_time + t) * spatial + pix];
    } else {
        uint src = t - left_time;
        destination[gid] = right[(c * right_time + src) * spatial + pix];
    }
}

kernel void stack_pairs(
    device const float* source [[buffer(0)]],
    device float* destination [[buffer(1)]],
    constant uint& channels [[buffer(2)]],
    constant uint& time [[buffer(3)]],
    constant uint& height [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint spatial = height * width;
    uint count = channels * (time * 2u) * spatial;
    if (gid >= count) {
        return;
    }
    uint pix = gid % spatial;
    uint rem = gid / spatial;
    uint tout = time * 2u;
    uint t = rem % tout;
    uint c = rem / tout;
    uint src_t = t / 2u;
    uint band = t - src_t * 2u;
    uint src_c = band * channels + c;
    destination[gid] = source[((src_c * time + src_t) * height + (pix / width)) * width + (pix % width)];
}

kernel void slice_time(
    device const float* source [[buffer(0)]],
    device float* destination [[buffer(1)]],
    constant uint& channels [[buffer(2)]],
    constant uint& source_time [[buffer(3)]],
    constant uint& height [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    constant uint& start [[buffer(6)]],
    constant uint& count_time [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    uint spatial = height * width;
    uint count = channels * count_time * spatial;
    if (gid >= count) {
        return;
    }
    uint pix = gid % spatial;
    uint rem = gid / spatial;
    uint t = rem % count_time;
    uint c = rem / count_time;
    uint src = start + t;
    destination[gid] = source[(c * source_time + src) * spatial + pix];
}

kernel void attend_bias(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* out [[buffer(3)]],
    device const int* buckets [[buffer(4)]],
    device const float* table [[buffer(5)]],
    constant uint& q_len [[buffer(6)]],
    constant uint& kv_len [[buffer(7)]],
    constant uint& heads [[buffer(8)]],
    constant uint& dim [[buffer(9)]],
    uint3 lane [[thread_position_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]])
{
    uint head = group.x;
    uint query = group.y;
    uint lid = lane.x;
    if (head >= heads || query >= q_len || kv_len == 0u) {
        return;
    }
    device const float* qq = q + (query * heads + head) * dim;
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    for (uint t = 0u; t < kv_len; t++) {
        device const float* kk = k + (t * heads + head) * dim;
        float partial = 0.0f;
        for (uint d = lid; d < dim; d += 32u) {
            partial += qq[d] * kk[d];
        }
        int bucket = buckets[query * kv_len + t];
        float dot = simd_sum(partial) + table[uint(bucket) * heads + head];
        float next = max(running_max, dot);
        running_sum = running_sum * exp(running_max - next) + exp(dot - next);
        running_max = next;
    }
    device float* dst = out + (query * heads + head) * dim;
    for (uint d = lid; d < dim; d += 32u) {
        dst[d] = 0.0f;
    }
    for (uint t = 0u; t < kv_len; t++) {
        device const float* kk = k + (t * heads + head) * dim;
        device const float* vv = v + (t * heads + head) * dim;
        float partial = 0.0f;
        for (uint d = lid; d < dim; d += 32u) {
            partial += qq[d] * kk[d];
        }
        int bucket = buckets[query * kv_len + t];
        float dot = simd_sum(partial) + table[uint(bucket) * heads + head];
        float weight = exp(dot - running_max) / running_sum;
        for (uint d = lid; d < dim; d += 32u) {
            dst[d] += weight * vv[d];
        }
    }
}

kernel void pixels_u8(
    device const float* source [[buffer(0)]],
    device uchar* destination [[buffer(1)]],
    constant uint& frames [[buffer(2)]],
    constant uint& height [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint spatial = height * width;
    uint count = frames * spatial;
    if (gid >= count) {
        return;
    }
    uint plane = frames * spatial;
    for (uint c = 0u; c < 3u; c++) {
        float value = clamp(source[c * plane + gid], -1.0f, 1.0f);
        float scaled = rint((value + 1.0f) * 127.5f);
        destination[gid * 3u + c] = uchar(clamp(scaled, 0.0f, 255.0f));
    }
}
