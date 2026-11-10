#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

using scale_plane = tensor_blockwise<tensor_plane_scales, device metal_fp8_ue8m0_format, 32, 1>;
using fp4_tensor = tensor<device metal_fp4_e2m1_format, dextents<int, 2>, tensor_inline, scale_plane>;
using fp8_tensor = tensor<device metal_fp8_e4m3_format, dextents<int, 2>, tensor_inline, scale_plane>;

struct GemmDims {
    int m;
    int n;
    int k;
    int stride_a;
    int stride_b;
};

kernel void gemm_f16_tn(
    device half *left [[buffer(0)]],
    device half *right [[buffer(1)]],
    device float *dest [[buffer(2)]],
    constant GemmDims &dims [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]])
{
    constexpr int tile_m = 64;
    constexpr int tile_n = 32;
    constexpr auto desc = matmul2d_descriptor(tile_m, tile_n, dynamic_length_v<int>, false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto a = tensor(left, dextents<int, 2>(dims.k, dims.m), array<int, 2>{1, dims.stride_a});
    auto b = tensor(right, dextents<int, 2>(dims.k, dims.n), array<int, 2>{1, dims.stride_b});
    auto c = tensor(dest, dextents<int, 2>(dims.n, dims.m), array<int, 2>{1, dims.n});
    int row = int(group.y) * tile_m;
    int col = int(group.x) * tile_n;
    auto tile_a = a.slice<dynamic_extent, tile_m>(0, row);
    auto tile_b = b.slice<dynamic_extent, tile_n>(0, col);
    auto tile_c = c.slice<tile_n, tile_m>(col, row);
    op.run(tile_a, tile_b, tile_c);
}

kernel void gemm_f16_tn64(
    device half *left [[buffer(0)]],
    device half *right [[buffer(1)]],
    device float *dest [[buffer(2)]],
    constant GemmDims &dims [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]])
{
    constexpr int tile_m = 64;
    constexpr int tile_n = 64;
    constexpr auto desc = matmul2d_descriptor(tile_m, tile_n, dynamic_length_v<int>, false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto a = tensor(left, dextents<int, 2>(dims.k, dims.m), array<int, 2>{1, dims.stride_a});
    auto b = tensor(right, dextents<int, 2>(dims.k, dims.n), array<int, 2>{1, dims.stride_b});
    auto c = tensor(dest, dextents<int, 2>(dims.n, dims.m), array<int, 2>{1, dims.n});
    int row = int(group.y) * tile_m;
    int col = int(group.x) * tile_n;
    auto tile_a = a.slice<dynamic_extent, tile_m>(0, row);
    auto tile_b = b.slice<dynamic_extent, tile_n>(0, col);
    auto tile_c = c.slice<tile_n, tile_m>(col, row);
    op.run(tile_a, tile_b, tile_c);
}

kernel void gemm_f16_nn(
    device half *left [[buffer(0)]],
    device half *right [[buffer(1)]],
    device float *dest [[buffer(2)]],
    constant GemmDims &dims [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]])
{
    constexpr int tile_m = 64;
    constexpr int tile_n = 32;
    constexpr auto desc = matmul2d_descriptor(tile_m, tile_n, dynamic_length_v<int>, false, false, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto a = tensor(left, dextents<int, 2>(dims.k, dims.m), array<int, 2>{1, dims.stride_a});
    auto b = tensor(right, dextents<int, 2>(dims.n, dims.k), array<int, 2>{1, dims.stride_b});
    auto c = tensor(dest, dextents<int, 2>(dims.n, dims.m), array<int, 2>{1, dims.n});
    int row = int(group.y) * tile_m;
    int col = int(group.x) * tile_n;
    auto tile_a = a.slice<dynamic_extent, tile_m>(0, row);
    auto tile_b = b.slice<tile_n, dynamic_extent>(col, 0);
    auto tile_c = c.slice<tile_n, tile_m>(col, row);
    op.run(tile_a, tile_b, tile_c);
}

kernel void gemm_fp4(
    device half *left [[buffer(0)]],
    device uchar *right [[buffer(1)]],
    device uchar *scales [[buffer(2)]],
    device float *dest [[buffer(3)]],
    constant GemmDims &dims [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]])
{
    constexpr int tile_m = 64;
    constexpr int tile_n = 32;
    constexpr auto desc = matmul2d_descriptor(tile_m, tile_n, dynamic_length_v<int>, false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto a = tensor(left, dextents<int, 2>(dims.k, dims.m), array<int, 2>{1, dims.stride_a});
    fp4_tensor b(right, dextents<int, 2>(dims.k, dims.n), scale_plane(scales));
    auto c = tensor(dest, dextents<int, 2>(dims.n, dims.m), array<int, 2>{1, dims.n});
    int row = int(group.y) * tile_m;
    int col = int(group.x) * tile_n;
    auto tile_a = a.slice<dynamic_extent, tile_m>(0, row);
    auto tile_b = b.slice<dynamic_extent, tile_n>(0, col);
    auto tile_c = c.slice<tile_n, tile_m>(col, row);
    op.run(tile_a, tile_b, tile_c);
}

kernel void gemm_fp8(
    device half *left [[buffer(0)]],
    device uchar *right [[buffer(1)]],
    device uchar *scales [[buffer(2)]],
    device float *dest [[buffer(3)]],
    constant GemmDims &dims [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]])
{
    constexpr int tile_m = 64;
    constexpr int tile_n = 32;
    constexpr auto desc = matmul2d_descriptor(tile_m, tile_n, dynamic_length_v<int>, false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto a = tensor(left, dextents<int, 2>(dims.k, dims.m), array<int, 2>{1, dims.stride_a});
    fp8_tensor b(right, dextents<int, 2>(dims.k, dims.n), scale_plane(scales));
    auto c = tensor(dest, dextents<int, 2>(dims.n, dims.m), array<int, 2>{1, dims.n});
    int row = int(group.y) * tile_m;
    int col = int(group.x) * tile_n;
    auto tile_a = a.slice<dynamic_extent, tile_m>(0, row);
    auto tile_b = b.slice<dynamic_extent, tile_n>(0, col);
    auto tile_c = c.slice<tile_n, tile_m>(col, row);
    op.run(tile_a, tile_b, tile_c);
}

kernel void gemm_i8(
    device int8_t *left [[buffer(0)]],
    device int8_t *right [[buffer(1)]],
    device int *dest [[buffer(2)]],
    constant GemmDims &dims [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]])
{
    constexpr int tile_m = 64;
    constexpr int tile_n = 128;
    constexpr auto desc = matmul2d_descriptor(tile_m, tile_n, dynamic_length_v<int>, false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto a = tensor(left, dextents<int, 2>(dims.k, dims.m), array<int, 2>{1, dims.stride_a});
    auto b = tensor(right, dextents<int, 2>(dims.k, dims.n), array<int, 2>{1, dims.stride_b});
    auto c = tensor(dest, dextents<int, 2>(dims.n, dims.m), array<int, 2>{1, dims.n});
    int row = int(group.y) * tile_m;
    int col = int(group.x) * tile_n;
    auto tile_a = a.slice<dynamic_extent, tile_m>(0, row);
    auto tile_b = b.slice<dynamic_extent, tile_n>(0, col);
    auto tile_c = c.slice<tile_n, tile_m>(col, row);
    op.run(tile_a, tile_b, tile_c);
}

struct ScaleArgs {
    int rows;
    int cols;
    int logical;
};

kernel void scale_i8(
    device int *acc [[buffer(0)]],
    device const float *row_scale [[buffer(1)]],
    device const float *col_scale [[buffer(2)]],
    device const half *bias [[buffer(3)]],
    constant ScaleArgs &shape [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    int rows = shape.rows;
    int cols = shape.cols;
    int logical = shape.logical;
    if (int(gid) >= rows * cols) {
        return;
    }
    int col = int(gid) % cols;
    int row = int(gid) / cols;
    float value = float(acc[gid]) * row_scale[row] * col_scale[col];
    if (col < logical) {
        value += float(bias[col]);
    }
    reinterpret_cast<device float *>(acc)[gid] = value;
}

kernel void gather_k(
    device const half *source [[buffer(0)]],
    device const uint *perm [[buffer(1)]],
    device half *dest [[buffer(2)]],
    constant int2 &shape [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    int rows = shape.x;
    int cols = shape.y;
    int count = rows * cols;
    if (int(gid) >= count) {
        return;
    }
    int row = int(gid) / cols;
    int col = int(gid) % cols;
    dest[gid] = source[row * cols + int(perm[col])];
}

kernel void bias_to_half(
    device const float *source [[buffer(0)]],
    device const half *bias [[buffer(1)]],
    device half *dest [[buffer(2)]],
    device float *rescale [[buffer(3)]],
    constant int3 &shape [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    int rows = shape.x;
    int cols = shape.y;
    int padded = shape.z;
    if (int(gid) >= rows) {
        return;
    }
    float peak = 0.0f;
    for (int col = 0; col < cols; ++col) {
        float value = source[gid * padded + col];
        if (bias != nullptr) {
            value += float(bias[col]);
        }
        peak = max(peak, fabs(value));
    }
    float limit = 65504.0f;
    float factor = peak > limit ? peak / limit : 1.0f;
    rescale[gid] = factor;
    float inv = 1.0f / factor;
    for (int col = 0; col < cols; ++col) {
        float value = source[gid * padded + col];
        if (bias != nullptr) {
            value += float(bias[col]);
        }
        dest[gid * cols + col] = half(value * inv);
    }
}

kernel void apply_rescale(
    device half *rows [[buffer(0)]],
    device const float *rescale [[buffer(1)]],
    constant int2 &shape [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    int cols = shape.y;
    if (int(gid) >= shape.x * cols) {
        return;
    }
    int row = int(gid) / cols;
    rows[gid] = half(float(rows[gid]) * rescale[row]);
}

kernel void rmsnorm(
    device const half *source [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device half *dest [[buffer(2)]],
    constant int2 &shape [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint width [[threads_per_threadgroup]])
{
    int cols = shape.y;
    threadgroup float scratch[256];
    float sum = 0.0f;
    const device half *line = source + row * cols;
    for (int col = int(lane); col < cols; col += int(width)) {
        float value = float(line[col]);
        sum += value * value;
    }
    scratch[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint span = width / 2; span > 0; span >>= 1) {
        if (lane < span) {
            scratch[lane] += scratch[lane + span];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(scratch[0] / float(cols) + 1e-5f);
    device half *out = dest + row * cols;
    for (int col = int(lane); col < cols; col += int(width)) {
        float value = float(line[col]) * inv;
        if (weight != nullptr) {
            value *= float(weight[col]);
        }
        out[col] = half(value);
    }
}

kernel void layernorm(
    device const half *source [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device const half *bias [[buffer(2)]],
    device half *dest [[buffer(3)]],
    constant int2 &shape [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint width [[threads_per_threadgroup]])
{
    int cols = shape.y;
    threadgroup float scratch[256];
    float sum = 0.0f;
    const device half *line = source + row * cols;
    for (int col = int(lane); col < cols; col += int(width)) {
        sum += float(line[col]);
    }
    scratch[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint span = width / 2; span > 0; span >>= 1) {
        if (lane < span) {
            scratch[lane] += scratch[lane + span];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = scratch[0] / float(cols);
    float var = 0.0f;
    for (int col = int(lane); col < cols; col += int(width)) {
        float delta = float(line[col]) - mean;
        var += delta * delta;
    }
    scratch[lane] = var;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint span = width / 2; span > 0; span >>= 1) {
        if (lane < span) {
            scratch[lane] += scratch[lane + span];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(scratch[0] / float(cols) + 1e-6f);
    device half *out = dest + row * cols;
    for (int col = int(lane); col < cols; col += int(width)) {
        float value = (float(line[col]) - mean) * inv;
        if (weight != nullptr) {
            value *= float(weight[col]);
        }
        if (bias != nullptr) {
            value += float(bias[col]);
        }
        out[col] = half(value);
    }
}

kernel void modulate(
    device const half *source [[buffer(0)]],
    device const half *scale [[buffer(1)]],
    device const half *shift [[buffer(2)]],
    device half *dest [[buffer(3)]],
    constant int3 &shape [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    int rows = shape.x;
    int cols = shape.y;
    int shared = shape.z;
    if (int(gid) >= rows * cols) {
        return;
    }
    int row = int(gid) / cols;
    int col = int(gid) % cols;
    int mod = shared ? 0 : row;
    float value = float(source[gid]) * (1.0f + float(scale[mod * cols + col])) + float(shift[mod * cols + col]);
    dest[gid] = half(value);
}

kernel void residual_gate(
    device const half *residual [[buffer(0)]],
    device const half *update [[buffer(1)]],
    device const half *gate [[buffer(2)]],
    device half *dest [[buffer(3)]],
    constant int3 &shape [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    int rows = shape.x;
    int cols = shape.y;
    int shared = shape.z;
    if (int(gid) >= rows * cols) {
        return;
    }
    int row = int(gid) / cols;
    int col = int(gid) % cols;
    int mod = shared ? 0 : row;
    float value = float(residual[gid]) + float(update[gid]) * float(gate[mod * cols + col]);
    dest[gid] = half(value);
}

kernel void gelu_tanh(
    device const half *source [[buffer(0)]],
    device half *dest [[buffer(1)]],
    constant int &count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (int(gid) >= count) {
        return;
    }
    float value = float(source[gid]);
    float inner = 0.7978845608f * (value + 0.044715f * value * value * value);
    dest[gid] = half(0.5f * value * (1.0f + tanh(inner)));
}

kernel void silu(device half *values [[buffer(0)]], constant int &count [[buffer(1)]], uint gid [[thread_position_in_grid]])
{
    if (int(gid) >= count) {
        return;
    }
    float value = float(values[gid]);
    values[gid] = half(value / (1.0f + exp(-value)));
}

kernel void rope3d(
    device half *tokens [[buffer(0)]],
    device const float *time_table [[buffer(1)]],
    device const float *height_table [[buffer(2)]],
    device const float *width_table [[buffer(3)]],
    constant int4 &grid [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    int tokens_n = grid.x;
    int heads = grid.y;
    int height = grid.z;
    int width = grid.w;
    int pairs = 64;
    int count = tokens_n * heads * pairs;
    if (int(gid) >= count) {
        return;
    }
    int pair = int(gid) % pairs;
    int head = (int(gid) / pairs) % heads;
    int token = int(gid) / (pairs * heads);
    int spatial = height * width;
    int frame = token / spatial;
    int rem = token - frame * spatial;
    int y = rem / width;
    int x = rem - y * width;
    const device float *table;
    int axis;
    int axis_pairs;
    int local;
    if (pair < 22) {
        table = time_table;
        axis = frame;
        axis_pairs = 22;
        local = pair;
    } else if (pair < 43) {
        table = height_table;
        axis = y;
        axis_pairs = 21;
        local = pair - 22;
    } else {
        table = width_table;
        axis = x;
        axis_pairs = 21;
        local = pair - 43;
    }
    float cosine = table[axis * (axis_pairs * 2) + local];
    float sine = table[axis * (axis_pairs * 2) + axis_pairs + local];
    int offset = ((token * heads + head) * 128) + pair * 2;
    float even = float(tokens[offset]);
    float odd = float(tokens[offset + 1]);
    tokens[offset] = half(even * cosine - odd * sine);
    tokens[offset + 1] = half(even * sine + odd * cosine);
}

kernel void add_bias_scores(
    device float *scores [[buffer(0)]],
    device const half *bias [[buffer(1)]],
    constant int3 &shape [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    int rows = shape.x;
    int cols = shape.y;
    int valid = shape.z;
    if (int(gid) >= rows * cols) {
        return;
    }
    int col = int(gid) % cols;
    if (col >= valid || int(gid) / cols >= valid) {
        scores[gid] = -1.0e30f;
        return;
    }
    if (bias != nullptr) {
        scores[gid] += float(bias[gid]);
    }
}

kernel void online_prepare(
    device const float *scores [[buffer(0)]],
    device float *running_max [[buffer(1)]],
    device float *running_sum [[buffer(2)]],
    device float *alpha [[buffer(3)]],
    device half *weights [[buffer(4)]],
    constant int2 &shape [[buffer(5)]],
    uint row [[thread_position_in_grid]])
{
    int cols = shape.y;
    if (int(row) >= shape.x) {
        return;
    }
    const device float *line = scores + row * cols;
    float peak = line[0];
    for (int col = 1; col < cols; ++col) {
        peak = max(peak, line[col]);
    }
    float previous = running_max[row];
    float next = max(previous, peak);
    float scale = exp(previous - next);
    float sum = 0.0f;
    device half *out = weights + row * cols;
    for (int col = 0; col < cols; ++col) {
        float value = exp(line[col] - next);
        sum += value;
        out[col] = half(value);
    }
    running_max[row] = next;
    running_sum[row] = running_sum[row] * scale + sum;
    alpha[row] = scale;
}

kernel void online_merge(
    device float *acc [[buffer(0)]],
    device const float *extra [[buffer(1)]],
    device const float *alpha [[buffer(2)]],
    constant int2 &shape [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    int cols = shape.y;
    if (int(gid) >= shape.x * cols) {
        return;
    }
    int row = int(gid) / cols;
    acc[gid] = acc[gid] * alpha[row] + extra[gid];
}

kernel void online_finish(
    device const float *acc [[buffer(0)]],
    device const float *running_sum [[buffer(1)]],
    device half *dest [[buffer(2)]],
    constant int3 &shape [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    int cols = shape.y;
    int stride = shape.z;
    if (int(gid) >= shape.x * cols) {
        return;
    }
    int row = int(gid) / cols;
    int col = int(gid) % cols;
    float denom = running_sum[row];
    float value = denom > 0.0f ? acc[gid] / denom : 0.0f;
    dest[row * stride + col] = half(value);
}

kernel void blend_latent(
    device float *latent [[buffer(0)]],
    device const float *image [[buffer(1)]],
    device const float *mask [[buffer(2)]],
    constant int &count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (int(gid) >= count) {
        return;
    }
    float keep = mask[gid];
    latent[gid] = (1.0f - keep) * image[gid] + keep * latent[gid];
}

struct ConvArgs {
    int out_c;
    int in_c;
    int frames;
    int height;
    int width;
    int kt;
    int kh;
    int kw;
    int pad_t;
    int pad_h;
    int pad_w;
    int stride_t;
    int stride_h;
    int stride_w;
    int out_t;
    int out_h;
    int out_w;
    int k_pad;
    int row0;
};

kernel void conv3d_f32(
    device const float *source [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device float *dest [[buffer(3)]],
    constant ConvArgs &args [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    int volume = args.out_c * args.out_t * args.out_h * args.out_w;
    if (int(gid) >= volume) {
        return;
    }
    int spatial = args.out_h * args.out_w;
    int plane = spatial * args.out_t;
    int oc = int(gid) / plane;
    int rest = int(gid) - oc * plane;
    int frame = rest / spatial;
    int pix = rest - frame * spatial;
    int y = pix / args.out_w;
    int x = pix - y * args.out_w;
    float acc = bias != nullptr ? bias[oc] : 0.0f;
    int taps = args.in_c * args.kt * args.kh * args.kw;
    for (int tap = 0; tap < taps; ++tap) {
        int kw = tap % args.kw;
        int kh = (tap / args.kw) % args.kh;
        int kt = (tap / (args.kw * args.kh)) % args.kt;
        int ic = tap / (args.kw * args.kh * args.kt);
        int in_t = frame * args.stride_t + kt - args.pad_t;
        int in_y = y * args.stride_h + kh - args.pad_h;
        int in_x = x * args.stride_w + kw - args.pad_w;
        if (in_t < 0 || in_y < 0 || in_x < 0 || in_t >= args.frames || in_y >= args.height || in_x >= args.width) {
            continue;
        }
        int in_index = ((ic * args.frames + in_t) * args.height + in_y) * args.width + in_x;
        acc += source[in_index] * weight[(oc * taps) + tap];
    }
    dest[gid] = acc;
}

kernel void im2col_f16(
    device const float *source [[buffer(0)]],
    device half *columns [[buffer(1)]],
    constant ConvArgs &args [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    int outputs = args.out_t * args.out_h * args.out_w;
    int taps = args.in_c * args.kt * args.kh * args.kw;
    int local = int(gid) / taps;
    int out_index = local + args.row0;
    if (out_index >= outputs) {
        return;
    }
    int spatial = args.out_h * args.out_w;
    int tap = int(gid) - local * taps;
    int stride = args.k_pad > 0 ? args.k_pad : taps;
    int frame = out_index / spatial;
    int pix = out_index - frame * spatial;
    int y = pix / args.out_w;
    int x = pix - y * args.out_w;
    int kw = tap % args.kw;
    int kh = (tap / args.kw) % args.kh;
    int kt = (tap / (args.kw * args.kh)) % args.kt;
    int ic = tap / (args.kw * args.kh * args.kt);
    int in_t = frame * args.stride_t + kt - args.pad_t;
    int in_y = y * args.stride_h + kh - args.pad_h;
    int in_x = x * args.stride_w + kw - args.pad_w;
    float value = 0.0f;
    if (in_t >= 0 && in_y >= 0 && in_x >= 0 && in_t < args.frames && in_y < args.height && in_x < args.width) {
        int in_index = ((ic * args.frames + in_t) * args.height + in_y) * args.width + in_x;
        value = source[in_index];
    }
    columns[local * stride + tap] = half(value);
}

struct RowArgs {
    int rows;
    int cols;
    int has_weight;
    int has_bias;
    float eps;
};

kernel void layernorm_f32(
    device const float *source [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device float *dest [[buffer(3)]],
    constant RowArgs &args [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint width [[threads_per_threadgroup]])
{
    if (int(row) >= args.rows) {
        return;
    }
    int cols = args.cols;
    threadgroup float scratch[256];
    const device float *line = source + row * cols;
    float sum = 0.0f;
    for (int col = int(lane); col < cols; col += int(width)) {
        sum += line[col];
    }
    scratch[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint span = width / 2; span > 0; span >>= 1) {
        if (lane < span) {
            scratch[lane] += scratch[lane + span];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = scratch[0] / float(cols);
    float var = 0.0f;
    for (int col = int(lane); col < cols; col += int(width)) {
        float delta = line[col] - mean;
        var += delta * delta;
    }
    scratch[lane] = var;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint span = width / 2; span > 0; span >>= 1) {
        if (lane < span) {
            scratch[lane] += scratch[lane + span];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(scratch[0] / float(cols) + args.eps);
    device float *out = dest + row * cols;
    for (int col = int(lane); col < cols; col += int(width)) {
        float value = (line[col] - mean) * inv;
        if (args.has_weight != 0) {
            value *= weight[col];
        }
        if (args.has_bias != 0) {
            value += bias[col];
        }
        out[col] = value;
    }
}

kernel void rmsnorm_f32(
    device const float *source [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *dest [[buffer(2)]],
    constant RowArgs &args [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint width [[threads_per_threadgroup]])
{
    if (int(row) >= args.rows) {
        return;
    }
    int cols = args.cols;
    threadgroup float scratch[256];
    const device float *line = source + row * cols;
    float sum = 0.0f;
    for (int col = int(lane); col < cols; col += int(width)) {
        float value = line[col];
        sum += value * value;
    }
    scratch[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint span = width / 2; span > 0; span >>= 1) {
        if (lane < span) {
            scratch[lane] += scratch[lane + span];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(scratch[0] / float(cols) + args.eps);
    device float *out = dest + row * cols;
    for (int col = int(lane); col < cols; col += int(width)) {
        float value = line[col] * inv;
        if (args.has_weight != 0) {
            value *= weight[col];
        }
        out[col] = value;
    }
}

kernel void modulate_f32(
    device const float *source [[buffer(0)]],
    device const float *proj_scale [[buffer(1)]],
    device const float *proj_shift [[buffer(2)]],
    device const float *mod_scale [[buffer(3)]],
    device const float *mod_shift [[buffer(4)]],
    device float *dest [[buffer(5)]],
    constant int2 &shape [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    int cols = shape.y;
    if (int(gid) >= shape.x * cols) {
        return;
    }
    int col = int(gid) % cols;
    float scale = proj_scale[gid] + mod_scale[col];
    float shift = proj_shift[gid] + mod_shift[col];
    dest[gid] = source[gid] * (1.0f + scale) + shift;
}

kernel void residual_f32(
    device const float *residual [[buffer(0)]],
    device const float *update [[buffer(1)]],
    device const float *proj_gate [[buffer(2)]],
    device const float *mod_gate [[buffer(3)]],
    device float *dest [[buffer(4)]],
    constant int2 &shape [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    int cols = shape.y;
    if (int(gid) >= shape.x * cols) {
        return;
    }
    int col = int(gid) % cols;
    float gate = proj_gate[gid] + mod_gate[col];
    dest[gid] = residual[gid] + update[gid] * gate;
}

kernel void silu_f32(
    device float *values [[buffer(0)]],
    constant int &count [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (int(gid) >= count) {
        return;
    }
    float value = values[gid];
    values[gid] = value / (1.0f + exp(-value));
}

kernel void rms_nchw_f32(
    device float *values [[buffer(0)]],
    device const float *gamma [[buffer(1)]],
    constant int2 &shape [[buffer(2)]],
    uint site [[thread_position_in_grid]])
{
    int channels = shape.x;
    int sites = shape.y;
    if (int(site) >= sites) {
        return;
    }
    float sum = 0.0f;
    for (int channel = 0; channel < channels; ++channel) {
        float value = values[channel * sites + site];
        sum += value * value;
    }
    float inv = rsqrt(sum / float(channels) + 1.0e-6f);
    for (int channel = 0; channel < channels; ++channel) {
        values[channel * sites + site] = values[channel * sites + site] * inv * gamma[channel];
    }
}

struct EpiArgs {
    int out_c;
    int padded_n;
    int out_t;
    int out_h;
    int out_w;
    float limit;
};

kernel void conv_epilogue_nchw(
    device const float *packed [[buffer(0)]],
    device const float *bias [[buffer(1)]],
    device float *dest [[buffer(2)]],
    constant EpiArgs &args [[buffer(3)]],
    uint pix [[thread_position_in_grid]])
{
    int spatial = args.out_h * args.out_w;
    int outputs = args.out_t * spatial;
    if (int(pix) >= outputs) {
        return;
    }
    int frame = int(pix) / spatial;
    int rem = int(pix) - frame * spatial;
    int y = rem / args.out_w;
    int x = rem - y * args.out_w;
    const device float *row = packed + int(pix) * args.padded_n;
    float peak = 0.0f;
    for (int channel = 0; channel < args.out_c; ++channel) {
        peak = max(peak, fabs(row[channel] + bias[channel]));
    }
    float factor = peak > args.limit ? peak / args.limit : 1.0f;
    int plane = args.out_t * spatial;
    for (int channel = 0; channel < args.out_c; ++channel) {
        float value = (row[channel] + bias[channel]) / factor;
        float stored = float(half(value)) * factor;
        dest[(channel * args.out_t + frame) * spatial + y * args.out_w + x] = stored;
    }
}

kernel void pack_time_tail(
    device const float *source [[buffer(0)]],
    device float *tail [[buffer(1)]],
    constant int *args [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    int channels = args[0];
    int frames = args[1];
    int height = args[2];
    int width = args[3];
    int spatial = height * width;
    int take = min(args[4], frames);
    int volume = channels * take * spatial;
    if (int(gid) >= volume) {
        return;
    }
    int per = take * spatial;
    int channel = int(gid) / per;
    int rem = int(gid) - channel * per;
    int local = rem / spatial;
    int pix = rem - local * spatial;
    int frame = frames - take + local;
    tail[int(gid)] = source[(channel * frames + frame) * spatial + pix];
}

kernel void nchw_prepend(
    device const float *previous [[buffer(0)]],
    device const float *source [[buffer(1)]],
    device float *dest [[buffer(2)]],
    constant int4 &shape [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    int channels = shape.x;
    int prev_t = shape.y;
    int src_t = shape.z;
    int spatial = shape.w;
    int total = prev_t + src_t;
    int volume = channels * total * spatial;
    if (int(gid) >= volume) {
        return;
    }
    int per = total * spatial;
    int channel = int(gid) / per;
    int rem = int(gid) - channel * per;
    int frame = rem / spatial;
    int pix = rem - frame * spatial;
    if (frame < prev_t) {
        dest[int(gid)] = previous[(channel * prev_t + frame) * spatial + pix];
    } else {
        dest[int(gid)] = source[(channel * src_t + frame - prev_t) * spatial + pix];
    }
}

kernel void gelu_tanh_f32(
    device float *values [[buffer(0)]],
    constant int &count [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (int(gid) >= count) {
        return;
    }
    float value = values[gid];
    float inner = 0.7978845608f * (value + 0.044715f * value * value * value);
    values[gid] = 0.5f * value * (1.0f + tanh(inner));
}

kernel void quantize_i8(
    device const float *source [[buffer(0)]],
    device char *codes [[buffer(1)]],
    device float *row_scale [[buffer(2)]],
    constant int2 &shape [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint width [[threads_per_threadgroup]])
{
    if (int(row) >= shape.x) {
        return;
    }
    int cols = shape.y;
    threadgroup float scratch[256];
    const device float *line = source + row * cols;
    float peak = 0.0f;
    for (int col = int(lane); col < cols; col += int(width)) {
        peak = max(peak, fabs(line[col]));
    }
    scratch[lane] = peak;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint span = width / 2; span > 0; span >>= 1) {
        if (lane < span) {
            scratch[lane] = max(scratch[lane], scratch[lane + span]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float top = scratch[0];
    float inverse = top > 0.0f ? 127.0f / top : 0.0f;
    if (lane == 0) {
        row_scale[row] = top > 0.0f ? top / 127.0f : 0.0f;
    }
    device char *out = codes + row * cols;
    for (int col = int(lane); col < cols; col += int(width)) {
        float rounded = rint(line[col] * inverse);
        rounded = clamp(rounded, -127.0f, 127.0f);
        out[col] = char(rounded);
    }
}

kernel void copy_f32(
    device const float *source [[buffer(0)]],
    device float *dest [[buffer(1)]],
    constant int &count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (int(gid) >= count) {
        return;
    }
    dest[gid] = source[gid];
}

kernel void cast_f16(
    device const float *source [[buffer(0)]],
    device half *dest [[buffer(1)]],
    constant int &count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (int(gid) >= count) {
        return;
    }
    dest[gid] = half(source[gid]);
}

kernel void rope_f32(
    device float *tokens [[buffer(0)]],
    device const float *time_table [[buffer(1)]],
    device const float *height_table [[buffer(2)]],
    device const float *width_table [[buffer(3)]],
    constant int4 &grid [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    int tokens_n = grid.x;
    int heads = grid.y;
    int height = grid.z;
    int width = grid.w;
    int pairs = 64;
    int count = tokens_n * heads * pairs;
    if (int(gid) >= count) {
        return;
    }
    int pair = int(gid) % pairs;
    int head = (int(gid) / pairs) % heads;
    int token = int(gid) / (pairs * heads);
    int spatial = height * width;
    int frame = token / spatial;
    int rem = token - frame * spatial;
    int y = rem / width;
    int x = rem - y * width;
    const device float *table;
    int axis;
    int axis_pairs;
    int local;
    if (pair < 22) {
        table = time_table;
        axis = frame;
        axis_pairs = 22;
        local = pair;
    } else if (pair < 43) {
        table = height_table;
        axis = y;
        axis_pairs = 21;
        local = pair - 22;
    } else {
        table = width_table;
        axis = x;
        axis_pairs = 21;
        local = pair - 43;
    }
    float cosine = table[axis * (axis_pairs * 2) + local];
    float sine = table[axis * (axis_pairs * 2) + axis_pairs + local];
    int offset = ((token * heads + head) * 128) + pair * 2;
    float even = tokens[offset];
    float odd = tokens[offset + 1];
    tokens[offset] = even * cosine - odd * sine;
    tokens[offset + 1] = even * sine + odd * cosine;
}

struct SoftArgs {
    int rows;
    int stride;
    int valid;
    float scale;
};

kernel void softmax_f16(
    device const float *scores [[buffer(0)]],
    device half *weights [[buffer(1)]],
    constant SoftArgs &args [[buffer(2)]],
    uint row [[thread_position_in_grid]])
{
    if (int(row) >= args.rows) {
        return;
    }
    const device float *line = scores + row * args.stride;
    float peak = -1.0e30f;
    for (int col = 0; col < args.valid; ++col) {
        peak = max(peak, line[col] * args.scale);
    }
    float sum = 0.0f;
    for (int col = 0; col < args.valid; ++col) {
        sum += exp(line[col] * args.scale - peak);
    }
    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    device half *out = weights + row * args.stride;
    for (int col = 0; col < args.valid; ++col) {
        out[col] = half(exp(line[col] * args.scale - peak) * inv);
    }
}

kernel void add_f32(
    device float *residual [[buffer(0)]],
    device const float *update [[buffer(1)]],
    constant int &count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (int(gid) >= count) {
        return;
    }
    residual[gid] += update[gid];
}

kernel void merge_heads_f32(
    device const float *source [[buffer(0)]],
    device float *dest [[buffer(1)]],
    constant int3 &shape [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    int rows = shape.x;
    int heads = shape.y;
    int dim = shape.z;
    if (int(gid) >= rows * heads * dim) {
        return;
    }
    int row = int(gid) / (heads * dim);
    int rem = int(gid) - row * heads * dim;
    int head = rem / dim;
    int col = rem - head * dim;
    dest[gid] = source[(head * rows + row) * dim + col];
}
