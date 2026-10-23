#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>

using namespace metal;
using namespace mpp::tensor_ops;

using scales_plane = tensor_blockwise<tensor_plane_scales,
                                      device metal_fp8_ue8m0_format,
                                      32, 1>;

constant float FP4_MAG[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
constant int TILE_M = 64;
constant int TILE_N = 32;

inline float fp4_mag(uint nibble) {
    float mag = FP4_MAG[nibble & 7u];
    return (nibble & 8u) != 0u ? -mag : mag;
}

inline float e8m0_scale(uchar exponent) {
    if (exponent == 0u) {
        return 0.0f;
    }
    return as_type<float>(uint(exponent) << 23);
}

inline float e4m3_to_f32(uchar code) {
    uint value = code;
    float sign = (value & 0x80u) != 0u ? -1.0f : 1.0f;
    uint exponent = (value >> 3) & 0xfu;
    uint mantissa = value & 0x7u;
    if (exponent == 0u) {
        return sign * (float(mantissa) / 8.0f) * exp2(-6.0f);
    }
    if (exponent == 15u && mantissa == 7u) {
        return sign * NAN;
    }
    return sign * (1.0f + float(mantissa) / 8.0f) * exp2(float(exponent) - 7.0f);
}

inline float dequant_fp4(device const uchar* weights, device const uchar* scales,
                         uint row, uint col, uint inner) {
    uint idx = row * inner + col;
    uchar byte = weights[idx >> 1];
    uint nibble = (idx & 1u) != 0u ? (byte >> 4) : (byte & 0x0fu);
    uint blocks = inner >> 5;
    uchar exponent = scales[row * blocks + (col >> 5)];
    return fp4_mag(nibble) * e8m0_scale(exponent);
}

inline float dequant_fp8(device const uchar* weights, device const uchar* scales,
                         uint row, uint col, uint inner) {
    uchar code = weights[row * inner + col];
    uint blocks = inner >> 5;
    uchar exponent = scales[row * blocks + (col >> 5)];
    return e4m3_to_f32(code) * e8m0_scale(exponent);
}

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

kernel void cast_f16(
    device const float* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) {
        dst[gid] = half(src[gid]);
    }
}

kernel void add_bias(
    device float* data [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& stride [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint n = rows * cols;
    if (gid >= n) {
        return;
    }
    uint row = gid / cols;
    uint col = gid - row * cols;
    data[row * stride + col] += bias[col];
}

kernel void add_vec(
    device float* dst [[buffer(0)]],
    device const float* src [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) {
        dst[gid] += src[gid];
    }
}

kernel void rmsnorm(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
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
        float v = x[row * width + i];
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
        y[row * width + i] = x[row * width + i] * inv * weight[i];
    }
}

kernel void rope_half(
    device const float* q [[buffer(0)]],
    device float* dst [[buffer(1)]],
    device const float* cos_table [[buffer(2)]],
    device const float* sin_table [[buffer(3)]],
    constant uint& seq [[buffer(4)]],
    constant uint& heads [[buffer(5)]],
    constant uint& dim [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    uint n = seq * heads * dim;
    if (gid >= n) {
        return;
    }
    uint split = dim >> 1;
    uint d = gid % dim;
    uint rest = gid / dim;
    uint pos = rest / heads;
    uint base = (pos * heads + rest % heads) * dim;
    float c = cos_table[pos * dim + d];
    float s = sin_table[pos * dim + d];
    float v = q[base + d];
    float other = d < split ? q[base + d + split] : q[base + d - split];
    dst[base + d] = d < split ? (v * c - other * s) : (v * c + other * s);
}

kernel void silu(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) {
        float v = x[gid];
        y[gid] = v / (1.0f + exp(-v));
    }
}

kernel void silu_mul(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) {
        float v = gate[gid];
        float activated = v / (1.0f + exp(-v));
        y[gid] = activated * up[gid];
    }
}

kernel void adaln(
    device const float* x [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device const float* shift [[buffer(2)]],
    device float* y [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint n = rows * width;
    if (gid >= n) {
        return;
    }
    uint col = gid % width;
    y[gid] = x[gid] * (1.0f + scale[col]) + shift[col];
}

kernel void channel_residual(
    device float* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    device const float* gate [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint n = rows * width;
    if (gid >= n) {
        return;
    }
    x[gid] += y[gid] * gate[gid % width];
}

kernel void residual(
    device float* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) {
        x[gid] += y[gid];
    }
}

kernel void quant_fp8(
    device const float* src [[buffer(0)]],
    device uchar* dst [[buffer(1)]],
    device uchar* scales [[buffer(2)]],
    constant uint& blocks [[buffer(3)]],
    uint lane [[thread_position_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]])
{
    if (group >= blocks || lane >= 32u) {
        return;
    }
    threadgroup float vals[32];
    threadgroup uchar shared_scale = 0u;
    vals[lane] = src[group * 32u + lane];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
        float amax = 0.0f;
        for (uint i = 0u; i < 32u; ++i) {
            amax = max(amax, fabs(vals[i]));
        }
        uchar exponent = 0u;
        if (amax > 0.0f) {
            float ratio = amax / 448.0f;
            uint bits = as_type<uint>(ratio);
            uint exp = (bits >> 23) & 255u;
            uint mantissa = bits & 0x7fffffu;
            if (exp == 255u) {
                exponent = 254u;
            } else if (exp == 0u) {
                exponent = 0u;
            } else {
                int power = mantissa == 0u ? int(exp) - 127 : int(exp) - 126;
                int biased = power + 127;
                if (biased < 1) {
                    exponent = 0u;
                } else if (biased > 254) {
                    exponent = 254u;
                } else {
                    exponent = uchar(biased);
                }
            }
        }
        shared_scale = exponent;
        scales[group] = exponent;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = e8m0_scale(shared_scale);
    vals[lane] = scale == 0.0f ? 0.0f : vals[lane] / scale;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 4u) {
        vec<float, 8> part;
        for (uint i = 0u; i < 8u; ++i) {
            part[i] = vals[lane * 8u + i];
        }
        auto packed = pack<metal_fp8_e4m3_format>(part);
        auto bytes = packed.as_storage_type();
        uint origin = group * 32u + lane * 8u;
        for (uint i = 0u; i < 8u; ++i) {
            dst[origin + i] = bytes[i];
        }
    }
}

inline float group_sum(float value, threadgroup float* parts, uint lane) {
    float grouped = simd_sum(value);
    if ((lane & 31u) == 0u) {
        parts[lane >> 5] = grouped;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return parts[0] + parts[1] + parts[2] + parts[3];
}

kernel void attend(
    device const float* query [[buffer(0)]],
    device const uchar* key [[buffer(1)]],
    device const uchar* key_scale [[buffer(2)]],
    device const uchar* value [[buffer(3)]],
    device const uchar* value_scale [[buffer(4)]],
    device float* destination [[buffer(5)]],
    constant uint& seq [[buffer(6)]],
    constant uint& kv_seq [[buffer(7)]],
    constant uint& heads [[buffer(8)]],
    constant uint& kv_heads [[buffer(9)]],
    constant uint& dim [[buffer(10)]],
    constant int& window [[buffer(11)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 local [[thread_position_in_threadgroup]])
{
    uint pos = tgid.x;
    uint head = tgid.y;
    uint lane = local.x;
    if (pos >= seq || head >= heads) {
        return;
    }
    threadgroup float parts[4];
    uint group = head / (heads / kv_heads);
    uint blocks = dim >> 5;
    float scale = rsqrt(float(dim));
    float query_value = query[(pos * heads + head) * dim + lane];
    float acc = 0.0f;
    float running = -INFINITY;
    float normalizer = 0.0f;
    for (uint key_pos = 0u; key_pos < kv_seq; ++key_pos) {
        bool live = window == -2 ? int(key_pos) <= int(pos) : (window < 0 || abs(int(pos) - int(key_pos)) <= window);
        uint block = (key_pos * kv_heads + group) * blocks + (lane >> 5);
        float key_value = live
            ? e4m3_to_f32(key[(key_pos * kv_heads + group) * dim + lane]) * e8m0_scale(key_scale[block])
            : 0.0f;
        float dot = group_sum(query_value * key_value, parts, lane);
        if (!live) {
            continue;
        }
        float score = dot * scale;
        float next = max(running, score);
        float alpha = exp(running - next);
        float weight = exp(score - next);
        normalizer = normalizer * alpha + weight;
        float value_value = e4m3_to_f32(value[(key_pos * kv_heads + group) * dim + lane]) * e8m0_scale(value_scale[block]);
        acc = acc * alpha + weight * value_value;
        running = next;
    }
    float inv = normalizer == 0.0f ? 0.0f : 1.0f / normalizer;
    destination[(pos * heads + head) * dim + lane] = acc * inv;
}

kernel void sinusoid_time(
    device float* dst [[buffer(0)]],
    constant float& tick [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= 256u) {
        return;
    }
    uint split = 128u;
    uint i = gid < split ? gid : gid - split;
    float freq = exp(-log(10000.0f) * float(i) / float(split));
    float angle = tick * 1000.0f * freq;
    dst[gid] = gid < split ? cos(angle) : sin(angle);
}

kernel void rope_tables(
    device float* cos_table [[buffer(0)]],
    device float* sin_table [[buffer(1)]],
    constant uint& seq [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    constant float& theta [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint n = seq * dim;
    if (gid >= n) {
        return;
    }
    uint split = dim >> 1;
    uint d = gid % dim;
    uint pos = gid / dim;
    uint i = d < split ? d : d - split;
    float freq = exp(-log(theta) * float(i) / float(split));
    float angle = float(pos) * freq;
    cos_table[gid] = cos(angle);
    sin_table[gid] = sin(angle);
}

kernel void euler(
    device float* x [[buffer(0)]],
    device const float* velocity [[buffer(1)]],
    constant float& dt [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) {
        x[gid] += dt * velocity[gid];
    }
}

kernel void snake(
    device const float* x [[buffer(0)]],
    device const float* alpha [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device float* y [[buffer(3)]],
    constant uint& channels [[buffer(4)]],
    constant uint& length [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint n = channels * length;
    if (gid >= n) {
        return;
    }
    uint channel = gid / length;
    float a = exp(alpha[channel]);
    float b = exp(beta[channel]);
    float v = x[gid];
    float s = sin(a * v);
    y[gid] = v + (s * s) / b;
}

kernel void conv1d(
    device const float* x [[buffer(0)]],
    device const uchar* weight [[buffer(1)]],
    device const uchar* scales [[buffer(2)]],
    device const float* bias [[buffer(3)]],
    device float* y [[buffer(4)]],
    constant uint& cin [[buffer(5)]],
    constant uint& cout [[buffer(6)]],
    constant uint& k [[buffer(7)]],
    constant uint& inner [[buffer(8)]],
    constant uint& length [[buffer(9)]],
    constant uint& pad [[buffer(10)]],
    constant uint& stride [[buffer(11)]],
    constant uint& dilation [[buffer(12)]],
    constant uint& has_bias [[buffer(13)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint oc = gid.x;
    uint pos = gid.y;
    uint out_len = (length + 2u * pad - dilation * (k - 1u) - 1u) / stride + 1u;
    if (oc >= cout || pos >= out_len) {
        return;
    }
    float acc = has_bias != 0u ? bias[oc] : 0.0f;
    for (uint ic = 0u; ic < cin; ++ic) {
        for (uint tap = 0u; tap < k; ++tap) {
            int src = int(pos * stride) + int(tap * dilation) - int(pad);
            if (src < 0 || src >= int(length)) {
                continue;
            }
            uint col = ic * k + tap;
            float w = dequant_fp8(weight, scales, oc, col, inner);
            acc += x[ic * length + uint(src)] * w;
        }
    }
    y[oc * out_len + pos] = acc;
}

kernel void conv_transpose1d(
    device const float* x [[buffer(0)]],
    device const uchar* weight [[buffer(1)]],
    device const uchar* scales [[buffer(2)]],
    device const float* bias [[buffer(3)]],
    device float* y [[buffer(4)]],
    constant uint& cin [[buffer(5)]],
    constant uint& cout [[buffer(6)]],
    constant uint& k [[buffer(7)]],
    constant uint& inner [[buffer(8)]],
    constant uint& length [[buffer(9)]],
    constant uint& pad [[buffer(10)]],
    constant uint& stride [[buffer(11)]],
    constant uint& has_bias [[buffer(12)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint oc = gid.x;
    uint pos = gid.y;
    uint out_len = length * stride;
    if (oc >= cout || pos >= out_len) {
        return;
    }
    float acc = has_bias != 0u ? bias[oc] : 0.0f;
    for (uint tap = 0u; tap < k; ++tap) {
        int numer = int(pos) + int(pad) - int(tap);
        if (numer < 0 || (uint(numer) % stride) != 0u) {
            continue;
        }
        uint src = uint(numer) / stride;
        if (src >= length) {
            continue;
        }
        for (uint ic = 0u; ic < cin; ++ic) {
            uint col = oc * k + tap;
            float w = dequant_fp8(weight, scales, ic, col, inner);
            acc += x[ic * length + src] * w;
        }
    }
    y[oc * out_len + pos] = acc;
}

kernel void gather_fp8(
    device const uchar* table [[buffer(0)]],
    device const uchar* scales [[buffer(1)]],
    device const int* ids [[buffer(2)]],
    device float* rows [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    constant uint& count [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    uint n = count * width;
    if (gid >= n) {
        return;
    }
    uint token = gid / width;
    uint col = gid - token * width;
    int id = ids[token];
    rows[gid] = dequant_fp8(table, scales, uint(id), col, inner);
}

kernel void copy_bytes(
    device const uchar* src [[buffer(0)]],
    device uchar* dst [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) {
        dst[gid] = src[gid];
    }
}

kernel void fill_f32(
    device float* dst [[buffer(0)]],
    constant float& value [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) {
        dst[gid] = value;
    }
}

kernel void copy_rows(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& src_stride [[buffer(4)]],
    constant uint& dst_stride [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint n = rows * cols;
    if (gid >= n) {
        return;
    }
    uint row = gid / cols;
    uint col = gid - row * cols;
    dst[row * dst_stride + col] = src[row * src_stride + col];
}

kernel void copy_span(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant uint& src_offset [[buffer(3)]],
    constant uint& dst_offset [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) {
        dst[dst_offset + gid] = src[src_offset + gid];
    }
}

kernel void transpose2d(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint n = rows * cols;
    if (gid >= n) {
        return;
    }
    uint row = gid / cols;
    uint col = gid - row * cols;
    dst[col * rows + row] = src[gid];
}

kernel void pack_latent(
    device const float* source [[buffer(0)]],
    device const float* mask [[buffer(1)]],
    device const float* noise [[buffer(2)]],
    device float* packed [[buffer(3)]],
    constant uint& channels [[buffer(4)]],
    constant uint& length [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint span = channels * length;
    uint n = span * 3u;
    if (gid >= n) {
        return;
    }
    uint plane = gid / span;
    uint offset = gid - plane * span;
    if (plane == 0u) {
        packed[gid] = source[offset];
    } else if (plane == 1u) {
        packed[gid] = mask[offset];
    } else {
        packed[gid] = noise[offset];
    }
}

kernel void block_amax(
    device const float* src [[buffer(0)]],
    device float* partial [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& stride [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]])
{
    float peak = 0.0f;
    for (uint i = group * stride + lane; i < n && i < (group + 1u) * stride; i += 256u) {
        peak = max(peak, fabs(src[i]));
    }
    float grouped = simd_max(peak);
    threadgroup float parts[8];
    if ((lane & 31u) == 0u) {
        parts[lane >> 5] = grouped;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
        float total = 0.0f;
        for (uint i = 0u; i < 8u; ++i) {
            total = max(total, parts[i]);
        }
        partial[group] = total;
    }
}

kernel void reduce_amax(
    device const float* partial [[buffer(0)]],
    device float* peak [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint lane [[thread_position_in_threadgroup]])
{
    float value = 0.0f;
    for (uint i = lane; i < n; i += 256u) {
        value = max(value, partial[i]);
    }
    float grouped = simd_max(value);
    threadgroup float parts[8];
    if ((lane & 31u) == 0u) {
        parts[lane >> 5] = grouped;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
        float total = 0.0f;
        for (uint i = 0u; i < 8u; ++i) {
            total = max(total, parts[i]);
        }
        peak[0] = total;
    }
}

kernel void scale_peak(
    device float* src [[buffer(0)]],
    device const float* peak [[buffer(1)]],
    constant float& target [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) {
        return;
    }
    float value = peak[0];
    if (value > 0.0f) {
        src[gid] *= target / value;
    }
}
