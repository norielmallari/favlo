// Input Gram and the FOEM column update.
// The Gram is half X transposed times X through matmul2d. A column block is 128,
// with an E8M0 scale every 32.

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>

using namespace metal;
using namespace mpp::tensor_ops;

constexpr constant uint FOEM_BLOCK = 128;
constexpr constant uint FP4_GROUP = 32;

constant float FP4_EDGE[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
constant float FP4_GRID[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

kernel void gram_transpose_f32(
    device const float *x [[buffer(0)]],
    device half *dest [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    constant uint &span [[buffer(3)]],
    constant uint &width [[buffer(4)]],
    constant uint &token_pad [[buffer(5)]],
    constant uint &width_pad [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    uint token = tgid.x * 64u + tid;
    uint col = tgid.y;
    if (token >= token_pad || col >= width_pad) {
        return;
    }
    half value = 0.0h;
    if (token < count && col < width && span >= count) {
        value = half(x[(ulong)token * width + col]);
    }
    dest[(ulong)col * token_pad + token] = value;
}

kernel void gram_transpose_f16(
    device const half *x [[buffer(0)]],
    device half *dest [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    constant uint &span [[buffer(3)]],
    constant uint &width [[buffer(4)]],
    constant uint &token_pad [[buffer(5)]],
    constant uint &width_pad [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    uint token = tgid.x * 64u + tid;
    uint col = tgid.y;
    if (token >= token_pad || col >= width_pad) {
        return;
    }
    half value = 0.0h;
    if (token < count && col < width && span >= count) {
        value = x[(ulong)token * width + col];
    }
    dest[(ulong)col * token_pad + token] = value;
}

kernel void gram_xtx(
    device half *transposed [[buffer(0)]],
    device float *scratch [[buffer(1)]],
    device float *acc [[buffer(2)]],
    constant uint &width [[buffer(3)]],
    constant uint &width_pad [[buffer(4)]],
    constant uint &token_pad [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    int kn = int(token_pad);
    int wide = int(width_pad);
    tensor<device half, dextents<int, 2>, tensor_inline> left(
        transposed, dextents<int, 2>(kn, wide), array<int, 2>{1, int(token_pad)});
    tensor<device half, dextents<int, 2>, tensor_inline> right(
        transposed, dextents<int, 2>(kn, wide), array<int, 2>{1, int(token_pad)});
    tensor<device float, dextents<int, 2>, tensor_inline> out(
        scratch, dextents<int, 2>(wide, wide), array<int, 2>{1, wide});
    constexpr auto desc = matmul2d_descriptor(
        64, 64, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto tile_left = left.slice(0, int(tgid.y) * 64);
    auto tile_right = right.slice(0, int(tgid.x) * 64);
    auto tile_out = out.slice(int(tgid.x) * 64, int(tgid.y) * 64);
    op.run(tile_left, tile_right, tile_out);
    threadgroup_barrier(mem_flags::mem_device);
    for (uint i = tid; i < 4096u; i += 128u) {
        uint local_row = i >> 6u;
        uint local_col = i & 63u;
        uint row = tgid.y * 64u + local_row;
        uint col = tgid.x * 64u + local_col;
        if (row < width && col < width) {
            acc[(ulong)row * width + col] += scratch[(ulong)row * width_pad + col];
        }
    }
}

static uint fp4_code(float value, float scale, thread float &restored) {
    float scaled = value / scale;
    float mag = fabs(scaled);
    uint choice = 0u;
    for (uint edge = 0u; edge < 7u; ++edge) {
        if (FP4_EDGE[edge] < mag) {
            choice = edge + 1u;
        }
    }
    if (choice < 7u && mag == FP4_EDGE[choice] && (choice & 1u) != 0u) {
        choice += 1u;
    }
    float sign = signbit(scaled) ? -1.0f : 1.0f;
    restored = sign * FP4_GRID[choice] * scale;
    return choice | (signbit(scaled) ? 8u : 0u);
}

static uint e8m0_scale(float amax, thread float &scale) {
    if (!(amax > 0.0f)) {
        scale = ldexp(1.0f, -127);
        return 0u;
    }
    float ratio = max(amax / 6.0f, ldexp(1.0f, -149));
    int exponent = 0;
    float mantissa = frexp(ratio, exponent);
    int power = exponent - (mantissa == 0.5f ? 1 : 0);
    int biased = clamp(power + 127, 0, 254);
    scale = ldexp(1.0f, biased - 127);
    return uint(biased);
}

kernel void foem_block(
    device float *work [[buffer(0)]],
    device const float *original [[buffer(1)]],
    device const float *upper [[buffer(2)]],
    device const uchar *dead [[buffer(3)]],
    device uchar *codes [[buffer(4)]],
    device uchar *scales [[buffer(5)]],
    device float *errors [[buffer(6)]],
    constant uint &rows [[buffer(7)]],
    constant uint &inner [[buffer(8)]],
    constant uint &b0 [[buffer(9)]],
    constant uint &width [[buffer(10)]],
    constant float &beta [[buffer(11)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= rows) {
        return;
    }
    device float *part = work + row * inner + b0;
    device const float *orig = original + row * inner + b0;
    float scale = 1.0f;
    uint scale_stride = inner >> 5;
    for (uint i = 0u; i < width; ++i) {
        uint column = b0 + i;
        if (dead[column] != 0u) {
            codes[row * inner + column] = 0u;
            errors[row * FOEM_BLOCK + i] = 0.0f;
            if ((column & 31u) == 0u) {
                scales[row * scale_stride + (column >> 5)] = 0u;
            }
            continue;
        }
        if ((column & 31u) == 0u) {
            float amax = 0.0f;
            for (uint k = 0u; k < FP4_GROUP; ++k) {
                amax = max(amax, fabs(part[i + k]));
            }
            uint biased = e8m0_scale(amax, scale);
            scales[row * scale_stride + (column >> 5)] = uchar(biased);
        }
        float restored = 0.0f;
        uint code = fp4_code(part[i], scale, restored);
        codes[row * inner + column] = uchar(code);
        float diagonal = upper[column * inner + column];
        float drift = part[i] - orig[i];
        float error = ((part[i] - restored) - beta * drift) / diagonal;
        errors[row * FOEM_BLOCK + i] = error;
        if (i + 1u < width) {
            for (uint j = i + 1u; j < width; ++j) {
                uint nxt = b0 + j;
                if (dead[nxt] != 0u) {
                    continue;
                }
                part[j] -= error * upper[column * inner + nxt];
            }
            uint nxt = column + 1u;
            if (dead[nxt] == 0u) {
                part[i + 1u] -= beta * (part[i + 1u] - orig[i + 1u]);
            }
        }
    }
}

kernel void foem_tail(
    device float *work [[buffer(0)]],
    device const float *original [[buffer(1)]],
    device const float *errors [[buffer(2)]],
    device const float *upper [[buffer(3)]],
    device const uchar *dead [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &inner [[buffer(6)]],
    constant uint &b0 [[buffer(7)]],
    constant uint &width [[buffer(8)]],
    constant float &beta [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint col = b0 + width + gid.x;
    uint row = gid.y;
    if (row >= rows || col >= inner || dead[col] != 0u) {
        return;
    }
    float acc = 0.0f;
    for (uint i = 0u; i < width; ++i) {
        acc += errors[row * FOEM_BLOCK + i] * upper[(b0 + i) * inner + col];
    }
    float value = work[row * inner + col] - acc;
    if (gid.x == 0u) {
        value -= beta * (value - original[row * inner + col]);
    }
    work[row * inner + col] = value;
}

kernel void foem_pack(
    device const uchar *codes [[buffer(0)]],
    device uchar *packed [[buffer(1)]],
    constant uint &rows [[buffer(2)]],
    constant uint &inner [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    uint pairs = inner >> 1;
    uint total = rows * pairs;
    if (tid >= total) {
        return;
    }
    uint row = tid / pairs;
    uint pair = tid - row * pairs;
    uint col = pair << 1;
    uint base = row * inner + col;
    packed[tid] = codes[base] | (codes[base + 1u] << 4);
}

// Float32 factor of the inverse Gram. The panel is 32. The matrix stays lower
// triangular in the workspace until the upper factor is written out.

static uint lower_tile(uint gid, thread uint &col) {
    float root = sqrt(float(8u * gid + 1u));
    uint row = uint(0.5f * (root - 1.0f));
    uint base = (row * (row + 1u)) >> 1;
    if (base > gid) {
        row -= 1u;
        base = (row * (row + 1u)) >> 1;
    } else if (base + row + 1u <= gid) {
        row += 1u;
        base = (row * (row + 1u)) >> 1;
    }
    col = gid - base;
    return row;
}

kernel void gram_prepare(
    device const float *gram [[buffer(0)]],
    device float *work [[buffer(1)]],
    device const uchar *dead [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    constant float &lam [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    ulong span = (ulong)n * n;
    if ((ulong)tid >= span) {
        return;
    }
    uint row = tid / n;
    uint col = tid - row * n;
    float value = gram[tid];
    if (row == col) {
        value = (dead[row] != 0u ? 1.0f : value) + lam;
    }
    work[tid] = value;
}

kernel void chol_panel(
    device float *a [[buffer(0)]],
    device uint *flag [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    constant uint &k [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]])
{
    if (flag[0] != 0u) {
        return;
    }
    threadgroup float tile[1024];
    for (uint item = tid; item < 1024u; item += 32u) {
        uint r = item >> 5;
        uint c = item & 31u;
        uint lo = r > c ? r : c;
        uint hi = r > c ? c : r;
        tile[item] = a[(ulong)(k + lo) * n + (k + hi)];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        for (uint j = 0u; j < 32u; ++j) {
            float sum = tile[j * 32u + j];
            for (uint t = 0u; t < j; ++t) {
                float value = tile[j * 32u + t];
                sum -= value * value;
            }
            if (!(sum > 0.0f)) {
                flag[0] = 1u;
                tile[j * 32u + j] = 1.0f;
                for (uint i = j + 1u; i < 32u; ++i) {
                    tile[i * 32u + j] = 0.0f;
                }
                continue;
            }
            float diag = sqrt(sum);
            tile[j * 32u + j] = diag;
            for (uint i = j + 1u; i < 32u; ++i) {
                float acc = tile[i * 32u + j];
                for (uint t = 0u; t < j; ++t) {
                    acc -= tile[i * 32u + t] * tile[j * 32u + t];
                }
                tile[i * 32u + j] = acc / diag;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint item = tid; item < 1024u; item += 32u) {
        uint r = item >> 5;
        uint c = item & 31u;
        if (c <= r) {
            a[(ulong)(k + r) * n + (k + c)] = tile[item];
        }
    }
}

kernel void chol_trsm(
    device float *a [[buffer(0)]],
    device uint *flag [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    constant uint &k [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    uint i = k + 32u + tgid * 256u + tid;
    if (flag[0] != 0u || i >= n) {
        return;
    }
    for (uint j = 0u; j < 32u; ++j) {
        float sum = a[(ulong)i * n + (k + j)];
        for (uint t = 0u; t < j; ++t) {
            sum -= a[(ulong)i * n + (k + t)] * a[(ulong)(k + j) * n + (k + t)];
        }
        a[(ulong)i * n + (k + j)] = sum / a[(ulong)(k + j) * n + (k + j)];
    }
}

kernel void chol_syrk(
    device float *a [[buffer(0)]],
    device uint *flag [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    constant uint &k [[buffer(3)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    if (flag[0] != 0u) {
        return;
    }
    uint tiles = (n - k - 32u) >> 5;
    uint tj = 0u;
    uint ti = lower_tile(gid, tj);
    if (ti >= tiles || tj > ti) {
        return;
    }
    uint row0 = k + 32u + ti * 32u;
    uint col0 = k + 32u + tj * 32u;
    threadgroup float left[1024];
    threadgroup float right[1024];
    for (uint item = tid; item < 1024u; item += 256u) {
        uint r = item >> 5;
        uint c = item & 31u;
        left[item] = a[(ulong)(row0 + r) * n + (k + c)];
        right[item] = a[(ulong)(col0 + r) * n + (k + c)];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 0u; step < 4u; ++step) {
        uint item = tid + step * 256u;
        uint r = item >> 5;
        uint c = item & 31u;
        if (ti == tj && c > r) {
            continue;
        }
        float acc = 0.0f;
        for (uint t = 0u; t < 32u; ++t) {
            acc += left[r * 32u + t] * right[c * 32u + t];
        }
        a[(ulong)(row0 + r) * n + (col0 + c)] -= acc;
    }
}

kernel void trinv_panel(
    device float *a [[buffer(0)]],
    constant uint &n [[buffer(1)]],
    constant uint &k [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float tile[1024];
    for (uint item = tid; item < 1024u; item += 32u) {
        uint r = item >> 5;
        uint c = item & 31u;
        tile[item] = c <= r ? a[(ulong)(k + r) * n + (k + c)] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        for (uint j = 0u; j < 32u; ++j) {
            tile[j * 32u + j] = 1.0f / tile[j * 32u + j];
            for (uint i = j + 1u; i < 32u; ++i) {
                float acc = 0.0f;
                for (uint t = j; t < i; ++t) {
                    acc += tile[i * 32u + t] * tile[t * 32u + j];
                }
                tile[i * 32u + j] = -acc / tile[i * 32u + i];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint item = tid; item < 1024u; item += 32u) {
        uint r = item >> 5;
        uint c = item & 31u;
        a[(ulong)(k + r) * n + (k + c)] = c <= r ? tile[item] : 0.0f;
    }
}

kernel void trinv_mul(
    device const float *a [[buffer(0)]],
    device float *temp [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    constant uint &k [[buffer(3)]],
    constant uint &m [[buffer(4)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= m) {
        return;
    }
    uint base = k + 32u;
    for (uint c = 0u; c < 32u; ++c) {
        float acc = 0.0f;
        for (uint t = 0u; t < 32u; ++t) {
            acc += a[(ulong)(base + row) * n + (k + t)] * a[(ulong)(k + t) * n + (k + c)];
        }
        temp[(ulong)row * 32u + c] = acc;
    }
}

kernel void trinv_apply(
    device float *a [[buffer(0)]],
    device const float *temp [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    constant uint &k [[buffer(3)]],
    constant uint &m [[buffer(4)]],
    uint tile [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    uint row0 = tile * 32u;
    if (row0 >= m) {
        return;
    }
    uint base = k + 32u;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    for (uint pp = 0u; pp < m; pp += 32u) {
        threadgroup float xpane[1024];
        threadgroup float tpane[1024];
        for (uint item = tid; item < 1024u; item += 256u) {
            uint t = item >> 5;
            uint lane = item & 31u;
            uint p = pp + t;
            float value = 0.0f;
            if (p < m && p <= row0 + lane) {
                value = a[(ulong)(base + row0 + lane) * n + (base + p)];
            }
            xpane[item] = value;
            tpane[item] = p < m ? temp[(ulong)p * 32u + lane] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint step = 0u; step < 4u; ++step) {
            uint item = tid + step * 256u;
            uint r = item >> 5;
            uint c = item & 31u;
            float dot = 0.0f;
            for (uint t = 0u; t < 32u; ++t) {
                dot += xpane[t * 32u + r] * tpane[t * 32u + c];
            }
            if (step == 0u) {
                acc0 += dot;
            } else if (step == 1u) {
                acc1 += dot;
            } else if (step == 2u) {
                acc2 += dot;
            } else {
                acc3 += dot;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint step = 0u; step < 4u; ++step) {
        uint item = tid + step * 256u;
        uint r = item >> 5;
        uint c = item & 31u;
        float acc = step == 0u ? acc0 : step == 1u ? acc1 : step == 2u ? acc2 : acc3;
        a[(ulong)(base + row0 + r) * n + (k + c)] = -acc;
    }
}

kernel void xtx_lower(
    device const float *x [[buffer(0)]],
    device float *dest [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    uint tiles = n >> 5;
    uint tj = 0u;
    uint ti = lower_tile(gid, tj);
    if (ti >= tiles || tj > ti) {
        return;
    }
    uint row0 = ti * 32u;
    uint col0 = tj * 32u;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    for (uint kk = 0u; kk < n; kk += 32u) {
        threadgroup float left[1024];
        threadgroup float right[1024];
        for (uint item = tid; item < 1024u; item += 256u) {
            uint t = item >> 5;
            uint c = item & 31u;
            uint krow = kk + t;
            float lv = 0.0f;
            float rv = 0.0f;
            if (krow < n && krow >= row0 + c) {
                lv = x[(ulong)krow * n + (row0 + c)];
            }
            if (krow < n && krow >= col0 + c) {
                rv = x[(ulong)krow * n + (col0 + c)];
            }
            left[item] = lv;
            right[item] = rv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint step = 0u; step < 4u; ++step) {
            uint item = tid + step * 256u;
            uint r = item >> 5;
            uint c = item & 31u;
            if (ti == tj && c > r) {
                continue;
            }
            float dot = 0.0f;
            for (uint t = 0u; t < 32u; ++t) {
                dot += left[t * 32u + r] * right[t * 32u + c];
            }
            if (step == 0u) {
                acc0 += dot;
            } else if (step == 1u) {
                acc1 += dot;
            } else if (step == 2u) {
                acc2 += dot;
            } else {
                acc3 += dot;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint step = 0u; step < 4u; ++step) {
        uint item = tid + step * 256u;
        uint r = item >> 5;
        uint c = item & 31u;
        if (ti == tj && c > r) {
            continue;
        }
        float acc = step == 0u ? acc0 : step == 1u ? acc1 : step == 2u ? acc2 : acc3;
        dest[(ulong)(row0 + r) * n + (col0 + c)] = acc;
        if (row0 != col0) {
            dest[(ulong)(col0 + c) * n + (row0 + r)] = acc;
        }
    }
}

kernel void transpose_upper(
    device const float *lower [[buffer(0)]],
    device float *upper [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= n) {
        return;
    }
    for (uint col = 0u; col < n; ++col) {
        float value = 0.0f;
        if (col >= row) {
            value = lower[(ulong)col * n + row];
        }
        upper[(ulong)row * n + col] = value;
    }
}

kernel void gram_quadratic(
    device const float *gram [[buffer(0)]],
    device const float *rows [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    constant uint &sample [[buffer(4)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    if (gid >= sample) {
        return;
    }
    device const float *row = rows + (ulong)gid * n;
    float acc = 0.0f;
    for (uint r = tid; r < n; r += 256u) {
        device const float *hrow = gram + (ulong)r * n;
        float dot = 0.0f;
        float left = row[r];
        for (uint c = 0u; c < n; ++c) {
            dot += hrow[c] * row[c];
        }
        acc += left * dot;
    }
    threadgroup float parts[256];
    parts[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 128u; step > 0u; step >>= 1u) {
        if (tid < step) {
            parts[tid] += parts[tid + step];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        out[gid] = parts[0];
    }
}
