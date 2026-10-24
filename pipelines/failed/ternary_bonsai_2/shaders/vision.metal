// Qwen3.5 vision tower. Layer norm, tanh GELU, 2D RoPE, and full attention.
// The fp4 projections reuse prefill_int2_n64.

constant float VISION_GELU = 0.7978845608028654f;
constant float VISION_THETA = 10000.0f;

kernel void vision_layernorm(
    device const float* x [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device float* y [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    if (row >= rows) {
        return;
    }
    threadgroup float red[8];
    threadgroup float red2[8];
    x += row * width;
    y += row * width;
    float sum = 0.0f;
    float sumsq = 0.0f;
    for (uint i = tid; i < width; i += 256u) {
        float value = x[i];
        sum += value;
        sumsq += value * value;
    }
    float part = simd_sum(sum);
    float part2 = simd_sum(sumsq);
    if (lane == 0u) {
        red[simd] = part;
        red2[simd] = part2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float total = 0.0f;
        float total2 = 0.0f;
        for (uint i = 0u; i < 8u; ++i) {
            total += red[i];
            total2 += red2[i];
        }
        float mean = total / float(width);
        float var = total2 / float(width) - mean * mean;
        red[0] = mean;
        red[1] = rsqrt(max(var, 0.0f) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = red[0];
    float inv = red[1];
    for (uint i = tid; i < width; i += 256u) {
        y[i] = (x[i] - mean) * inv * float(weight[i]) + float(bias[i]);
    }
}

kernel void vision_gelu_tanh(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) {
        return;
    }
    float value = x[tid];
    float cube = value * value * value;
    float inner = VISION_GELU * (value + 0.044715f * cube);
    // tanh(x) overflows to NaN once exp(2x) leaves float32. The tails are already saturated.
    float saturated = clamp(inner, -10.0f, 10.0f);
    y[tid] = 0.5f * value * (1.0f + tanh(saturated));
}

// Abramowitz and Stegun 7.1.26. Absolute error stays under 1.5e-7.
static float erf_as(float x) {
    float ax = fabs(x);
    float t = 1.0f / (1.0f + 0.3275911f * ax);
    float poly = 0.254829592f + t * (-0.284496736f + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f)));
    float value = 1.0f - t * poly * exp(-ax * ax);
    return copysign(value, x);
}

kernel void vision_gelu_erf(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) {
        return;
    }
    float value = x[tid];
    y[tid] = 0.5f * value * (1.0f + erf_as(value * 0.7071067811865476f));
}

kernel void vision_bias(
    device float* rows [[buffer(0)]],
    device const half* bias [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant uint& stored [[buffer(3)]],
    constant uint& logical [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    uint span = count * stored;
    if (tid >= span) {
        return;
    }
    uint col = tid % stored;
    if (col < logical) {
        rows[tid] += float(bias[col]);
    } else {
        rows[tid] = 0.0f;
    }
}

kernel void vision_to_half(
    device const float* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& row_pad [[buffer(3)]],
    constant uint& logical [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    constant uint& src_stride [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    uint span = row_pad * inner;
    if (tid >= span) {
        return;
    }
    uint row = tid / inner;
    uint col = tid % inner;
    if (row >= rows || col >= logical) {
        dst[tid] = half(0.0f);
        return;
    }
    dst[tid] = half(src[row * src_stride + col]);
}

kernel void dequant_int2_rows(
    device const uchar* weights [[buffer(0)]],
    device const uchar* scales [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& inner [[buffer(4)]],
    constant uint& logical [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    if (row >= rows) {
        return;
    }
    uint blocks = inner >> 5;
    device const uchar* packed = weights + row * (inner >> 2);
    device const uchar* row_scale = scales + row * blocks;
    device float* dest = destination + row * logical;
    for (uint block = 0u; block < blocks; ++block) {
        float factor = e8m0_code(row_scale[block]);
        uint element = (block << 5) + lane;
        if (element >= logical) {
            continue;
        }
        uchar byte = packed[element >> 2];
        uint shift = (element & 3u) * 2u;
        dest[element] = float(int2_code(byte >> shift)) * factor;
    }
}

kernel void vision_pos_add(
    device const float* table [[buffer(0)]],
    device float* tokens [[buffer(1)]],
    constant uint& side [[buffer(2)]],
    constant uint& grid_h [[buffer(3)]],
    constant uint& grid_w [[buffer(4)]],
    constant uint& dim [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint seq [[threadgroup_position_in_grid]])
{
    uint blocks_w = grid_w >> 1;
    uint n = grid_h * grid_w;
    if (seq >= n) {
        return;
    }
    uint block = seq >> 2;
    uint intra = seq & 3u;
    uint bh = block / blocks_w;
    uint bw = block - bh * blocks_w;
    uint ih = intra >> 1;
    uint iw = intra & 1u;
    uint ph = (bh << 1) + ih;
    uint pw = (bw << 1) + iw;
    float gy = grid_h == 1u ? 0.0f : float(ph) * float(side - 1u) / float(grid_h - 1u);
    float gx = grid_w == 1u ? 0.0f : float(pw) * float(side - 1u) / float(grid_w - 1u);
    uint y0 = uint(floor(gy));
    uint x0 = uint(floor(gx));
    uint y1 = min(y0 + 1u, side - 1u);
    uint x1 = min(x0 + 1u, side - 1u);
    float fy = gy - float(y0);
    float fx = gx - float(x0);
    for (uint col = tid; col < dim; col += 256u) {
        float v00 = table[(y0 * side + x0) * dim + col];
        float v01 = table[(y0 * side + x1) * dim + col];
        float v10 = table[(y1 * side + x0) * dim + col];
        float v11 = table[(y1 * side + x1) * dim + col];
        float top = v00 + (v01 - v00) * fx;
        float bot = v10 + (v11 - v10) * fx;
        tokens[seq * dim + col] += top + (bot - top) * fy;
    }
}

kernel void vision_rope(
    device float* query [[buffer(0)]],
    device float* key [[buffer(1)]],
    device const uint* coords [[buffer(2)]],
    constant uint& tokens [[buffer(3)]],
    constant uint& heads [[buffer(4)]],
    constant uint& dim [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    uint token = tgid.x;
    uint head = tgid.y;
    if (token >= tokens || head >= heads) {
        return;
    }
    uint mid = dim / 2u;
    uint span = dim / 4u;
    device float* q = query + (token * heads + head) * dim;
    device float* k = key + (token * heads + head) * dim;
    uint hpos = coords[token * 2u];
    uint wpos = coords[token * 2u + 1u];
    for (uint i = tid; i < mid; i += 32u) {
        uint freq = i < span ? i : i - span;
        uint pos = i < span ? hpos : wpos;
        float angle = float(pos) * pow(VISION_THETA, -float(freq) / float(span));
        float c = cos(angle);
        float s = sin(angle);
        float q0 = q[i];
        float q1 = q[i + mid];
        q[i] = q0 * c - q1 * s;
        q[i + mid] = q1 * c + q0 * s;
        float k0 = k[i];
        float k1 = k[i + mid];
        k[i] = k0 * c - k1 * s;
        k[i + mid] = k1 * c + k0 * s;
    }
}

kernel void vision_split_qkv(
    device const float* mixed [[buffer(0)]],
    device float* query [[buffer(1)]],
    device float* key [[buffer(2)]],
    device float* value [[buffer(3)]],
    constant uint& tokens [[buffer(4)]],
    constant uint& hidden [[buffer(5)]],
    constant uint& stored [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    uint span = tokens * hidden * 3u;
    if (tid >= span) {
        return;
    }
    uint hidden3 = hidden * 3u;
    uint token = tid / hidden3;
    uint local = tid - token * hidden3;
    uint part = local / hidden;
    uint channel = local - part * hidden;
    float value_f = mixed[token * stored + local];
    if (part == 0u) {
        query[token * hidden + channel] = value_f;
    } else if (part == 1u) {
        key[token * hidden + channel] = value_f;
    } else {
        value[token * hidden + channel] = value_f;
    }
}

kernel void vision_attend(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& tokens [[buffer(4)]],
    constant uint& heads [[buffer(5)]],
    constant uint& dim [[buffer(6)]],
    uint tid [[thread_index_in_simdgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    uint qrow = tgid.x;
    uint head = tgid.y;
    if (qrow >= tokens || head >= heads) {
        return;
    }
    float scale = rsqrt(float(dim));
    device const float* q = query + (qrow * heads + head) * dim;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float running_max = -1.0e30f;
    float running_sum = 0.0f;
    for (uint krow = 0u; krow < tokens; ++krow) {
        device const float* k = key + (krow * heads + head) * dim;
        float partial = 0.0f;
        for (uint d = tid; d < dim; d += 32u) {
            partial += q[d] * k[d];
        }
        float dot = simd_sum(partial) * scale;
        float next_max = max(running_max, dot);
        float alpha = exp(running_max - next_max);
        float weight = exp(dot - next_max);
        device const float* v = value + (krow * heads + head) * dim;
        acc0 = acc0 * alpha + weight * (tid < dim ? v[tid] : 0.0f);
        acc1 = acc1 * alpha + weight * (tid + 32u < dim ? v[tid + 32u] : 0.0f);
        acc2 = acc2 * alpha + weight * (tid + 64u < dim ? v[tid + 64u] : 0.0f);
        running_sum = running_sum * alpha + weight;
        running_max = next_max;
    }
    float inv = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
    device float* dest = out + (qrow * heads + head) * dim;
    if (tid < dim) {
        dest[tid] = acc0 * inv;
    }
    if (tid + 32u < dim) {
        dest[tid + 32u] = acc1 * inv;
    }
    if (tid + 64u < dim) {
        dest[tid + 64u] = acc2 * inv;
    }
}

kernel void vision_merge_pack(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& groups [[buffer(2)]],
    constant uint& hidden [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]])
{
    if (group >= groups) {
        return;
    }
    uint width = hidden * 4u;
    for (uint col = tid; col < width; col += 256u) {
        uint local = col / hidden;
        uint channel = col - local * hidden;
        dst[group * width + col] = src[((group * 4u) + local) * hidden + channel];
    }
}

kernel void scatter_hidden_rows(
    device const float* src [[buffer(0)]],
    device float* dest [[buffer(1)]],
    device const uint* local_rows [[buffer(2)]],
    device const uint* src_rows [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    if (row >= n) {
        return;
    }
    uint dst = local_rows[row];
    uint src_row = src_rows[row];
    for (uint col = tid; col < width; col += 256u) {
        dest[dst * width + col] = src[src_row * width + col];
    }
}
