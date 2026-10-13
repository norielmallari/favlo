// Routed experts. Weights stay in one atlas. A negative table entry is a miss.
// Concatenated after act.metal.

static float int4_weight(
    device const uchar* packed,
    device const uchar* row_scale,
    uint element)
{
    uint block = element >> 5;
    float factor = e8m0_scale(row_scale[block]);
    uchar byte = packed[element >> 1];
    uint nibble = (element & 1u) ? (byte >> 4) : (byte & 0x0Fu);
    int code = int(nibble);
    if (code > 7) {
        code -= 16;
    }
    return float(code) * factor;
}

kernel void gather_half(
    device const float* src [[buffer(0)]],
    device const uint* tokens [[buffer(1)]],
    device half* dst [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& logical [[buffer(4)]],
    constant uint& stored [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = n * stored;
    if (tid >= total) {
        return;
    }
    uint row = tid / stored;
    uint col = tid - row * stored;
    float value = 0.0f;
    if (col < logical && row < n) {
        value = src[tokens[row] * logical + col];
    }
    dst[tid] = half(clamp(value, -65504.0f, 65504.0f));
}

// Unpack eight codes once and apply them to every token in the tile.
static void codes8(
    uint word,
    float scale,
    thread float& c0,
    thread float& c1,
    thread float& c2,
    thread float& c3,
    thread float& c4,
    thread float& c5,
    thread float& c6,
    thread float& c7)
{
    c0 = float(int4_code(word & 0x0fu)) * scale;
    c1 = float(int4_code((word >> 4) & 0x0fu)) * scale;
    c2 = float(int4_code((word >> 8) & 0x0fu)) * scale;
    c3 = float(int4_code((word >> 12) & 0x0fu)) * scale;
    c4 = float(int4_code((word >> 16) & 0x0fu)) * scale;
    c5 = float(int4_code((word >> 20) & 0x0fu)) * scale;
    c6 = float(int4_code((word >> 24) & 0x0fu)) * scale;
    c7 = float(int4_code((word >> 28) & 0x0fu)) * scale;
}

static float apply8(
    float c0, float c1, float c2, float c3, float c4, float c5, float c6, float c7,
    device const half* act)
{
    device const half4* v = (device const half4*)act;
    half4 h0 = v[0];
    half4 h1 = v[1];
    return c0 * float(h0.x) + c1 * float(h0.y) + c2 * float(h0.z) + c3 * float(h0.w)
        + c4 * float(h1.x) + c5 * float(h1.y) + c6 * float(h1.z) + c7 * float(h1.w);
}

kernel void moe_grouped(
    device const half* activation [[buffer(0)]],
    device const uchar* atlas [[buffer(1)]],
    device const long* table [[buffer(2)]],
    device const uint* group_expert [[buffer(3)]],
    device const uint* group_offset [[buffer(4)]],
    device float* destination [[buffer(5)]],
    device uint* miss [[buffer(6)]],
    constant uint& table_base [[buffer(7)]],
    constant uint& cols [[buffer(8)]],
    constant uint& inner [[buffer(9)]],
    constant uint& weight_off [[buffer(10)]],
    constant uint& scale_off [[buffer(11)]],
    constant uint& miss_index [[buffer(12)]],
    uint lane [[thread_index_in_simdgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    uint group = tg / cols;
    uint row = tg - group * cols;
    uint begin = group_offset[group];
    uint end = group_offset[group + 1u];
    uint n = end - begin;
    long base = table[table_base + group_expert[group]];
    if (base < 0) {
        if (row == 0u && lane == 0u) {
            miss[miss_index] = 1u;
        }
        if (lane == 0u) {
            for (uint local = 0u; local < n; ++local) {
                destination[(begin + local) * cols + row] = 0.0f;
            }
        }
        return;
    }
    device const uchar* image = atlas + base;
    device const uchar* packed = image + weight_off + row * (inner >> 1);
    device const uchar* row_scale = image + scale_off + row * (inner >> 5);
    uint blocks = inner >> 5;
    uint niter = (blocks + 31u) >> 5;
    // Partial dots live in threadgroup memory so the accumulators do not spill.
    // Codes for one 16-byte word are unpacked once and reused across the tile.
    threadgroup float part[8][32];
    for (uint tile = 0u; tile < n; tile += 8u) {
        uint nt = min(n - tile, 8u);
        for (uint t = 0u; t < nt; ++t) {
            part[t][lane] = 0.0f;
        }
        for (uint iter = 0u; iter < niter; ++iter) {
            uint chunk = lane + iter * 32u;
            if (chunk >= blocks) {
                continue;
            }
            uint4 weight = load_chunk(packed, chunk);
            float scale = e8m0_scale(row_scale[chunk]);
            device const half* act = activation + ((begin + tile) * inner + (chunk << 5));
            float c0, c1, c2, c3, c4, c5, c6, c7;
            uint words[4] = {weight.x, weight.y, weight.z, weight.w};
            for (uint w = 0u; w < 4u; ++w) {
                codes8(words[w], scale, c0, c1, c2, c3, c4, c5, c6, c7);
                for (uint t = 0u; t < nt; ++t) {
                    part[t][lane] += apply8(
                        c0, c1, c2, c3, c4, c5, c6, c7, act + t * inner + w * 8u
                    );
                }
            }
        }
        for (uint t = 0u; t < nt; ++t) {
            float sum = simd_sum(part[t][lane]);
            if (lane == 0u) {
                destination[(begin + tile + t) * cols + row] = sum;
            }
        }
    }
}

kernel void moe_activate(
    device const float* gate_up [[buffer(0)]],
    device half* dst [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& moe [[buffer(3)]],
    constant uint& stored [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = n * stored;
    if (tid >= total) {
        return;
    }
    uint row = tid / stored;
    uint col = tid - row * stored;
    float value = 0.0f;
    if (col < moe) {
        device const float* pair = gate_up + row * moe * 2u;
        value = silu_f(pair[col]) * pair[moe + col];
    }
    dst[tid] = half(clamp(value, -65504.0f, 65504.0f));
}

kernel void moe_scatter(
    device const float* shared [[buffer(0)]],
    device const float* partial [[buffer(1)]],
    device const uint* inverse [[buffer(2)]],
    device const float* weights [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint& count [[buffer(5)]],
    constant uint& topk [[buffer(6)]],
    constant uint& hidden [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = count * hidden;
    if (tid >= total) {
        return;
    }
    uint token = tid / hidden;
    uint row = tid - token * hidden;
    float acc = shared[tid];
    for (uint slot = 0u; slot < topk; ++slot) {
        uint local = inverse[token * topk + slot];
        float scale = weights[token * topk + slot];
        acc += scale * partial[local * hidden + row];
    }
    out[tid] = acc;
}

kernel void moe_decode_gate(
    device const float* activation [[buffer(0)]],
    device const uchar* atlas [[buffer(1)]],
    device const long* table [[buffer(2)]],
    device const uint* ids [[buffer(3)]],
    device float* destination [[buffer(4)]],
    device uint* miss [[buffer(5)]],
    constant uint& topk [[buffer(6)]],
    constant uint& table_base [[buffer(7)]],
    constant uint& cols [[buffer(8)]],
    constant uint& inner [[buffer(9)]],
    constant uint& weight_off [[buffer(10)]],
    constant uint& scale_off [[buffer(11)]],
    constant uint& miss_index [[buffer(12)]],
    uint lane [[thread_index_in_simdgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    uint slot = tg / cols;
    uint row = tg - slot * cols;
    if (slot >= topk) {
        return;
    }
    long base = table[table_base + ids[slot]];
    if (base < 0) {
        if (row == 0u && lane == 0u) {
            miss[miss_index] = 1u;
        }
        if (lane == 0u) {
            destination[slot * cols + row] = 0.0f;
        }
        return;
    }
    device const uchar* image = atlas + base;
    device const uchar* packed = image + weight_off + row * (inner >> 1);
    device const uchar* row_scale = image + scale_off + row * (inner >> 5);
    uint blocks = inner >> 5;
    uint niter = (blocks + 31u) >> 5;
    float acc = 0.0f;
    for (uint iter = 0u; iter < niter; ++iter) {
        uint chunk = lane + iter * 32u;
        if (chunk < blocks) {
            uint4 weight = load_chunk(packed, chunk);
            float scale = e8m0_scale(row_scale[chunk]);
            acc += dot_chunk_f32(weight, scale, activation + (chunk << 5));
        }
    }
    float sum = simd_sum(acc);
    if (lane == 0u) {
        destination[slot * cols + row] = sum;
    }
}

kernel void moe_decode_down(
    device const half* activation [[buffer(0)]],
    device const uchar* atlas [[buffer(1)]],
    device const long* table [[buffer(2)]],
    device const uint* ids [[buffer(3)]],
    device const float* weights [[buffer(4)]],
    device const float* shared [[buffer(5)]],
    device float* destination [[buffer(6)]],
    device uint* miss [[buffer(7)]],
    constant uint& topk [[buffer(8)]],
    constant uint& table_base [[buffer(9)]],
    constant uint& cols [[buffer(10)]],
    constant uint& inner [[buffer(11)]],
    constant uint& weight_off [[buffer(12)]],
    constant uint& scale_off [[buffer(13)]],
    constant uint& miss_index [[buffer(14)]],
    uint lane [[thread_index_in_simdgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    if (row >= cols) {
        return;
    }
    float acc = shared[row];
    for (uint slot = 0u; slot < topk; ++slot) {
        long base = table[table_base + ids[slot]];
        if (base < 0) {
            if (row == 0u && lane == 0u) {
                miss[miss_index] = 1u;
            }
            continue;
        }
        device const uchar* image = atlas + base;
        device const uchar* packed = image + weight_off + row * (inner >> 1);
        device const uchar* row_scale = image + scale_off + row * (inner >> 5);
        uint blocks = inner >> 5;
        uint niter = (blocks + 31u) >> 5;
        device const half* act = activation + slot * inner;
        float dot = 0.0f;
        for (uint iter = 0u; iter < niter; ++iter) {
            uint chunk = lane + iter * 32u;
            if (chunk < blocks) {
                uint4 weight = load_chunk(packed, chunk);
                float scale = e8m0_scale(row_scale[chunk]);
                dot += dot_chunk_half(weight, scale, act + (chunk << 5));
            }
        }
        acc += weights[slot] * simd_sum(dot);
    }
    if (lane == 0u) {
        destination[row] = acc;
    }
}
