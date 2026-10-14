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

static int int8_code(uchar byte) {
    int code = int(byte);
    return code > 127 ? code - 256 : code;
}

kernel void gemv_int8_f32(
    device const float* activation [[buffer(0)]],
    device const uchar* weights [[buffer(1)]],
    device const uchar* scales [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    constant uint& logical [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    if (row >= cols) {
        return;
    }
    uint blocks = inner >> 5;
    device const uchar* packed = weights + (ulong)row * inner;
    device const uchar* row_scale = scales + (ulong)row * blocks;
    float acc = 0.0f;
    for (uint block = lane; block < blocks; block += 32u) {
        uint base = block << 5;
        if (base >= logical) {
            break;
        }
        float factor = e8m0_scale(row_scale[block]);
        for (uint i = 0u; i < 32u; ++i) {
            uint elem = base + i;
            if (elem >= logical) {
                break;
            }
            acc += float(int8_code(packed[elem])) * factor * activation[elem];
        }
    }
    float sum = simd_sum(acc);
    if (lane == 0u) {
        destination[row] = sum;
    }
}

kernel void gemv_int8_half(
    device const half* activation [[buffer(0)]],
    device const uchar* weights [[buffer(1)]],
    device const uchar* scales [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    constant uint& logical [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    if (row >= cols) {
        return;
    }
    uint blocks = inner >> 5;
    device const uchar* packed = weights + (ulong)row * inner;
    device const uchar* row_scale = scales + (ulong)row * blocks;
    float acc = 0.0f;
    for (uint block = lane; block < blocks; block += 32u) {
        uint base = block << 5;
        if (base >= logical) {
            break;
        }
        float factor = e8m0_scale(row_scale[block]);
        for (uint i = 0u; i < 32u; ++i) {
            uint elem = base + i;
            if (elem >= logical) {
                break;
            }
            acc += float(int8_code(packed[elem])) * factor * float(activation[elem]);
        }
    }
    float sum = simd_sum(acc);
    if (lane == 0u) {
        destination[row] = sum;
    }
}

kernel void gather_f32(
    device const float* src [[buffer(0)]],
    device const uint* index [[buffer(1)]],
    device float* dst [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n) {
        return;
    }
    dst[tid] = src[index[tid]];
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
        if (local == 0xffffffffu) {
            continue;
        }
        float scale = weights[token * topk + slot];
        acc += scale * partial[local * hidden + row];
    }
    out[tid] = acc;
}

// Add one expert's rows into the running mix. ``tokens`` and ``slots`` are
// the assignment order. Distinct experts can land on the same token, so the
// add is atomic across dispatches.
kernel void moe_add_weighted(
    device const float* down [[buffer(0)]],
    device const uint* tokens [[buffer(1)]],
    device const uint* slots [[buffer(2)]],
    device const float* weight [[buffer(3)]],
    device atomic_float* dst [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    constant uint& hidden [[buffer(6)]],
    constant uint& topk [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = n * hidden;
    if (tid >= total) {
        return;
    }
    uint row = tid / hidden;
    uint col = tid - row * hidden;
    uint token = tokens[row];
    float scale = weight[token * topk + slots[row]];
    atomic_fetch_add_explicit(
        dst + token * hidden + col, scale * down[tid], memory_order_relaxed
    );
}

static bool tier_accepts(uint want, device const uchar* tier, uint index) {
    if (want == 255u) {
        return true;
    }
    return uint(tier[index]) == want;
}

kernel void moe_decode_gate(
    device const float* activation [[buffer(0)]],
    device const uchar* atlas [[buffer(1)]],
    device const long* table [[buffer(2)]],
    device const uint* ids [[buffer(3)]],
    device float* destination [[buffer(4)]],
    device uint* miss [[buffer(5)]],
    device const uchar* tier [[buffer(6)]],
    constant uint& topk [[buffer(7)]],
    constant uint& table_base [[buffer(8)]],
    constant uint& cols [[buffer(9)]],
    constant uint& inner [[buffer(10)]],
    constant uint& weight_off [[buffer(11)]],
    constant uint& scale_off [[buffer(12)]],
    constant uint& miss_index [[buffer(13)]],
    constant uint& want [[buffer(14)]],
    uint lane [[thread_index_in_simdgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    uint slot = tg / cols;
    uint row = tg - slot * cols;
    if (slot >= topk) {
        return;
    }
    if (!tier_accepts(want, tier, table_base + ids[slot])) {
        if (lane == 0u) {
            destination[slot * cols + row] = 0.0f;
        }
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
    device const uchar* tier [[buffer(8)]],
    constant uint& topk [[buffer(9)]],
    constant uint& table_base [[buffer(10)]],
    constant uint& cols [[buffer(11)]],
    constant uint& inner [[buffer(12)]],
    constant uint& weight_off [[buffer(13)]],
    constant uint& scale_off [[buffer(14)]],
    constant uint& miss_index [[buffer(15)]],
    constant uint& want [[buffer(16)]],
    uint lane [[thread_index_in_simdgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    if (row >= cols) {
        return;
    }
    float acc = shared[row];
    for (uint slot = 0u; slot < topk; ++slot) {
        if (!tier_accepts(want, tier, table_base + ids[slot])) {
            continue;
        }
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

static int int2_signed(uint bits) {
    int code = int(bits & 3u);
    return code - ((code >> 1) << 2);
}

static void codes8_int2(
    uint bits16,
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
    c0 = float(int2_signed(bits16)) * scale;
    c1 = float(int2_signed(bits16 >> 2)) * scale;
    c2 = float(int2_signed(bits16 >> 4)) * scale;
    c3 = float(int2_signed(bits16 >> 6)) * scale;
    c4 = float(int2_signed(bits16 >> 8)) * scale;
    c5 = float(int2_signed(bits16 >> 10)) * scale;
    c6 = float(int2_signed(bits16 >> 12)) * scale;
    c7 = float(int2_signed(bits16 >> 14)) * scale;
}

static float dot_int2_f32(device const uchar* packed, uint chunk, float scale, device const float* act) {
    device const uint* words = (device const uint*)(packed + chunk * 8u);
    float acc = 0.0f;
    uint elem = 0u;
    uint pair[2] = {words[0], words[1]};
    for (uint w = 0u; w < 2u; ++w) {
        uint bits = pair[w];
        for (uint c = 0u; c < 16u; ++c) {
            float value = clamp(act[elem], -65504.0f, 65504.0f);
            acc += float(int2_signed(bits >> (c * 2u))) * float(half(value)) * scale;
            elem += 1u;
        }
    }
    return acc;
}

static float dot_int2_half(device const uchar* packed, uint chunk, float scale, device const half* act) {
    device const uint* words = (device const uint*)(packed + chunk * 8u);
    float acc = 0.0f;
    uint elem = 0u;
    uint pair[2] = {words[0], words[1]};
    for (uint w = 0u; w < 2u; ++w) {
        uint bits = pair[w];
        for (uint c = 0u; c < 16u; ++c) {
            acc += float(int2_signed(bits >> (c * 2u))) * float(act[elem]) * scale;
            elem += 1u;
        }
    }
    return acc;
}

kernel void moe_grouped_int2(
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
    device const uchar* packed = image + weight_off + row * (inner >> 2);
    device const uchar* row_scale = image + scale_off + row * (inner >> 5);
    uint blocks = inner >> 5;
    uint niter = (blocks + 31u) >> 5;
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
            device const uint* words = (device const uint*)(packed + chunk * 8u);
            uint w0 = words[0];
            uint w1 = words[1];
            float scale = e8m0_scale(row_scale[chunk]);
            device const half* act = activation + ((begin + tile) * inner + (chunk << 5));
            uint halves[4] = {w0 & 0xffffu, w0 >> 16, w1 & 0xffffu, w1 >> 16};
            float c0, c1, c2, c3, c4, c5, c6, c7;
            for (uint w = 0u; w < 4u; ++w) {
                codes8_int2(halves[w], scale, c0, c1, c2, c3, c4, c5, c6, c7);
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

kernel void moe_decode_gate_int2(
    device const float* activation [[buffer(0)]],
    device const uchar* atlas [[buffer(1)]],
    device const long* table [[buffer(2)]],
    device const uint* ids [[buffer(3)]],
    device float* destination [[buffer(4)]],
    device uint* miss [[buffer(5)]],
    device const uchar* tier [[buffer(6)]],
    constant uint& topk [[buffer(7)]],
    constant uint& table_base [[buffer(8)]],
    constant uint& cols [[buffer(9)]],
    constant uint& inner [[buffer(10)]],
    constant uint& weight_off [[buffer(11)]],
    constant uint& scale_off [[buffer(12)]],
    constant uint& miss_index [[buffer(13)]],
    constant uint& want [[buffer(14)]],
    uint lane [[thread_index_in_simdgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    uint slot = tg / cols;
    uint row = tg - slot * cols;
    if (slot >= topk) {
        return;
    }
    if (!tier_accepts(want, tier, table_base + ids[slot])) {
        if (lane == 0u) {
            destination[slot * cols + row] = 0.0f;
        }
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
    device const uchar* packed = image + weight_off + row * (inner >> 2);
    device const uchar* row_scale = image + scale_off + row * (inner >> 5);
    uint blocks = inner >> 5;
    uint niter = (blocks + 31u) >> 5;
    float acc = 0.0f;
    for (uint iter = 0u; iter < niter; ++iter) {
        uint chunk = lane + iter * 32u;
        if (chunk < blocks) {
            float scale = e8m0_scale(row_scale[chunk]);
            acc += dot_int2_f32(packed, chunk, scale, activation + (chunk << 5));
        }
    }
    float sum = simd_sum(acc);
    if (lane == 0u) {
        destination[slot * cols + row] = sum;
    }
}

kernel void moe_decode_down_int2(
    device const half* activation [[buffer(0)]],
    device const uchar* atlas [[buffer(1)]],
    device const long* table [[buffer(2)]],
    device const uint* ids [[buffer(3)]],
    device const float* weights [[buffer(4)]],
    device const float* shared [[buffer(5)]],
    device float* destination [[buffer(6)]],
    device uint* miss [[buffer(7)]],
    device const uchar* tier [[buffer(8)]],
    constant uint& topk [[buffer(9)]],
    constant uint& table_base [[buffer(10)]],
    constant uint& cols [[buffer(11)]],
    constant uint& inner [[buffer(12)]],
    constant uint& weight_off [[buffer(13)]],
    constant uint& scale_off [[buffer(14)]],
    constant uint& miss_index [[buffer(15)]],
    constant uint& want [[buffer(16)]],
    uint lane [[thread_index_in_simdgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    if (row >= cols) {
        return;
    }
    float acc = shared[row];
    for (uint slot = 0u; slot < topk; ++slot) {
        if (!tier_accepts(want, tier, table_base + ids[slot])) {
            continue;
        }
        long base = table[table_base + ids[slot]];
        if (base < 0) {
            if (row == 0u && lane == 0u) {
                miss[miss_index] = 1u;
            }
            continue;
        }
        device const uchar* image = atlas + base;
        device const uchar* packed = image + weight_off + row * (inner >> 2);
        device const uchar* row_scale = image + scale_off + row * (inner >> 5);
        uint blocks = inner >> 5;
        uint niter = (blocks + 31u) >> 5;
        device const half* act = activation + slot * inner;
        float dot = 0.0f;
        for (uint iter = 0u; iter < niter; ++iter) {
            uint chunk = lane + iter * 32u;
            if (chunk < blocks) {
                float scale = e8m0_scale(row_scale[chunk]);
                dot += dot_int2_half(packed, chunk, scale, act + (chunk << 5));
            }
        }
        acc += weights[slot] * simd_sum(dot);
    }
    if (lane == 0u) {
        destination[row] = acc;
    }
}
