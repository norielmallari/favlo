// Batched dots, the Gated DeltaNet scan, and the sparse-attention scan.
// Concatenated after act.metal, so silu_f and sigmoid_f are already defined.

// E8M0 bias 127 is the IEEE-754 binary32 exponent field. Exponent 0 flushes to zero.
static float e8m0_scale(uchar exponent) {
    return as_type<float>(uint(exponent) << 23);
}

static float int4_from_chunk(uint4 chunk, uint lane) {
    uint byte_i = lane >> 1;
    uint shift = (byte_i & 3u) * 8u;
    uint word_i = byte_i >> 2;
    uint wordv = word_i == 0u ? chunk.x : word_i == 1u ? chunk.y : word_i == 2u ? chunk.z : chunk.w;
    uint byte = (wordv >> shift) & 0xffu;
    uint nibble = (lane & 1u) ? (byte >> 4) : (byte & 0x0fu);
    return fp4_e2m1(nibble);
}

// One lane owns one 32-wide block. Consecutive lanes read consecutive blocks.
static uint4 load_chunk(device const uchar* packed, uint chunk) {
    device const uint* words = (device const uint*)(packed + chunk * 16u);
    return uint4(words[0], words[1], words[2], words[3]);
}

static float dot_word_f32(uint word, float scale, device const float* act) {
    float acc = 0.0f;
    uint elem = 0u;
    for (uint b = 0u; b < 4u; ++b) {
        uint byte = (word >> (b * 8u)) & 0xffu;
        float lo = fp4_e2m1(byte & 0x0fu);
        float hi = fp4_e2m1(byte >> 4);
        float v0 = clamp(act[elem], -65504.0f, 65504.0f);
        float v1 = clamp(act[elem + 1u], -65504.0f, 65504.0f);
        acc += (lo * float(half(v0)) + hi * float(half(v1))) * scale;
        elem += 2u;
    }
    return acc;
}

static float dot_ready_word(uint word, thread const float* act) {
    float acc = k_fp4_e2m1[word & 15u] * act[0];
    acc += k_fp4_e2m1[(word >> 4) & 15u] * act[1];
    acc += k_fp4_e2m1[(word >> 8) & 15u] * act[2];
    acc += k_fp4_e2m1[(word >> 12) & 15u] * act[3];
    acc += k_fp4_e2m1[(word >> 16) & 15u] * act[4];
    acc += k_fp4_e2m1[(word >> 20) & 15u] * act[5];
    acc += k_fp4_e2m1[(word >> 24) & 15u] * act[6];
    acc += k_fp4_e2m1[(word >> 28) & 15u] * act[7];
    return acc;
}

static float dot_ready(uint4 chunk, thread const float* act) {
    return dot_ready_word(chunk.x, act)
        + dot_ready_word(chunk.y, act + 8)
        + dot_ready_word(chunk.z, act + 16)
        + dot_ready_word(chunk.w, act + 24);
}

static void ready_block(device const float* act, thread float ready[32]) {
    for (uint i = 0u; i < 32u; ++i) {
        ready[i] = float(half(clamp(act[i], -65504.0f, 65504.0f)));
    }
}

static float dot_chunk_f32(uint4 chunk, float scale, device const float* act) {
    float ready[32];
    ready_block(act, ready);
    return scale * dot_ready(chunk, ready);
}

static float dot_word_half(uint word, float scale, device const half* act) {
    float acc = 0.0f;
    uint elem = 0u;
    for (uint b = 0u; b < 4u; ++b) {
        uint byte = (word >> (b * 8u)) & 0xffu;
        float lo = fp4_e2m1(byte & 0x0fu);
        float hi = fp4_e2m1(byte >> 4);
        acc += (lo * float(act[elem]) + hi * float(act[elem + 1u])) * scale;
        elem += 2u;
    }
    return acc;
}

static float dot_half_word(uint word, device const half* act) {
    float acc = k_fp4_e2m1[word & 15u] * float(act[0]);
    acc += k_fp4_e2m1[(word >> 4) & 15u] * float(act[1]);
    acc += k_fp4_e2m1[(word >> 8) & 15u] * float(act[2]);
    acc += k_fp4_e2m1[(word >> 12) & 15u] * float(act[3]);
    acc += k_fp4_e2m1[(word >> 16) & 15u] * float(act[4]);
    acc += k_fp4_e2m1[(word >> 20) & 15u] * float(act[5]);
    acc += k_fp4_e2m1[(word >> 24) & 15u] * float(act[6]);
    acc += k_fp4_e2m1[(word >> 28) & 15u] * float(act[7]);
    return acc;
}

static float dot_chunk_half(uint4 chunk, float scale, device const half* act) {
    float acc = dot_half_word(chunk.x, act);
    acc += dot_half_word(chunk.y, act + 8);
    acc += dot_half_word(chunk.z, act + 16);
    acc += dot_half_word(chunk.w, act + 24);
    return acc * scale;
}

kernel void copy_bytes(
    device const uchar* src [[buffer(0)]],
    device uchar* dst [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n) {
        dst[tid] = src[tid];
    }
}

kernel void gemv_f32_batch(
    device const float* activation [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    uint token = tg / cols;
    uint row = tg - token * cols;
    if (token >= count || row >= cols) {
        return;
    }
    device const float* act = activation + token * inner;
    device const float* rowp = weights + row * inner;
    float acc = 0.0f;
    for (uint i = lane; i < inner; i += 32u) {
        acc += rowp[i] * act[i];
    }
    float sum = simd_sum(acc);
    if (lane == 0u) {
        destination[token * cols + row] = sum;
    }
}

kernel void gemv_f16_batch(
    device const half* activation [[buffer(0)]],
    device const half* weights [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    uint token = tg / cols;
    uint row = tg - token * cols;
    if (token >= count || row >= cols) {
        return;
    }
    device const half* act = activation + token * inner;
    device const half* rowp = weights + row * inner;
    float acc = 0.0f;
    for (uint i = lane; i < inner; i += 32u) {
        acc += float(rowp[i]) * float(act[i]);
    }
    float sum = simd_sum(acc);
    if (lane == 0u) {
        destination[token * cols + row] = sum;
    }
}

kernel void decode_int4_batch(
    device const half* activation [[buffer(0)]],
    device const uchar* weights [[buffer(1)]],
    device const uchar* scales [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    constant uint& inner [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    uint token = tg / cols;
    uint row = tg - token * cols;
    if (token >= count || row >= cols) {
        return;
    }
    device const half* act = activation + token * inner;
    uint blocks = inner >> 5;
    device const uchar* packed = weights + row * (inner >> 1);
    device const uchar* row_scale = scales + row * blocks;
    float acc = 0.0f;
    for (uint block = 0; block < blocks; ++block) {
        float factor = e8m0_scale(row_scale[block]);
        uint element = (block << 5) + lane;
        uchar byte = packed[element >> 1];
        uint nibble = (element & 1u) ? (byte >> 4) : (byte & 0x0Fu);
        acc += fp4_e2m1(nibble) * factor * float(act[element]);
    }
    float sum = simd_sum(acc);
    if (lane == 0u) {
        destination[token * cols + row] = sum;
    }
}

kernel void topk_batch(
    device const float* logits [[buffer(0)]],
    device uint* index [[buffer(1)]],
    device float* weight [[buffer(2)]],
    device float* probs [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    constant uint& k [[buffer(6)]],
    uint token [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    if (token >= count) {
        return;
    }
    device const float* row = logits + token * n;
    device float* scratch = probs + token * n;
    device uint* out_i = index + token * k;
    device float* out_w = weight + token * k;
    float peak = -INFINITY;
    for (uint i = lane; i < n; i += 32u) {
        peak = max(peak, row[i]);
    }
    peak = simd_max(peak);
    float sum = 0.0f;
    for (uint i = lane; i < n; i += 32u) {
        float prob = exp(row[i] - peak);
        scratch[i] = prob;
        sum += prob;
    }
    sum = simd_sum(sum);
    float inv = 1.0f / sum;
    for (uint i = lane; i < n; i += 32u) {
        scratch[i] *= inv;
    }
    for (uint slot = 0u; slot < k; ++slot) {
        float best = -INFINITY;
        uint best_i = 0xffffffffu;
        for (uint i = lane; i < n; i += 32u) {
            float value = scratch[i];
            if (value > best) {
                best = value;
                best_i = i;
            }
        }
        float vmax = simd_max(best);
        uint winner = best == vmax ? best_i : 0xffffffffu;
        uint chosen_i = simd_min(winner);
        float chosen_v = best_i == chosen_i ? best : -INFINITY;
        chosen_v = simd_max(chosen_v);
        if (lane == 0u) {
            out_i[slot] = chosen_i;
            out_w[slot] = chosen_v;
        }
        if (chosen_i != 0xffffffffu && (chosen_i & 31u) == lane) {
            scratch[chosen_i] = -1.0f;
        }
    }
    if (lane == 0u) {
        float chosen = 0.0f;
        for (uint slot = 0u; slot < k; ++slot) {
            chosen += out_w[slot];
        }
        float inv_chosen = 1.0f / chosen;
        for (uint slot = 0u; slot < k; ++slot) {
            out_w[slot] *= inv_chosen;
        }
    }
}

kernel void gdn_conv_scan(
    device float* history [[buffer(0)]],
    device const float* qkv [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device float* mixed [[buffer(3)]],
    device float* snap [[buffer(4)]],
    constant uint& count [[buffer(5)]],
    constant uint& channels [[buffer(6)]],
    constant uint& taps [[buffer(7)]],
    constant uint& save [[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= channels) {
        return;
    }
    uint hist = taps - 1u;
    for (uint token = 0u; token < count; ++token) {
        device const float* current = qkv + token * channels;
        float acc = 0.0f;
        for (uint tap = 0u; tap < hist; ++tap) {
            acc += weight[tid * taps + tap] * history[tap * channels + tid];
        }
        acc += weight[tid * taps + hist] * current[tid];
        mixed[token * channels + tid] = silu_f(acc);
        for (uint tap = 0u; tap + 1u < hist; ++tap) {
            history[tap * channels + tid] = history[(tap + 1u) * channels + tid];
        }
        if (hist > 0u) {
            history[(hist - 1u) * channels + tid] = current[tid];
        }
        if (save != 0u && token < 4u) {
            for (uint tap = 0u; tap < hist; ++tap) {
                snap[(token * hist + tap) * channels + tid] = history[tap * channels + tid];
            }
        }
    }
}

kernel void gdn_state_scan(
    device float* state [[buffer(0)]],
    device const float* mixed [[buffer(1)]],
    device const float* raw_a [[buffer(2)]],
    device const float* raw_b [[buffer(3)]],
    device const float* z [[buffer(4)]],
    device const float* dt_bias [[buffer(5)]],
    device const float* a_log [[buffer(6)]],
    device const float* norm_w [[buffer(7)]],
    device float* heads [[buffer(8)]],
    device float* snap [[buffer(9)]],
    constant uint& count [[buffer(10)]],
    constant float& eps [[buffer(11)]],
    constant uint& save [[buffer(12)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float red[8];
    threadgroup float key_h[128];
    threadgroup float query_h[128];
    threadgroup float decay_h;
    threadgroup float beta_h;
    uint qhead = head / 3u;
    device float* shard = state + head * 128u * 128u;
    for (uint token = 0u; token < count; ++token) {
        device const float* base = mixed + token * 10240u;
        float qv = base[qhead * 128u + tid];
        float kv = base[2048u + qhead * 128u + tid];
        float qpart = simd_sum(qv * qv);
        float kpart = simd_sum(kv * kv);
        if ((tid & 31u) == 0u) {
            red[tid >> 5u] = qpart;
            red[4u + (tid >> 5u)] = kpart;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float qsum = red[0] + red[1] + red[2] + red[3];
        float ksum = red[4] + red[5] + red[6] + red[7];
        qv *= rsqrt(qsum + eps) * rsqrt(128.0f);
        kv *= rsqrt(ksum + eps);
        if (tid == 0u) {
            float raw = raw_a[token * 48u + head];
            decay_h = exp(-exp(a_log[head]) * softplus_f(raw + dt_bias[head]));
            beta_h = sigmoid_f(raw_b[token * 48u + head]);
        }
        query_h[tid] = qv;
        key_h[tid] = kv;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float dot = 0.0f;
        for (uint dk = 0u; dk < 128u; ++dk) {
            dot += shard[dk * 128u + tid] * key_h[dk];
        }
        float decay = decay_h;
        float value = base[4096u + head * 128u + tid];
        float error = value - decay * dot;
        float write = beta_h;
        for (uint dk = 0u; dk < 128u; ++dk) {
            uint index = dk * 128u + tid;
            shard[index] = decay * shard[index] + write * key_h[dk] * error;
        }
        if (save != 0u && token < 4u) {
            device float* dst = snap + (token * 48u + head) * 128u * 128u;
            for (uint dk = 0u; dk < 128u; ++dk) {
                dst[dk * 128u + tid] = shard[dk * 128u + tid];
            }
        }
        float acc = 0.0f;
        for (uint dk = 0u; dk < 128u; ++dk) {
            acc += shard[dk * 128u + tid] * query_h[dk];
        }
        float ss = simd_sum(acc * acc);
        if ((tid & 31u) == 0u) {
            red[tid >> 5u] = ss;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float sumsq = red[0] + red[1] + red[2] + red[3];
        float inv = rsqrt(sumsq / 128.0f + eps);
        float y = acc * inv * norm_w[tid];
        float gate = sigmoid_f(z[token * 6144u + head * 128u + tid]);
        heads[token * 6144u + head * 128u + tid] = y * gate;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void split_q_batch(
    device const float* packed [[buffer(0)]],
    device float* query [[buffer(1)]],
    device float* gate [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant uint& heads [[buffer(4)]],
    constant uint& dim [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    uint span = heads * dim;
    uint total = count * span;
    if (tid >= total) {
        return;
    }
    uint token = tid / span;
    uint within = tid - token * span;
    uint head = within / dim;
    uint col = within - head * dim;
    uint src = token * heads * dim * 2u + head * dim * 2u + col;
    query[tid] = packed[src];
    gate[tid] = packed[src + dim];
}

kernel void rope_batch(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    constant uint& rotary [[buffer(3)]],
    constant uint& position [[buffer(4)]],
    constant float& theta [[buffer(5)]],
    constant uint& heads [[buffer(6)]],
    constant uint& count [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = count * heads * cols;
    if (tid >= total) {
        return;
    }
    uint pairs = rotary >> 1;
    uint row = tid / cols;
    uint col = tid - row * cols;
    uint token = row / heads;
    float value = x[row * cols + col];
    if (col >= rotary) {
        y[row * cols + col] = value;
        return;
    }
    uint pair = col < pairs ? col : col - pairs;
    float freq = pow(theta, -float(pair) / float(pairs));
    float angle = float(position + token) * freq;
    float c = cos(angle);
    float s = sin(angle);
    float first = x[row * cols + pair];
    float second = x[row * cols + pairs + pair];
    if (col < pairs) {
        y[row * cols + col] = first * c - second * s;
    } else {
        y[row * cols + col] = first * s + second * c;
    }
}

static ushort f32_to_bf16(float value) {
    return as_type<ushort>(bfloat(value));
}

static float bf16_to_f32(ushort bits) {
    return float(as_type<bfloat>(bits));
}

static float e4m3_to_f32(uchar code) {
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

// Same ceil as formats.kv_e8m0_byte. Exponent 0 in the shader scale flushes to zero.
static uchar kv_e8m0(float amax) {
    if (!(amax > 0.0f)) {
        return 0u;
    }
    float ratio = amax / 448.0f;
    if (!isfinite(ratio)) {
        return 254u;
    }
    uint bits = as_type<uint>(ratio);
    uint exponent = (bits >> 23) & 255u;
    uint mantissa = bits & 0x7fffffu;
    if (exponent == 255u) {
        return 254u;
    }
    if (exponent == 0u) {
        return 0u;
    }
    int power = mantissa == 0u ? int(exponent) - 127 : int(exponent) - 126;
    int biased = power + 127;
    if (biased < 0) {
        return 0u;
    }
    if (biased > 254) {
        return 254u;
    }
    return uchar(biased);
}

static float load_kv(
    device const uchar* body,
    device const ushort* tail,
    uint token,
    uint channel,
    uint mode,
    uint complete)
{
    if (mode == 0u) {
        device const ushort* words = (device const ushort*)body;
        return bf16_to_f32(words[token * 512u + channel]);
    }
    if (token < complete) {
        device const uchar* rec = body + token * 528u;
        return e4m3_to_f32(rec[channel]) * e8m0_scale(rec[512u + (channel >> 5)]);
    }
    return bf16_to_f32(tail[(token - complete) * 512u + channel]);
}

// Eight consecutive channels. A lane owns lane*8, so the simdgroup covers
// the 256-wide head in order and one E8M0 scale covers the eight.
static void load_kv8(
    device const uchar* body,
    device const ushort* tail,
    uint token,
    uint channel,
    uint mode,
    uint complete,
    thread float out[8])
{
    if (mode == 0u) {
        device const ushort* words = ((device const ushort*)body) + token * 512u + channel;
        for (uint i = 0u; i < 8u; ++i) {
            out[i] = bf16_to_f32(words[i]);
        }
        return;
    }
    if (token < complete) {
        device const uchar* rec = body + token * 528u;
        float scale = e8m0_scale(rec[512u + (channel >> 5)]);
        for (uint i = 0u; i < 8u; ++i) {
            out[i] = e4m3_to_f32(rec[channel + i]) * scale;
        }
        return;
    }
    device const ushort* words = tail + (token - complete) * 512u + channel;
    for (uint i = 0u; i < 8u; ++i) {
        out[i] = bf16_to_f32(words[i]);
    }
}

// Phase 0 writes completed E4M3 blocks, or every new token when mode is BF16.
// Phase 1 writes the unfinished BF16 tail after those blocks. The host orders
// the two dispatches so phase 1 does not replace the tail phase 0 still reads.
kernel void kv_store(
    device const float* key [[buffer(0)]],
    device const float* value [[buffer(1)]],
    device uchar* key_body [[buffer(2)]],
    device uchar* value_body [[buffer(3)]],
    device ushort* key_tail [[buffer(4)]],
    device ushort* value_tail [[buffer(5)]],
    constant uint& count [[buffer(6)]],
    constant uint& base [[buffer(7)]],
    constant uint& mode [[buffer(8)]],
    constant uint& phase [[buffer(9)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    threadgroup float vals[32];
    threadgroup float scaled[32];
    threadgroup uint shared_scale;
    uint plane = group_id & 1u;
    uint rest = group_id >> 1;
    uint group = rest & 15u;
    uint token_i = rest >> 4;
    uint channel = group * 32u + lane;
    device const float* src = plane == 0u ? key : value;
    device uchar* body = plane == 0u ? key_body : value_body;
    device ushort* tail = plane == 0u ? key_tail : value_tail;
    if (mode == 0u) {
        float item = src[token_i * 512u + channel];
        device ushort* dst = (device ushort*)body;
        dst[(base + token_i) * 512u + channel] = f32_to_bf16(item);
        return;
    }
    uint old_complete = base & ~3u;
    uint new_end = base + count;
    uint new_complete = new_end & ~3u;
    if (phase == 0u) {
        uint abs_token = old_complete + token_i;
        float item = abs_token < base
            ? bf16_to_f32(tail[(abs_token - old_complete) * 512u + channel])
            : src[(abs_token - base) * 512u + channel];
        vals[lane] = item;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0u) {
            float amax = 0.0f;
            for (uint i = 0u; i < 32u; ++i) {
                amax = max(amax, fabs(vals[i]));
            }
            shared_scale = kv_e8m0(amax);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint scale = shared_scale;
        scaled[lane] = scale == 0u ? 0.0f : vals[lane] / e8m0_scale(uchar(scale));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0u) {
            device uchar* rec = body + abs_token * 528u;
            uint origin = group * 32u;
            for (uint chunk = 0u; chunk < 4u; ++chunk) {
                vec<float, 8> part;
                for (uint i = 0u; i < 8u; ++i) {
                    part[i] = scaled[chunk * 8u + i];
                }
                auto packed = pack<metal_fp8_e4m3_format>(part);
                auto bytes = packed.as_storage_type();
                for (uint i = 0u; i < 8u; ++i) {
                    rec[origin + chunk * 8u + i] = bytes[i];
                }
            }
            rec[512u + group] = uchar(scale);
        }
        return;
    }
    if (new_complete == old_complete) {
        uint slot = (base - old_complete) + token_i;
        tail[slot * 512u + channel] = f32_to_bf16(src[token_i * 512u + channel]);
        return;
    }
    uint abs_token = new_complete + token_i;
    tail[token_i * 512u + channel] = f32_to_bf16(src[(abs_token - base) * 512u + channel]);
}

kernel void indexer_write(
    device const float* index_qk [[buffer(0)]],
    device ushort* cache [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant uint& base [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = count * 128u;
    if (tid >= total) {
        return;
    }
    uint token = tid / 128u;
    uint col = tid - token * 128u;
    float item = index_qk[token * 640u + 512u + col];
    cache[(base + token) * 128u + col] = f32_to_bf16(item);
}

// One threadgroup owns one key/value head and one token. Twelve simdgroups are
// the query heads that share it, so the key and the value are loaded once.
static void store_kv8(
    threadgroup float* tile,
    device const uchar* body,
    device const ushort* tail,
    uint token,
    uint kv,
    uint mode,
    uint complete,
    uint lane)
{
    float tmp[8];
    load_kv8(body, tail, token, kv * 256u + lane * 8u, mode, complete, tmp);
    uint local = lane * 8u;
    for (uint i = 0u; i < 8u; ++i) {
        tile[local + i] = tmp[i];
    }
}

static void fold_online(
    float score,
    threadgroup const float* value,
    thread float& running_max,
    thread float& running_sum,
    thread float& acc0,
    thread float& acc1,
    thread float& acc2,
    thread float& acc3,
    thread float& acc4,
    thread float& acc5,
    thread float& acc6,
    thread float& acc7)
{
    float next_max = max(running_max, score);
    float decay = running_max == -INFINITY ? 0.0f : exp(running_max - next_max);
    float add = exp(score - next_max);
    running_sum = running_sum * decay + add;
    acc0 = acc0 * decay + add * value[0];
    acc1 = acc1 * decay + add * value[1];
    acc2 = acc2 * decay + add * value[2];
    acc3 = acc3 * decay + add * value[3];
    acc4 = acc4 * decay + add * value[4];
    acc5 = acc5 * decay + add * value[5];
    acc6 = acc6 * decay + add * value[6];
    acc7 = acc7 * decay + add * value[7];
    running_max = next_max;
}

kernel void qsa_attend(
    device const float* query [[buffer(0)]],
    device const float* gate [[buffer(1)]],
    device const uchar* key [[buffer(2)]],
    device const uchar* value [[buffer(3)]],
    device const ushort* key_tail [[buffer(4)]],
    device const ushort* value_tail [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant uint& count [[buffer(7)]],
    constant uint& base [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    constant uint& groups [[buffer(10)]],
    constant uint& mode [[buffer(11)]],
    constant uint& complete [[buffer(12)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]])
{
    threadgroup float shared_k[8 * 256];
    threadgroup float shared_v[8 * 256];
    uint kv = gid.x;
    uint token = gid.y;
    if (kv >= 2u || token >= count) {
        return;
    }
    uint head = kv * 12u + simd;
    device const float* qh = query + (token * 24u + head) * 256u;
    uint ch = lane * 8u;
    float q0 = qh[ch];
    float q1 = qh[ch + 1u];
    float q2 = qh[ch + 2u];
    float q3 = qh[ch + 3u];
    float q4 = qh[ch + 4u];
    float q5 = qh[ch + 5u];
    float q6 = qh[ch + 6u];
    float q7 = qh[ch + 7u];
    uint visible = base + token + 1u;
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
    for (uint t0 = 0u; t0 < visible; t0 += 8u) {
        uint ntile = min(8u, visible - t0);
        if (simd == 0u) {
            for (uint s = 0u; s < ntile; ++s) {
                store_kv8(shared_k + s * 256u, key, key_tail, t0 + s, kv, mode, complete, lane);
                store_kv8(shared_v + s * 256u, value, value_tail, t0 + s, kv, mode, complete, lane);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = 0u; s < ntile; ++s) {
            threadgroup const float* ks = shared_k + s * 256u + ch;
            float partial = q0 * ks[0] + q1 * ks[1] + q2 * ks[2] + q3 * ks[3];
            partial += q4 * ks[4] + q5 * ks[5] + q6 * ks[6] + q7 * ks[7];
            fold_online(
                simd_sum(partial) * scale,
                shared_v + s * 256u + ch,
                running_max,
                running_sum,
                acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = running_sum == 0.0f ? 0.0f : 1.0f / running_sum;
    device float* out = output + (token * 24u + head) * 256u;
    device const float* gh = gate + (token * 24u + head) * 256u;
    out[ch] = acc0 * inv * sigmoid_f(gh[ch]);
    out[ch + 1u] = acc1 * inv * sigmoid_f(gh[ch + 1u]);
    out[ch + 2u] = acc2 * inv * sigmoid_f(gh[ch + 2u]);
    out[ch + 3u] = acc3 * inv * sigmoid_f(gh[ch + 3u]);
    out[ch + 4u] = acc4 * inv * sigmoid_f(gh[ch + 4u]);
    out[ch + 5u] = acc5 * inv * sigmoid_f(gh[ch + 5u]);
    out[ch + 6u] = acc6 * inv * sigmoid_f(gh[ch + 6u]);
    out[ch + 7u] = acc7 * inv * sigmoid_f(gh[ch + 7u]);
}

// Non-negative finite scores rank by their IEEE bits. context.top_blocks
// walks the same four bytes and keeps the lower index on a tie.
static uint score_bits(float value) {
    if (!isfinite(value) || value < 0.0f) {
        value = 0.0f;
    }
    return as_type<uint>(value);
}

static bool radix_active_tg(
    uint bits, threadgroup const uint* shift, threadgroup const uint* byte, uint nprefix
) {
    for (uint p = 0u; p < nprefix; ++p) {
        if (((bits >> shift[p]) & 255u) != byte[p]) {
            return false;
        }
    }
    return true;
}

// One threadgroup owns one block. The pooled key is shared across the queries.
kernel void indexer_block_score(
    device const float* index_qk [[buffer(0)]],
    device const ushort* cache [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant uint& base [[buffer(4)]],
    constant uint& n_blocks [[buffer(5)]],
    uint block [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    if (block >= n_blocks) {
        return;
    }
    threadgroup float pooled[128];
    threadgroup float warp[4];
    for (uint d = lane; d < 128u; d += tg_size) {
        float sum = 0.0f;
        uint token = block * 4u;
        sum += bf16_to_f32(cache[(token + 0u) * 128u + d]);
        sum += bf16_to_f32(cache[(token + 1u) * 128u + d]);
        sum += bf16_to_f32(cache[(token + 2u) * 128u + d]);
        sum += bf16_to_f32(cache[(token + 3u) * 128u + d]);
        pooled[d] = sum * 0.25f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // A decode span is at most four queries. Every lane owns one channel and
    // the head dot is a simd sum. A prefill chunk keeps one query per lane.
    if (count <= 4u) {
        for (uint query = 0u; query < count; ++query) {
            uint pos = base + query;
            uint slot = query * n_blocks + block;
            if (block >= pos / 4u) {
                if (lane == 0u) {
                    scores[slot] = 0.0f;
                }
                continue;
            }
            float acc = 0.0f;
            for (uint head = 0u; head < 4u; ++head) {
                float product = 0.0f;
                if (lane < 128u) {
                    uint q = query * 640u + head * 128u + lane;
                    product = index_qk[q] * pooled[lane];
                }
                float partial = simd_sum(product);
                if ((lane & 31u) == 0u && lane < 128u) {
                    warp[lane >> 5u] = partial;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (lane == 0u) {
                    float dot = warp[0] + warp[1] + warp[2] + warp[3];
                    acc += max(dot, 0.0f);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (lane == 0u) {
                scores[slot] = acc;
            }
        }
        return;
    }
    for (uint query = lane; query < count; query += tg_size) {
        uint pos = base + query;
        if (block >= pos / 4u) {
            scores[query * n_blocks + block] = 0.0f;
            continue;
        }
        float acc = 0.0f;
        for (uint head = 0u; head < 4u; ++head) {
            device const float* qh = index_qk + query * 640u + head * 128u;
            float dot = 0.0f;
            for (uint d = 0u; d < 128u; ++d) {
                dot += qh[d] * pooled[d];
            }
            acc += max(dot, 0.0f);
        }
        scores[query * n_blocks + block] = acc;
    }
}

// One threadgroup per query. Each lane owns a contiguous run of blocks.
// The histogram and the index-ordered collect match context.top_blocks.
kernel void indexer_select(
    device const float* scores [[buffer(0)]],
    device uint* indices [[buffer(1)]],
    device uint* counts [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant uint& base [[buffer(4)]],
    constant uint& n_blocks [[buffer(5)]],
    constant uint& stride [[buffer(6)]],
    uint query [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    if (query >= count) {
        return;
    }
    uint pos = base + query;
    uint tail = (pos / 4u) * 4u;
    uint n_complete = tail / 4u;
    device uint* out = indices + query * stride;
    if (n_complete <= 512u) {
        if (lane == 0u) {
            for (uint t = 0u; t <= pos; ++t) {
                out[t] = t;
            }
            counts[query] = pos + 1u;
        }
        return;
    }
    device const float* row = scores + query * n_blocks;
    threadgroup atomic_uint hist[256];
    threadgroup uint prefix_shift[4];
    threadgroup uint prefix_byte[4];
    threadgroup uint nprefix;
    threadgroup uint picked[512];
    threadgroup uint npicked;
    threadgroup uint need;
    threadgroup uint cut;
    threadgroup uint remaining;
    threadgroup uint stop;
    threadgroup uint chunk_base[256];
    if (lane == 0u) {
        nprefix = 0u;
        npicked = 0u;
        need = 512u;
        stop = 0u;
    }
    if (lane < 256u) {
        atomic_store_explicit(&hist[lane], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint chunk = (n_complete + 255u) / 256u;
    uint start = lane * chunk;
    uint end = min(start + chunk, n_complete);

    uint running = 1u;
    for (uint pass = 0u; pass < 4u && running != 0u; ++pass) {
        uint shift = 24u - pass * 8u;
        if (lane < 256u) {
            atomic_store_explicit(&hist[lane], 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint block = start; block < end; ++block) {
            uint bits = score_bits(row[block]);
            if (radix_active_tg(bits, prefix_shift, prefix_byte, nprefix)) {
                uint byte = (bits >> shift) & 255u;
                atomic_fetch_add_explicit(&hist[byte], 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0u) {
            uint active_n = 0u;
            for (uint bucket = 0u; bucket < 256u; ++bucket) {
                active_n += atomic_load_explicit(&hist[bucket], memory_order_relaxed);
            }
            if (active_n <= need) {
                cut = 256u;
                remaining = 0u;
            } else {
                uint rem = need;
                uint chosen = 0u;
                for (int bucket = 255; bucket >= 0; --bucket) {
                    uint have = atomic_load_explicit(&hist[uint(bucket)], memory_order_relaxed);
                    if (have >= rem) {
                        chosen = uint(bucket);
                        break;
                    }
                    rem -= have;
                }
                cut = chosen;
                remaining = rem;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint local = 0u;
        for (uint block = start; block < end; ++block) {
            uint bits = score_bits(row[block]);
            if (!radix_active_tg(bits, prefix_shift, prefix_byte, nprefix)) {
                continue;
            }
            uint byte = (bits >> shift) & 255u;
            bool take = cut == 256u || byte > cut;
            if (take) {
                local += 1u;
            }
        }
        if (lane < 256u) {
            chunk_base[lane] = local;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0u) {
            uint sum = 0u;
            for (uint i = 0u; i < 256u; ++i) {
                uint have = chunk_base[i];
                chunk_base[i] = npicked + sum;
                sum += have;
            }
            npicked += sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint slot = lane < 256u ? chunk_base[lane] : 0u;
        for (uint block = start; block < end; ++block) {
            uint bits = score_bits(row[block]);
            if (!radix_active_tg(bits, prefix_shift, prefix_byte, nprefix)) {
                continue;
            }
            uint byte = (bits >> shift) & 255u;
            if (cut == 256u || byte > cut) {
                picked[slot] = block;
                slot += 1u;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (cut == 256u || shift == 0u || remaining == 0u) {
            if (shift == 0u && cut != 256u && remaining > 0u) {
                uint matched = 0u;
                for (uint block = start; block < end; ++block) {
                    uint bits = score_bits(row[block]);
                    if (!radix_active_tg(bits, prefix_shift, prefix_byte, nprefix)) {
                        continue;
                    }
                    if (((bits >> shift) & 255u) == cut) {
                        matched += 1u;
                    }
                }
                if (lane < 256u) {
                    chunk_base[lane] = matched;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (lane == 0u) {
                    uint sum = 0u;
                    for (uint i = 0u; i < 256u; ++i) {
                        uint have = chunk_base[i];
                        chunk_base[i] = sum;
                        sum += have;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                uint placed = 0u;
                uint origin = lane < 256u ? chunk_base[lane] : 0u;
                for (uint block = start; block < end; ++block) {
                    uint bits = score_bits(row[block]);
                    if (!radix_active_tg(bits, prefix_shift, prefix_byte, nprefix)) {
                        continue;
                    }
                    if (((bits >> shift) & 255u) != cut) {
                        continue;
                    }
                    if (origin + placed < remaining) {
                        picked[npicked + origin + placed] = block;
                    }
                    placed += 1u;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (lane == 0u) {
                    npicked += remaining;
                    stop = 1u;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            } else {
                if (lane == 0u) {
                    stop = 1u;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        } else {
            if (lane == 0u) {
                prefix_shift[nprefix] = shift;
                prefix_byte[nprefix] = cut;
                nprefix += 1u;
                need = remaining;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        running = stop == 0u ? 1u : 0u;
    }
    if (lane == 0u) {
        for (uint i = 1u; i < npicked; ++i) {
            uint value = picked[i];
            uint j = i;
            while (j > 0u && picked[j - 1u] > value) {
                picked[j] = picked[j - 1u];
                j -= 1u;
            }
            picked[j] = value;
        }
        uint n = 0u;
        for (uint i = 0u; i < npicked; ++i) {
            uint block_start = picked[i] * 4u;
            out[n] = block_start;
            out[n + 1u] = block_start + 1u;
            out[n + 2u] = block_start + 2u;
            out[n + 3u] = block_start + 3u;
            n += 4u;
        }
        for (uint t = tail; t <= pos && n < stride; ++t) {
            out[n] = t;
            n += 1u;
        }
        counts[query] = n;
    }
}

kernel void qsa_attend_selected(
    device const float* query [[buffer(0)]],
    device const float* gate [[buffer(1)]],
    device const uchar* key [[buffer(2)]],
    device const uchar* value [[buffer(3)]],
    device const ushort* key_tail [[buffer(4)]],
    device const ushort* value_tail [[buffer(5)]],
    device const uint* indices [[buffer(6)]],
    device const uint* counts [[buffer(7)]],
    device float* output [[buffer(8)]],
    constant uint& count [[buffer(9)]],
    constant uint& stride [[buffer(10)]],
    constant float& scale [[buffer(11)]],
    constant uint& groups [[buffer(12)]],
    constant uint& kv_mode [[buffer(13)]],
    constant uint& complete [[buffer(14)]],
    device const uint* cache_mode [[buffer(15)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]])
{
    threadgroup float shared_k[8 * 256];
    threadgroup float shared_v[8 * 256];
    uint kv = gid.x;
    uint token = gid.y;
    if (kv >= 2u || token >= count) {
        return;
    }
    uint head = kv * 12u + simd;
    device const float* qh = query + (token * 24u + head) * 256u;
    uint ch = lane * 8u;
    float q0 = qh[ch];
    float q1 = qh[ch + 1u];
    float q2 = qh[ch + 2u];
    float q3 = qh[ch + 3u];
    float q4 = qh[ch + 4u];
    float q5 = qh[ch + 5u];
    float q6 = qh[ch + 6u];
    float q7 = qh[ch + 7u];
    device const uint* ids = indices + token * stride;
    uint nsel = counts[token];
    if (nsel > stride) {
        nsel = stride;
    }
    // Buffer 13 does not reach load_kv8 on this kernel. The mode is a device word.
    uint kv_load = cache_mode[0];
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
    for (uint slot0 = 0u; slot0 < nsel; slot0 += 8u) {
        uint ntile = min(8u, nsel - slot0);
        if (simd == 0u) {
            for (uint s = 0u; s < ntile; ++s) {
                uint picked = ids[slot0 + s];
                store_kv8(shared_k + s * 256u, key, key_tail, picked, kv, kv_load, complete, lane);
                store_kv8(shared_v + s * 256u, value, value_tail, picked, kv, kv_load, complete, lane);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = 0u; s < ntile; ++s) {
            threadgroup const float* ks = shared_k + s * 256u + ch;
            float partial = q0 * ks[0] + q1 * ks[1] + q2 * ks[2] + q3 * ks[3];
            partial += q4 * ks[4] + q5 * ks[5] + q6 * ks[6] + q7 * ks[7];
            fold_online(
                simd_sum(partial) * scale,
                shared_v + s * 256u + ch,
                running_max,
                running_sum,
                acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = running_sum == 0.0f ? 0.0f : 1.0f / running_sum;
    device float* out = output + (token * 24u + head) * 256u;
    device const float* gh = gate + (token * 24u + head) * 256u;
    out[ch] = acc0 * inv * sigmoid_f(gh[ch]);
    out[ch + 1u] = acc1 * inv * sigmoid_f(gh[ch + 1u]);
    out[ch + 2u] = acc2 * inv * sigmoid_f(gh[ch + 2u]);
    out[ch + 3u] = acc3 * inv * sigmoid_f(gh[ch + 3u]);
    out[ch + 4u] = acc4 * inv * sigmoid_f(gh[ch + 4u]);
    out[ch + 5u] = acc5 * inv * sigmoid_f(gh[ch + 5u]);
    out[ch + 6u] = acc6 * inv * sigmoid_f(gh[ch + 6u]);
    out[ch + 7u] = acc7 * inv * sigmoid_f(gh[ch + 7u]);
}

// Four rows per simdgroup. Each lane reads its own 16-byte block so a simdgroup
// streams 512-byte bursts instead of reloading one broadcast block.
kernel void gemv_int4_f32(
    device const float* activation [[buffer(0)]],
    device const uchar* weights [[buffer(1)]],
    device const uchar* scales [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    constant uint& logical [[buffer(6)]],
    constant uint& epilogue [[buffer(7)]],
    constant float& epi_scale [[buffer(8)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    uint row0 = (tg * 4u + simd) * 4u;
    if (row0 >= cols) {
        return;
    }
    uint blocks = inner >> 5;
    uint stride = inner >> 1;
    uint niter = (blocks + 31u) >> 5;
    bool m1 = row0 + 1u < cols;
    bool m2 = row0 + 2u < cols;
    bool m3 = row0 + 3u < cols;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (uint iter = 0u; iter < niter; ++iter) {
        uint chunk = lane + iter * 32u;
        uint base = chunk << 5;
        if (chunk < blocks && base < logical) {
            device const float* act = activation + base;
            uint4 w0 = load_chunk(weights + (ulong)row0 * stride, chunk);
            float s0 = e8m0_scale((scales + (ulong)row0 * blocks)[chunk]);
            if (base + 32u <= logical) {
                float ready[32];
                ready_block(act, ready);
                acc0 += s0 * dot_ready(w0, ready);
                if (m1) {
                    uint4 w1 = load_chunk(weights + (ulong)(row0 + 1u) * stride, chunk);
                    float s1 = e8m0_scale((scales + (ulong)(row0 + 1u) * blocks)[chunk]);
                    acc1 += s1 * dot_ready(w1, ready);
                }
                if (m2) {
                    uint4 w2 = load_chunk(weights + (ulong)(row0 + 2u) * stride, chunk);
                    float s2 = e8m0_scale((scales + (ulong)(row0 + 2u) * blocks)[chunk]);
                    acc2 += s2 * dot_ready(w2, ready);
                }
                if (m3) {
                    uint4 w3 = load_chunk(weights + (ulong)(row0 + 3u) * stride, chunk);
                    float s3 = e8m0_scale((scales + (ulong)(row0 + 3u) * blocks)[chunk]);
                    acc3 += s3 * dot_ready(w3, ready);
                }
            } else {
                for (uint sub = 0u; sub < 32u; ++sub) {
                    if (base + sub >= logical) {
                        break;
                    }
                    float value = clamp(act[sub], -65504.0f, 65504.0f);
                    float halved = float(half(value));
                    uint4 chunks[4] = {w0, uint4(0), uint4(0), uint4(0)};
                    float scales4[4] = {s0, 0.0f, 0.0f, 0.0f};
                    if (m1) {
                        chunks[1] = load_chunk(weights + (ulong)(row0 + 1u) * stride, chunk);
                        scales4[1] = e8m0_scale((scales + (ulong)(row0 + 1u) * blocks)[chunk]);
                    }
                    if (m2) {
                        chunks[2] = load_chunk(weights + (ulong)(row0 + 2u) * stride, chunk);
                        scales4[2] = e8m0_scale((scales + (ulong)(row0 + 2u) * blocks)[chunk]);
                    }
                    if (m3) {
                        chunks[3] = load_chunk(weights + (ulong)(row0 + 3u) * stride, chunk);
                        scales4[3] = e8m0_scale((scales + (ulong)(row0 + 3u) * blocks)[chunk]);
                    }
                    float codes[4];
                    for (uint r = 0u; r < 4u; ++r) {
                        codes[r] = int4_from_chunk(chunks[r], sub) * scales4[r] * halved;
                    }
                    acc0 += codes[0];
                    acc1 += codes[1];
                    acc2 += codes[2];
                    acc3 += codes[3];
                }
            }
        }
    }
    float sums[4] = {simd_sum(acc0), simd_sum(acc1), simd_sum(acc2), simd_sum(acc3)};
    if (lane == 0u) {
        bool live[4] = {true, m1, m2, m3};
        for (uint r = 0u; r < 4u; ++r) {
            if (!live[r]) {
                continue;
            }
            float sum = sums[r];
            if (epilogue == 1u) {
                sum = silu_f(sum);
            } else if (epilogue == 2u) {
                sum = sigmoid_f(sum);
            } else if (epilogue == 3u) {
                sum = silu_f(sum * epi_scale);
            } else if (epilogue == 4u) {
                sum = 2.0f * sigmoid_f(sum * 0.25f);
            }
            destination[row0 + r] = sum;
        }
    }
}

// One weight row, up to four tokens. One row per simdgroup keeps the weight
// chunk in registers so the tokens share that load.
kernel void gemv_int4_span(
    device const float* activation [[buffer(0)]],
    device const uchar* weights [[buffer(1)]],
    device const uchar* scales [[buffer(2)]],
    device float* destination [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    constant uint& inner [[buffer(6)]],
    constant uint& logical [[buffer(7)]],
    constant uint& epilogue [[buffer(8)]],
    constant float& epi_scale [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    uint row = tg * 4u + simd;
    if (row >= cols || count == 0u || count > 4u) {
        return;
    }
    uint blocks = inner >> 5;
    uint stride = inner >> 1;
    uint niter = (blocks + 31u) >> 5;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (uint iter = 0u; iter < niter; ++iter) {
        uint chunk = lane + iter * 32u;
        uint base = chunk << 5;
        if (chunk < blocks && base < logical) {
            uint4 w = load_chunk(weights + (ulong)row * stride, chunk);
            float scale = e8m0_scale((scales + (ulong)row * blocks)[chunk]);
            if (base + 32u <= logical) {
                if (count > 0u) {
                    acc0 += dot_chunk_f32(w, scale, activation + base);
                }
                if (count > 1u) {
                    acc1 += dot_chunk_f32(w, scale, activation + logical + base);
                }
                if (count > 2u) {
                    acc2 += dot_chunk_f32(w, scale, activation + 2u * logical + base);
                }
                if (count > 3u) {
                    acc3 += dot_chunk_f32(w, scale, activation + 3u * logical + base);
                }
            } else {
                for (uint sub = 0u; sub < 32u; ++sub) {
                    if (base + sub >= logical) {
                        break;
                    }
                    float code = int4_from_chunk(w, sub) * scale;
                    float v0 = clamp(activation[base + sub], -65504.0f, 65504.0f);
                    acc0 += code * float(half(v0));
                    if (count > 1u) {
                        float v1 = clamp(activation[logical + base + sub], -65504.0f, 65504.0f);
                        acc1 += code * float(half(v1));
                    }
                    if (count > 2u) {
                        float v2 = clamp(
                            activation[2u * logical + base + sub], -65504.0f, 65504.0f
                        );
                        acc2 += code * float(half(v2));
                    }
                    if (count > 3u) {
                        float v3 = clamp(
                            activation[3u * logical + base + sub], -65504.0f, 65504.0f
                        );
                        acc3 += code * float(half(v3));
                    }
                }
            }
        }
    }
    float s0 = simd_sum(acc0);
    float s1 = simd_sum(acc1);
    float s2 = simd_sum(acc2);
    float s3 = simd_sum(acc3);
    if (lane == 0u) {
        float sums[4] = {s0, s1, s2, s3};
        for (uint t = 0u; t < count; ++t) {
            float sum = sums[t];
            if (epilogue == 1u) {
                sum = silu_f(sum);
            } else if (epilogue == 2u) {
                sum = sigmoid_f(sum);
            } else if (epilogue == 3u) {
                sum = silu_f(sum * epi_scale);
            } else if (epilogue == 4u) {
                sum = 2.0f * sigmoid_f(sum * 0.25f);
            }
            destination[(ulong)t * cols + row] = sum;
        }
    }
}
