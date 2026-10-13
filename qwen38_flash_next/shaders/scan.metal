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
    int code = int(nibble);
    if (code > 7) {
        code -= 16;
    }
    return float(code);
}

// One lane owns one 32-wide block. Consecutive lanes read consecutive blocks.
static uint4 load_chunk(device const uchar* packed, uint chunk) {
    device const uint* words = (device const uint*)(packed + chunk * 16u);
    return uint4(words[0], words[1], words[2], words[3]);
}

static int int4_code(uint nibble) {
    int code = int(nibble);
    return code - ((code >> 3) << 4);
}

static float dot_word_f32(uint word, float scale, device const float* act) {
    float acc = 0.0f;
    uint elem = 0u;
    for (uint b = 0u; b < 4u; ++b) {
        uint byte = (word >> (b * 8u)) & 0xffu;
        int lo = int4_code(byte & 0x0fu);
        int hi = int4_code(byte >> 4);
        float v0 = clamp(act[elem], -65504.0f, 65504.0f);
        float v1 = clamp(act[elem + 1u], -65504.0f, 65504.0f);
        acc += (float(lo) * float(half(v0)) + float(hi) * float(half(v1))) * scale;
        elem += 2u;
    }
    return acc;
}

static float dot_chunk_f32(uint4 chunk, float scale, device const float* act) {
    return dot_word_f32(chunk.x, scale, act)
        + dot_word_f32(chunk.y, scale, act + 8)
        + dot_word_f32(chunk.z, scale, act + 16)
        + dot_word_f32(chunk.w, scale, act + 24);
}

static float dot_word_half(uint word, float scale, device const half* act) {
    float acc = 0.0f;
    uint elem = 0u;
    for (uint b = 0u; b < 4u; ++b) {
        uint byte = (word >> (b * 8u)) & 0xffu;
        int lo = int4_code(byte & 0x0fu);
        int hi = int4_code(byte >> 4);
        acc += (float(lo) * float(act[elem]) + float(hi) * float(act[elem + 1u])) * scale;
        elem += 2u;
    }
    return acc;
}

static float dot_chunk_half(uint4 chunk, float scale, device const half* act) {
    return dot_word_half(chunk.x, scale, act)
        + dot_word_half(chunk.y, scale, act + 8)
        + dot_word_half(chunk.z, scale, act + 16)
        + dot_word_half(chunk.w, scale, act + 24);
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
        int code = int(nibble);
        if (code > 7) {
            code -= 16;
        }
        acc += float(code) * factor * float(act[element]);
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
    uint token [[thread_position_in_grid]])
{
    if (token >= count) {
        return;
    }
    device const float* row = logits + token * n;
    device float* scratch = probs + token * n;
    device uint* out_i = index + token * k;
    device float* out_w = weight + token * k;
    float peak = row[0];
    for (uint i = 1u; i < n; ++i) {
        peak = max(peak, row[i]);
    }
    float sum = 0.0f;
    for (uint i = 0u; i < n; ++i) {
        float prob = exp(row[i] - peak);
        scratch[i] = prob;
        sum += prob;
    }
    for (uint i = 0u; i < n; ++i) {
        scratch[i] /= sum;
    }
    for (uint slot = 0u; slot < k; ++slot) {
        uint best_i = 0u;
        float best = -1.0f;
        for (uint i = 0u; i < n; ++i) {
            if (scratch[i] > best) {
                best = scratch[i];
                best_i = i;
            }
        }
        out_i[slot] = best_i;
        out_w[slot] = best;
        scratch[best_i] = -1.0f;
    }
    float chosen = 0.0f;
    for (uint slot = 0u; slot < k; ++slot) {
        chosen += out_w[slot];
    }
    for (uint slot = 0u; slot < k; ++slot) {
        out_w[slot] /= chosen;
    }
}

kernel void gdn_conv_scan(
    device float* history [[buffer(0)]],
    device const float* qkv [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device float* mixed [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    constant uint& channels [[buffer(5)]],
    constant uint& taps [[buffer(6)]],
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
    constant uint& count [[buffer(9)]],
    constant float& eps [[buffer(10)]],
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

kernel void indexer_write(
    device const float* index_qk [[buffer(0)]],
    device float* cache [[buffer(1)]],
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
    cache[(base + token) * 128u + col] = index_qk[token * 640u + 512u + col];
}

// Sixteen simdgroups split the visible keys and values. One simdgroup walking
// the whole cache leaves the GPU idle once the sequence is long.
kernel void qsa_attend(
    device const float* query [[buffer(0)]],
    device const float* gate [[buffer(1)]],
    device const float* key [[buffer(2)]],
    device const float* value [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& count [[buffer(5)]],
    constant uint& base [[buffer(6)]],
    constant float& scale [[buffer(7)]],
    constant uint& groups [[buffer(8)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]])
{
    threadgroup float scores[2052];
    threadgroup float red[16];
    uint kv = head / 12u;
    for (uint token = 0u; token < count; ++token) {
        device const float* qh = query + (token * 24u + head) * 256u;
        uint visible = base + token + 1u;
        for (uint t = simd; t < visible; t += groups) {
            device const float* kt = key + (t * 2u + kv) * 256u;
            float dot = 0.0f;
            for (uint d = lane; d < 256u; d += 32u) {
                dot += qh[d] * kt[d];
            }
            float score = simd_sum(dot) * scale;
            if (lane == 0u) {
                scores[t] = score;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float local_max = -INFINITY;
        for (uint t = simd; t < visible; t += groups) {
            local_max = max(local_max, scores[t]);
        }
        local_max = simd_max(local_max);
        if (lane == 0u) {
            red[simd] = local_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float maxs = -INFINITY;
        for (uint s = 0u; s < groups; ++s) {
            maxs = max(maxs, red[s]);
        }
        for (uint t = simd; t < visible; t += groups) {
            if (lane == 0u) {
                scores[t] = exp(scores[t] - maxs);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float local_sum = 0.0f;
        for (uint t = simd; t < visible; t += groups) {
            local_sum += scores[t];
        }
        if (lane == 0u) {
            red[simd] = local_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float denom = 0.0f;
        for (uint s = 0u; s < groups; ++s) {
            denom += red[s];
        }
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
        float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
        for (uint t = simd; t < visible; t += groups) {
            float weight = scores[t] / denom;
            device const float* vt = value + (t * 2u + kv) * 256u;
            acc0 += weight * vt[lane];
            acc1 += weight * vt[lane + 32u];
            acc2 += weight * vt[lane + 64u];
            acc3 += weight * vt[lane + 96u];
            acc4 += weight * vt[lane + 128u];
            acc5 += weight * vt[lane + 160u];
            acc6 += weight * vt[lane + 192u];
            acc7 += weight * vt[lane + 224u];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        device float* out = output + (token * 24u + head) * 256u;
        device const float* gh = gate + (token * 24u + head) * 256u;
        for (uint s = 0u; s < groups; ++s) {
            if (simd == s) {
                float gated0 = acc0 * sigmoid_f(gh[lane]);
                float gated1 = acc1 * sigmoid_f(gh[lane + 32u]);
                float gated2 = acc2 * sigmoid_f(gh[lane + 64u]);
                float gated3 = acc3 * sigmoid_f(gh[lane + 96u]);
                float gated4 = acc4 * sigmoid_f(gh[lane + 128u]);
                float gated5 = acc5 * sigmoid_f(gh[lane + 160u]);
                float gated6 = acc6 * sigmoid_f(gh[lane + 192u]);
                float gated7 = acc7 * sigmoid_f(gh[lane + 224u]);
                if (s == 0u) {
                    out[lane] = gated0;
                    out[lane + 32u] = gated1;
                    out[lane + 64u] = gated2;
                    out[lane + 96u] = gated3;
                    out[lane + 128u] = gated4;
                    out[lane + 160u] = gated5;
                    out[lane + 192u] = gated6;
                    out[lane + 224u] = gated7;
                } else {
                    out[lane] += gated0;
                    out[lane + 32u] += gated1;
                    out[lane + 64u] += gated2;
                    out[lane + 96u] += gated3;
                    out[lane + 128u] += gated4;
                    out[lane + 160u] += gated5;
                    out[lane + 192u] += gated6;
                    out[lane + 224u] += gated7;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
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
                acc0 += dot_chunk_f32(w0, s0, act);
                if (m1) {
                    uint4 w1 = load_chunk(weights + (ulong)(row0 + 1u) * stride, chunk);
                    float s1 = e8m0_scale((scales + (ulong)(row0 + 1u) * blocks)[chunk]);
                    acc1 += dot_chunk_f32(w1, s1, act);
                }
                if (m2) {
                    uint4 w2 = load_chunk(weights + (ulong)(row0 + 2u) * stride, chunk);
                    float s2 = e8m0_scale((scales + (ulong)(row0 + 2u) * blocks)[chunk]);
                    acc2 += dot_chunk_f32(w2, s2, act);
                }
                if (m3) {
                    uint4 w3 = load_chunk(weights + (ulong)(row0 + 3u) * stride, chunk);
                    float s3 = e8m0_scale((scales + (ulong)(row0 + 3u) * blocks)[chunk]);
                    acc3 += dot_chunk_f32(w3, s3, act);
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
