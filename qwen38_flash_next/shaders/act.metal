
#include <metal_stdlib>
using namespace metal;

static float silu_f(float v) {
    if (v >= 0.0f) {
        return v / (1.0f + exp(-v));
    }
    float e = exp(v);
    return v * e / (1.0f + e);
}

static float sigmoid_f(float v) {
    if (v >= 0.0f) {
        return 1.0f / (1.0f + exp(-v));
    }
    float e = exp(v);
    return e / (1.0f + e);
}

static float softplus_f(float v) {
    return log(1.0f + exp(-fabs(v))) + max(v, 0.0f);
}

kernel void silu_f32(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n) {
        y[tid] = silu_f(x[tid]);
    }
}

kernel void sigmoid_f32(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n) {
        y[tid] = sigmoid_f(x[tid]);
    }
}

kernel void softplus_f32(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n) {
        y[tid] = softplus_f(x[tid]);
    }
}

kernel void scale_f32(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant float& factor [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n) {
        y[tid] = x[tid] * factor;
    }
}

kernel void mul_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n) {
        y[tid] = a[tid] * b[tid];
    }
}

kernel void add_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n) {
        y[tid] = a[tid] + b[tid];
    }
}

kernel void add_into_f32(
    device float* dst [[buffer(0)]],
    device const float* src [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n) {
        dst[tid] += src[tid];
    }
}

kernel void copy_f32(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n) {
        dst[tid] = src[tid];
    }
}

kernel void axpy_slot(
    device float* dst [[buffer(0)]],
    device const float* src [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& slot [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n) {
        dst[tid] += scales[slot] * src[tid];
    }
}

kernel void scale_rows_f32(
    device const float* x [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = count * cols;
    if (tid >= total) {
        return;
    }
    y[tid] = x[tid] * scales[tid / cols];
}

kernel void rms_rows(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    constant uint& zero_centered [[buffer(5)]],
    constant uint& weight_rows [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    device const float* rowp = x + row * cols;
    float acc = 0.0f;
    for (uint i = lane; i < cols; i += 32u) {
        float value = rowp[i];
        acc += value * value;
    }
    float inv = rsqrt(simd_sum(acc) / float(cols) + eps);
    uint wrow = weight_rows == 0u ? 0u : (row % weight_rows);
    device const float* gain = weight + wrow * cols;
    device float* out = y + row * cols;
    for (uint i = lane; i < cols; i += 32u) {
        float factor = gain[i];
        if (zero_centered != 0u) {
            factor = 1.0f + factor;
        }
        out[i] = rowp[i] * inv * factor;
    }
}

kernel void l2_rows(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    constant float& eps [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    device const float* rowp = x + row * cols;
    float acc = 0.0f;
    for (uint i = lane; i < cols; i += 32u) {
        float value = rowp[i];
        acc += value * value;
    }
    float inv = rsqrt(simd_sum(acc) + eps);
    device float* out = y + row * cols;
    for (uint i = lane; i < cols; i += 32u) {
        out[i] = rowp[i] * inv;
    }
}

kernel void rope_rows(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    constant uint& rotary [[buffer(3)]],
    constant uint& position [[buffer(4)]],
    constant float& theta [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = rows * cols;
    if (tid >= total) {
        return;
    }
    uint pairs = rotary >> 1;
    uint row = tid / cols;
    uint col = tid - row * cols;
    float value = x[row * cols + col];
    if (col >= rotary) {
        y[row * cols + col] = value;
        return;
    }
    uint pair = col < pairs ? col : col - pairs;
    float freq = pow(theta, -float(pair) / float(pairs));
    float angle = float(position) * freq;
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

kernel void causal_conv(
    device float* history [[buffer(0)]],
    device const float* current [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& channels [[buffer(4)]],
    constant uint& taps [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= channels) {
        return;
    }
    uint hist = taps - 1u;
    float acc = 0.0f;
    for (uint tap = 0u; tap < hist; ++tap) {
        float sample = history[tap * channels + tid];
        acc += weight[tid * taps + tap] * sample;
    }
    acc += weight[tid * taps + hist] * current[tid];
    out[tid] = acc;
    for (uint tap = 0u; tap + 1u < hist; ++tap) {
        history[tap * channels + tid] = history[(tap + 1u) * channels + tid];
    }
    if (hist > 0u) {
        history[(hist - 1u) * channels + tid] = current[tid];
    }
}

kernel void dilated_conv(
    device float* history [[buffer(0)]],
    device const float* current [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& channels [[buffer(4)]],
    constant uint& taps [[buffer(5)]],
    constant uint& dilation [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= channels) {
        return;
    }
    uint span = (taps - 1u) * dilation;
    float acc = 0.0f;
    for (uint tap = 0u; tap < taps; ++tap) {
        uint frame = tap * dilation;
        float sample = current[tid];
        if (frame != span) {
            sample = history[frame * channels + tid];
        }
        acc += weight[tid * taps + tap] * sample;
    }
    out[tid] = acc;
    for (uint frame = 0u; frame + 1u < span; ++frame) {
        uint next = (frame + 1u) * channels + tid;
        history[frame * channels + tid] = history[next];
    }
    if (span > 0u) {
        history[(span - 1u) * channels + tid] = current[tid];
    }
}

kernel void repeat_rows(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& times [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = rows * times * cols;
    if (tid >= total) {
        return;
    }
    uint col = tid % cols;
    uint src_row = (tid / cols) / times;
    dst[tid] = src[src_row * cols + col];
}

kernel void residual_write(
    device float* residual [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device const float* mixed [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant uint& branches [[buffer(4)]],
    constant uint& hidden [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    uint span = branches * hidden;
    uint total = count * span;
    if (tid >= total) {
        return;
    }
    uint token = tid / span;
    uint within = tid - token * span;
    uint branch = within / hidden;
    uint col = within - branch * hidden;
    float scale = scales[token * branches + branch];
    residual[tid] += scale * mixed[token * hidden + col];
}

kernel void branch_mean(
    device const float* gate [[buffer(0)]],
    device const float* hats [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant uint& branches [[buffer(4)]],
    constant uint& hidden [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = count * hidden;
    if (tid >= total) {
        return;
    }
    uint col = tid % hidden;
    uint token = tid / hidden;
    float acc = 0.0f;
    for (uint branch = 0u; branch < branches; ++branch) {
        uint index = (token * branches + branch) * hidden + col;
        acc += gate[index] * hats[index];
    }
    out[tid] = acc / float(branches);
}

kernel void gdn_alpha(
    device const float* raw [[buffer(0)]],
    device const float* dt_bias [[buffer(1)]],
    device const float* a_log [[buffer(2)]],
    device float* alpha [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n) {
        return;
    }
    alpha[tid] = exp(-exp(a_log[tid]) * softplus_f(raw[tid] + dt_bias[tid]));
}

kernel void ple_gate(
    device const float* query [[buffer(0)]],
    device const float* keyed [[buffer(1)]],
    device float* gate [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant float& inv_sqrt [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    device const float* q = query + row * cols;
    device const float* k = keyed + row * cols;
    float acc = 0.0f;
    for (uint i = lane; i < cols; i += 32u) {
        acc += q[i] * k[i];
    }
    float dot = simd_sum(acc) * inv_sqrt;
    if (lane != 0u) {
        return;
    }
    float mag = fabs(dot);
    if (mag < 1.0e-6f) {
        mag = 1.0e-6f;
    }
    float sign = 0.0f;
    if (dot < 0.0f) {
        sign = -1.0f;
    } else if (dot > 0.0f) {
        sign = 1.0f;
    }
    gate[row] = sigmoid_f(sign * sqrt(mag));
}

kernel void gate_broadcast(
    device const float* gate [[buffer(0)]],
    device const float* value [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& branches [[buffer(3)]],
    constant uint& hidden [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = branches * hidden;
    if (tid >= total) {
        return;
    }
    out[tid] = gate[tid / hidden] * value[tid % hidden];
}

kernel void split_q(
    device const float* packed [[buffer(0)]],
    device float* query [[buffer(1)]],
    device float* gate [[buffer(2)]],
    constant uint& heads [[buffer(3)]],
    constant uint& dim [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = heads * dim;
    if (tid >= total) {
        return;
    }
    uint head = tid / dim;
    uint col = tid - head * dim;
    query[tid] = packed[head * dim * 2u + col];
    gate[tid] = packed[head * dim * 2u + dim + col];
}

kernel void sigmoid_mul(
    device const float* left [[buffer(0)]],
    device const float* gate [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n) {
        out[tid] = left[tid] * sigmoid_f(gate[tid]);
    }
}

kernel void topk_softmax(
    device const float* logits [[buffer(0)]],
    device uint* index [[buffer(1)]],
    device float* weight [[buffer(2)]],
    device float* probs [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0u) {
        return;
    }
    float peak = logits[0];
    for (uint i = 1u; i < n; ++i) {
        peak = max(peak, logits[i]);
    }
    float sum = 0.0f;
    for (uint i = 0u; i < n; ++i) {
        float prob = exp(logits[i] - peak);
        probs[i] = prob;
        sum += prob;
    }
    for (uint i = 0u; i < n; ++i) {
        probs[i] /= sum;
    }
    for (uint slot = 0u; slot < k; ++slot) {
        uint best_i = 0u;
        float best = -1.0f;
        for (uint i = 0u; i < n; ++i) {
            if (probs[i] > best) {
                best = probs[i];
                best_i = i;
            }
        }
        index[slot] = best_i;
        weight[slot] = best;
        probs[best_i] = -1.0f;
    }
    float chosen = 0.0f;
    for (uint slot = 0u; slot < k; ++slot) {
        chosen += weight[slot];
    }
    for (uint slot = 0u; slot < k; ++slot) {
        weight[slot] /= chosen;
    }
}

kernel void argmax_partial(
    device const float* values [[buffer(0)]],
    device float* peaks [[buffer(1)]],
    device uint* index [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& grid [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= grid) {
        return;
    }
    float best = -INFINITY;
    uint best_i = 0u;
    for (uint i = gid; i < n; i += grid) {
        float value = values[i];
        if (value > best) {
            best = value;
            best_i = i;
        }
    }
    peaks[gid] = best;
    index[gid] = best_i;
}

kernel void argmax_final(
    device const float* peaks [[buffer(0)]],
    device const uint* index [[buffer(1)]],
    device uint* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0u) {
        return;
    }
    float best = peaks[0];
    uint best_i = index[0];
    for (uint i = 1u; i < n; ++i) {
        if (peaks[i] > best) {
            best = peaks[i];
            best_i = index[i];
        }
    }
    out[0] = best_i;
}

kernel void cast_f16_pad(
    device const float* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    constant uint& logical [[buffer(2)]],
    constant uint& stored [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= stored) {
        return;
    }
    float value = 0.0f;
    if (tid < logical) {
        value = src[tid];
    }
    value = clamp(value, -65504.0f, 65504.0f);
    dst[tid] = half(value);
}

kernel void cast_prefill(
    device const float* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant uint& logical [[buffer(3)]],
    constant uint& padded [[buffer(4)]],
    constant uint& stored [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = padded * stored;
    if (tid >= total) {
        return;
    }
    uint row = tid / stored;
    uint col = tid - row * stored;
    float value = 0.0f;
    if (row < count && col < logical) {
        value = src[row * logical + col];
    }
    dst[tid] = half(clamp(value, -65504.0f, 65504.0f));
}

kernel void compact_rows(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& packed [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = count * rows;
    if (tid >= total) {
        return;
    }
    uint col = tid % rows;
    uint row = tid / rows;
    dst[tid] = src[row * packed + col];
}

kernel void embed_gather(
    device const uchar* image [[buffer(0)]],
    device const uint* tokens [[buffer(1)]],
    device float* residual [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant uint& logical [[buffer(4)]],
    constant uint& data_row [[buffer(5)]],
    constant uint& scale_row [[buffer(6)]],
    constant uint& data_base [[buffer(7)]],
    constant uint& branches [[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = count * logical;
    if (tid >= total) {
        return;
    }
    uint token_i = tid / logical;
    uint col = tid - token_i * logical;
    uint token = tokens[token_i];
    device const uchar* data = image + token * data_row;
    device const uchar* scales = image + data_base + token * scale_row;
    uchar byte = data[col >> 1];
    uint nibble = (col & 1u) != 0u ? (byte >> 4) : (byte & 0x0Fu);
    int code = int(nibble);
    if (code > 7) {
        code -= 16;
    }
    int exponent = int(scales[col >> 5]) - 127;
    float value = float(code) * ldexp(1.0f, exponent);
    for (uint branch = 0u; branch < branches; ++branch) {
        uint at = (token_i * branches + branch) * logical + col;
        residual[at] = value;
    }
}

kernel void dequant_ngram(
    device const uchar* slots [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& width [[buffer(2)]],
    constant uint& slot_bytes [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    uint rows = 16u;
    if (tid >= rows * width) {
        return;
    }
    uint row = tid / width;
    uint col = tid - row * width;
    device const uchar* slot = slots + row * slot_bytes;
    uchar byte = slot[col >> 1];
    uint nibble = (col & 1u) != 0u ? (byte >> 4) : (byte & 0x0Fu);
    int code = int(nibble);
    if (code > 7) {
        code -= 16;
    }
    int exponent = int(slot[80u + (col >> 5)]) - 127;
    out[tid] = float(code) * ldexp(1.0f, exponent);
}

kernel void broadcast_rows(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= cols) {
        return;
    }
    float value = src[tid];
    for (uint row = 0u; row < rows; ++row) {
        dst[row * cols + tid] = value;
    }
}

// Each thread keeps the top k scores it sees. The draw kernel merges those lists.
kernel void sample_topk_partial(
    device const float* logits [[buffer(0)]],
    device float* top_scores [[buffer(1)]],
    device uint* top_ids [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    constant float& temperature [[buffer(5)]],
    constant uint& grid [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    float scores[32];
    uint ids[32];
    uint keep = k > 32u ? 32u : k;
    for (uint s = 0u; s < 32u; ++s) {
        scores[s] = -INFINITY;
        ids[s] = 0u;
    }
    if (gid < grid && keep > 0u) {
        for (uint i = gid; i < n; i += grid) {
            float score = logits[i] / temperature;
            if (!(score > scores[keep - 1u])) {
                continue;
            }
            uint pos = keep - 1u;
            while (pos > 0u && score > scores[pos - 1u]) {
                scores[pos] = scores[pos - 1u];
                ids[pos] = ids[pos - 1u];
                pos -= 1u;
            }
            scores[pos] = score;
            ids[pos] = i;
        }
    }
    if (gid < grid) {
        for (uint s = 0u; s < 32u; ++s) {
            top_scores[gid * 32u + s] = scores[s];
            top_ids[gid * 32u + s] = ids[s];
        }
    }
}

kernel void sample_topk_draw(
    device const float* top_scores [[buffer(0)]],
    device const uint* top_ids [[buffer(1)]],
    device uint* out [[buffer(2)]],
    constant uint& grid [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    constant float& top_p [[buffer(5)]],
    constant float& uniform [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0u) {
        return;
    }
    float scores[32];
    uint ids[32];
    float probs[32];
    uint keep = k > 32u ? 32u : k;
    for (uint s = 0u; s < 32u; ++s) {
        scores[s] = -INFINITY;
        ids[s] = 0u;
        probs[s] = 0.0f;
    }
    for (uint g = 0u; g < grid; ++g) {
        for (uint s = 0u; s < keep; ++s) {
            float score = top_scores[g * 32u + s];
            uint id = top_ids[g * 32u + s];
            if (!(score > scores[keep - 1u])) {
                continue;
            }
            uint pos = keep - 1u;
            while (pos > 0u && score > scores[pos - 1u]) {
                scores[pos] = scores[pos - 1u];
                ids[pos] = ids[pos - 1u];
                pos -= 1u;
            }
            scores[pos] = score;
            ids[pos] = id;
        }
    }
    float peak = scores[0];
    float sum = 0.0f;
    for (uint s = 0u; s < keep; ++s) {
        float prob = exp(scores[s] - peak);
        probs[s] = prob;
        sum += prob;
    }
    if (top_p < 1.0f && sum > 0.0f) {
        float cursor = 0.0f;
        uint kept = keep;
        for (uint s = 0u; s < keep; ++s) {
            cursor += probs[s] / sum;
            if (cursor > top_p) {
                kept = s + 1u;
                break;
            }
        }
        sum = 0.0f;
        for (uint s = 0u; s < keep; ++s) {
            if (s >= kept) {
                probs[s] = 0.0f;
            }
            sum += probs[s];
        }
    }
    for (uint a = 1u; a < keep; ++a) {
        float prob = probs[a];
        uint id = ids[a];
        uint b = a;
        while (b > 0u && ids[b - 1u] > id) {
            probs[b] = probs[b - 1u];
            ids[b] = ids[b - 1u];
            b -= 1u;
        }
        probs[b] = prob;
        ids[b] = id;
    }
    if (!(sum > 0.0f)) {
        out[0] = ids[0];
        return;
    }
    float cursor = 0.0f;
    for (uint s = 0u; s < keep; ++s) {
        cursor += probs[s] / sum;
        if (uniform < cursor) {
            out[0] = ids[s];
            return;
        }
    }
    out[0] = ids[keep - 1u];
}

kernel void sample_f32(
    device const float* logits [[buffer(0)]],
    device float* scratch [[buffer(1)]],
    device uint* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant float& temperature [[buffer(4)]],
    constant float& top_p [[buffer(5)]],
    constant uint& top_k [[buffer(6)]],
    constant float& uniform [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0u) {
        return;
    }
    float peak = -INFINITY;
    for (uint i = 0u; i < n; ++i) {
        float score = logits[i] / temperature;
        scratch[i] = score;
        peak = max(peak, score);
    }
    if (top_k > 0u && top_k < n) {
        float previous = INFINITY;
        float cutoff = peak;
        for (uint nth = 0u; nth < top_k; ++nth) {
            float next = -INFINITY;
            for (uint i = 0u; i < n; ++i) {
                float value = scratch[i];
                if (value < previous && value > next) {
                    next = value;
                }
            }
            cutoff = next;
            previous = next;
        }
        for (uint i = 0u; i < n; ++i) {
            if (scratch[i] < cutoff) {
                scratch[i] = -INFINITY;
            }
        }
        peak = cutoff;
    }
    float sum = 0.0f;
    for (uint i = 0u; i < n; ++i) {
        float prob = 0.0f;
        if (scratch[i] > -INFINITY) {
            prob = exp(scratch[i] - peak);
        }
        scratch[i] = prob;
        sum += prob;
    }
    if (!(sum > 0.0f)) {
        uint best = 0u;
        float best_v = logits[0];
        for (uint i = 1u; i < n; ++i) {
            if (logits[i] > best_v) {
                best_v = logits[i];
                best = i;
            }
        }
        out[0] = best;
        return;
    }
    for (uint i = 0u; i < n; ++i) {
        scratch[i] /= sum;
    }
    if (top_p < 1.0f) {
        float cursor = 0.0f;
        while (true) {
            uint best = 0u;
            float best_p = 0.0f;
            for (uint i = 0u; i < n; ++i) {
                if (scratch[i] > best_p) {
                    best_p = scratch[i];
                    best = i;
                }
            }
            if (!(best_p > 0.0f)) {
                break;
            }
            scratch[best] = -best_p;
            cursor += best_p;
            if (cursor > top_p) {
                break;
            }
        }
        sum = 0.0f;
        for (uint i = 0u; i < n; ++i) {
            if (scratch[i] < 0.0f) {
                scratch[i] = -scratch[i];
                sum += scratch[i];
            } else {
                scratch[i] = 0.0f;
            }
        }
        for (uint i = 0u; i < n; ++i) {
            scratch[i] /= sum;
        }
    }
    float cursor = 0.0f;
    for (uint i = 0u; i < n; ++i) {
        cursor += scratch[i];
        if (uniform < cursor) {
            out[0] = i;
            return;
        }
    }
    out[0] = n - 1u;
}
