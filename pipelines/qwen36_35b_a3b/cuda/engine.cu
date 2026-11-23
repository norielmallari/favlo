// Qwen3.6-35B-A3B language step.
// Integer GEMMs keep both operands in the weight's type. The epilogue is the
// fp32 row scale times the fp32 activation scale. The key/value cache is int8
// with one E8M0 scale byte per 32 channels. Gated DeltaNet follows Yang, Kautz,
// and Hatamizadeh, arXiv:2412.06464: decay the state, then add beta (v - S k) k^T.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cmath>

#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API
#endif

static int check(cudaError_t err) { return err == cudaSuccess ? 0 : static_cast<int>(err); }

static cudaStream_t g_stream = nullptr;
static cudaStream_t g_main = nullptr;
static cudaStream_t g_side = nullptr;
constexpr int kGraphs = 82;
constexpr int kForks = 128;
static cudaGraphExec_t g_graph[kGraphs] = {};
static cudaEvent_t g_forks[kForks] = {};
static cudaEvent_t g_joins[kForks] = {};
static int g_fork_at = 0;
static int g_splits = 1;
constexpr int kAttendHeads = 16;
constexpr int kAttendDim = 256;
constexpr int kAttendSplits = 128;
constexpr int kCopies = 32;
static float* g_attend_peak = nullptr;
static float* g_attend_sum = nullptr;
static float* g_attend_acc = nullptr;
static cudaEvent_t g_copies[kCopies] = {};
static int g_copy_next = 0;
static int* g_step = nullptr;
__device__ int* g_step_ptr;
static int8_t* g_zero = nullptr;
__device__ const int8_t* g_zero_ptr;
static int8_t* g_codes = nullptr;
static float* g_scales = nullptr;
static int8_t* g_expand = nullptr;
static int g_code_limit = 0;
static int g_scale_limit = 0;
static int g_expand_limit = 0;

__device__ int step_at(int index) { return reinterpret_cast<volatile int*>(g_step_ptr)[index]; }

static int scratch_for(int tokens, int inner) {
    int codes = tokens * inner;
    int scales = tokens * (inner / 32);
    if (codes <= g_code_limit && scales <= g_scale_limit) {
        return 0;
    }
    cudaFree(g_codes);
    cudaFree(g_scales);
    g_codes = nullptr;
    g_scales = nullptr;
    g_code_limit = 0;
    g_scale_limit = 0;
    int status = check(cudaMalloc(&g_codes, static_cast<size_t>(codes)));
    if (status != 0) {
        return status;
    }
    status = check(cudaMalloc(&g_scales, static_cast<size_t>(scales) * sizeof(float)));
    if (status != 0) {
        cudaFree(g_codes);
        g_codes = nullptr;
        return status;
    }
    g_code_limit = codes;
    g_scale_limit = scales;
    return 0;
}

static int expand_for(int count) {
    if (count <= g_expand_limit) {
        return 0;
    }
    cudaFree(g_expand);
    g_expand = nullptr;
    g_expand_limit = 0;
    int status = check(cudaMalloc(&g_expand, static_cast<size_t>(count)));
    if (status != 0) {
        return status;
    }
    g_expand_limit = count;
    return 0;
}

__device__ int sext4(int nibble) {
    int n = nibble & 0xF;
    return n >= 8 ? n - 16 : n;
}

__device__ float block_sum(float value) {
    __shared__ float buf[256];
    int lane = threadIdx.x;
    buf[lane] = value;
    __syncthreads();
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (lane < offset) {
            buf[lane] += buf[lane + offset];
        }
        __syncthreads();
    }
    return buf[0];
}

__global__ void quant_kernel(const float* source, void* codes, float* scales, int inner, int bits, int tokens) {
    int token = blockIdx.x;
    if (token >= tokens) {
        return;
    }
    const float* row = source + static_cast<long long>(token) * inner;
    int blocks = inner >> 5;
    int qmax = bits == 4 ? 7 : 127;
    for (int block = threadIdx.x; block < blocks; block += blockDim.x) {
        const float* slice = row + block * 32;
        float peak = 0.0f;
        for (int i = 0; i < 32; ++i) {
            peak = fmaxf(peak, fabsf(slice[i]));
        }
        float scale = peak == 0.0f ? 0.0f : peak / static_cast<float>(qmax);
        scales[static_cast<long long>(token) * blocks + block] = scale;
        float inv = scale == 0.0f ? 0.0f : 1.0f / scale;
        if (bits == 8) {
            auto* out = static_cast<int8_t*>(codes) + static_cast<long long>(token) * inner + block * 32;
            for (int i = 0; i < 32; ++i) {
                int code = static_cast<int>(rintf(slice[i] * inv));
                code = max(-127, min(127, code));
                out[i] = static_cast<int8_t>(code);
            }
        } else {
            auto* out = static_cast<uint8_t*>(codes) + (static_cast<long long>(token) * inner + block * 32) / 2;
            for (int i = 0; i < 32; i += 2) {
                int lo = static_cast<int>(rintf(slice[i] * inv));
                int hi = static_cast<int>(rintf(slice[i + 1] * inv));
                lo = max(-7, min(7, lo));
                hi = max(-7, min(7, hi));
                out[i / 2] = static_cast<uint8_t>((lo & 0xF) | ((hi & 0xF) << 4));
            }
        }
    }
}

__device__ int dot32_i8(const int8_t* weight, const int8_t* act) {
    int acc = 0;
    const int* w = reinterpret_cast<const int*>(weight);
    const int* a = reinterpret_cast<const int*>(act);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        acc = __dp4a(w[i], a[i], acc);
    }
    return acc;
}

__device__ int dot32_i4(const uint8_t* weight, const uint8_t* act) {
    int acc = 0;
    for (int i = 0; i < 16; ++i) {
        int wb = weight[i];
        int ab = act[i];
        acc += sext4(wb) * sext4(ab);
        acc += sext4(wb >> 4) * sext4(ab >> 4);
    }
    return acc;
}

__global__ void gemv_kernel(
    const void* weight,
    const float* wscale,
    const void* codes,
    const float* ascales,
    float* destination,
    int rows,
    int inner,
    int bits,
    int tokens) {
    int row = blockIdx.x;
    int token = blockIdx.y;
    if (row >= rows || token >= tokens) {
        return;
    }
    int blocks = inner >> 5;
    const float* scale = ascales + static_cast<long long>(token) * blocks;
    float acc = 0.0f;
    for (int block = threadIdx.x; block < blocks; block += blockDim.x) {
        int dot = 0;
        if (bits == 8) {
            const auto* w = static_cast<const int8_t*>(weight) + (static_cast<long long>(row) * inner + block * 32);
            const auto* a = static_cast<const int8_t*>(codes) + (static_cast<long long>(token) * inner + block * 32);
            dot = dot32_i8(w, a);
        } else {
            const auto* w = static_cast<const uint8_t*>(weight) + (static_cast<long long>(row) * inner + block * 32) / 2;
            const auto* a = static_cast<const uint8_t*>(codes) + (static_cast<long long>(token) * inner + block * 32) / 2;
            dot = dot32_i4(w, a);
        }
        acc += static_cast<float>(dot) * scale[block];
    }
    acc = block_sum(acc);
    if (threadIdx.x == 0) {
        destination[static_cast<long long>(token) * rows + row] = acc * wscale[row];
    }
}

// One block converts the int4 activation once. Every warp in the block reuses it.
__global__ void gemv_i4_cached(
    const void* weight,
    const float* wscale,
    const void* codes,
    const float* ascales,
    float* destination,
    int rows,
    int inner) {
    extern __shared__ __align__(16) char shared_mix[];
    int packed = inner >> 1;
    int blocks = inner >> 5;
    auto* act = reinterpret_cast<uint8_t*>(shared_mix);
    auto* scales = reinterpret_cast<float*>(act + packed);
    const auto* raw = static_cast<const uint8_t*>(codes);
    for (int col = threadIdx.x; col < packed; col += blockDim.x) {
        act[col] = raw[col];
    }
    for (int block = threadIdx.x; block < blocks; block += blockDim.x) {
        scales[block] = ascales[block];
    }
    __syncthreads();
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps = blockDim.x >> 5;
    int row = blockIdx.x * warps + warp;
    if (row >= rows) {
        return;
    }
    const auto* line = static_cast<const uint8_t*>(weight) + (static_cast<long long>(row) * inner) / 2;
    float acc = 0.0f;
    for (int block = lane; block < blocks; block += 32) {
        int dot = dot32_i4(line + block * 16, act + block * 16);
        acc += static_cast<float>(dot) * scales[block];
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0) {
        destination[row] = acc * wscale[row];
    }
}

__global__ void gemv_f16_rows_kernel(
    const __half* weight, const float* source, float* destination, int rows, int inner) {
    int row = blockIdx.x;
    int token = blockIdx.y;
    if (row >= rows) {
        return;
    }
    const __half* line = weight + static_cast<long long>(row) * inner;
    const float* row_in = source + static_cast<long long>(token) * inner;
    float acc = 0.0f;
    for (int col = threadIdx.x; col < inner; col += blockDim.x) {
        acc += __half2float(line[col]) * row_in[col];
    }
    acc = block_sum(acc);
    if (threadIdx.x == 0) {
        destination[static_cast<long long>(token) * rows + row] = acc;
    }
}

__global__ void gemv_f16_kernel(const __half* weight, const float* source, float* destination, int rows, int inner) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const __half* line = weight + static_cast<long long>(row) * inner;
    float acc = 0.0f;
    for (int col = threadIdx.x; col < inner; col += blockDim.x) {
        acc += __half2float(line[col]) * source[col];
    }
    acc = block_sum(acc);
    if (threadIdx.x == 0) {
        destination[row] = acc;
    }
}

__global__ void embed_kernel(const int8_t* table, const float* scale, int token, float* destination, int inner) {
    float factor = scale[token];
    const int8_t* row = table + static_cast<long long>(token) * inner;
    for (int col = threadIdx.x; col < inner; col += blockDim.x) {
        destination[col] = static_cast<float>(row[col]) * factor;
    }
}

__global__ void rmsnorm_kernel(
    const float* source, const __half* weight, float* destination, int width, float eps, int heads) {
    int head = blockIdx.x;
    if (head >= heads) {
        return;
    }
    const float* row = source + static_cast<long long>(head) * width;
    float sum = 0.0f;
    for (int col = threadIdx.x; col < width; col += blockDim.x) {
        float value = row[col];
        sum += value * value;
    }
    sum = block_sum(sum);
    float inv = rsqrtf(sum / static_cast<float>(width) + eps);
    float* out = destination + static_cast<long long>(head) * width;
    for (int col = threadIdx.x; col < width; col += blockDim.x) {
        float factor = weight == nullptr ? 1.0f : 1.0f + __half2float(weight[col]);
        out[col] = row[col] * inv * factor;
    }
}

__global__ void add_kernel(float* stream, const float* delta, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < width) {
        stream[col] += delta[col];
    }
}

__global__ void silu_mul_kernel(const float* gate, const float* up, float* destination, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < width) {
        float value = gate[col];
        float silu = value / (1.0f + expf(-value));
        destination[col] = silu * up[col];
    }
}

__global__ void scale_add_kernel(float* destination, const float* source, float scale, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < width) {
        destination[col] += source[col] * scale;
    }
}

__global__ void scale_add_at_kernel(
    float* destination, const float* source, const float* scale, int width) {
    float factor = scale[0];
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < width) {
        destination[col] += source[col] * factor;
    }
}

__global__ void sig_add_kernel(float* destination, const float* source, const float* coeff, int width) {
    float scale = 1.0f / (1.0f + expf(-coeff[0]));
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < width) {
        destination[col] += source[col] * scale;
    }
}

__global__ void l2norm_tokens_kernel(float* values, int heads, int dim, int stride, int tokens) {
    int id = blockIdx.x;
    int token = id / heads;
    int head = id - token * heads;
    if (token >= tokens) {
        return;
    }
    float* row = values + static_cast<long long>(token) * stride + static_cast<long long>(head) * dim;
    float sum = 0.0f;
    for (int col = threadIdx.x; col < dim; col += blockDim.x) {
        float value = row[col];
        sum += value * value;
    }
    sum = block_sum(sum);
    float inv = rsqrtf(sum + 1.0e-6f);
    for (int col = threadIdx.x; col < dim; col += blockDim.x) {
        row[col] *= inv;
    }
}

__global__ void l2norm_kernel(float* values, int heads, int dim) {
    int head = blockIdx.x;
    if (head >= heads) {
        return;
    }
    float* row = values + static_cast<long long>(head) * dim;
    float sum = 0.0f;
    for (int col = threadIdx.x; col < dim; col += blockDim.x) {
        float value = row[col];
        sum += value * value;
    }
    sum = block_sum(sum);
    float inv = rsqrtf(sum + 1.0e-6f);
    for (int col = threadIdx.x; col < dim; col += blockDim.x) {
        row[col] *= inv;
    }
}

__global__ void repeat2_tokens_kernel(
    const float* source, float* destination, int heads, int dim, int source_stride, int dest_stride, int tokens) {
    int span = heads * 2;
    int id = blockIdx.x;
    int token = id / span;
    int out_head = id - token * span;
    if (token >= tokens) {
        return;
    }
    const float* row = source + static_cast<long long>(token) * source_stride + static_cast<long long>(out_head >> 1) * dim;
    float* dest = destination + static_cast<long long>(token) * dest_stride + static_cast<long long>(out_head) * dim;
    for (int col = threadIdx.x; col < dim; col += blockDim.x) {
        dest[col] = row[col];
    }
}

__global__ void repeat2_kernel(const float* source, float* destination, int heads, int dim) {
    int out_head = blockIdx.x;
    if (out_head >= heads * 2) {
        return;
    }
    const float* row = source + static_cast<long long>(out_head >> 1) * dim;
    float* dest = destination + static_cast<long long>(out_head) * dim;
    for (int col = threadIdx.x; col < dim; col += blockDim.x) {
        dest[col] = row[col];
    }
}

__global__ void conv_silu_kernel(
    __half* history, const __half* taps, const float* source, float* destination, int channels, int count) {
    int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= channels) {
        return;
    }
    const __half* weight = taps + static_cast<long long>(channel) * 4;
    float w0 = __half2float(weight[0]);
    float w1 = __half2float(weight[1]);
    float w2 = __half2float(weight[2]);
    float w3 = __half2float(weight[3]);
    float older = __half2float(history[channel]);
    float mid = __half2float(history[channels + channel]);
    float newer = __half2float(history[2 * channels + channel]);
    for (int token = 0; token < count; ++token) {
        float current = source[static_cast<long long>(token) * channels + channel];
        float acc = w0 * older + w1 * mid + w2 * newer + w3 * current;
        destination[static_cast<long long>(token) * channels + channel] = acc / (1.0f + expf(-acc));
        older = mid;
        mid = newer;
        newer = current;
    }
    history[channel] = __float2half(older);
    history[channels + channel] = __float2half(mid);
    history[2 * channels + channel] = __float2half(newer);
}

__device__ float warp_reduce_sum(float value) {
    value += __shfl_xor_sync(0xffffffff, value, 16);
    value += __shfl_xor_sync(0xffffffff, value, 8);
    value += __shfl_xor_sync(0xffffffff, value, 4);
    value += __shfl_xor_sync(0xffffffff, value, 2);
    value += __shfl_xor_sync(0xffffffff, value, 1);
    return value;
}

__device__ float dot4(float4 left, float4 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z + left.w * right.w;
}

// One warp owns one state row. The 128 columns are one coalesced vector, so a
// 512-token tile does not stride 512 bytes between neighboring threads.
__global__ void gdn_kernel(
    float* state,
    const float* query,
    const float* key,
    const float* value,
    const float* a_proj,
    const float* b_proj,
    const __half* a_log,
    const __half* dt_bias,
    const __half* norm,
    const float* gate,
    float* destination,
    float eps,
    int norm_span,
    int count,
    int value_stride) {
    int head = blockIdx.x;
    if (head >= 32) {
        return;
    }
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    __shared__ __align__(16) float ks[128];
    __shared__ __align__(16) float qs[128];
    __shared__ __align__(16) float vs[128];
    __shared__ float ys[128];
    float* head_state = state + (static_cast<long long>(head) * 128) * 128;
    for (int token = 0; token < count; ++token) {
        const float* qrow = query + static_cast<long long>(token) * 32 * 128;
        const float* krow = key + static_cast<long long>(token) * 32 * 128;
        const float* vrow = value + static_cast<long long>(token) * value_stride;
        ks[threadIdx.x] = krow[head * 128 + threadIdx.x];
        qs[threadIdx.x] = qrow[head * 128 + threadIdx.x] * rsqrtf(128.0f);
        vs[threadIdx.x] = vrow[head * 128 + threadIdx.x];
        __syncthreads();
        float aa = a_proj[token * 32 + head];
        float bb = b_proj[token * 32 + head];
        float shifted = aa + __half2float(dt_bias[head]);
        float soft = shifted > 20.0f ? shifted : log1pf(expf(shifted));
        float decay = expf(-expf(__half2float(a_log[head])) * soft);
        float beta = 1.0f / (1.0f + expf(-bb));
        for (int row0 = 0; row0 < 128; row0 += 4) {
            int row = row0 + warp;
            float4* line = reinterpret_cast<float4*>(head_state + row * 128);
            float4 chunk = line[lane];
            float4 key4 = reinterpret_cast<const float4*>(ks)[lane];
            chunk.x *= decay;
            chunk.y *= decay;
            chunk.z *= decay;
            chunk.w *= decay;
            float sk = warp_reduce_sum(dot4(chunk, key4));
            float delta = beta * (vs[row] - sk);
            chunk.x += delta * key4.x;
            chunk.y += delta * key4.y;
            chunk.z += delta * key4.z;
            chunk.w += delta * key4.w;
            line[lane] = chunk;
            float y = warp_reduce_sum(dot4(chunk, reinterpret_cast<const float4*>(qs)[lane]));
            if (lane == 0) {
                ys[row] = y;
            }
        }
        __syncthreads();
        int row = threadIdx.x;
        float sum = 0.0f;
        for (int col = 0; col < 128; ++col) {
            sum += ys[col] * ys[col];
        }
        float inv = rsqrtf(sum / 128.0f + eps);
        int weight_at = norm_span == 0 ? row : head * 128 + row;
        float gain = __half2float(norm[weight_at]);
        float z = gate[static_cast<long long>(token) * 32 * 128 + head * 128 + row];
        float silu = z / (1.0f + expf(-z));
        destination[static_cast<long long>(token) * 32 * 128 + head * 128 + row] = ys[row] * inv * gain * silu;
        __syncthreads();
    }
}

__global__ void split_q_kernel(const float* packed, float* query, float* gate, int heads) {
    int head = blockIdx.x;
    int col = threadIdx.x;
    if (head >= heads || col >= 256) {
        return;
    }
    const float* row = packed + static_cast<long long>(head) * 512;
    if (col < 256) {
        query[head * 256 + col] = row[col];
    }
}

__global__ void split_gate_kernel(const float* packed, float* gate, int heads) {
    int head = blockIdx.x;
    int col = threadIdx.x;
    if (head >= heads || col >= 256) {
        return;
    }
    gate[head * 256 + col] = packed[static_cast<long long>(head) * 512 + 256 + col];
}

__device__ void rotate_pair(float& left, float& right, float theta, int index, int position) {
    float freq = powf(theta, -static_cast<float>(index) / 32.0f);
    float angle = static_cast<float>(position) * freq;
    float cos = cosf(angle);
    float sin = sinf(angle);
    float a = left;
    float b = right;
    left = a * cos - b * sin;
    right = a * sin + b * cos;
}

__global__ void rope_kernel(
    float* query, int q_heads, float* key, int k_heads, int position, float theta, int count) {
    int token = blockIdx.y;
    int head = blockIdx.x;
    int index = threadIdx.x;
    if (token >= count || index >= 32) {
        return;
    }
    int origin = position < 0 ? step_at(2) : position;
    float* row = nullptr;
    if (head < q_heads) {
        row = query + (static_cast<long long>(token) * q_heads + head) * 256;
    } else if (head < q_heads + k_heads) {
        row = key + (static_cast<long long>(token) * k_heads + (head - q_heads)) * 256;
    } else {
        return;
    }
    rotate_pair(row[index], row[index + 32], theta, index, origin + token);
}

__device__ int e8m0_of(float scale) {
    if (!(scale > 0.0f)) {
        return 0;
    }
    int exponent = static_cast<int>(ceilf(log2f(scale)));
    exponent = max(-127, min(127, exponent));
    return exponent + 127;
}

__global__ void store_kv_kernel(
    int8_t* cache,
    const float* source,
    int tokens,
    int layer,
    int heads,
    int token,
    int full,
    int count) {
    int which = blockIdx.y;
    int head = blockIdx.x;
    int col = threadIdx.x;
    if (which >= count || head >= heads || col >= 256) {
        return;
    }
    int origin = token < 0 ? step_at(1) : token;
    long long index = (static_cast<long long>(origin + which) * full + layer) * heads + head;
    source += static_cast<long long>(which) * heads * 256;
    int8_t* record = cache + index * 264;
    __shared__ float row[256];
    row[col] = source[head * 256 + col];
    __syncthreads();
    if (col < 8) {
        float peak = 0.0f;
        for (int i = 0; i < 32; ++i) {
            peak = fmaxf(peak, fabsf(row[col * 32 + i]));
        }
        float scale = peak == 0.0f ? 0.0f : peak / 127.0f;
        reinterpret_cast<unsigned char*>(record)[256 + col] = static_cast<unsigned char>(e8m0_of(scale));
    }
    __syncthreads();
    int biased = static_cast<unsigned char>(record[256 + col / 32]);
    float unit = biased == 0 ? 0.0f : exp2f(static_cast<float>(biased - 127));
    float inv = unit == 0.0f ? 0.0f : 1.0f / unit;
    int code = static_cast<int>(rintf(row[col] * inv));
    record[col] = static_cast<int8_t>(max(-127, min(127, code)));
    (void)tokens;
}

__device__ float dequant_kv(const int8_t* record, int col) {
    int biased = static_cast<unsigned char>(record[256 + col / 32]);
    float unit = biased == 0 ? 0.0f : exp2f(static_cast<float>(biased - 127));
    return static_cast<float>(record[col]) * unit;
}

#include "tile.cu"

__global__ void attend_kernel(
    const int8_t* k_cache,
    const int8_t* v_cache,
    const float* query,
    float* destination,
    int count,
    int layer,
    int full,
    float scale) {
    int head = blockIdx.x;
    int col = threadIdx.x;
    if (head >= 16 || col >= 256) {
        return;
    }
    int kv = head >> 3;
    float q = query[head * 256 + col];
    float running_max = -1.0e30f;
    float running_sum = 0.0f;
    float acc = 0.0f;
    int seen = count < 0 ? step_at(1) + 1 : count;
    for (int token = 0; token < seen; ++token) {
        long long index = (static_cast<long long>(token) * full + layer) * 2 + kv;
        float part = q * dequant_kv(k_cache + index * 264, col);
        float score = block_sum(part) * scale;
        if (score > running_max) {
            float factor = expf(running_max - score);
            acc *= factor;
            running_sum *= factor;
            running_max = score;
        }
        float prob = expf(score - running_max);
        running_sum += prob;
        acc += prob * dequant_kv(v_cache + index * 264, col);
    }
    destination[head * 256 + col] = running_sum == 0.0f ? 0.0f : acc / running_sum;
}

// Each block owns one slice of the key span. The join mixes those partials.
__global__ void attend_split_kernel(
    const int8_t* k_cache,
    const int8_t* v_cache,
    const float* query,
    int count,
    int layer,
    int full,
    float scale,
    int splits,
    float* peaks,
    float* sums,
    float* acc) {
    int head = blockIdx.x;
    int split = blockIdx.y;
    int col = threadIdx.x;
    if (head >= kAttendHeads || col >= kAttendDim || split >= splits) {
        return;
    }
    int kv = head >> 3;
    int seen = count < 0 ? step_at(1) + 1 : count;
    int chunk = (seen + splits - 1) / splits;
    int pos0 = split * chunk;
    int pos1 = pos0 + chunk;
    if (pos1 > seen) {
        pos1 = seen;
    }
    float q = query[head * kAttendDim + col];
    float running_max = -1.0e30f;
    float running_sum = 0.0f;
    float mine = 0.0f;
    for (int token = pos0; token < pos1; ++token) {
        long long index = (static_cast<long long>(token) * full + layer) * 2 + kv;
        float part = q * dequant_kv(k_cache + index * 264, col);
        float score = block_sum(part) * scale;
        if (score > running_max) {
            float factor = expf(running_max - score);
            mine *= factor;
            running_sum *= factor;
            running_max = score;
        }
        float prob = expf(score - running_max);
        running_sum += prob;
        mine += prob * dequant_kv(v_cache + index * 264, col);
    }
    int part = split * kAttendHeads + head;
    if (col == 0) {
        peaks[part] = running_max;
        sums[part] = running_sum;
    }
    acc[static_cast<long long>(part) * kAttendDim + col] = mine;
}

__global__ void attend_join_kernel(
    float* destination, int splits, const float* peaks, const float* sums, const float* acc) {
    int head = blockIdx.x;
    int col = threadIdx.x;
    if (head >= kAttendHeads || col >= kAttendDim) {
        return;
    }
    float running_max = -1.0e30f;
    float running_sum = 0.0f;
    float mine = 0.0f;
    for (int split = 0; split < splits; ++split) {
        int part = split * kAttendHeads + head;
        float part_peak = peaks[part];
        float part_sum = sums[part];
        if (part_sum == 0.0f && part_peak < -1.0e20f) {
            continue;
        }
        float next = fmaxf(running_max, part_peak);
        float left = expf(running_max - next);
        float right = expf(part_peak - next);
        running_sum = running_sum * left + part_sum * right;
        mine = acc[static_cast<long long>(part) * kAttendDim + col] * right + mine * left;
        running_max = next;
    }
    destination[head * kAttendDim + col] = running_sum == 0.0f ? 0.0f : mine / running_sum;
}

__global__ void gate_mul_kernel(float* values, const float* gate, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < width) {
        float g = gate[col];
        values[col] *= 1.0f / (1.0f + expf(-g));
    }
}

__global__ void zero_kernel(float* destination, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < width) {
        destination[col] = 0.0f;
    }
}

__global__ void topk_tokens_kernel(const float* scores, int* ids, float* weights, int n, int k) {
    scores += static_cast<long long>(blockIdx.x) * n;
    ids += static_cast<long long>(blockIdx.x) * k;
    weights += static_cast<long long>(blockIdx.x) * k;
    __shared__ float vals[256];
    __shared__ int index[256];
    int lane = threadIdx.x;
    vals[lane] = lane < n ? scores[lane] : -1.0e30f;
    index[lane] = lane;
    __syncthreads();
    for (int pick = 0; pick < k; ++pick) {
        float best = -1.0e30f;
        int at = 0;
        for (int i = 0; i < n; ++i) {
            if (vals[i] > best) {
                best = vals[i];
                at = i;
            }
        }
        if (lane == 0) {
            ids[pick] = index[at];
            weights[pick] = best;
            vals[at] = -1.0e30f;
        }
        __syncthreads();
    }
    if (lane == 0) {
        float peak = weights[0];
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            peak = fmaxf(peak, weights[i]);
        }
        for (int i = 0; i < k; ++i) {
            weights[i] = expf(weights[i] - peak);
            sum += weights[i];
        }
        float inv = sum == 0.0f ? 0.0f : 1.0f / sum;
        for (int i = 0; i < k; ++i) {
            weights[i] *= inv;
        }
    }
}

__global__ void topk_kernel(const float* scores, int* ids, float* weights, int n, int k) {
    __shared__ float vals[256];
    __shared__ int index[256];
    int lane = threadIdx.x;
    vals[lane] = lane < n ? scores[lane] : -1.0e30f;
    index[lane] = lane;
    __syncthreads();
    for (int pick = 0; pick < k; ++pick) {
        float best = -1.0e30f;
        int at = 0;
        for (int i = 0; i < n; ++i) {
            if (vals[i] > best) {
                best = vals[i];
                at = i;
            }
        }
        if (lane == 0) {
            ids[pick] = index[at];
            weights[pick] = best;
            vals[at] = -1.0e30f;
        }
        __syncthreads();
    }
    if (lane == 0) {
        float peak = weights[0];
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            peak = fmaxf(peak, weights[i]);
        }
        for (int i = 0; i < k; ++i) {
            weights[i] = expf(weights[i] - peak);
            sum += weights[i];
        }
        float inv = sum == 0.0f ? 0.0f : 1.0f / sum;
        for (int i = 0; i < k; ++i) {
            weights[i] *= inv;
        }
    }
}

__global__ void penalize_kernel(
    float* logits, const int* counts, float presence, float frequency, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) {
        return;
    }
    int seen = counts[col];
    if (seen > 0) {
        logits[col] -= presence;
    }
    logits[col] -= frequency * static_cast<float>(seen);
}

__global__ void argmax_partial(const float* logits, float* values, int* index, int n) {
    __shared__ float vals[256];
    __shared__ int ids[256];
    int start = blockIdx.x * 256;
    int lane = threadIdx.x;
    int at = start + lane;
    vals[lane] = at < n ? logits[at] : -1.0e30f;
    ids[lane] = at;
    __syncthreads();
    for (int offset = 128; offset > 0; offset >>= 1) {
        if (lane < offset) {
            if (vals[lane + offset] > vals[lane]) {
                vals[lane] = vals[lane + offset];
                ids[lane] = ids[lane + offset];
            }
        }
        __syncthreads();
    }
    if (lane == 0) {
        values[blockIdx.x] = vals[0];
        index[blockIdx.x] = ids[0];
    }
}

__global__ void argmax_finish(const float* values, const int* index, int* token, int n) {
    float best = -1.0e30f;
    int at = 0;
    for (int i = 0; i < n; ++i) {
        if (values[i] > best) {
            best = values[i];
            at = index[i];
        }
    }
    token[0] = at;
}

__global__ void sample_draw(
    const float* logits,
    const int* counts,
    int* token,
    int n,
    float temperature,
    float top_p,
    int top_k,
    float presence,
    float frequency,
    unsigned seed) {
    // One block walks the vocabulary, keeps a short top list, and draws.
    constexpr int kCap = 64;
    __shared__ float best[kCap];
    __shared__ int ids[kCap];
    if (threadIdx.x == 0) {
        for (int i = 0; i < kCap; ++i) {
            best[i] = -1.0e30f;
            ids[i] = 0;
        }
        int keep = top_k > 0 && top_k < kCap ? top_k : kCap;
        for (int i = 0; i < n; ++i) {
            int seen = counts == nullptr ? 0 : counts[i];
            float score = logits[i];
            if (seen > 0) {
                score -= presence;
            }
            score -= frequency * static_cast<float>(seen);
            score /= temperature;
            float worst = best[keep - 1];
            if (score <= worst) {
                continue;
            }
            int slot = keep - 1;
            while (slot > 0 && score > best[slot - 1]) {
                best[slot] = best[slot - 1];
                ids[slot] = ids[slot - 1];
                slot -= 1;
            }
            best[slot] = score;
            ids[slot] = i;
        }
        float peak = best[0];
        float sum = 0.0f;
        int keep_n = keep;
        for (int i = 0; i < keep; ++i) {
            best[i] = expf(best[i] - peak);
            sum += best[i];
        }
        if (top_p > 0.0f && top_p < 1.0f && sum > 0.0f) {
            float ran = 0.0f;
            keep_n = 0;
            for (int i = 0; i < keep; ++i) {
                ran += best[i] / sum;
                keep_n = i + 1;
                if (ran >= top_p) {
                    break;
                }
            }
            sum = 0.0f;
            for (int i = 0; i < keep_n; ++i) {
                sum += best[i];
            }
        }
        float draw = static_cast<float>(seed % 1000003) / 1000003.0f * sum;
        float cursor = 0.0f;
        int chosen = ids[0];
        for (int i = 0; i < keep_n; ++i) {
            cursor += best[i];
            if (draw <= cursor) {
                chosen = ids[i];
                break;
            }
        }
        token[0] = chosen;
    }
}

__global__ void count_kernel(int* counts, const int* token) {
    if (threadIdx.x == 0) {
        counts[token[0]] += 1;
    }
}

__global__ void expand_i4_kernel(const uint8_t* packed, int8_t* expanded, int rows, int inner) {
    long long count = static_cast<long long>(rows) * inner;
    long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
        index += stride) {
        int nibble = packed[index >> 1];
        nibble = (index & 1) ? (nibble >> 4) : (nibble & 0xF);
        if (nibble & 8) {
            nibble -= 16;
        }
        expanded[index] = static_cast<int8_t>(nibble);
    }
}

__global__ void quant_i4_as_i8_kernel(
    const float* source, int8_t* codes, float* scales, int inner, int tokens) {
    int token = blockIdx.x;
    if (token >= tokens) {
        return;
    }
    int blocks = inner >> 5;
    const float* row = source + static_cast<long long>(token) * inner;
    for (int block = threadIdx.x; block < blocks; block += blockDim.x) {
        const float* slice = row + block * 32;
        float peak = 0.0f;
        for (int i = 0; i < 32; ++i) {
            peak = fmaxf(peak, fabsf(slice[i]));
        }
        float scale = peak == 0.0f ? 0.0f : peak / 7.0f;
        scales[static_cast<long long>(token) * blocks + block] = scale;
        float inv = scale == 0.0f ? 0.0f : 1.0f / scale;
        int8_t* out = codes + static_cast<long long>(token) * inner + block * 32;
        for (int i = 0; i < 32; ++i) {
            int code = static_cast<int>(rintf(slice[i] * inv));
            code = max(-7, min(7, code));
            out[i] = static_cast<int8_t>(code);
        }
    }
}

__global__ void gather_rows_kernel(
    const float* source, float* destination, const int* index, int rows, int width) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || col >= width) {
        return;
    }
    destination[static_cast<long long>(row) * width + col] =
        source[static_cast<long long>(index[row]) * width + col];
}

__global__ void silu_rows_kernel(
    const float* packed, float* destination, int width, int stride, int tokens) {
    int token = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= tokens || col >= width) {
        return;
    }
    const float* row = packed + static_cast<long long>(token) * stride;
    float gate = row[col];
    float value = row[width + col];
    destination[static_cast<long long>(token) * width + col] = gate / (1.0f + expf(-gate)) * value;
}

__global__ void scatter_scale_kernel(
    float* destination,
    const float* source,
    const int* token,
    const int* slot,
    const float* route,
    int rows,
    int width,
    int topk) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || col >= width) {
        return;
    }
    int at = token[row];
    float scale = route[static_cast<long long>(at) * topk + slot[row]];
    destination[static_cast<long long>(at) * width + col] +=
        source[static_cast<long long>(row) * width + col] * scale;
}

__global__ void sig_rows_kernel(
    float* destination, const float* source, const float* coeff, int tokens, int width) {
    int token = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= tokens || col >= width) {
        return;
    }
    float scale = 1.0f / (1.0f + expf(-coeff[token]));
    destination[static_cast<long long>(token) * width + col] +=
        source[static_cast<long long>(token) * width + col] * scale;
}

__device__ float row_dot(
    const uint8_t* row, const int8_t* codes, const float* scales, int inner, int bits) {
    int blocks = inner >> 5;
    float acc = 0.0f;
    for (int block = 0; block < blocks; ++block) {
        int sum = 0;
        if (bits == 8) {
            const int* weight = reinterpret_cast<const int*>(row + block * 32);
            const int* act = reinterpret_cast<const int*>(codes + block * 32);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                sum = __dp4a(weight[i], act[i], sum);
            }
        } else {
            const uint8_t* weight = row + block * 16;
            const int* act = reinterpret_cast<const int*>(codes + block * 32);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                int low = sext4(weight[i * 2]);
                int high = sext4(weight[i * 2] >> 4);
                int low2 = sext4(weight[i * 2 + 1]);
                int high2 = sext4(weight[i * 2 + 1] >> 4);
                int packed = (low & 0xFF) | ((high & 0xFF) << 8) | ((low2 & 0xFF) << 16) | ((high2 & 0xFF) << 24);
                sum = __dp4a(packed, act[i], sum);
            }
        }
        acc += static_cast<float>(sum) * scales[block];
    }
    return acc;
}

__device__ void quant_lane(
    const float* source, int8_t* codes, float* scales, int inner, int qmax, int lane, int threads) {
    int blocks = inner >> 5;
    for (int block = lane; block < blocks; block += threads) {
        const float* slice = source + block * 32;
        float peak = 0.0f;
        for (int i = 0; i < 32; ++i) {
            peak = fmaxf(peak, fabsf(slice[i]));
        }
        float scale = peak == 0.0f ? 0.0f : peak / static_cast<float>(qmax);
        scales[block] = scale;
        float inv = scale == 0.0f ? 0.0f : 1.0f / scale;
        int8_t* out = codes + block * 32;
        for (int i = 0; i < 32; ++i) {
            int code = static_cast<int>(rintf(slice[i] * inv));
            code = max(-qmax, min(qmax, code));
            out[i] = static_cast<int8_t>(code);
        }
    }
}

__global__ void quant_matrix_kernel(
    const float* source, int8_t* codes, float* scales, int inner, int rows, int qmax) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    int blocks = inner >> 5;
    const float* in = source + static_cast<long long>(row) * inner;
    for (int block = threadIdx.x; block < blocks; block += blockDim.x) {
        const float* slice = in + block * 32;
        float peak = 0.0f;
        for (int i = 0; i < 32; ++i) {
            peak = fmaxf(peak, fabsf(slice[i]));
        }
        float scale = peak == 0.0f ? 0.0f : peak / static_cast<float>(qmax);
        scales[static_cast<long long>(row) * blocks + block] = scale;
        float inv = scale == 0.0f ? 0.0f : 1.0f / scale;
        int8_t* out = codes + static_cast<long long>(row) * inner + block * 32;
        for (int i = 0; i < 32; ++i) {
            int code = static_cast<int>(rintf(slice[i] * inv));
            code = max(-qmax, min(qmax, code));
            out[i] = static_cast<int8_t>(code);
        }
    }
}

__device__ void warp_cached_product(
    const uint8_t* row,
    const int8_t* codes,
    const float* scales,
    const int* code_index,
    int index_base,
    int width,
    int inner,
    int bits,
    int lane,
    float wscale,
    float* dest,
    int dest_stride,
    float* moe,
    const int* tokens,
    const int* slots,
    const float* route,
    int hidden,
    int topk,
    int out_row) {
    constexpr int kKeep = 2;
    int held[kKeep][8];
    int blocks = inner >> 5;
#pragma unroll
    for (int slot = 0; slot < kKeep; ++slot) {
        int block = lane + slot * 32;
        if (block >= blocks) {
            continue;
        }
        if (bits == 8) {
            const int* weight = reinterpret_cast<const int*>(row + block * 32);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                held[slot][i] = weight[i];
            }
        } else {
            const uint8_t* weight = row + block * 16;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                int low = sext4(weight[i * 2]);
                int high = sext4(weight[i * 2] >> 4);
                int low2 = sext4(weight[i * 2 + 1]);
                int high2 = sext4(weight[i * 2 + 1] >> 4);
                held[slot][i] = (low & 0xFF) | ((high & 0xFF) << 8) | ((low2 & 0xFF) << 16) | ((high2 & 0xFF) << 24);
            }
        }
    }
    for (int item = 0; item < width; ++item) {
        int at = code_index != nullptr ? code_index[item] : index_base + item;
        float partial = 0.0f;
#pragma unroll
        for (int slot = 0; slot < kKeep; ++slot) {
            int block = lane + slot * 32;
            if (block >= blocks) {
                continue;
            }
            const int* act = reinterpret_cast<const int*>(codes + static_cast<long long>(at) * inner + block * 32);
            int sum = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                sum = __dp4a(held[slot][i], act[i], sum);
            }
            partial += static_cast<float>(sum) * scales[static_cast<long long>(at) * (inner >> 5) + block];
        }
        float acc = warp_sum(partial) * wscale;
        if (lane == 0) {
            if (dest != nullptr) {
                dest[static_cast<long long>(item) * dest_stride] = acc;
            } else {
                int token = tokens[item];
                float mix = route[static_cast<long long>(token) * topk + slots[item]];
                atomicAdd(moe + static_cast<long long>(token) * hidden + out_row, acc * mix);
            }
        }
    }
}

__global__ void tc_rows_kernel(
    const unsigned long long* device,
    const int* bits,
    const int* group_n,
    const int* group_off,
    const int* token_index,
    const int* slot_index,
    const int* plane_off,
    const int* scale_off,
    const int8_t* codes8,
    const float* scales8,
    const int8_t* codes4,
    const float* scales4,
    float* dest,
    const float* route,
    int inner,
    int rows,
    int dest_stride,
    int hidden,
    int topk,
    int selection_codes,
    int atomic_mix) {
    constexpr int kK = 32;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int row0 = (blockIdx.x * (blockDim.x >> 5) + warp) * 16;
    int expert = blockIdx.y;
    if (row0 >= rows) {
        return;
    }
    int group = lane >> 2;
    int inside = lane & 3;
    int row_lo = row0 + group;
    int row_hi = row_lo + 8;
    int bit = bits[expert];
    int width = group_n[expert];
    int base = group_off[expert];
    const auto* image = reinterpret_cast<const uint8_t*>(device[expert]);
    int poff = plane_off != nullptr ? plane_off[expert] : 0;
    const uint8_t* plane = image + poff;
    const float* wscale = reinterpret_cast<const float*>(image + scale_off[expert]);
    const int8_t* codes = selection_codes != 0 ? codes8 : (bit == 8 ? codes8 : codes4);
    const float* scales = selection_codes != 0 ? scales8 : (bit == 8 ? scales8 : scales4);
    int stride = bit == 8 ? inner : inner >> 1;
    int kblocks = inner >> 5;
    const int* tokens = token_index + base;
    const int* slots = slot_index != nullptr ? slot_index + base : nullptr;
    __shared__ __align__(16) signed char a_tile[4][16 * kK];
    __shared__ __align__(16) signed char b_tile[4][8 * kK];
    signed char* a_row = a_tile[warp];
    signed char* b_row = b_tile[warp];
    for (int n0 = 0; n0 < width; n0 += 8) {
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        float acc3 = 0.0f;
        for (int k0 = 0; k0 < inner; k0 += kK) {
            if (bit == 8) {
                int elem = lane * 16;
                int local = elem >> 5;
                int koff = elem & 31;
                const int8_t* src = reinterpret_cast<const int8_t*>(plane) +
                    (static_cast<long long>(row0 + local) * stride + k0 + koff);
                *reinterpret_cast<int4*>(a_row + local * kK + koff) = *reinterpret_cast<const int4*>(src);
            } else if (lane < 16) {
                const uint8_t* src = plane + static_cast<long long>(row0 + lane) * stride + (k0 >> 1);
                signed char* dst = a_row + lane * kK;
#pragma unroll
                for (int byte = 0; byte < 16; ++byte) {
                    int packed = src[byte];
                    dst[byte * 2] = static_cast<signed char>(sext4(packed));
                    dst[byte * 2 + 1] = static_cast<signed char>(sext4(packed >> 4));
                }
            }
            int local_tok = n0 + lane;
            if (lane < 8 && local_tok < width) {
                int at = selection_codes != 0 ? base + local_tok : tokens[local_tok];
                const int8_t* src = codes + static_cast<long long>(at) * inner + k0;
                *reinterpret_cast<int4*>(b_row + lane * kK) = *reinterpret_cast<const int4*>(src);
                *reinterpret_cast<int4*>(b_row + lane * kK + 16) = *reinterpret_cast<const int4*>(src + 16);
            } else if (lane < 8) {
                *reinterpret_cast<int4*>(b_row + lane * kK) = make_int4(0, 0, 0, 0);
                *reinterpret_cast<int4*>(b_row + lane * kK + 16) = make_int4(0, 0, 0, 0);
            }
            __syncwarp();
            const signed char* a_base = a_row + group * kK;
            unsigned a0 = *reinterpret_cast<const unsigned*>(a_base + inside * 4);
            unsigned a1 = *reinterpret_cast<const unsigned*>(a_base + 8 * kK + inside * 4);
            unsigned a2 = *reinterpret_cast<const unsigned*>(a_base + 16 + inside * 4);
            unsigned a3 = *reinterpret_cast<const unsigned*>(a_base + 8 * kK + 16 + inside * 4);
            const signed char* b_base = b_row + group * kK;
            unsigned b0 = *reinterpret_cast<const unsigned*>(b_base + inside * 4);
            unsigned b1 = *reinterpret_cast<const unsigned*>(b_base + 16 + inside * 4);
            int d0, d1, d2, d3;
            mma_s8(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
            int tok_lo = n0 + inside * 2;
            int tok_hi = tok_lo + 1;
            float scale_lo = 0.0f;
            float scale_hi = 0.0f;
            if (tok_lo < width) {
                int at = selection_codes != 0 ? base + tok_lo : tokens[tok_lo];
                scale_lo = scales[static_cast<long long>(at) * kblocks + (k0 >> 5)];
            }
            if (tok_hi < width) {
                int at = selection_codes != 0 ? base + tok_hi : tokens[tok_hi];
                scale_hi = scales[static_cast<long long>(at) * kblocks + (k0 >> 5)];
            }
            acc0 += static_cast<float>(d0) * scale_lo;
            acc1 += static_cast<float>(d1) * scale_hi;
            acc2 += static_cast<float>(d2) * scale_lo;
            acc3 += static_cast<float>(d3) * scale_hi;
            __syncwarp();
        }
        int tok_lo = n0 + inside * 2;
        int tok_hi = tok_lo + 1;
        float lo = row_lo < rows ? wscale[row_lo] : 0.0f;
        float hi = row_hi < rows ? wscale[row_hi] : 0.0f;
        if (atomic_mix == 0) {
            if (row_lo < rows && tok_lo < width) {
                dest[(static_cast<long long>(base) + tok_lo) * dest_stride + row_lo] = acc0 * lo;
            }
            if (row_lo < rows && tok_hi < width) {
                dest[(static_cast<long long>(base) + tok_hi) * dest_stride + row_lo] = acc1 * lo;
            }
            if (row_hi < rows && tok_lo < width) {
                dest[(static_cast<long long>(base) + tok_lo) * dest_stride + row_hi] = acc2 * hi;
            }
            if (row_hi < rows && tok_hi < width) {
                dest[(static_cast<long long>(base) + tok_hi) * dest_stride + row_hi] = acc3 * hi;
            }
        } else {
            if (row_lo < rows && tok_lo < width) {
                int token = tokens[tok_lo];
                float mix = route[static_cast<long long>(token) * topk + slots[tok_lo]];
                atomicAdd(dest + static_cast<long long>(token) * hidden + row_lo, acc0 * lo * mix);
            }
            if (row_lo < rows && tok_hi < width) {
                int token = tokens[tok_hi];
                float mix = route[static_cast<long long>(token) * topk + slots[tok_hi]];
                atomicAdd(dest + static_cast<long long>(token) * hidden + row_lo, acc1 * lo * mix);
            }
            if (row_hi < rows && tok_lo < width) {
                int token = tokens[tok_lo];
                float mix = route[static_cast<long long>(token) * topk + slots[tok_lo]];
                atomicAdd(dest + static_cast<long long>(token) * hidden + row_hi, acc2 * hi * mix);
            }
            if (row_hi < rows && tok_hi < width) {
                int token = tokens[tok_hi];
                float mix = route[static_cast<long long>(token) * topk + slots[tok_hi]];
                atomicAdd(dest + static_cast<long long>(token) * hidden + row_hi, acc3 * hi * mix);
            }
        }
    }
}

__global__ void gate_rows_kernel(
    const unsigned long long* device,
    const int* bits,
    const int* group_n,
    const int* group_off,
    const int* token_index,
    const int* gate_scale,
    const int8_t* codes8,
    const float* scales8,
    const int8_t* codes4,
    const float* scales4,
    float* gate_out,
    int hidden,
    int gate_rows) {
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    int expert = blockIdx.y;
    if (row >= gate_rows) {
        return;
    }
    int bit = bits[expert];
    int width = group_n[expert];
    int base = group_off[expert];
    const auto* image = reinterpret_cast<const uint8_t*>(device[expert]);
    int stride = bit == 8 ? hidden : hidden >> 1;
    const uint8_t* weight = image + static_cast<long long>(row) * stride;
    float wscale = reinterpret_cast<const float*>(image + gate_scale[expert])[row];
    const int8_t* codes = bit == 8 ? codes8 : codes4;
    const float* scales = bit == 8 ? scales8 : scales4;
    warp_cached_product(
        weight, codes, scales, token_index + base, 0, width, hidden, bit, lane, wscale,
        gate_out + static_cast<long long>(base) * gate_rows + row, gate_rows, nullptr, nullptr, nullptr, nullptr,
        0, 0, 0);
}

__global__ void silu_quant_kernel(
    const float* gate,
    int8_t* codes,
    float* scales,
    const int* bits,
    const int* group_n,
    int groups,
    int gate_rows,
    int inner,
    int sel_base) {
    int sel = blockIdx.x + sel_base;
    int block = threadIdx.x;
    int blocks = inner >> 5;
    if (block >= blocks) {
        return;
    }
    int cursor = 0;
    int qmax = 127;
    for (int expert = 0; expert < groups; ++expert) {
        int count = group_n[expert];
        if (sel < cursor + count) {
            qmax = bits[expert] == 4 ? 7 : 127;
            break;
        }
        cursor += count;
    }
    const float* row = gate + static_cast<long long>(sel) * gate_rows;
    int col = block * 32;
    float vals[32];
    float peak = 0.0f;
    for (int i = 0; i < 32; ++i) {
        float value = row[col + i];
        float up = row[inner + col + i];
        vals[i] = value / (1.0f + expf(-value)) * up;
        peak = fmaxf(peak, fabsf(vals[i]));
    }
    float scale = peak == 0.0f ? 0.0f : peak / static_cast<float>(qmax);
    scales[static_cast<long long>(sel) * blocks + block] = scale;
    float inv = scale == 0.0f ? 0.0f : 1.0f / scale;
    int8_t* out = codes + static_cast<long long>(sel) * inner + col;
    for (int i = 0; i < 32; ++i) {
        int code = static_cast<int>(rintf(vals[i] * inv));
        code = max(-qmax, min(qmax, code));
        out[i] = static_cast<int8_t>(code);
    }
}

__global__ void down_rows_kernel(
    const unsigned long long* device,
    const int* bits,
    const int* group_n,
    const int* group_off,
    const int* token_index,
    const int* slot_index,
    const int* down,
    const int* down_scale,
    const int8_t* codes,
    const float* scales,
    float* moe,
    const float* route,
    int down_inner,
    int down_rows,
    int hidden,
    int topk) {
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    int expert = blockIdx.y;
    if (row >= down_rows) {
        return;
    }
    int bit = bits[expert];
    int width = group_n[expert];
    int base = group_off[expert];
    const auto* image = reinterpret_cast<const uint8_t*>(device[expert]);
    int stride = bit == 8 ? down_inner : down_inner >> 1;
    const uint8_t* weight = image + down[expert] + static_cast<long long>(row) * stride;
    float wscale = reinterpret_cast<const float*>(image + down_scale[expert])[row];
    warp_cached_product(
        weight, codes, scales, nullptr, base, width, down_inner, bit, lane, wscale, nullptr, 0, moe, token_index + base,
        slot_index + base, route, hidden, topk, row);
}

__global__ void fused_expert_kernel(
    const unsigned long long* device,
    const int* bits,
    const int* group_n,
    const int* group_off,
    const int* token_index,
    const int* slot_index,
    const int* gate_scale,
    const int* down,
    const int* down_scale,
    const float* normed,
    float* moe,
    const float* route,
    int hidden,
    int gate_rows,
    int down_inner,
    int down_rows,
    int topk) {
    int expert = blockIdx.x;
    int lane = threadIdx.x;
    int width = group_n[expert];
    const auto* image = reinterpret_cast<const uint8_t*>(device[expert]);
    int bit = bits[expert];
    int qmax = bit == 4 ? 7 : 127;
    int gate_stride = bit == 8 ? hidden : hidden >> 1;
    int down_stride = bit == 8 ? down_inner : down_inner >> 1;
    const int* tokens = token_index + group_off[expert];
    const int* slots = slot_index + group_off[expert];
    const float* gate_wscale = reinterpret_cast<const float*>(image + gate_scale[expert]);
    const float* down_wscale = reinterpret_cast<const float*>(image + down_scale[expert]);
    const uint8_t* gate_w = image;
    const uint8_t* down_w = image + down[expert];
    __shared__ __align__(16) int8_t codes[2048];
    __shared__ float scales[64];
    __shared__ float gate_out[1024];
    __shared__ float activated[512];
    for (int item = 0; item < width; ++item) {
        int token = tokens[item];
        quant_lane(
            normed + static_cast<long long>(token) * hidden, codes, scales, hidden, qmax, lane, blockDim.x);
        __syncthreads();
        for (int row = lane; row < gate_rows; row += blockDim.x) {
            gate_out[row] = row_dot(gate_w + static_cast<long long>(row) * gate_stride, codes, scales, hidden, bit) *
                gate_wscale[row];
        }
        __syncthreads();
        for (int col = lane; col < down_inner; col += blockDim.x) {
            float gate = gate_out[col];
            activated[col] = gate / (1.0f + expf(-gate)) * gate_out[down_inner + col];
        }
        __syncthreads();
        quant_lane(activated, codes, scales, down_inner, qmax, lane, blockDim.x);
        __syncthreads();
        float mix = route[static_cast<long long>(token) * topk + slots[item]];
        for (int row = lane; row < down_rows; row += blockDim.x) {
            float value =
                row_dot(down_w + static_cast<long long>(row) * down_stride, codes, scales, down_inner, bit) *
                down_wscale[row] * mix;
            atomicAdd(moe + static_cast<long long>(token) * hidden + row, value);
        }
        __syncthreads();
    }
}

extern "C" {

API int engine_open() {
    int device = 0;
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }
    cudaDeviceProp prop{};
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }
    if (prop.major * 10 + prop.minor < 86) {
        return 86;
    }
    if (g_main != nullptr) {
        return 0;
    }
    int status = check(cudaStreamCreate(&g_main));
    if (status != 0) {
        return status;
    }
    status = check(cudaStreamCreate(&g_side));
    if (status != 0) {
        return status;
    }
    g_stream = g_main;
    g_splits = prop.multiProcessorCount * 2;
    if (g_splits > kAttendSplits) {
        g_splits = kAttendSplits;
    }
    if (g_splits < 1) {
        g_splits = 1;
    }
    for (int copy = 0; copy < kCopies; ++copy) {
        status = check(cudaEventCreateWithFlags(&g_copies[copy], cudaEventDisableTiming));
        if (status != 0) {
            return status;
        }
    }
    size_t parts = static_cast<size_t>(g_splits) * kAttendHeads;
    status = check(cudaMalloc(&g_attend_peak, parts * sizeof(float)));
    if (status != 0) {
        return status;
    }
    status = check(cudaMalloc(&g_attend_sum, parts * sizeof(float)));
    if (status != 0) {
        return status;
    }
    status = check(cudaMalloc(&g_attend_acc, parts * kAttendDim * sizeof(float)));
    if (status != 0) {
        return status;
    }
    for (int fork = 0; fork < kForks; ++fork) {
        status = check(cudaEventCreateWithFlags(&g_forks[fork], cudaEventDisableTiming));
        if (status != 0) {
            return status;
        }
        status = check(cudaEventCreateWithFlags(&g_joins[fork], cudaEventDisableTiming));
        if (status != 0) {
            return status;
        }
    }
    status = check(cudaMalloc(&g_step, sizeof(int) * 3));
    if (status != 0) {
        return status;
    }
    status = check(cudaMemset(g_step, 0, sizeof(int) * 3));
    if (status != 0) {
        return status;
    }
    status = check(cudaMemcpyToSymbol(g_step_ptr, &g_step, sizeof(g_step)));
    if (status != 0) {
        return status;
    }
    status = check(cudaMalloc(&g_zero, 16));
    if (status != 0) {
        return status;
    }
    status = check(cudaMemset(g_zero, 0, 16));
    if (status != 0) {
        return status;
    }
    return check(cudaMemcpyToSymbol(g_zero_ptr, &g_zero, sizeof(g_zero)));
}

API void engine_close() {
    for (int slot = 0; slot < kGraphs; ++slot) {
        if (g_graph[slot] != nullptr) {
            cudaGraphExecDestroy(g_graph[slot]);
            g_graph[slot] = nullptr;
        }
    }
}

API void* engine_alloc(size_t nbytes) {
    void* pointer = nullptr;
    if (cudaMalloc(&pointer, nbytes) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    return pointer;
}

API void engine_free(void* pointer) {
    if (pointer != nullptr) {
        cudaFree(pointer);
    }
}

API void* engine_pin(size_t nbytes) {
    void* pointer = nullptr;
    if (cudaHostAlloc(&pointer, nbytes, cudaHostAllocDefault) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    return pointer;
}

API void* engine_host_alloc(size_t nbytes) {
    void* pointer = nullptr;
    if (cudaHostAlloc(&pointer, nbytes == 0 ? 1 : nbytes, cudaHostAllocWriteCombined) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    return pointer;
}

API void engine_host_free(void* pointer) {
    if (pointer != nullptr) {
        cudaFreeHost(pointer);
    }
}

API void engine_unpin(void* pointer) {
    if (pointer != nullptr) {
        cudaFreeHost(pointer);
    }
}

API int engine_register(void* pointer, size_t nbytes) {
    if (cudaHostRegister(pointer, nbytes, cudaHostRegisterDefault) != cudaSuccess) {
        cudaGetLastError();
        return 1;
    }
    return 0;
}

API int engine_register_ro(void* pointer, size_t nbytes) {
    if (cudaHostRegister(pointer, nbytes, cudaHostRegisterReadOnly) != cudaSuccess) {
        cudaGetLastError();
        return 1;
    }
    return 0;
}

API void engine_unregister(void* pointer) {
    if (pointer != nullptr) {
        cudaHostUnregister(pointer);
    }
}

API int engine_h2d(void* destination, const void* source, size_t nbytes) {
    return check(cudaMemcpyAsync(destination, source, nbytes, cudaMemcpyHostToDevice, g_stream));
}

API int engine_copy_async(void* destination, const void* source, size_t nbytes, int* event_id) {
    if (destination == nullptr || source == nullptr || nbytes == 0 || event_id == nullptr || g_side == nullptr) {
        return 1;
    }
    if (g_copy_next >= kCopies) {
        int status = check(cudaStreamSynchronize(g_stream));
        if (status != 0) {
            return status;
        }
        g_copy_next = 0;
    }
    int slot = g_copy_next++;
    int status = check(cudaMemcpyAsync(destination, source, nbytes, cudaMemcpyHostToDevice, g_side));
    if (status != 0) {
        return status;
    }
    status = check(cudaEventRecord(g_copies[slot], g_side));
    if (status != 0) {
        return status;
    }
    *event_id = slot;
    return 0;
}

API int engine_wait_copy(int event_id) {
    if (event_id < 0 || event_id >= kCopies || g_copies[event_id] == nullptr) {
        return 1;
    }
    return check(cudaStreamWaitEvent(g_stream, g_copies[event_id], 0));
}

API int engine_d2h(void* destination, const void* source, size_t nbytes) {
    int status = check(cudaMemcpyAsync(destination, source, nbytes, cudaMemcpyDeviceToHost, g_stream));
    if (status != 0) {
        return status;
    }
    return check(cudaStreamSynchronize(g_stream));
}

API int engine_zero(void* destination, size_t nbytes) {
    return check(cudaMemsetAsync(destination, 0, nbytes, g_stream));
}

API int engine_sync() { return check(cudaStreamSynchronize(g_stream)); }

API int engine_mem_info(size_t* free_bytes, size_t* total_bytes) {
    return check(cudaMemGetInfo(free_bytes, total_bytes));
}

API int engine_quant(const float* source, void* codes, float* scales, int inner, int bits, int tokens) {
    quant_kernel<<<tokens, 128, 0, g_stream>>>(source, codes, scales, inner, bits, tokens);
    return check(cudaGetLastError());
}

API int engine_gemv(
    const void* weight,
    const float* wscale,
    const void* codes,
    const float* ascales,
    float* destination,
    int rows,
    int inner,
    int bits,
    int tokens) {
    dim3 grid(rows, tokens);
    if (tokens == 1 && inner <= 8192 && (inner & 31) == 0) {
        int threads = 256;
        int warps = threads / 32;
        int blocks = (rows + warps - 1) / warps;
        if (bits == 8) {
            size_t smem = static_cast<size_t>(inner) + static_cast<size_t>(inner / 32) * sizeof(float);
            gemv_i8_cached<<<blocks, threads, smem, g_stream>>>(
                static_cast<const int8_t*>(weight), wscale, static_cast<const int8_t*>(codes), ascales, destination,
                rows, inner);
        } else {
            size_t smem = static_cast<size_t>(inner / 2) + static_cast<size_t>(inner / 32) * sizeof(float);
            gemv_i4_cached<<<blocks, threads, smem, g_stream>>>(
                weight, wscale, codes, ascales, destination, rows, inner);
        }
        return check(cudaGetLastError());
    }
    if (bits == 8) {
        gemv_i8_warp<<<grid, 32, 0, g_stream>>>(
            static_cast<const int8_t*>(weight),
            wscale,
            static_cast<const int8_t*>(codes),
            ascales,
            destination,
            rows,
            inner);
    } else {
        gemv_kernel<<<grid, 128, 0, g_stream>>>(weight, wscale, codes, ascales, destination, rows, inner, bits, tokens);
    }
    return check(cudaGetLastError());
}

API int engine_gemv_f16(const void* weight, const float* source, float* destination, int rows, int inner) {
    gemv_f16_kernel<<<rows, 128, 0, g_stream>>>(static_cast<const __half*>(weight), source, destination, rows, inner);
    return check(cudaGetLastError());
}

API int engine_gemv_f16_tokens(
    const void* weight, const float* source, float* destination, int rows, int inner, int tokens) {
    if (tokens < 1 || rows < 1) {
        return 0;
    }
    dim3 grid(rows, tokens);
    gemv_f16_rows_kernel<<<grid, 128, 0, g_stream>>>(
        static_cast<const __half*>(weight), source, destination, rows, inner);
    return check(cudaGetLastError());
}

API int engine_topk_tokens(const float* scores, int* ids, float* weights, int n, int k, int tokens) {
    if (tokens < 1 || n < 1 || n > 256 || k < 1 || k > 8) {
        return 1;
    }
    topk_tokens_kernel<<<tokens, 256, 0, g_stream>>>(scores, ids, weights, n, k);
    return check(cudaGetLastError());
}

API int engine_embed(const void* table, const float* scale, int token, float* destination, int inner) {
    embed_kernel<<<1, 256, 0, g_stream>>>(static_cast<const int8_t*>(table), scale, token, destination, inner);
    return check(cudaGetLastError());
}

API int engine_rmsnorm(
    const float* source, const void* weight, float* destination, int width, float eps, int heads) {
    rmsnorm_kernel<<<heads, 128, 0, g_stream>>>(
        source, static_cast<const __half*>(weight), destination, width, eps, heads);
    return check(cudaGetLastError());
}

API int engine_add(float* stream, const float* delta, int width) {
    int threads = 256;
    int blocks = (width + threads - 1) / threads;
    add_kernel<<<blocks, threads, 0, g_stream>>>(stream, delta, width);
    return check(cudaGetLastError());
}

API int engine_silu_mul(const float* gate, const float* up, float* destination, int width) {
    int threads = 256;
    int blocks = (width + threads - 1) / threads;
    silu_mul_kernel<<<blocks, threads, 0, g_stream>>>(gate, up, destination, width);
    return check(cudaGetLastError());
}

API int engine_scale_add(float* destination, const float* source, float scale, int width) {
    int threads = 256;
    int blocks = (width + threads - 1) / threads;
    scale_add_kernel<<<blocks, threads, 0, g_stream>>>(destination, source, scale, width);
    return check(cudaGetLastError());
}

API int engine_scale_add_at(
    float* destination, const float* source, const float* scale, int width) {
    int threads = 256;
    int blocks = (width + threads - 1) / threads;
    scale_add_at_kernel<<<blocks, threads, 0, g_stream>>>(destination, source, scale, width);
    return check(cudaGetLastError());
}

API int engine_sig_add(float* destination, const float* source, const float* coeff, int width) {
    int threads = 256;
    int blocks = (width + threads - 1) / threads;
    sig_add_kernel<<<blocks, threads, 0, g_stream>>>(destination, source, coeff, width);
    return check(cudaGetLastError());
}

API int engine_l2norm(float* values, int heads, int dim) {
    l2norm_kernel<<<heads, 64, 0, g_stream>>>(values, heads, dim);
    return check(cudaGetLastError());
}

API int engine_l2norm_tokens(float* values, int heads, int dim, int stride, int tokens) {
    if (tokens < 1 || heads < 1) {
        return 0;
    }
    l2norm_tokens_kernel<<<heads * tokens, 64, 0, g_stream>>>(values, heads, dim, stride, tokens);
    return check(cudaGetLastError());
}

API int engine_repeat2(const float* source, float* destination, int heads, int dim) {
    repeat2_kernel<<<heads * 2, 128, 0, g_stream>>>(source, destination, heads, dim);
    return check(cudaGetLastError());
}

API int engine_repeat2_tokens(
    const float* source, float* destination, int heads, int dim, int source_stride, int dest_stride, int tokens) {
    if (tokens < 1 || heads < 1) {
        return 0;
    }
    repeat2_tokens_kernel<<<heads * 2 * tokens, 128, 0, g_stream>>>(
        source, destination, heads, dim, source_stride, dest_stride, tokens);
    return check(cudaGetLastError());
}

API int engine_conv_silu(
    void* history, const void* taps, const float* source, float* destination, int channels, int count) {
    int threads = 256;
    int blocks = (channels + threads - 1) / threads;
    conv_silu_kernel<<<blocks, threads, 0, g_stream>>>(
        static_cast<__half*>(history),
        static_cast<const __half*>(taps),
        source,
        destination,
        channels,
        count);
    return check(cudaGetLastError());
}

API int engine_gdn(
    float* state,
    const float* query,
    const float* key,
    const float* value,
    const float* a_proj,
    const float* b_proj,
    const void* a_log,
    const void* dt_bias,
    const void* norm,
    const float* gate,
    float* destination,
    float eps,
    int norm_span,
    int count,
    int value_stride) {
    gdn_kernel<<<32, 128, 0, g_stream>>>(
        state,
        query,
        key,
        value,
        a_proj,
        b_proj,
        static_cast<const __half*>(a_log),
        static_cast<const __half*>(dt_bias),
        static_cast<const __half*>(norm),
        gate,
        destination,
        eps,
        norm_span,
        count,
        value_stride);
    return check(cudaGetLastError());
}

API int engine_split_q(const float* packed, float* query, float* gate, int heads) {
    split_q_kernel<<<heads, 256, 0, g_stream>>>(packed, query, gate, heads);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }
    split_gate_kernel<<<heads, 256, 0, g_stream>>>(packed, gate, heads);
    return check(cudaGetLastError());
}

API int engine_rope(float* query, int q_heads, float* key, int k_heads, int position, float theta, int count) {
    dim3 grid(q_heads + k_heads, count);
    rope_kernel<<<grid, 32, 0, g_stream>>>(query, q_heads, key, k_heads, position, theta, count);
    return check(cudaGetLastError());
}

API int engine_store_kv(
    void* cache, const float* source, int tokens, int layer, int heads, int token, int full, int count) {
    dim3 grid(heads, count);
    store_kv_kernel<<<grid, 256, 0, g_stream>>>(
        static_cast<int8_t*>(cache), source, tokens, layer, heads, token, full, count);
    return check(cudaGetLastError());
}

API int engine_attend(
    const void* k_cache,
    const void* v_cache,
    const float* query,
    float* destination,
    int count,
    int layer,
    int full,
    float scale) {
    if (g_attend_peak != nullptr && g_splits > 1) {
        dim3 grid(kAttendHeads, g_splits);
        attend_split_kernel<<<grid, kAttendDim, 0, g_stream>>>(
            static_cast<const int8_t*>(k_cache), static_cast<const int8_t*>(v_cache), query, count, layer, full, scale,
            g_splits, g_attend_peak, g_attend_sum, g_attend_acc);
        attend_join_kernel<<<kAttendHeads, kAttendDim, 0, g_stream>>>(
            destination, g_splits, g_attend_peak, g_attend_sum, g_attend_acc);
        return check(cudaGetLastError());
    }
    attend_kernel<<<16, 256, 0, g_stream>>>(
        static_cast<const int8_t*>(k_cache),
        static_cast<const int8_t*>(v_cache),
        query,
        destination,
        count,
        layer,
        full,
        scale);
    return check(cudaGetLastError());
}

API int engine_gate_mul(float* values, const float* gate, int width) {
    int threads = 256;
    int blocks = (width + threads - 1) / threads;
    gate_mul_kernel<<<blocks, threads, 0, g_stream>>>(values, gate, width);
    return check(cudaGetLastError());
}

API int engine_zero_f32(float* destination, int width) {
    int threads = 256;
    int blocks = (width + threads - 1) / threads;
    zero_kernel<<<blocks, threads, 0, g_stream>>>(destination, width);
    return check(cudaGetLastError());
}

API int engine_topk(const float* scores, int* ids, float* weights, int n, int k) {
    topk_kernel<<<1, 256, 0, g_stream>>>(scores, ids, weights, n, k);
    return check(cudaGetLastError());
}

API int engine_argmax(const float* logits, float* partial_v, int* partial_i, int* token, int n) {
    int blocks = (n + 255) / 256;
    argmax_partial<<<blocks, 256, 0, g_stream>>>(logits, partial_v, partial_i, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }
    argmax_finish<<<1, 1, 0, g_stream>>>(partial_v, partial_i, token, blocks);
    return check(cudaGetLastError());
}

API int engine_sample(
    const float* logits,
    const int* counts,
    int* token,
    int n,
    float temperature,
    float top_p,
    int top_k,
    float presence,
    float frequency,
    unsigned seed) {
    sample_draw<<<1, 1, 0, g_stream>>>(logits, counts, token, n, temperature, top_p, top_k, presence, frequency, seed);
    return check(cudaGetLastError());
}

API int engine_count(int* counts, const int* token) {
    count_kernel<<<1, 1, 0, g_stream>>>(counts, token);
    return check(cudaGetLastError());
}

API int engine_penalize(float* logits, const int* counts, float presence, float frequency, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    penalize_kernel<<<blocks, threads, 0, g_stream>>>(logits, counts, presence, frequency, n);
    return check(cudaGetLastError());
}

static int gemm_launch(
    const void* weight, const float* scale, float* destination, int rows, int inner, int tokens, cudaStream_t stream) {
    dim3 tile((tokens + 63) / 64, (rows + 63) / 64);
    gemm_i8<<<tile, 128, 0, stream>>>(
        static_cast<const int8_t*>(weight), scale, g_codes, g_scales, destination, rows, inner, tokens);
    return check(cudaGetLastError());
}

static int quantize_activation(const float* source, int inner, int tokens) {
    int status = scratch_for(tokens, inner);
    if (status != 0) {
        return status;
    }
    int blocks = tokens * (inner >> 5);
    int grid = (blocks + 127) / 128;
    quant_batch<<<grid, 128, 0, g_stream>>>(source, g_codes, g_scales, inner, tokens);
    return check(cudaGetLastError());
}

API int engine_reserve(int tokens, int inner) {
    if (tokens < 1 || inner <= 0) {
        return 1;
    }
    return scratch_for(tokens, inner);
}

API int engine_gemm(
    const void* weight, const float* scale, const float* source, float* destination, int rows, int inner, int tokens) {
    if (tokens < 1 || inner <= 0 || (inner & 31) != 0) {
        return 1;
    }
    int status = quantize_activation(source, inner, tokens);
    if (status != 0) {
        return status;
    }
    return gemm_launch(weight, scale, destination, rows, inner, tokens, g_stream);
}

API int engine_gemm_i4(
    const void* weight, const float* scale, const float* source, float* destination, int rows, int inner,
    int tokens) {
    if (tokens < 1 || rows < 1 || inner <= 0 || (inner & 31) != 0) {
        return 1;
    }
    int status = scratch_for(tokens, inner);
    if (status != 0) {
        return status;
    }
    status = expand_for(rows * inner);
    if (status != 0) {
        return status;
    }
    int threads = 256;
    int blocks = (rows * inner + threads - 1) / threads;
    if (blocks > 2048) {
        blocks = 2048;
    }
    expand_i4_kernel<<<blocks, threads, 0, g_stream>>>(
        static_cast<const uint8_t*>(weight), g_expand, rows, inner);
    status = check(cudaGetLastError());
    if (status != 0) {
        return status;
    }
    quant_i4_as_i8_kernel<<<tokens, 128, 0, g_stream>>>(source, g_codes, g_scales, inner, tokens);
    status = check(cudaGetLastError());
    if (status != 0) {
        return status;
    }
    return gemm_launch(g_expand, scale, destination, rows, inner, tokens, g_stream);
}

API int engine_gather_rows(const float* source, float* destination, const int* index, int rows, int width) {
    if (rows < 1 || width < 1) {
        return 0;
    }
    int threads = 256;
    dim3 grid((width + threads - 1) / threads, rows);
    gather_rows_kernel<<<grid, threads, 0, g_stream>>>(source, destination, index, rows, width);
    return check(cudaGetLastError());
}

API int engine_silu_rows(const float* packed, float* destination, int width, int stride, int tokens) {
    if (tokens < 1 || width < 1) {
        return 0;
    }
    int threads = 256;
    dim3 grid((width + threads - 1) / threads, tokens);
    silu_rows_kernel<<<grid, threads, 0, g_stream>>>(packed, destination, width, stride, tokens);
    return check(cudaGetLastError());
}

API int engine_scatter_scale(
    float* destination,
    const float* source,
    const int* token,
    const int* slot,
    const float* route,
    int rows,
    int width,
    int topk) {
    if (rows < 1 || width < 1) {
        return 0;
    }
    int threads = 256;
    dim3 grid((width + threads - 1) / threads, rows);
    scatter_scale_kernel<<<grid, threads, 0, g_stream>>>(
        destination, source, token, slot, route, rows, width, topk);
    return check(cudaGetLastError());
}

API int engine_sig_rows(float* destination, const float* source, const float* coeff, int tokens, int width) {
    if (tokens < 1 || width < 1) {
        return 0;
    }
    int threads = 256;
    dim3 grid((width + threads - 1) / threads, tokens);
    sig_rows_kernel<<<grid, threads, 0, g_stream>>>(destination, source, coeff, tokens, width);
    return check(cudaGetLastError());
}

static unsigned long long* g_fused_ptr = nullptr;
static int* g_fused_meta = nullptr;
static int* g_fused_tok = nullptr;
static int* g_fused_slot = nullptr;
static int8_t* g_codes8 = nullptr;
static float* g_scales8 = nullptr;
static int8_t* g_codes4 = nullptr;
static float* g_scales4 = nullptr;
static float* g_gate = nullptr;
static int8_t* g_act = nullptr;
static float* g_act_scale = nullptr;

static int fused_alloc(void** pointer, size_t bytes) {
    return check(cudaMalloc(pointer, bytes));
}

static int fused_ready() {
    if (g_fused_ptr != nullptr) {
        return 0;
    }
    int status = check(cudaMalloc(&g_fused_ptr, sizeof(unsigned long long) * 256));
    if (status != 0) {
        return status;
    }
    status = check(cudaMalloc(&g_fused_meta, sizeof(int) * 256 * 6));
    if (status != 0) {
        return status;
    }
    status = check(cudaMalloc(&g_fused_tok, sizeof(int) * 8192));
    if (status != 0) {
        return status;
    }
    status = check(cudaMalloc(&g_fused_slot, sizeof(int) * 8192));
    if (status != 0) {
        return status;
    }
    status = fused_alloc(reinterpret_cast<void**>(&g_codes8), static_cast<size_t>(512) * 2048);
    if (status != 0) {
        return status;
    }
    status = fused_alloc(reinterpret_cast<void**>(&g_scales8), static_cast<size_t>(512) * 64 * sizeof(float));
    if (status != 0) {
        return status;
    }
    status = fused_alloc(reinterpret_cast<void**>(&g_codes4), static_cast<size_t>(512) * 2048);
    if (status != 0) {
        return status;
    }
    status = fused_alloc(reinterpret_cast<void**>(&g_scales4), static_cast<size_t>(512) * 64 * sizeof(float));
    if (status != 0) {
        return status;
    }
    status = fused_alloc(reinterpret_cast<void**>(&g_gate), static_cast<size_t>(4096) * 1024 * sizeof(float));
    if (status != 0) {
        return status;
    }
    status = fused_alloc(reinterpret_cast<void**>(&g_act), static_cast<size_t>(4096) * 512);
    if (status != 0) {
        return status;
    }
    return fused_alloc(reinterpret_cast<void**>(&g_act_scale), static_cast<size_t>(4096) * 16 * sizeof(float));
}

static cudaEvent_t g_copied[256] = {};
static cudaEvent_t g_consumed[256] = {};
static int g_pipe_n = 0;

static int pipe_events(int count) {
    while (g_pipe_n < count) {
        int status = check(cudaEventCreateWithFlags(&g_copied[g_pipe_n], cudaEventDisableTiming));
        if (status != 0) {
            return status;
        }
        status = check(cudaEventCreateWithFlags(&g_consumed[g_pipe_n], cudaEventDisableTiming));
        if (status != 0) {
            return status;
        }
        g_pipe_n += 1;
    }
    return 0;
}

static int copy_expert(
    unsigned long long destination, unsigned long long source, int bytes, cudaStream_t stream) {
    if (bytes < 1) {
        return 0;
    }
    return check(cudaMemcpyAsync(
        reinterpret_cast<void*>(destination), reinterpret_cast<const void*>(source), static_cast<size_t>(bytes),
        cudaMemcpyHostToDevice, stream));
}

static int slot_owner(const unsigned long long* device, int next, int live) {
    unsigned long long slot = device[next];
    int owner = -1;
    if (slot == 0) {
        return owner;
    }
    for (int group = 0; group < live; ++group) {
        if (device[group] == slot) {
            owner = group;
        }
    }
    return owner;
}

static int copy_span(
    const unsigned long long* device,
    const unsigned long long* host_src,
    const int* copy_bytes,
    int begin,
    int end,
    cudaStream_t stream) {
    for (int group = begin; group < end; ++group) {
        int status = copy_expert(device[group], host_src[group], copy_bytes[group], stream);
        if (status != 0) {
            return status;
        }
    }
    return 0;
}

static bool span_alias(const unsigned long long* device, int live, int begin, int end) {
    for (int group = begin; group < end; ++group) {
        if (slot_owner(device, group, live) >= 0) {
            return true;
        }
    }
    return false;
}

static void launch_span(
    int begin,
    int count,
    int hidden,
    int gate_rows,
    int down_inner,
    int down_rows,
    int topk,
    int sel0,
    int seln,
    int groups,
    float* moe,
    const float* route) {
    dim3 gate_grid((gate_rows + 3) / 4, count);
    gate_rows_kernel<<<gate_grid, 128, 0, g_stream>>>(
        g_fused_ptr + begin, g_fused_meta + begin, g_fused_meta + 256 + begin, g_fused_meta + 512 + begin,
        g_fused_tok, g_fused_meta + 768 + begin, g_codes8, g_scales8, g_codes4, g_scales4, g_gate, hidden,
        gate_rows);
    if (seln < 1) {
        return;
    }
    silu_quant_kernel<<<seln, 32, 0, g_stream>>>(
        g_gate, g_act, g_act_scale, g_fused_meta, g_fused_meta + 256, groups, gate_rows, down_inner, sel0);
    dim3 down_grid((down_rows + 3) / 4, count);
    down_rows_kernel<<<down_grid, 128, 0, g_stream>>>(
        g_fused_ptr + begin, g_fused_meta + begin, g_fused_meta + 256 + begin, g_fused_meta + 512 + begin,
        g_fused_tok, g_fused_slot, g_fused_meta + 1024 + begin, g_fused_meta + 1280 + begin, g_act,
        g_act_scale, moe, route, down_inner, down_rows, hidden, topk);
}

API int engine_grouped_mlp(
    const unsigned long long* device,
    const unsigned long long* host_src,
    const int* copy_bytes,
    const int* bits,
    const int* token_index,
    const int* slot_index,
    const int* group_n,
    const int* gate_scale,
    const int* down,
    const int* down_scale,
    int groups,
    float* normed,
    float* gather,
    float* up,
    float* silu,
    float* proj,
    float* moe,
    const float* route,
    int* token_buf,
    int* slot_buf,
    int hidden,
    int gate_rows,
    int down_inner,
    int down_rows,
    int topk) {
    (void)gather;
    (void)up;
    (void)silu;
    (void)proj;
    (void)token_buf;
    (void)slot_buf;
    if (groups < 1) {
        return 0;
    }
    if (groups > 256 || hidden > 2048 || gate_rows > 1024 || down_inner > 512 || down_rows > 2048) {
        return 1;
    }
    int status = fused_ready();
    if (status != 0) {
        return status;
    }
    static int off[256];
    int cursor = 0;
    for (int group = 0; group < groups; ++group) {
        off[group] = cursor;
        cursor += group_n[group];
    }
    if (cursor > 4096) {
        return 1;
    }
    int ntok = 0;
    for (int i = 0; i < cursor; ++i) {
        int token = token_index[i] + 1;
        if (token > ntok) {
            ntok = token;
        }
    }
    if (ntok > 512) {
        return 1;
    }
    status = pipe_events(groups);
    if (status != 0) {
        return status;
    }
    cudaStream_t copy_stream = g_side != nullptr ? g_side : g_stream;
    constexpr int kWave = 64;
    int first = groups < kWave ? groups : kWave;
    status = copy_span(device, host_src, copy_bytes, 0, first, copy_stream);
    if (status != 0) {
        return status;
    }
    status = check(cudaEventRecord(g_copied[0], copy_stream));
    if (status != 0) {
        return status;
    }
    status = check(cudaMemcpyAsync(
        g_fused_ptr, device, sizeof(unsigned long long) * static_cast<size_t>(groups), cudaMemcpyHostToDevice,
        g_stream));
    if (status != 0) {
        return status;
    }
    int* meta[6] = {const_cast<int*>(bits), const_cast<int*>(group_n), off, const_cast<int*>(gate_scale),
        const_cast<int*>(down), const_cast<int*>(down_scale)};
    for (int part = 0; part < 6; ++part) {
        status = check(cudaMemcpyAsync(
            g_fused_meta + part * 256, meta[part], sizeof(int) * static_cast<size_t>(groups), cudaMemcpyHostToDevice,
            g_stream));
        if (status != 0) {
            return status;
        }
    }
    if (cursor > 0) {
        status = check(cudaMemcpyAsync(
            g_fused_tok, token_index, sizeof(int) * static_cast<size_t>(cursor), cudaMemcpyHostToDevice, g_stream));
        if (status != 0) {
            return status;
        }
        status = check(cudaMemcpyAsync(
            g_fused_slot, slot_index, sizeof(int) * static_cast<size_t>(cursor), cudaMemcpyHostToDevice, g_stream));
        if (status != 0) {
            return status;
        }
    }
    if (ntok > 0) {
        quant_matrix_kernel<<<ntok, 64, 0, g_stream>>>(normed, g_codes8, g_scales8, hidden, ntok, 127);
        quant_matrix_kernel<<<ntok, 64, 0, g_stream>>>(normed, g_codes4, g_scales4, hidden, ntok, 7);
    }
    int wave = 0;
    for (int begin = 0; begin < groups; begin += kWave, ++wave) {
        int end = begin + kWave;
        if (end > groups) {
            end = groups;
        }
        status = check(cudaStreamWaitEvent(g_stream, g_copied[wave], 0));
        if (status != 0) {
            return status;
        }
        int sel0 = off[begin];
        int sel1 = end < groups ? off[end] : cursor;
        launch_span(
            begin, end - begin, hidden, gate_rows, down_inner, down_rows, topk, sel0, sel1 - sel0, groups,
            moe, route);
        status = check(cudaEventRecord(g_consumed[wave], g_stream));
        if (status != 0) {
            return status;
        }
        if (end >= groups) {
            continue;
        }
        int next = end + kWave;
        if (next > groups) {
            next = groups;
        }
        if (span_alias(device, end, end, next)) {
            status = check(cudaStreamWaitEvent(copy_stream, g_consumed[wave], 0));
            if (status != 0) {
                return status;
            }
        }
        status = copy_span(device, host_src, copy_bytes, end, next, copy_stream);
        if (status != 0) {
            return status;
        }
        status = check(cudaEventRecord(g_copied[wave + 1], copy_stream));
        if (status != 0) {
            return status;
        }
    }
    return check(cudaGetLastError());
}

API int engine_gemm_pair(
    const void* weight_a,
    const float* scale_a,
    float* destination_a,
    int rows_a,
    const void* weight_b,
    const float* scale_b,
    float* destination_b,
    int rows_b,
    const float* source,
    int inner,
    int tokens) {
    if (tokens < 1 || inner <= 0 || (inner & 31) != 0 || rows_a < 1 || rows_b < 1) {
        return 1;
    }
    int status = quantize_activation(source, inner, tokens);
    if (status != 0) {
        return status;
    }
    cudaStreamCaptureStatus capturing = cudaStreamCaptureStatusNone;
    unsigned long long capture_id = 0;
    status = check(cudaStreamGetCaptureInfo(g_stream, &capturing, &capture_id));
    if (status != 0) {
        return status;
    }
    int fork = 0;
    if (capturing == cudaStreamCaptureStatusActive) {
        if (g_fork_at >= kForks) {
            status = gemm_launch(weight_a, scale_a, destination_a, rows_a, inner, tokens, g_stream);
            if (status != 0) {
                return status;
            }
            return gemm_launch(weight_b, scale_b, destination_b, rows_b, inner, tokens, g_stream);
        }
        fork = g_fork_at++;
    }
    status = check(cudaEventRecord(g_forks[fork], g_stream));
    if (status != 0) {
        return status;
    }
    status = check(cudaStreamWaitEvent(g_side, g_forks[fork], 0));
    if (status != 0) {
        return status;
    }
    status = gemm_launch(weight_a, scale_a, destination_a, rows_a, inner, tokens, g_stream);
    if (status != 0) {
        return status;
    }
    status = gemm_launch(weight_b, scale_b, destination_b, rows_b, inner, tokens, g_side);
    if (status != 0) {
        return status;
    }
    status = check(cudaEventRecord(g_joins[fork], g_side));
    if (status != 0) {
        return status;
    }
    return check(cudaStreamWaitEvent(g_stream, g_joins[fork], 0));
}

API int engine_attend_prefill(
    const void* k_cache,
    const void* v_cache,
    const float* query,
    const float* gate,
    float* destination,
    int length,
    int layer,
    int full,
    float scale,
    int count) {
    int tiles = (count + kPrefillTokens - 1) / kPrefillTokens;
    attend_prefill<<<tiles * 2, kQueryWarps * 32, 0, g_stream>>>(
        query,
        gate,
        static_cast<const int8_t*>(k_cache),
        static_cast<const int8_t*>(v_cache),
        destination,
        length,
        layer,
        full,
        scale,
        count);
    return check(cudaGetLastError());
}

API int engine_set_step(int token, int position, int rope) {
    int host[3] = {token, position, rope};
    return check(cudaMemcpyAsync(g_step, host, sizeof(host), cudaMemcpyHostToDevice, g_stream));
}

API int engine_capture_begin(int slot) {
    if (slot < 0 || slot >= kGraphs) {
        return 1;
    }
    cudaStream_t created = nullptr;
    int status = check(cudaStreamCreate(&created));
    if (status != 0) {
        return status;
    }
    cudaStream_t previous = g_stream;
    g_stream = created;
    g_fork_at = 0;
    status = check(cudaStreamBeginCapture(created, cudaStreamCaptureModeGlobal));
    if (status != 0) {
        cudaStreamDestroy(created);
        g_stream = previous;
    }
    return status;
}

API int engine_capture_end(int slot) {
    if (slot < 0 || slot >= kGraphs) {
        return 1;
    }
    cudaGraph_t graph = nullptr;
    cudaStream_t captured = g_stream;
    int status = check(cudaStreamEndCapture(captured, &graph));
    g_stream = g_main;
    if (captured != nullptr) {
        cudaStreamDestroy(captured);
    }
    if (status != 0) {
        return status;
    }
    if (g_graph[slot] != nullptr) {
        cudaGraphExecDestroy(g_graph[slot]);
        g_graph[slot] = nullptr;
    }
    status = check(cudaGraphInstantiate(&g_graph[slot], graph, 0));
    cudaGraphDestroy(graph);
    return status;
}

API int engine_replay(int slot) {
    if (slot < 0 || slot >= kGraphs || g_graph[slot] == nullptr) {
        return 1;
    }
    return check(cudaGraphLaunch(g_graph[slot], g_stream));
}

}  // extern "C"
