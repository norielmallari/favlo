// Qwen3.5-4B language step. int8 weights stay int8: one fp32 scale per row,
// the activation is int8 with one fp32 scale per 32 channels. Gated DeltaNet
// is Yang, Kautz, and Hatamizadeh, arXiv:2412.06464. The key/value cache is
// int8 with one E8M0 scale byte per 32 channels.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

static int8_t* g_codes = nullptr;
static float* g_scales = nullptr;
static int g_code_limit = 0;
static int g_scale_limit = 0;
static cudaStream_t g_stream = nullptr;
static cudaStream_t g_main = nullptr;
static cudaStream_t g_side = nullptr;
static cudaGraphExec_t g_graph[4] = {nullptr, nullptr, nullptr, nullptr};

// One fork per paired product inside a capture. Reusing one event in a graph is an error.
constexpr int kForks = 128;
static cudaEvent_t g_forks[kForks] = {};
static cudaEvent_t g_joins[kForks] = {};
static int g_fork_at = 0;

// token, cache position, RoPE position. The pointer is fixed; the three ints
// are ordinary device memory so a replay sees the latest step.
static int* g_step = nullptr;
__device__ int* g_step_ptr;
static int8_t* g_zero = nullptr;
__device__ const int8_t* g_zero_ptr;

// Decode attention splits the keys across this many blocks per head so the
// SMs stay occupied. FlashDecoding, Dao et al., the 2023 streaming-attention note.
constexpr int kAttHeads = 16;
constexpr int kAttTokens = 128;
constexpr int kAttSplits = 32;
static float* g_att_max = nullptr;
static float* g_att_sum = nullptr;
static float* g_att_acc = nullptr;

// Per-block top-k candidates. The draw kernel reduces these to one token.
constexpr int kSampleBlocks = 80;
constexpr int kSampleK = 64;
static float* g_cand_v = nullptr;
static int* g_cand_i = nullptr;

__device__ int step_at(int index) {
    return reinterpret_cast<volatile int*>(g_step_ptr)[index];
}

static int check(cudaError_t status) {
    return status == cudaSuccess ? 0 : static_cast<int>(status);
}

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

__device__ float block_sum(float value) {
    __shared__ float warp[32];
    int lane = threadIdx.x & 31;
    int group = threadIdx.x >> 5;
    int groups = blockDim.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    if (lane == 0) {
        warp[group] = value;
    }
    __syncthreads();
    if (group == 0) {
        value = lane < groups ? warp[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffff, value, offset);
        }
        if (lane == 0) {
            warp[0] = value;
        }
    }
    __syncthreads();
    // A second call reuses this buffer. Every thread has to finish the read
    // before any thread starts the next reduction.
    float result = warp[0];
    __syncthreads();
    return result;
}

__device__ float e8m0(unsigned byte) {
    unsigned bits = byte == 0u ? (1u << 22) : (byte << 23);
    return __uint_as_float(bits);
}

__device__ __forceinline__ float i8byte(unsigned word, int shift) {
    return static_cast<float>(static_cast<signed char>((word >> shift) & 0xffu));
}

// Streaming load. A decode scan reads each key once, so the line must not
// push the weight tiles out of L2.
__device__ __forceinline__ uint2 load_cs(const void* ptr) {
    uint2 out;
    asm volatile("ld.global.cs.v2.u32 {%0, %1}, [%2];"
                 : "=r"(out.x), "=r"(out.y)
                 : "l"(ptr));
    return out;
}

__device__ float silu(float value) {
    return value / (1.0f + expf(-value));
}

__global__ void quant_blocks(const float* source, int8_t* codes, float* scales, int blocks) {
    int block = blockIdx.x * blockDim.x + threadIdx.x;
    if (block >= blocks) {
        return;
    }
    const float* row = source + block * 32;
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) {
        amax = fmaxf(amax, fabsf(row[i]));
    }
    float scale = amax > 0.0f ? amax / 127.0f : 0.0f;
    scales[block] = scale;
    float inv = scale == 0.0f ? 0.0f : 1.0f / scale;
    for (int i = 0; i < 32; ++i) {
        int code = static_cast<int>(rintf(row[i] * inv));
        code = max(-127, min(127, code));
        codes[block * 32 + i] = static_cast<int8_t>(code);
    }
}

__global__ void quant_batch(
    const float* source, int8_t* codes, float* scales, int inner, int tokens) {
    int blocks = inner >> 5;
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int total = tokens * blocks;
    if (id >= total) {
        return;
    }
    int token = id / blocks;
    int block = id - token * blocks;
    const float* row = source + static_cast<long long>(token) * inner + block * 32;
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) {
        amax = fmaxf(amax, fabsf(row[i]));
    }
    float scale = amax > 0.0f ? amax / 127.0f : 0.0f;
    scales[id] = scale;
    float inv = scale == 0.0f ? 0.0f : 1.0f / scale;
    int8_t* out = codes + static_cast<long long>(token) * inner + block * 32;
    for (int i = 0; i < 32; ++i) {
        int code = static_cast<int>(rintf(row[i] * inv));
        code = max(-127, min(127, code));
        out[i] = static_cast<int8_t>(code);
    }
}

// One warp owns one output row and reuses that row across the token tile.
// Ampere __dp4a is the int8 product on sm_86 and every later RTX arch.
__device__ int dot16(int4 left, int4 right) {
    int acc = __dp4a(left.x, right.x, 0);
    acc = __dp4a(left.y, right.y, acc);
    acc = __dp4a(left.z, right.z, acc);
    return __dp4a(left.w, right.w, acc);
}

// Butterfly leaves the total in every lane. A shfl_down loop compiles as a
// divergent call per key, and that call is the attention scan.
__device__ __forceinline__ float warp_sum(float value) {
    float next;
    asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 31, 0xffffffff;"
                 : "=f"(next)
                 : "f"(value), "r"(16));
    value += next;
    asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 31, 0xffffffff;"
                 : "=f"(next)
                 : "f"(value), "r"(8));
    value += next;
    asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 31, 0xffffffff;"
                 : "=f"(next)
                 : "f"(value), "r"(4));
    value += next;
    asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 31, 0xffffffff;"
                 : "=f"(next)
                 : "f"(value), "r"(2));
    value += next;
    asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 31, 0xffffffff;"
                 : "=f"(next)
                 : "f"(value), "r"(1));
    return value + next;
}

// One m16n8k32 covers a whole activation-scale group. The fragment map is the
// one PTX specifies for mma.m16n8k32 .s8, so the scale stays in registers.
__device__ void mma_s8(int& d0, int& d1, int& d2, int& d3, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    int z0 = 0;
    int z1 = 0;
    int z2 = 0;
    int z3 = 0;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(z0), "r"(z1), "r"(z2), "r"(z3));
}

// m16n8k16 accumulates in fp32. FlashAttention-2's value product, Dao, arXiv:2307.08691.
__device__ void mma_f16(
    float& d0, float& d1, float& d2, float& d3,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ unsigned pack_half2(float x, float y) {
    __half2 packed = __floats2half2_rn(x, y);
    return *reinterpret_cast<unsigned*>(&packed);
}

__device__ float quad_max(float value) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, 1));
    value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, 2));
    return value;
}

__device__ float quad_sum(float value) {
    value += __shfl_xor_sync(0xffffffff, value, 1);
    value += __shfl_xor_sync(0xffffffff, value, 2);
    return value;
}

__device__ void cp_async16(void* shared, const void* global) {
    unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(shared));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(address), "l"(global));
}

__device__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ void cp_async_wait() {
    asm volatile("cp.async.wait_group 0;\n" ::);
}

// `ready` is how many of the newest commit groups may stay in flight.
// The immediate form is what PTX accepts, so the count is a branch.
__device__ void cp_async_wait_n(int ready) {
    if (ready <= 0) {
        asm volatile("cp.async.wait_group 0;\n" ::);
    } else if (ready == 1) {
        asm volatile("cp.async.wait_group 1;\n" ::);
    } else {
        asm volatile("cp.async.wait_group 2;\n" ::);
    }
}

// One 16-byte chunk of A and of B. Every thread commits one group so the wait
// stays uniform on a partial tile. Out-of-range rows copy the zero pad.
__device__ void stage_tile(
    signed char* a_dst,
    signed char* b_dst,
    const int8_t* weight,
    const int8_t* codes,
    int row0,
    int tok0,
    int k0,
    int rows,
    int tokens,
    int inner,
    int b_bytes,
    int tid) {
    int byte = tid * 16;
    int local = byte >> 5;
    int koff = byte & 31;
    const int8_t* a_src = g_zero_ptr;
    if (row0 + local < rows) {
        a_src = weight + static_cast<long long>(row0 + local) * inner + k0 + koff;
    }
    cp_async16(a_dst + byte, a_src);
    // The token tile is narrower than the row tile, so half the threads
    // have no B bytes. An empty commit keeps the wait uniform.
    if (byte < b_bytes) {
        const int8_t* b_src = g_zero_ptr;
        if (tok0 + local < tokens) {
            b_src = codes + static_cast<long long>(tok0 + local) * inner + k0 + koff;
        }
        cp_async16(b_dst + byte, b_src);
    }
    cp_async_commit();
}

// One scale per token for this K group. Staging it once removes a strided
// global load from every MMA in the inner loop.
__device__ void stage_scales(
    float* tile_scale,
    const float* code_scale,
    int tok0,
    int tokens,
    int kblocks,
    int k0,
    int tid) {
    if (tid >= 64) {
        return;
    }
    int tok = tok0 + tid;
    float value = 0.0f;
    if (tok < tokens) {
        value = code_scale[static_cast<long long>(tok) * kblocks + (k0 >> 5)];
    }
    tile_scale[tid] = value;
}

// 64 rows by 64 tokens, four warps. Two blocks fit on each SM and their
// K barriers are independent, which the single 128-row block could not do.
__global__ void gemm_i8(
    const int8_t* weight,
    const float* scale,
    const int8_t* codes,
    const float* code_scale,
    float* destination,
    int rows,
    int inner,
    int tokens) {
    constexpr int kRows = 64;
    constexpr int kTokens = 64;
    constexpr int kK = 32;
    int row0 = blockIdx.y * kRows;
    int tok0 = blockIdx.x * kTokens;
    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int group = lane >> 2;
    int inside = lane & 3;
    int row_lo = row0 + warp * 16 + group;
    int row_hi = row_lo + 8;
    // Two K stages. The activation scales for the 64 tokens live in shared
    // memory so the MMA loop does not gather them from global memory.
    __shared__ __align__(16) signed char As[2][kRows * kK];
    __shared__ __align__(16) signed char Bs[2][kTokens * kK];
    __shared__ float kscale[2][kTokens];
    int kblocks = inner >> 5;
    int b_bytes = kTokens * kK;
    float mine[8][4];
#pragma unroll
    for (int tile = 0; tile < 8; ++tile) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            mine[tile][i] = 0.0f;
        }
    }
    stage_tile(As[0], Bs[0], weight, codes, row0, tok0, 0, rows, tokens, inner, b_bytes, tid);
    stage_scales(kscale[0], code_scale, tok0, tokens, kblocks, 0, tid);
    cp_async_wait();
    __syncthreads();
    int stage = 0;
    for (int k0 = 0; k0 < inner; k0 += kK) {
        int next = stage ^ 1;
        if (k0 + kK < inner) {
            stage_tile(
                As[next],
                Bs[next],
                weight,
                codes,
                row0,
                tok0,
                k0 + kK,
                rows,
                tokens,
                inner,
                b_bytes,
                tid);
            stage_scales(kscale[next], code_scale, tok0, tokens, kblocks, k0 + kK, tid);
        }
        const signed char* a_base = As[stage] + (warp * 16 + group) * kK;
        unsigned a0 = *reinterpret_cast<const unsigned*>(a_base + inside * 4);
        unsigned a1 = *reinterpret_cast<const unsigned*>(a_base + 8 * kK + inside * 4);
        unsigned a2 = *reinterpret_cast<const unsigned*>(a_base + 16 + inside * 4);
        unsigned a3 = *reinterpret_cast<const unsigned*>(a_base + 8 * kK + 16 + inside * 4);
#pragma unroll
        for (int tile = 0; tile < 8; ++tile) {
            const signed char* column = Bs[stage] + (tile * 8 + group) * kK;
            unsigned b0 = *reinterpret_cast<const unsigned*>(column + inside * 4);
            unsigned b1 = *reinterpret_cast<const unsigned*>(column + 16 + inside * 4);
            int d0, d1, d2, d3;
            mma_s8(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
            int local = tile * 8 + inside * 2;
            float scale_lo = kscale[stage][local];
            float scale_hi = kscale[stage][local + 1];
            mine[tile][0] += static_cast<float>(d0) * scale_lo;
            mine[tile][1] += static_cast<float>(d1) * scale_hi;
            mine[tile][2] += static_cast<float>(d2) * scale_lo;
            mine[tile][3] += static_cast<float>(d3) * scale_hi;
        }
        if (k0 + kK < inner) {
            cp_async_wait();
            __syncthreads();
        }
        stage = next;
    }
#pragma unroll
    for (int tile = 0; tile < 8; ++tile) {
        int col_lo = inside * 2;
        int tok_lo = tok0 + tile * 8 + col_lo;
        int tok_hi = tok_lo + 1;
        if (row_lo < rows && tok_lo < tokens) {
            destination[static_cast<long long>(tok_lo) * rows + row_lo] = mine[tile][0] * scale[row_lo];
        }
        if (row_lo < rows && tok_hi < tokens) {
            destination[static_cast<long long>(tok_hi) * rows + row_lo] = mine[tile][1] * scale[row_lo];
        }
        if (row_hi < rows && tok_lo < tokens) {
            destination[static_cast<long long>(tok_lo) * rows + row_hi] = mine[tile][2] * scale[row_hi];
        }
        if (row_hi < rows && tok_hi < tokens) {
            destination[static_cast<long long>(tok_hi) * rows + row_hi] = mine[tile][3] * scale[row_hi];
        }
    }
}

__global__ void gemv_i8(
    const int8_t* weight,
    const float* scale,
    const int8_t* codes,
    const float* code_scale,
    float* destination,
    int rows,
    int inner) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const int4* line = reinterpret_cast<const int4*>(weight + static_cast<long long>(row) * inner);
    const int4* act = reinterpret_cast<const int4*>(codes);
    int vecs = inner >> 4;
    float acc = 0.0f;
    for (int vec = threadIdx.x; vec < vecs; vec += 32) {
        int dot = dot16(line[vec], act[vec]);
        acc += static_cast<float>(dot) * code_scale[vec >> 1];
    }
    acc = warp_sum(acc);
    if (threadIdx.x == 0) {
        destination[row] = acc * scale[row];
    }
}

__global__ void embed_row(const int8_t* table, const float* scale, int token, float* destination, int inner) {
    if (token < 0) {
        token = step_at(0);
    }
    float factor = scale[token];
    const int8_t* row = table + static_cast<long long>(token) * inner;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < inner; i += stride) {
        destination[i] = static_cast<float>(row[i]) * factor;
    }
}

__global__ void rmsnorm_kernel(
    const float* source,
    const __half* gain,
    float* destination,
    int width,
    float eps,
    float shift) {
    const float* row = source + static_cast<long long>(blockIdx.x) * width;
    float* out = destination + static_cast<long long>(blockIdx.x) * width;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        float value = row[i];
        sum += value * value;
    }
    sum = block_sum(sum);
    float inv = rsqrtf(sum / static_cast<float>(width) + eps);
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        float factor = __half2float(gain[i]) + shift;
        out[i] = row[i] * inv * factor;
    }
}

__global__ void residual_kernel(float* stream, const float* delta, int width) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < width) {
        stream[index] += delta[index];
    }
}

__global__ void silu_mul_kernel(const float* gate, const float* up, float* destination, int width) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < width) {
        destination[index] = silu(gate[index]) * up[index];
    }
}

__global__ void rope_kernel(
    float* data,
    int heads,
    int stride,
    int rope_dim,
    int position,
    float theta,
    int count) {
    int origin = position < 0 ? step_at(2) : position;
    int pairs = rope_dim >> 1;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int span = heads * pairs;
    if (gid >= count * span) {
        return;
    }
    int token = gid / span;
    int local = gid - token * span;
    int head = local / pairs;
    int i = local % pairs;
    float freq = powf(theta, -2.0f * static_cast<float>(i) / static_cast<float>(rope_dim));
    float angle = static_cast<float>(origin + token) * freq;
    float c = cosf(angle);
    float s = sinf(angle);
    float* row = data + (static_cast<long long>(token) * heads + head) * stride;
    float x0 = row[i];
    float x1 = row[i + pairs];
    row[i] = x0 * c - x1 * s;
    row[i + pairs] = x1 * c + x0 * s;
}

__global__ void head_norm_kernel(
    float* data,
    const __half* gain,
    int stride,
    int dim,
    float eps,
    float shift) {
    int head = blockIdx.x;
    int tid = threadIdx.x;
    if (tid >= dim) {
        return;
    }
    float value = data[static_cast<long long>(head) * stride + tid];
    float total = block_sum(value * value);
    float inv = rsqrtf(total / static_cast<float>(dim) + eps);
    data[static_cast<long long>(head) * stride + tid] = value * inv * (__half2float(gain[tid]) + shift);
}

__global__ void conv_silu_kernel(
    const __half* taps, float* mixed, float* history, int channels, int count) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= channels) {
        return;
    }
    float* prior = history + gid * 3;
    float x0 = prior[0];
    float x1 = prior[1];
    float x2 = prior[2];
    const __half* weight = taps + gid * 4;
    float w0 = __half2float(weight[0]);
    float w1 = __half2float(weight[1]);
    float w2 = __half2float(weight[2]);
    float w3 = __half2float(weight[3]);
    for (int token = 0; token < count; ++token) {
        float* sample = mixed + static_cast<long long>(token) * channels + gid;
        float x3 = *sample;
        *sample = silu(w0 * x0 + w1 * x1 + w2 * x2 + w3 * x3);
        x0 = x1;
        x1 = x2;
        x2 = x3;
    }
    prior[0] = x0;
    prior[1] = x1;
    prior[2] = x2;
}

__global__ void gdn_kernel(
    float* state,
    const float* mixed,
    const float* proj_a,
    const float* proj_b,
    const __half* a_log,
    const __half* dt_bias,
    float* output,
    int key_heads,
    int value_heads,
    int count) {
    constexpr int D = 128;
    int head = blockIdx.x;
    int tid = threadIdx.x;
    if (tid >= D) {
        return;
    }
    int repeat = value_heads / key_heads;
    int key = head / repeat;
    int key_dim = key_heads * D;
    int mixed_span = key_dim * 2 + value_heads * D;
    float* matrix = state + static_cast<long long>(head) * D * D;
    __shared__ float qs[D];
    __shared__ float ks[D];
    for (int token = 0; token < count; ++token) {
        const float* mix = mixed + static_cast<long long>(token) * mixed_span;
        float qv = mix[key * D + tid];
        float kv = mix[key_dim + key * D + tid];
        float vv = mix[2 * key_dim + head * D + tid];
        float nq = block_sum(qv * qv);
        float nk = block_sum(kv * kv);
        qs[tid] = qv * rsqrtf(nq + 1.0e-6f) * rsqrtf(static_cast<float>(D));
        ks[tid] = kv * rsqrtf(nk + 1.0e-6f);
        __syncthreads();
        float beta = 1.0f / (1.0f + expf(-proj_b[static_cast<long long>(token) * value_heads + head]));
        float raw = proj_a[static_cast<long long>(token) * value_heads + head] + __half2float(dt_bias[head]);
        float soft = raw > 20.0f ? raw : logf(1.0f + expf(raw));
        float alpha = expf(-expf(__half2float(a_log[head])) * soft);
        float sk = 0.0f;
        for (int column = 0; column < D; ++column) {
            sk = fmaf(matrix[column * D + tid], ks[column], sk);
        }
        float delta = beta * (vv - alpha * sk);
        float out = 0.0f;
        for (int column = 0; column < D; ++column) {
            float next = fmaf(alpha, matrix[column * D + tid], ks[column] * delta);
            matrix[column * D + tid] = next;
            out = fmaf(next, qs[column], out);
        }
        output[(static_cast<long long>(token) * value_heads + head) * D + tid] = out;
        __syncthreads();
    }
}

__global__ void gated_norm_kernel(
    float* data,
    const float* gate,
    const __half* gain,
    int dim,
    float eps) {
    int head = blockIdx.x;
    int tid = threadIdx.x;
    if (tid >= dim) {
        return;
    }
    long long at = static_cast<long long>(head) * dim + tid;
    float value = data[at];
    float total = block_sum(value * value);
    float inv = rsqrtf(total / static_cast<float>(dim) + eps);
    data[at] = value * inv * __half2float(gain[tid]) * silu(gate[at]);
}

__global__ void store_kv_kernel(
    const float* source,
    int8_t* data,
    unsigned char* scale,
    int position,
    int heads,
    int dim,
    int count) {
    int flat = blockIdx.x;
    int token = flat / heads;
    int head = flat - token * heads;
    int tid = threadIdx.x;
    if (token >= count || tid >= dim) {
        return;
    }
    if (position < 0) {
        position = step_at(1);
    }
    float value = source[(static_cast<long long>(token) * heads + head) * dim + tid];
    position += token;
    float amax = fabsf(value);
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset));
    }
    int exponent = 0;
    int code = 0;
    if (amax > 0.0f) {
        float power = ceilf(log2f(amax / 127.0f));
        exponent = static_cast<int>(power) + 127;
        exponent = max(1, min(254, exponent));
        float factor = exp2f(static_cast<float>(exponent - 127));
        code = static_cast<int>(rintf(value / factor));
        code = max(-127, min(127, code));
    }
    long long slot = (static_cast<long long>(position) * heads + head) * dim + tid;
    data[slot] = static_cast<int8_t>(code);
    if ((tid & 31) == 0) {
        long long scale_at = (static_cast<long long>(position) * heads + head) * (dim >> 5) + (tid >> 5);
        scale[scale_at] = static_cast<unsigned char>(exponent);
    }
}

// Online softmax split across warps in the block, then one merge.
// Dao, Fu, Ermon, Rudra, and Ré, FlashAttention, arXiv:2205.14135.
__global__ void attend_kernel(
    const float* query_gate,
    const int8_t* key_data,
    const unsigned char* key_scale,
    const int8_t* value_data,
    const unsigned char* value_scale,
    float* output,
    int length,
    int heads,
    int kv_heads,
    int dim,
    float scale,
    int count) {
    constexpr int kSplits = 8;
    int flat = blockIdx.x;
    int token = flat / heads;
    int head = flat - token * heads;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (token >= count || warp >= kSplits || (dim & 31) != 0) {
        return;
    }
    int groups = dim >> 5;
    int kv = head / (heads / kv_heads);
    const float* query_row =
        query_gate + (static_cast<long long>(token) * heads + head) * (dim * 2);
    float query[8];
    float gate[8];
    float acc[8];
#pragma unroll
    for (int group = 0; group < 8; ++group) {
        int channel = group * 32 + lane;
        query[group] = channel < dim ? query_row[channel] : 0.0f;
        gate[group] = channel < dim ? query_row[dim + channel] : 0.0f;
        acc[group] = 0.0f;
    }
    float running_max = -1.0e30f;
    float running_sum = 0.0f;
    int seen = (length < 0 ? step_at(1) + 1 : length) + token;
    for (int position = warp; position < seen; position += kSplits) {
        long long slot = static_cast<long long>(position) * kv_heads + kv;
        const int8_t* key_row = key_data + slot * dim;
        const int8_t* value_row = value_data + slot * dim;
        const unsigned char* key_sc = key_scale + slot * groups;
        const unsigned char* value_sc = value_scale + slot * groups;
        float partial = 0.0f;
        float value[8];
#pragma unroll
        for (int group = 0; group < 8; ++group) {
            int channel = group * 32 + lane;
            float key = 0.0f;
            value[group] = 0.0f;
            if (channel < dim) {
                key = static_cast<float>(key_row[channel]) * e8m0(key_sc[group]);
                value[group] = static_cast<float>(value_row[channel]) * e8m0(value_sc[group]);
            }
            partial += query[group] * key;
        }
        float score = warp_sum(partial) * scale;
        float next = fmaxf(running_max, score);
        float rescale = __expf(running_max - next);
        float prob = __expf(score - next);
        running_sum = running_sum * rescale + prob;
#pragma unroll
        for (int group = 0; group < 8; ++group) {
            acc[group] = acc[group] * rescale + prob * value[group];
        }
        running_max = next;
    }
    __shared__ float red_max[kSplits];
    __shared__ float red_sum[kSplits];
    __shared__ float red_acc[kSplits][256];
    if (lane == 0) {
        red_max[warp] = running_max;
        red_sum[warp] = running_sum;
    }
#pragma unroll
    for (int group = 0; group < 8; ++group) {
        int channel = group * 32 + lane;
        if (channel < dim) {
            red_acc[warp][channel] = acc[group];
        }
    }
    __syncthreads();
    if (warp != 0) {
        return;
    }
    float merged_max = -1.0e30f;
    for (int split = 0; split < kSplits; ++split) {
        merged_max = fmaxf(merged_max, red_max[split]);
    }
    float merged_sum = 0.0f;
    float merged[8];
#pragma unroll
    for (int group = 0; group < 8; ++group) {
        merged[group] = 0.0f;
    }
    for (int split = 0; split < kSplits; ++split) {
        float rescale = __expf(red_max[split] - merged_max);
        merged_sum += red_sum[split] * rescale;
#pragma unroll
        for (int group = 0; group < 8; ++group) {
            int channel = group * 32 + lane;
            float part = channel < dim ? red_acc[split][channel] : 0.0f;
            merged[group] += part * rescale;
        }
    }
    float inv = merged_sum == 0.0f ? 0.0f : 1.0f / merged_sum;
    float* out = output + (static_cast<long long>(token) * heads + head) * dim;
#pragma unroll
    for (int group = 0; group < 8; ++group) {
        int channel = group * 32 + lane;
        if (channel < dim) {
            out[channel] = merged[group] * inv * (1.0f / (1.0f + __expf(-gate[group])));
        }
    }
}

// Shorter contexts do not fill 32 chunks. The graph launch grid stays fixed,
// so the count has to be chosen from the step buffer on the device.
__device__ int attend_splits(int longest, int count) {
    // A prefill tile already has a job per token and head. Extra splits cut the
    // key walk into pieces that never fill the bus, then write a partial each.
    if (count > 1) {
        return longest >= 1024 ? 4 : 1;
    }
    if (longest >= 1536) {
        return 32;
    }
    if (longest >= 768) {
        return 16;
    }
    if (longest >= 256) {
        return 8;
    }
    return 4;
}

// Blocks walk the (token, head, chunk) jobs. A fixed grid keeps the CUDA graph
// valid while a short tile does not schedule 65,536 blocks.
__global__ void attend_split(
    const float* query_gate,
    const int8_t* key_data,
    const unsigned char* key_scale,
    const int8_t* value_data,
    const unsigned char* value_scale,
    float* partial_max,
    float* partial_sum,
    float* partial_acc,
    int length,
    int heads,
    int kv_heads,
    int dim,
    float scale,
    int count) {
    constexpr int kWarpSplits = 8;
    __shared__ float red_max[kWarpSplits];
    __shared__ float red_sum[kWarpSplits];
    __shared__ float red_acc[kWarpSplits][256];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int origin = length < 0 ? step_at(1) + 1 : length;
    int longest = origin + count - 1;
    int splits = attend_splits(longest, count);
    int jobs = count * heads * splits;
    for (int job = blockIdx.x; job < jobs; job += gridDim.x) {
    int split = job % splits;
    int rest = job / splits;
    int head = rest % heads;
    int token = rest / heads;
    int slot = ((token * heads + head) * splits) + split;
    int seen = origin + token;
    int chunk = (seen + splits - 1) / splits;
    int begin = split * chunk;
    int end = begin + chunk;
    if (end > seen) {
        end = seen;
    }
    if (begin >= seen) {
        if (threadIdx.x == 0) {
            partial_max[slot] = -1.0e30f;
            partial_sum[slot] = 0.0f;
        }
        if (threadIdx.x < dim) {
            partial_acc[slot * 256 + threadIdx.x] = 0.0f;
        }
        __syncthreads();
        continue;
    }
    int groups = dim >> 5;
    int kv = head / (heads / kv_heads);
    const float* query_row = query_gate + (static_cast<long long>(token) * heads + head) * (dim * 2);
    // Eight contiguous channels keep the key and value on one 8-byte load.
    int channel0 = lane * 8;
    int scale_group = lane >> 2;
    float query[8];
    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        query[i] = query_row[channel0 + i];
        acc[i] = 0.0f;
    }
    float running_max = -1.0e30f;
    float running_sum = 0.0f;
    for (int position = begin + warp; position < end;) {
        long long at = static_cast<long long>(position) * kv_heads + kv;
        int position2 = position + kWarpSplits;
        int paired = position2 < end;
        long long at2 = static_cast<long long>(position2) * kv_heads + kv;
        uint2 key = load_cs(key_data + at * dim + channel0);
        uint2 value = load_cs(value_data + at * dim + channel0);
        uint2 key2;
        uint2 value2;
        if (paired) {
            key2 = load_cs(key_data + at2 * dim + channel0);
            value2 = load_cs(value_data + at2 * dim + channel0);
        }
        float key_factor = e8m0(key_scale[at * groups + scale_group]);
        float value_factor = e8m0(value_scale[at * groups + scale_group]);
        float partial = query[0] * i8byte(key.x, 0);
        partial += query[1] * i8byte(key.x, 8);
        partial += query[2] * i8byte(key.x, 16);
        partial += query[3] * i8byte(key.x, 24);
        partial += query[4] * i8byte(key.y, 0);
        partial += query[5] * i8byte(key.y, 8);
        partial += query[6] * i8byte(key.y, 16);
        partial += query[7] * i8byte(key.y, 24);
        float score = warp_sum(partial * key_factor) * scale;
        float next = fmaxf(running_max, score);
        float rescale = __expf(running_max - next);
        float prob = __expf(score - next);
        running_sum = running_sum * rescale + prob;
        float weight = prob * value_factor;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            acc[i] = acc[i] * rescale + weight * i8byte(value.x, i * 8);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            acc[i + 4] = acc[i + 4] * rescale + weight * i8byte(value.y, i * 8);
        }
        running_max = next;
        if (paired) {
            float key_factor2 = e8m0(key_scale[at2 * groups + scale_group]);
            float value_factor2 = e8m0(value_scale[at2 * groups + scale_group]);
            float partial2 = query[0] * i8byte(key2.x, 0);
            partial2 += query[1] * i8byte(key2.x, 8);
            partial2 += query[2] * i8byte(key2.x, 16);
            partial2 += query[3] * i8byte(key2.x, 24);
            partial2 += query[4] * i8byte(key2.y, 0);
            partial2 += query[5] * i8byte(key2.y, 8);
            partial2 += query[6] * i8byte(key2.y, 16);
            partial2 += query[7] * i8byte(key2.y, 24);
            float score2 = warp_sum(partial2 * key_factor2) * scale;
            float next2 = fmaxf(running_max, score2);
            float rescale2 = __expf(running_max - next2);
            float prob2 = __expf(score2 - next2);
            running_sum = running_sum * rescale2 + prob2;
            float weight2 = prob2 * value_factor2;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                acc[i] = acc[i] * rescale2 + weight2 * i8byte(value2.x, i * 8);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                acc[i + 4] = acc[i + 4] * rescale2 + weight2 * i8byte(value2.y, i * 8);
            }
            running_max = next2;
        }
        position += paired ? kWarpSplits * 2 : kWarpSplits;
    }
    if (lane == 0) {
        red_max[warp] = running_max;
        red_sum[warp] = running_sum;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        red_acc[warp][channel0 + i] = acc[i];
    }
    __syncthreads();
    if (warp == 0) {
        float merged_max = -1.0e30f;
        for (int part = 0; part < kWarpSplits; ++part) {
            merged_max = fmaxf(merged_max, red_max[part]);
        }
        float merged_sum = 0.0f;
        float merged[8];
#pragma unroll
        for (int group = 0; group < 8; ++group) {
            merged[group] = 0.0f;
        }
        for (int part = 0; part < kWarpSplits; ++part) {
            float rescale = __expf(red_max[part] - merged_max);
            merged_sum += red_sum[part] * rescale;
#pragma unroll
            for (int group = 0; group < 8; ++group) {
                int channel = group * 32 + lane;
                float piece = channel < dim ? red_acc[part][channel] : 0.0f;
                merged[group] += piece * rescale;
            }
        }
        if (lane == 0) {
            partial_max[slot] = merged_max;
            partial_sum[slot] = merged_sum;
        }
#pragma unroll
        for (int group = 0; group < 8; ++group) {
            int channel = group * 32 + lane;
            if (channel < dim) {
                partial_acc[slot * 256 + channel] = merged[group];
            }
        }
    }
    __syncthreads();
    }
}

__global__ void attend_merge(
    const float* query_gate,
    float* output,
    const float* partial_max,
    const float* partial_sum,
    const float* partial_acc,
    int length,
    int heads,
    int dim,
    int count) {
    int head = blockIdx.x % heads;
    int token = blockIdx.x / heads;
    int lane = threadIdx.x;
    int origin = length < 0 ? step_at(1) + 1 : length;
    int splits = attend_splits(origin + count - 1, count);
    float merged_max = -1.0e30f;
    int base = (token * heads + head) * splits;
    for (int split = 0; split < splits; ++split) {
        merged_max = fmaxf(merged_max, partial_max[base + split]);
    }
    const float* query_row = query_gate + (static_cast<long long>(token) * heads + head) * (dim * 2);
    float gate[8];
    float merged[8];
#pragma unroll
    for (int group = 0; group < 8; ++group) {
        int channel = group * 32 + lane;
        gate[group] = channel < dim ? query_row[dim + channel] : 0.0f;
        merged[group] = 0.0f;
    }
    float merged_sum = 0.0f;
    for (int split = 0; split < splits; ++split) {
        int slot = base + split;
        float rescale = __expf(partial_max[slot] - merged_max);
        merged_sum += partial_sum[slot] * rescale;
#pragma unroll
        for (int group = 0; group < 8; ++group) {
            int channel = group * 32 + lane;
            merged[group] += partial_acc[slot * 256 + channel] * rescale;
        }
    }
    float inv = merged_sum == 0.0f ? 0.0f : 1.0f / merged_sum;
    float* out = output + (static_cast<long long>(token) * heads + head) * dim;
#pragma unroll
    for (int group = 0; group < 8; ++group) {
        int channel = group * 32 + lane;
        if (channel < dim) {
            out[channel] = merged[group] * inv * (1.0f / (1.0f + __expf(-gate[group])));
        }
    }
}

// Sixteen query rows fill one m16n8k32. The value product is m16n8k16 over the
// same sixteen keys. FlashAttention, Dao, Fu, Ermon, Rudra, and Ré,
// arXiv:2205.14135. FlashAttention-2, Dao, arXiv:2307.08691.
constexpr int kPrefillTokens = 16;
constexpr int kPrefillKeys = 16;

__device__ __forceinline__ void stage_kv_rows(
    int8_t* tile,
    const int8_t* table,
    int base,
    int rows,
    int kv_heads,
    int kv,
    int tid,
    int threads) {
    constexpr int kDim = 256;
    constexpr int kVec = 16;
    int vecs = kDim / kVec;
    int items = rows * vecs;
    for (int item = tid; item < items; item += threads) {
        int row = item / vecs;
        int vec = item - row * vecs;
        long long at = (static_cast<long long>(base + row) * kv_heads + kv) * kDim;
        const int4* from = reinterpret_cast<const int4*>(table + at);
        reinterpret_cast<int4*>(tile)[item] = from[vec];
    }
}

__device__ __forceinline__ void stage_kv_scales(
    unsigned char* tile,
    const unsigned char* table,
    int base,
    int rows,
    int kv_heads,
    int kv,
    int tid,
    int threads) {
    constexpr int kBytes = 8;
    int items = rows * kBytes;
    for (int item = tid; item < items; item += threads) {
        int row = item / kBytes;
        int col = item - row * kBytes;
        long long at = (static_cast<long long>(base + row) * kv_heads + kv) * kBytes + col;
        tile[item] = table[at];
    }
}

__device__ void quant_query_group(int8_t* codes, float* scale_out, const float* row) {
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        amax = fmaxf(amax, fabsf(row[i]));
    }
    float qs = amax > 0.0f ? amax / 127.0f : 0.0f;
    *scale_out = qs;
    float inv = qs == 0.0f ? 0.0f : 1.0f / qs;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        int code = static_cast<int>(rintf(row[i] * inv));
        code = max(-127, min(127, code));
        codes[i] = static_cast<int8_t>(code);
    }
}

__global__ void attend_prefill(
    const float* query_gate,
    const int8_t* key_data,
    const unsigned char* key_scale,
    const int8_t* value_data,
    const unsigned char* value_scale,
    float* output,
    int length,
    int heads,
    int kv_heads,
    float scale,
    int count) {
    constexpr int kDim = 256;
    constexpr int kGroups = kDim / 32;
    __shared__ __align__(16) int8_t key_tile[kPrefillKeys * kDim];
    __shared__ __align__(16) int8_t value_tile[kPrefillKeys * kDim];
    __shared__ unsigned char key_bytes[kPrefillKeys * kGroups];
    __shared__ unsigned char value_bytes[kPrefillKeys * kGroups];
    __shared__ __align__(16) int8_t query_codes[4 * kPrefillTokens * kDim];
    __shared__ float query_scales[4 * kPrefillTokens * kGroups];
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int frag = lane >> 2;
    int inside = lane & 3;
    int group = heads / kv_heads;
    int kv = blockIdx.x % kv_heads;
    int head = kv * group + warp;
    int token0 = (blockIdx.x / kv_heads) * kPrefillTokens;
    int origin = length < 0 ? step_at(1) + 1 : length;
    int furthest = origin + token0 + kPrefillTokens - 1;
    int last = origin + count - 1;
    if (furthest > last) {
        furthest = last;
    }
    constexpr int kChannelTiles = kDim / 8;
    float acc[kChannelTiles][4];
#pragma unroll
    for (int tile = 0; tile < kChannelTiles; ++tile) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            acc[tile][i] = 0.0f;
        }
    }
    float max_lo = -1.0e30f;
    float max_hi = -1.0e30f;
    float sum_lo = 0.0f;
    float sum_hi = 0.0f;
    if (lane < kPrefillTokens) {
        int token = token0 + lane;
        for (int kgroup = 0; kgroup < kGroups; ++kgroup) {
            int query = warp * kPrefillTokens + lane;
            int at = (query * kGroups + kgroup) * 32;
            if (token < count) {
                const float* row = query_gate
                    + (static_cast<long long>(token) * heads + head) * (kDim * 2)
                    + kgroup * 32;
                quant_query_group(query_codes + at, query_scales + query * kGroups + kgroup, row);
            } else {
                query_scales[query * kGroups + kgroup] = 0.0f;
#pragma unroll
                for (int i = 0; i < 32; ++i) {
                    query_codes[at + i] = 0;
                }
            }
        }
    }
    __syncthreads();
    int query_lo = frag;
    int query_hi = frag + 8;
    int token_lo = token0 + query_lo;
    int token_hi = token0 + query_hi;
    bool live_lo = token_lo < count;
    bool live_hi = token_hi < count;
    int seen_lo = origin + token_lo;
    int seen_hi = origin + token_hi;
    int tid = threadIdx.x;
    int threads = blockDim.x;
    for (int base = 0; base < furthest; base += kPrefillKeys) {
        int rows = furthest - base;
        if (rows > kPrefillKeys) {
            rows = kPrefillKeys;
        }
        stage_kv_rows(key_tile, key_data, base, rows, kv_heads, kv, tid, threads);
        stage_kv_rows(value_tile, value_data, base, rows, kv_heads, kv, tid, threads);
        stage_kv_scales(key_bytes, key_scale, base, rows, kv_heads, kv, tid, threads);
        stage_kv_scales(value_bytes, value_scale, base, rows, kv_heads, kv, tid, threads);
        __syncthreads();
        float score_lo[2][2];
        float score_hi[2][2];
#pragma unroll
        for (int oct = 0; oct < 2; ++oct) {
            score_lo[oct][0] = 0.0f;
            score_lo[oct][1] = 0.0f;
            score_hi[oct][0] = 0.0f;
            score_hi[oct][1] = 0.0f;
        }
        for (int kgroup = 0; kgroup < kGroups; ++kgroup) {
            int code_lo = ((warp * kPrefillTokens + frag) * kGroups + kgroup) * 32;
            int code_hi = ((warp * kPrefillTokens + frag + 8) * kGroups + kgroup) * 32;
            const int8_t* low = query_codes + code_lo;
            const int8_t* high = query_codes + code_hi;
            unsigned a0 = *reinterpret_cast<const unsigned*>(low + inside * 4);
            unsigned a2 = *reinterpret_cast<const unsigned*>(low + 16 + inside * 4);
            unsigned a1 = *reinterpret_cast<const unsigned*>(high + inside * 4);
            unsigned a3 = *reinterpret_cast<const unsigned*>(high + 16 + inside * 4);
            float query0 = query_scales[(warp * kPrefillTokens + frag) * kGroups + kgroup];
            float query1 = query_scales[(warp * kPrefillTokens + frag + 8) * kGroups + kgroup];
#pragma unroll
            for (int oct = 0; oct < 2; ++oct) {
                int row0 = oct * 8;
                int key_index = row0 + frag;
                unsigned b0 = 0;
                unsigned b1 = 0;
                if (key_index < rows) {
                    const int8_t* key_row = key_tile + key_index * kDim + kgroup * 32;
                    b0 = *reinterpret_cast<const unsigned*>(key_row + inside * 4);
                    b1 = *reinterpret_cast<const unsigned*>(key_row + 16 + inside * 4);
                }
                int d0, d1, d2, d3;
                mma_s8(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
                int col0 = row0 + inside * 2;
                float key0 = 0.0f;
                float key1 = 0.0f;
                if (col0 < rows) {
                    key0 = e8m0(key_bytes[col0 * kGroups + kgroup]);
                }
                if (col0 + 1 < rows) {
                    key1 = e8m0(key_bytes[(col0 + 1) * kGroups + kgroup]);
                }
                score_lo[oct][0] += static_cast<float>(d0) * query0 * key0;
                score_lo[oct][1] += static_cast<float>(d1) * query0 * key1;
                score_hi[oct][0] += static_cast<float>(d2) * query1 * key0;
                score_hi[oct][1] += static_cast<float>(d3) * query1 * key1;
            }
        }
#pragma unroll
        for (int oct = 0; oct < 2; ++oct) {
#pragma unroll
            for (int col = 0; col < 2; ++col) {
                int pos = base + oct * 8 + inside * 2 + col;
                if (!live_lo || pos >= seen_lo || pos >= base + rows) {
                    score_lo[oct][col] = -1.0e30f;
                } else {
                    score_lo[oct][col] *= scale;
                }
                if (!live_hi || pos >= seen_hi || pos >= base + rows) {
                    score_hi[oct][col] = -1.0e30f;
                } else {
                    score_hi[oct][col] *= scale;
                }
            }
        }
        float local_lo = fmaxf(
            fmaxf(score_lo[0][0], score_lo[0][1]), fmaxf(score_lo[1][0], score_lo[1][1]));
        float local_hi = fmaxf(
            fmaxf(score_hi[0][0], score_hi[0][1]), fmaxf(score_hi[1][0], score_hi[1][1]));
        float tile_lo = quad_max(local_lo);
        float tile_hi = quad_max(local_hi);
        float prob_lo[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
        float prob_hi[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
        float alpha_lo = 1.0f;
        float alpha_hi = 1.0f;
        float part_lo = 0.0f;
        float part_hi = 0.0f;
        if (tile_lo > -1.0e20f) {
            float next = fmaxf(max_lo, tile_lo);
            alpha_lo = __expf(max_lo - next);
#pragma unroll
            for (int oct = 0; oct < 2; ++oct) {
#pragma unroll
                for (int col = 0; col < 2; ++col) {
                    float raw = __expf(score_lo[oct][col] - next);
                    float prob = __half2float(__float2half(raw));
                    prob_lo[oct][col] = prob;
                    part_lo += prob;
                }
            }
            max_lo = next;
        }
        if (tile_hi > -1.0e20f) {
            float next = fmaxf(max_hi, tile_hi);
            alpha_hi = __expf(max_hi - next);
#pragma unroll
            for (int oct = 0; oct < 2; ++oct) {
#pragma unroll
                for (int col = 0; col < 2; ++col) {
                    float raw = __expf(score_hi[oct][col] - next);
                    float prob = __half2float(__float2half(raw));
                    prob_hi[oct][col] = prob;
                    part_hi += prob;
                }
            }
            max_hi = next;
        }
        part_lo = quad_sum(part_lo);
        part_hi = quad_sum(part_hi);
        if (tile_lo > -1.0e20f) {
            sum_lo = sum_lo * alpha_lo + part_lo;
        }
        if (tile_hi > -1.0e20f) {
            sum_hi = sum_hi * alpha_hi + part_hi;
        }
#pragma unroll
        for (int tile = 0; tile < kChannelTiles; ++tile) {
            acc[tile][0] *= alpha_lo;
            acc[tile][1] *= alpha_lo;
            acc[tile][2] *= alpha_hi;
            acc[tile][3] *= alpha_hi;
            int channel = tile * 8 + frag;
            int scale_group = channel >> 5;
            auto value_at = [&](int key) {
                if (key >= rows) {
                    return 0.0f;
                }
                float factor = e8m0(value_bytes[key * kGroups + scale_group]);
                return static_cast<float>(value_tile[key * kDim + channel]) * factor;
            };
            unsigned b0 = pack_half2(value_at(inside * 2), value_at(inside * 2 + 1));
            unsigned b1 = pack_half2(value_at(8 + inside * 2), value_at(8 + inside * 2 + 1));
            unsigned p0 = pack_half2(prob_lo[0][0], prob_lo[0][1]);
            unsigned p1 = pack_half2(prob_hi[0][0], prob_hi[0][1]);
            unsigned p2 = pack_half2(prob_lo[1][0], prob_lo[1][1]);
            unsigned p3 = pack_half2(prob_hi[1][0], prob_hi[1][1]);
            mma_f16(
                acc[tile][0], acc[tile][1], acc[tile][2], acc[tile][3], p0, p1, p2, p3, b0, b1);
        }
        __syncthreads();
    }
    float inv_lo = sum_lo == 0.0f ? 0.0f : 1.0f / sum_lo;
    float inv_hi = sum_hi == 0.0f ? 0.0f : 1.0f / sum_hi;
    const float* row_lo = nullptr;
    const float* row_hi = nullptr;
    if (live_lo) {
        row_lo = query_gate + (static_cast<long long>(token_lo) * heads + head) * (kDim * 2);
    }
    if (live_hi) {
        row_hi = query_gate + (static_cast<long long>(token_hi) * heads + head) * (kDim * 2);
    }
#pragma unroll
    for (int tile = 0; tile < kChannelTiles; ++tile) {
        int c0 = tile * 8 + inside * 2;
        int c1 = c0 + 1;
        if (live_lo) {
            float* out = output + (static_cast<long long>(token_lo) * heads + head) * kDim;
            float g0 = row_lo[kDim + c0];
            float g1 = row_lo[kDim + c1];
            out[c0] = acc[tile][0] * inv_lo * (1.0f / (1.0f + __expf(-g0)));
            out[c1] = acc[tile][1] * inv_lo * (1.0f / (1.0f + __expf(-g1)));
        }
        if (live_hi) {
            float* out = output + (static_cast<long long>(token_hi) * heads + head) * kDim;
            float g0 = row_hi[kDim + c0];
            float g1 = row_hi[kDim + c1];
            out[c0] = acc[tile][2] * inv_hi * (1.0f / (1.0f + __expf(-g0)));
            out[c1] = acc[tile][3] * inv_hi * (1.0f / (1.0f + __expf(-g1)));
        }
    }
}


// One block owns a strided slice of the vocabulary. ``k`` passes keep that
// slice's top-k after the presence penalty. A later draw reduces the blocks.
__global__ void sample_partial(
    const float* logits,
    const int* counts,
    float* cand_v,
    int* cand_i,
    int vocab,
    int k,
    float presence,
    float frequency) {
    __shared__ float rv[256];
    __shared__ int ri[256];
    __shared__ int taken[64];
    int tid = threadIdx.x;
    for (int pass = 0; pass < k; ++pass) {
        float best = -1.0e30f;
        int best_i = 0x7fffffff;
        int stride = blockDim.x * gridDim.x;
        for (int i = blockIdx.x * blockDim.x + tid; i < vocab; i += stride) {
            bool skip = false;
            for (int prev = 0; prev < pass; ++prev) {
                if (taken[prev] == i) {
                    skip = true;
                    break;
                }
            }
            if (skip) {
                continue;
            }
            float value = logits[i];
            int seen = counts[i];
            if (seen > 0) {
                value -= presence + frequency * static_cast<float>(seen);
            }
            if (value > best || (value == best && i < best_i)) {
                best = value;
                best_i = i;
            }
        }
        rv[tid] = best;
        ri[tid] = best_i;
        __syncthreads();
        for (int offset = 128; offset > 0; offset >>= 1) {
            if (tid < offset) {
                float other_v = rv[tid + offset];
                int other_i = ri[tid + offset];
                bool better = other_v > rv[tid] || (other_v == rv[tid] && other_i < ri[tid]);
                if (better) {
                    rv[tid] = other_v;
                    ri[tid] = other_i;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            taken[pass] = ri[0];
            cand_v[blockIdx.x * k + pass] = rv[0];
            cand_i[blockIdx.x * k + pass] = ri[0];
        }
        __syncthreads();
    }
}

// Reduce the block candidates, apply temperature and nucleus, write one token.
__global__ void sample_draw(
    const float* cand_v,
    const int* cand_i,
    int* counts,
    int* token_out,
    int n_cand,
    int k,
    float temperature,
    float top_p,
    float draw) {
    __shared__ float rv[256];
    __shared__ int ri[256];
    __shared__ float top_v[64];
    __shared__ int top_i[64];
    int tid = threadIdx.x;
    int limit = temperature <= 0.0f ? 1 : k;
    for (int pass = 0; pass < limit; ++pass) {
        float best = -1.0e30f;
        int best_i = 0x7fffffff;
        for (int i = tid; i < n_cand; i += blockDim.x) {
            int index = cand_i[i];
            if (index < 0 || index == 0x7fffffff) {
                continue;
            }
            bool skip = false;
            for (int prev = 0; prev < pass; ++prev) {
                if (top_i[prev] == index) {
                    skip = true;
                    break;
                }
            }
            if (skip) {
                continue;
            }
            float value = cand_v[i];
            if (value > best || (value == best && index < best_i)) {
                best = value;
                best_i = index;
            }
        }
        rv[tid] = best;
        ri[tid] = best_i;
        __syncthreads();
        for (int offset = 128; offset > 0; offset >>= 1) {
            if (tid < offset) {
                float other_v = rv[tid + offset];
                int other_i = ri[tid + offset];
                bool better = other_v > rv[tid] || (other_v == rv[tid] && other_i < ri[tid]);
                if (better) {
                    rv[tid] = other_v;
                    ri[tid] = other_i;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            top_v[pass] = rv[0];
            top_i[pass] = ri[0];
        }
        __syncthreads();
    }
    if (tid != 0) {
        return;
    }
    int picked = top_i[0];
    if (picked < 0 || picked == 0x7fffffff) {
        token_out[0] = 0;
        counts[0] += 1;
        return;
    }
    if (temperature <= 0.0f) {
        token_out[0] = picked;
        counts[picked] += 1;
        return;
    }
    float peak = top_v[0];
    float inv = 1.0f / fmaxf(temperature, 1.0e-5f);
    float weight[64];
    float total = 0.0f;
    int live = 0;
    for (int i = 0; i < limit; ++i) {
        int index = top_i[i];
        if (index < 0 || index == 0x7fffffff || top_v[i] <= -1.0e29f) {
            break;
        }
        float term = expf((top_v[i] - peak) * inv);
        weight[live] = term;
        top_i[live] = index;
        total += term;
        live += 1;
    }
    if (live == 0 || total <= 0.0f) {
        token_out[0] = picked;
        counts[picked] += 1;
        return;
    }
    int keep = live;
    if (top_p > 0.0f && top_p < 1.0f) {
        float cumulative = 0.0f;
        for (int i = 0; i < live; ++i) {
            cumulative += weight[i] / total;
            if (cumulative >= top_p) {
                keep = i + 1;
                break;
            }
        }
    }
    float kept = 0.0f;
    for (int i = 0; i < keep; ++i) {
        kept += weight[i];
    }
    float target = draw * kept;
    float run = 0.0f;
    picked = top_i[keep - 1];
    for (int i = 0; i < keep; ++i) {
        run += weight[i];
        if (target < run) {
            picked = top_i[i];
            break;
        }
    }
    token_out[0] = picked;
    counts[picked] += 1;
}

__device__ float warp_sum32(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__global__ void vision_layernorm_kernel(
    const float* source, const __half* weight, const __half* bias, float* destination, int width, float eps) {
    int row = blockIdx.x;
    const float* in = source + static_cast<long long>(row) * width;
    float* out = destination + static_cast<long long>(row) * width;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        sum += in[i];
    }
    sum = block_sum(sum);
    float mean = sum / static_cast<float>(width);
    float var = 0.0f;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        float delta = in[i] - mean;
        var += delta * delta;
    }
    var = block_sum(var);
    float inv = rsqrtf(var / static_cast<float>(width) + eps);
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        out[i] = (in[i] - mean) * inv * __half2float(weight[i]) + __half2float(bias[i]);
    }
}

__global__ void vision_bias_kernel(float* rows, const __half* bias, int cols, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int n = cols * count;
    if (index >= n) {
        return;
    }
    rows[index] += __half2float(bias[index % cols]);
}

__global__ void vision_gelu_kernel(float* values, int count, int erf_mode) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    float x = values[index];
    if (erf_mode) {
        float ax = fabsf(x);
        float t = 1.0f / (1.0f + 0.3275911f * ax);
        float poly = fmaf(fmaf(fmaf(fmaf(1.061405429f, t, -1.453152027f), t, 1.421413741f), t, -0.284496736f), t, 0.254829592f);
        poly *= t;
        float erf = copysignf(1.0f - poly * expf(-ax * ax), x);
        values[index] = 0.5f * x * (1.0f + erf);
        return;
    }
    if (fabsf(x) >= 8.0f) {
        values[index] = x > 0.0f ? x : 0.0f;
        return;
    }
    float inner = 0.7978845608028654f * fmaf(0.044715f * x * x, x, x);
    values[index] = 0.5f * x * (1.0f + tanhf(inner));
}

__global__ void vision_split_qkv_kernel(
    const float* packed, float* query, float* key, float* value, int tokens) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int token = blockIdx.y;
    if (col >= 1024 || token >= tokens) {
        return;
    }
    long long base = static_cast<long long>(token) * 3072 + col;
    long long out = static_cast<long long>(token) * 1024 + col;
    query[out] = packed[base];
    key[out] = packed[base + 1024];
    value[out] = packed[base + 2048];
}

__global__ void vision_rope_kernel(
    float* values, const float* cosv, const float* sinv, int tokens, int heads) {
    int token = blockIdx.x;
    if (token >= tokens) {
        return;
    }
    int pairs = heads * 32;
    for (int index = threadIdx.x; index < pairs; index += blockDim.x) {
        int head = index / 32;
        int pair = index - head * 32;
        float* row = values + (static_cast<long long>(token) * heads + head) * 64;
        float c = cosv[static_cast<long long>(token) * 64 + pair];
        float s = sinv[static_cast<long long>(token) * 64 + pair];
        float x0 = row[pair];
        float x1 = row[32 + pair];
        row[pair] = x0 * c - x1 * s;
        row[32 + pair] = x1 * c + x0 * s;
    }
}

__global__ void vision_attend_kernel(
    const float* query, const float* key, const float* value, float* destination, int tokens, int heads) {
    int head = blockIdx.x;
    int qindex = blockIdx.y;
    int tid = threadIdx.x;
    if (head >= heads || qindex >= tokens || tid >= 64) {
        return;
    }
    const float* qrow = query + (static_cast<long long>(qindex) * heads + head) * 64;
    __shared__ float red[2];
    int lane = tid & 31;
    int simd = tid >> 5;
    float running_max = -1.0e30f;
    float running_sum = 0.0f;
    for (int token = 0; token < tokens; ++token) {
        const float* krow = key + (static_cast<long long>(token) * heads + head) * 64;
        float part = warp_sum32(qrow[tid] * krow[tid]);
        if (lane == 0) {
            red[simd] = part;
        }
        __syncthreads();
        float score = (red[0] + red[1]) * 0.125f;
        __syncthreads();
        float next = fmaxf(running_max, score);
        running_sum = running_sum * __expf(running_max - next) + __expf(score - next);
        running_max = next;
    }
    float acc = 0.0f;
    for (int token = 0; token < tokens; ++token) {
        const float* krow = key + (static_cast<long long>(token) * heads + head) * 64;
        const float* vrow = value + (static_cast<long long>(token) * heads + head) * 64;
        float part = warp_sum32(qrow[tid] * krow[tid]);
        if (lane == 0) {
            red[simd] = part;
        }
        __syncthreads();
        float score = (red[0] + red[1]) * 0.125f;
        __syncthreads();
        acc += (__expf(score - running_max) / running_sum) * vrow[tid];
    }
    destination[(static_cast<long long>(qindex) * heads + head) * 64 + tid] = acc;
}

__global__ void vision_dequant_kernel(const int8_t* codes, const float* scale, float* destination, int inner, int rows) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    float factor = scale[row];
    const int8_t* src = codes + static_cast<long long>(row) * inner;
    float* dest = destination + static_cast<long long>(row) * inner;
    for (int i = threadIdx.x; i < inner; i += blockDim.x) {
        dest[i] = static_cast<float>(src[i]) * factor;
    }
}

__global__ void splice_rows_kernel(
    const float* rows, const unsigned* index, float* hidden, int position0, int count, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    if (row >= count || col >= width) {
        return;
    }
    unsigned slot = index[position0 + row];
    if (slot == 0xffffffffu) {
        return;
    }
    hidden[static_cast<long long>(row) * width + col] =
        rows[static_cast<long long>(slot) * width + col];
}

__constant__ int k_mrope_axis[32] = {
    0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1,
};

__global__ void rope_mrope_kernel(
    float* data,
    const unsigned* positions,
    int heads,
    int stride,
    int rope_dim,
    int position0,
    int count,
    int prompt_len,
    float theta) {
    int pairs = rope_dim >> 1;
    int span = heads * pairs;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count * span) {
        return;
    }
    int row = gid / span;
    int local = gid - row * span;
    int head = local / pairs;
    int i = local % pairs;
    unsigned pos = positions[static_cast<long long>(k_mrope_axis[i]) * prompt_len + position0 + row];
    float freq = powf(theta, -2.0f * static_cast<float>(i) / static_cast<float>(rope_dim));
    float angle = static_cast<float>(pos) * freq;
    float c = cosf(angle);
    float s = sinf(angle);
    float* rowp = data + (static_cast<long long>(row) * heads + head) * stride;
    float x0 = rowp[i];
    float x1 = rowp[i + pairs];
    rowp[i] = x0 * c - x1 * s;
    rowp[i + pairs] = x1 * c + x0 * s;
}

#if defined(_WIN32)
#define API __declspec(dllexport)
#else
#define API
#endif

extern "C" {

API int engine_open(void) {
    int device = 0;
    int status = check(cudaSetDevice(0));
    if (status != 0) {
        return status;
    }
    status = check(cudaGetDevice(&device));
    if (status != 0) {
        return status;
    }
    cudaDeviceProp prop{};
    status = check(cudaGetDeviceProperties(&prop, device));
    if (status != 0) {
        return status;
    }
    if (prop.major * 10 + prop.minor < 86) {
        return 1001;
    }
    int status_step = check(cudaStreamCreate(&g_main));
    if (status_step != 0) {
        return status_step;
    }
    status_step = check(cudaStreamCreate(&g_side));
    if (status_step != 0) {
        return status_step;
    }
    g_stream = g_main;
    for (int fork = 0; fork < kForks; ++fork) {
        status_step = check(cudaEventCreateWithFlags(&g_forks[fork], cudaEventDisableTiming));
        if (status_step != 0) {
            return status_step;
        }
        status_step = check(cudaEventCreateWithFlags(&g_joins[fork], cudaEventDisableTiming));
        if (status_step != 0) {
            return status_step;
        }
    }
    status_step = check(cudaMalloc(&g_step, sizeof(int) * 3));
    if (status_step != 0) {
        return status_step;
    }
    status_step = check(cudaMemset(g_step, 0, sizeof(int) * 3));
    if (status_step != 0) {
        return status_step;
    }
    status_step = check(cudaMemcpyToSymbol(g_step_ptr, &g_step, sizeof(g_step)));
    if (status_step != 0) {
        return status_step;
    }
    status_step = check(cudaMalloc(&g_zero, 16));
    if (status_step != 0) {
        return status_step;
    }
    status_step = check(cudaMemset(g_zero, 0, 16));
    if (status_step != 0) {
        return status_step;
    }
    status_step = check(cudaMemcpyToSymbol(g_zero_ptr, &g_zero, sizeof(g_zero)));
    if (status_step != 0) {
        return status_step;
    }
    int partials = kAttTokens * kAttHeads * kAttSplits;
    status_step = check(cudaMalloc(&g_att_max, static_cast<size_t>(partials) * sizeof(float)));
    if (status_step != 0) {
        return status_step;
    }
    status_step = check(cudaMalloc(&g_att_sum, static_cast<size_t>(partials) * sizeof(float)));
    if (status_step != 0) {
        return status_step;
    }
    status_step = check(cudaMalloc(&g_att_acc, static_cast<size_t>(partials) * 256 * sizeof(float)));
    if (status_step != 0) {
        return status_step;
    }
    size_t cand = static_cast<size_t>(kSampleBlocks) * kSampleK;
    status_step = check(cudaMalloc(&g_cand_v, cand * sizeof(float)));
    if (status_step != 0) {
        return status_step;
    }
    status_step = check(cudaMalloc(&g_cand_i, cand * sizeof(int)));
    if (status_step != 0) {
        return status_step;
    }
    return scratch_for(1, 16384);
}

API void engine_close(void) {
    cudaFree(g_codes);
    cudaFree(g_scales);
    cudaFree(g_step);
    cudaFree(g_zero);
    cudaFree(g_att_max);
    cudaFree(g_att_sum);
    cudaFree(g_att_acc);
    cudaFree(g_cand_v);
    cudaFree(g_cand_i);
    g_codes = nullptr;
    g_scales = nullptr;
    g_step = nullptr;
    g_zero = nullptr;
    g_att_max = nullptr;
    g_att_sum = nullptr;
    g_att_acc = nullptr;
    g_cand_v = nullptr;
    g_cand_i = nullptr;
    g_code_limit = 0;
    g_scale_limit = 0;
    for (int slot = 0; slot < 4; ++slot) {
        if (g_graph[slot] != nullptr) {
            cudaGraphExecDestroy(g_graph[slot]);
            g_graph[slot] = nullptr;
        }
    }
    if (g_main != nullptr) {
        cudaStreamDestroy(g_main);
        g_main = nullptr;
    }
    if (g_side != nullptr) {
        cudaStreamDestroy(g_side);
        g_side = nullptr;
    }
    g_stream = nullptr;
    for (int fork = 0; fork < kForks; ++fork) {
        if (g_forks[fork] != nullptr) {
            cudaEventDestroy(g_forks[fork]);
            g_forks[fork] = nullptr;
        }
        if (g_joins[fork] != nullptr) {
            cudaEventDestroy(g_joins[fork]);
            g_joins[fork] = nullptr;
        }
    }
}

API void* engine_alloc(size_t bytes) {
    void* pointer = nullptr;
    if (cudaMalloc(&pointer, bytes) != cudaSuccess) {
        return nullptr;
    }
    return pointer;
}

API void engine_free(void* pointer) {
    cudaFree(pointer);
}

API int engine_on_device(const void* pointer) {
    cudaPointerAttributes attr{};
    if (cudaPointerGetAttributes(&attr, pointer) != cudaSuccess) {
        return 0;
    }
    return attr.type == cudaMemoryTypeDevice && attr.hostPointer == nullptr ? 1 : 0;
}

API int engine_h2d(void* destination, const void* source, size_t bytes) {
    return check(cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice));
}

API int engine_register(void* pointer, size_t bytes) {
    if (cudaHostRegister(pointer, bytes, cudaHostRegisterDefault) != cudaSuccess) {
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

API int engine_d2h(void* destination, const void* source, size_t bytes) {
    return check(cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost));
}

API int engine_copy(void* destination, const void* source, size_t bytes) {
    if (bytes == 0 || g_stream == nullptr) {
        return bytes == 0 ? 0 : 1;
    }
    return check(cudaMemcpyAsync(
        destination, source, bytes, cudaMemcpyDeviceToDevice, g_stream));
}

API int engine_zero(void* pointer, size_t bytes) {
    return check(cudaMemset(pointer, 0, bytes));
}

API int engine_gemv(
    const void* weight, const float* scale, const float* source, float* destination, int rows, int inner) {
    if (inner <= 0 || (inner & 31) != 0) {
        return 1;
    }
    int status = scratch_for(1, inner);
    if (status != 0) {
        return status;
    }
    int blocks = inner >> 5;
    int grid = (blocks + 127) / 128;
    quant_blocks<<<grid, 128, 0, g_stream>>>(source, g_codes, g_scales, blocks);
    gemv_i8<<<rows, 32, 0, g_stream>>>(
        static_cast<const int8_t*>(weight), scale, g_codes, g_scales, destination, rows, inner);
    return check(cudaGetLastError());
}

static int gemm_launch(
    const void* weight,
    const float* scale,
    float* destination,
    int rows,
    int inner,
    int tokens,
    cudaStream_t stream) {
    dim3 tile((tokens + 63) / 64, (rows + 63) / 64);
    gemm_i8<<<tile, 128, 0, stream>>>(
        static_cast<const int8_t*>(weight),
        scale,
        g_codes,
        g_scales,
        destination,
        rows,
        inner,
        tokens);
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

API int engine_gemm(
    const void* weight,
    const float* scale,
    const float* source,
    float* destination,
    int rows,
    int inner,
    int tokens) {
    if (tokens < 1 || inner <= 0 || (inner & 31) != 0) {
        return 1;
    }
    int status = quantize_activation(source, inner, tokens);
    if (status != 0) {
        return status;
    }
    return gemm_launch(weight, scale, destination, rows, inner, tokens, g_stream);
}

// Two products of one activation. The second runs on the side stream so a tile
// that does not fill the device shares it with the other product.
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

API int engine_embed(const void* table, const float* scale, int token, float* destination, int inner) {
    embed_row<<<8, 256, 0, g_stream>>>(static_cast<const int8_t*>(table), scale, token, destination, inner);
    return check(cudaGetLastError());
}

API int engine_rmsnorm(
    const float* source,
    const void* gain,
    float* destination,
    int width,
    float eps,
    float shift,
    int count) {
    if (count < 1) {
        return 1;
    }
    rmsnorm_kernel<<<count, 256, 0, g_stream>>>(
        source, static_cast<const __half*>(gain), destination, width, eps, shift);
    return check(cudaGetLastError());
}

API int engine_add(float* stream, const float* delta, int width, int count) {
    int n = width * count;
    residual_kernel<<<(n + 255) / 256, 256, 0, g_stream>>>(stream, delta, n);
    return check(cudaGetLastError());
}

API int engine_silu_mul(const float* gate, const float* up, float* destination, int width, int count) {
    int n = width * count;
    silu_mul_kernel<<<(n + 255) / 256, 256, 0, g_stream>>>(gate, up, destination, n);
    return check(cudaGetLastError());
}

API int engine_rope(
    float* data, int heads, int stride, int rope_dim, int position, float theta, int count) {
    int pairs = count * heads * (rope_dim / 2);
    rope_kernel<<<(pairs + 255) / 256, 256, 0, g_stream>>>(data, heads, stride, rope_dim, position, theta, count);
    return check(cudaGetLastError());
}

API int engine_head_norm(
    float* data, const void* gain, int heads, int stride, int dim, float eps, float shift, int count) {
    head_norm_kernel<<<heads * count, dim, 0, g_stream>>>(
        data, static_cast<const __half*>(gain), stride, dim, eps, shift);
    return check(cudaGetLastError());
}

API int engine_conv_silu(const void* taps, float* mixed, float* history, int channels, int count) {
    conv_silu_kernel<<<(channels + 255) / 256, 256, 0, g_stream>>>(
        static_cast<const __half*>(taps), mixed, history, channels, count);
    return check(cudaGetLastError());
}

API int engine_gdn(
    float* state,
    const float* mixed,
    const float* proj_a,
    const float* proj_b,
    const void* a_log,
    const void* dt_bias,
    float* output,
    int key_heads,
    int value_heads,
    int count) {
    gdn_kernel<<<value_heads, 128, 0, g_stream>>>(
        state,
        mixed,
        proj_a,
        proj_b,
        static_cast<const __half*>(a_log),
        static_cast<const __half*>(dt_bias),
        output,
        key_heads,
        value_heads,
        count);
    return check(cudaGetLastError());
}

API int engine_gated_norm(
    float* data, const float* gate, const void* gain, int heads, int dim, float eps, int count) {
    gated_norm_kernel<<<heads * count, dim, 0, g_stream>>>(
        data, gate, static_cast<const __half*>(gain), dim, eps);
    return check(cudaGetLastError());
}

API int engine_store_kv(
    const float* source, void* data, void* scale, int position, int heads, int dim, int count) {
    store_kv_kernel<<<heads * count, dim, 0, g_stream>>>(
        source,
        static_cast<int8_t*>(data),
        static_cast<unsigned char*>(scale),
        position,
        heads,
        dim,
        count);
    return check(cudaGetLastError());
}

API int engine_attend(
    const float* query,
    const void* key_data,
    const void* key_scale,
    const void* value_data,
    const void* value_scale,
    float* output,
    int length,
    int heads,
    int kv_heads,
    int dim,
    float scale,
    int count) {
    if (dim != 256 || heads < 1 || heads > kAttHeads || kv_heads < 1 || heads % kv_heads != 0) {
        return 1;
    }
    if (count > 1) {
        int group = heads / kv_heads;
        int threads = group * 32;
        if (group <= 4 && threads <= 1024) {
            int tiles = (count + kPrefillTokens - 1) / kPrefillTokens;
            attend_prefill<<<tiles * kv_heads, threads, 0, g_stream>>>(
                query,
                static_cast<const int8_t*>(key_data),
                static_cast<const unsigned char*>(key_scale),
                static_cast<const int8_t*>(value_data),
                static_cast<const unsigned char*>(value_scale),
                output,
                length,
                heads,
                kv_heads,
                scale,
                count);
            return check(cudaGetLastError());
        }
    }
    if (count == 1 || count == 128) {
        // Decode keeps one block per chunk. Prefill reuses a fixed grid so the
        // captured graph does not launch a block for every empty chunk.
        int grid = count == 1 ? heads * kAttSplits : 4096;
        attend_split<<<grid, 256, 0, g_stream>>>(
            query,
            static_cast<const int8_t*>(key_data),
            static_cast<const unsigned char*>(key_scale),
            static_cast<const int8_t*>(value_data),
            static_cast<const unsigned char*>(value_scale),
            g_att_max,
            g_att_sum,
            g_att_acc,
            length,
            heads,
            kv_heads,
            dim,
            scale,
            count);
        attend_merge<<<count * heads, 32, 0, g_stream>>>(
            query, output, g_att_max, g_att_sum, g_att_acc, length, heads, dim, count);
        return check(cudaGetLastError());
    }
    attend_kernel<<<heads * count, 256, 0, g_stream>>>(
        query,
        static_cast<const int8_t*>(key_data),
        static_cast<const unsigned char*>(key_scale),
        static_cast<const int8_t*>(value_data),
        static_cast<const unsigned char*>(value_scale),
        output,
        length,
        heads,
        kv_heads,
        dim,
        scale,
        count);
    return check(cudaGetLastError());
}

API int engine_reserve(int tokens, int inner) {
    if (tokens < 1 || inner <= 0) {
        return 1;
    }
    return scratch_for(tokens, inner);
}

API int engine_set_step(int token, int position, int rope) {
    int host[3] = {token, position, rope};
    return check(cudaMemcpy(g_step, host, sizeof(host), cudaMemcpyHostToDevice));
}

static int graph_slot(int slot) {
    return slot >= 0 && slot < 4 ? 0 : 1;
}

API int engine_capture_begin(int slot) {
    if (graph_slot(slot) != 0) {
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
    if (graph_slot(slot) != 0) {
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
    if (graph_slot(slot) != 0 || g_graph[slot] == nullptr) {
        return 1;
    }
    return check(cudaGraphLaunch(g_graph[slot], g_stream));
}

API int engine_sample(
    const float* logits,
    int* counts,
    int* token_out,
    int vocab,
    int top_k,
    float temperature,
    float top_p,
    float presence,
    float frequency,
    float draw) {
    if (vocab < 1 || top_k < 1 || top_k > kSampleK || g_cand_v == nullptr) {
        return 1;
    }
    int k = temperature <= 0.0f ? 1 : top_k;
    sample_partial<<<kSampleBlocks, 256, 0, g_stream>>>(
        logits, counts, g_cand_v, g_cand_i, vocab, k, presence, frequency);
    sample_draw<<<1, 256, 0, g_stream>>>(
        g_cand_v,
        g_cand_i,
        counts,
        token_out,
        kSampleBlocks * k,
        k,
        temperature,
        top_p,
        draw);
    return check(cudaGetLastError());
}

API int engine_sync(void) {
    return check(cudaDeviceSynchronize());
}

API int engine_vision_layernorm(
    const float* source, const void* weight, const void* bias, float* destination, int width, float eps, int rows) {
    if (rows < 1 || width < 1) {
        return 1;
    }
    vision_layernorm_kernel<<<rows, 256, 0, g_stream>>>(
        source, static_cast<const __half*>(weight), static_cast<const __half*>(bias), destination, width, eps);
    return check(cudaGetLastError());
}

API int engine_vision_bias(float* rows, const void* bias, int cols, int count) {
    int n = cols * count;
    if (n < 1) {
        return 0;
    }
    vision_bias_kernel<<<(n + 255) / 256, 256, 0, g_stream>>>(rows, static_cast<const __half*>(bias), cols, count);
    return check(cudaGetLastError());
}

API int engine_vision_gelu(float* values, int count, int erf_mode) {
    if (count < 1) {
        return 0;
    }
    vision_gelu_kernel<<<(count + 255) / 256, 256, 0, g_stream>>>(values, count, erf_mode);
    return check(cudaGetLastError());
}

API int engine_vision_split_qkv(const float* packed, float* query, float* key, float* value, int tokens) {
    if (tokens < 1) {
        return 1;
    }
    vision_split_qkv_kernel<<<dim3((1024 + 31) / 32, tokens), 32, 0, g_stream>>>(packed, query, key, value, tokens);
    return check(cudaGetLastError());
}

API int engine_vision_rope(float* values, const float* cosv, const float* sinv, int tokens, int heads) {
    if (tokens < 1 || heads < 1) {
        return 1;
    }
    vision_rope_kernel<<<tokens, 256, 0, g_stream>>>(values, cosv, sinv, tokens, heads);
    return check(cudaGetLastError());
}

API int engine_vision_attend(
    const float* query, const float* key, const float* value, float* destination, int tokens, int heads) {
    if (tokens < 1 || heads < 1 || tokens > 65535) {
        return 1;
    }
    vision_attend_kernel<<<dim3(heads, tokens), 64, 0, g_stream>>>(query, key, value, destination, tokens, heads);
    return check(cudaGetLastError());
}

API int engine_vision_dequant(const void* codes, const float* scale, float* destination, int inner, int rows) {
    if (rows < 1 || inner < 1) {
        return 1;
    }
    vision_dequant_kernel<<<rows, 256, 0, g_stream>>>(
        static_cast<const int8_t*>(codes), scale, destination, inner, rows);
    return check(cudaGetLastError());
}

API int engine_splice_rows(
    const float* rows, const unsigned* index, float* hidden, int position0, int count, int width) {
    if (count < 1 || width < 1) {
        return 0;
    }
    splice_rows_kernel<<<dim3((width + 255) / 256, count), 256, 0, g_stream>>>(
        rows, index, hidden, position0, count, width);
    return check(cudaGetLastError());
}

API int engine_rope_mrope(
    float* data,
    const unsigned* positions,
    int heads,
    int stride,
    int rope_dim,
    int position0,
    int count,
    int prompt_len,
    float theta) {
    if (count < 1 || rope_dim != 64) {
        return 1;
    }
    int span = heads * (rope_dim >> 1);
    int n = count * span;
    rope_mrope_kernel<<<(n + 255) / 256, 256, 0, g_stream>>>(
        data, positions, heads, stride, rope_dim, position0, count, prompt_len, theta);
    return check(cudaGetLastError());
}

}
