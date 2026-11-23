// FLUX.2 [klein] 4B CUDA step.
// Activations are fp16. int4 weights are dequantized inside the product and
// multiplied with mma.sync.aligned.m16n8k16, which accumulates in fp32.
// The low nibble is the even column. Codes are -7..7. One fp32 scale per row.

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>

namespace {

constexpr int kBM = 128;
constexpr int kBN = 128;
constexpr int kBK = 32;
constexpr int kNTiles = kBN / 8;
constexpr int kGemmThreads = (kBM / 16) * 32;

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
    }
}

// A fp16 store saturates at the finite range. Inf would poison the next norm.
__device__ __half finite_half(float value) {
    return __float2half(fminf(fmaxf(value, -65504.f), 65504.f));
}

__device__ unsigned pack_half2(__half a, __half b) {
    unsigned low = __half_as_ushort(a);
    unsigned high = __half_as_ushort(b);
    return low | (high << 16);
}

__device__ void mma_16x8x16(float d[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3]));
}

__device__ __half dequant_nibble(const uint8_t* row, int k, float scale) {
    uint8_t byte = row[k >> 1];
    int nibble = (k & 1) ? (byte >> 4) : (byte & 0x0f);
    int code = nibble >= 8 ? nibble - 16 : nibble;
    return __float2half(static_cast<float>(code) * scale);
}

__global__ void gemm_int4_kernel(
    const __half* __restrict__ act,
    const uint8_t* __restrict__ codes,
    const float* __restrict__ scales,
    const __half* __restrict__ bias,
    __half* __restrict__ out,
    int rows,
    int cols,
    int inner) {
    __shared__ __half tile_a[kBM][kBK];
    __shared__ __half tile_b[kBN][kBK];

    int block_m = blockIdx.y * kBM;
    int block_n = blockIdx.x * kBN;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int tid = threadIdx.x;

    float acc[kNTiles][4];
#pragma unroll
    for (int tile = 0; tile < kNTiles; ++tile) {
#pragma unroll
        for (int lane_out = 0; lane_out < 4; ++lane_out) {
            acc[tile][lane_out] = 0.f;
        }
    }

    int packed_stride = inner >> 1;
    for (int k0 = 0; k0 < inner; k0 += kBK) {
        for (int item = tid; item < kBM * kBK; item += blockDim.x) {
            int row = item / kBK;
            int col = item - row * kBK;
            int global_m = block_m + row;
            int global_k = k0 + col;
            __half value = __float2half(0.f);
            if (global_m < rows && global_k < inner) {
                value = act[static_cast<long long>(global_m) * inner + global_k];
            }
            tile_a[row][col] = value;
        }
        for (int item = tid; item < kBN * kBK; item += blockDim.x) {
            int row = item / kBK;
            int col = item - row * kBK;
            int global_n = block_n + row;
            int global_k = k0 + col;
            __half value = __float2half(0.f);
            if (global_n < cols && global_k < inner) {
                const uint8_t* weight = codes + static_cast<long long>(global_n) * packed_stride;
                value = dequant_nibble(weight, global_k, scales[global_n]);
            }
            tile_b[row][col] = value;
        }
        __syncthreads();

        int row_base = warp * 16;
#pragma unroll
        for (int kk = 0; kk < kBK; kk += 16) {
            int group = lane >> 2;
            int pair = (lane & 3) * 2;
            unsigned a0 = pack_half2(tile_a[row_base + group][kk + pair], tile_a[row_base + group][kk + pair + 1]);
            unsigned a1 = pack_half2(tile_a[row_base + group][kk + pair + 8], tile_a[row_base + group][kk + pair + 9]);
            unsigned a2 = pack_half2(tile_a[row_base + group + 8][kk + pair], tile_a[row_base + group + 8][kk + pair + 1]);
            unsigned a3 = pack_half2(
                tile_a[row_base + group + 8][kk + pair + 8], tile_a[row_base + group + 8][kk + pair + 9]);
#pragma unroll
            for (int tile = 0; tile < kNTiles; ++tile) {
                int n_base = tile * 8;
                int n_local = lane >> 2;
                int k_local = (lane & 3) * 2;
                unsigned b0 = pack_half2(
                    tile_b[n_base + n_local][kk + k_local], tile_b[n_base + n_local][kk + k_local + 1]);
                unsigned b1 = pack_half2(
                    tile_b[n_base + n_local][kk + k_local + 8], tile_b[n_base + n_local][kk + k_local + 9]);
                mma_16x8x16(acc[tile], a0, a2, a1, a3, b0, b1);
            }
        }
        __syncthreads();
    }

    int row0 = warp * 16 + (lane >> 2);
    int pair = (lane & 3) * 2;
#pragma unroll
    for (int tile = 0; tile < kNTiles; ++tile) {
        int col_base = block_n + tile * 8;
        int rows_out[2] = {row0, row0 + 8};
        int cols_out[2] = {col_base + pair, col_base + pair + 1};
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            int row = block_m + rows_out[slot >> 1];
            int col = cols_out[slot & 1];
            if (row < rows && col < cols) {
                float value = acc[tile][slot];
                if (bias != nullptr) {
                    value += __half2float(bias[col]);
                }
                out[static_cast<long long>(row) * cols + col] = finite_half(value);
            }
        }
    }
}

__global__ void rmsnorm_kernel(const __half* x, const __half* weight, __half* y, int rows, int dim, float eps) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const __half* in = x + static_cast<long long>(row) * dim;
    __half* out = y + static_cast<long long>(row) * dim;
    float sum = 0.f;
    for (int col = threadIdx.x; col < dim; col += blockDim.x) {
        float value = __half2float(in[col]);
        sum += value * value;
    }
    __shared__ float share[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) {
        share[warp] = sum;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.f;
        int warps = (blockDim.x + 31) >> 5;
        for (int index = 0; index < warps; ++index) {
            total += share[index];
        }
        share[0] = rsqrtf(total / static_cast<float>(dim) + eps);
    }
    __syncthreads();
    float scale = share[0];
    for (int col = threadIdx.x; col < dim; col += blockDim.x) {
        float value = __half2float(in[col]) * scale;
        if (weight != nullptr) {
            value *= __half2float(weight[col]);
        }
        out[col] = finite_half(value);
    }
}

__global__ void layernorm_kernel(const __half* x, __half* y, int rows, int dim, float eps) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const __half* in = x + static_cast<long long>(row) * dim;
    __half* out = y + static_cast<long long>(row) * dim;
    float sum = 0.f;
    float squares = 0.f;
    for (int col = threadIdx.x; col < dim; col += blockDim.x) {
        float value = __half2float(in[col]);
        sum += value;
        squares += value * value;
    }
    __shared__ float share[64];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    float sum_w = sum;
    float sq_w = squares;
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_w += __shfl_down_sync(0xffffffff, sum_w, offset);
        sq_w += __shfl_down_sync(0xffffffff, sq_w, offset);
    }
    if (lane == 0) {
        share[warp] = sum_w;
        share[32 + warp] = sq_w;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.f;
        float total_sq = 0.f;
        int warps = (blockDim.x + 31) >> 5;
        for (int index = 0; index < warps; ++index) {
            total += share[index];
            total_sq += share[32 + index];
        }
        float mean = total / static_cast<float>(dim);
        float var = total_sq / static_cast<float>(dim) - mean * mean;
        share[0] = mean;
        share[1] = rsqrtf(var + eps);
    }
    __syncthreads();
    float mean = share[0];
    float inv = share[1];
    for (int col = threadIdx.x; col < dim; col += blockDim.x) {
        out[col] = finite_half((__half2float(in[col]) - mean) * inv);
    }
}

__global__ void silu_kernel(const __half* x, __half* y, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    float value = __half2float(x[index]);
    float gate = value / (1.f + expf(-value));
    y[index] = __float2half(gate);
}

__global__ void swiglu_kernel(const __half* x, __half* y, int rows, int cols) {
    int half = cols >> 1;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int count = rows * half;
    if (index >= count) {
        return;
    }
    int row = index / half;
    int col = index - row * half;
    const __half* in = x + static_cast<long long>(row) * cols;
    float gate = __half2float(in[col]);
    float up = __half2float(in[half + col]);
    y[index] = finite_half((gate / (1.f + expf(-gate))) * up);
}

__global__ void silu_mul_kernel(const __half* gate, const __half* up, __half* y, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    float value = __half2float(gate[index]);
    y[index] = finite_half((value / (1.f + expf(-value))) * __half2float(up[index]));
}

__global__ void modulate_kernel(const __half* x, const __half* shift, const __half* scale, __half* y, int rows, int dim) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int count = rows * dim;
    if (index >= count) {
        return;
    }
    int col = index % dim;
    float value = __half2float(x[index]);
    float shifted = (1.f + __half2float(scale[col])) * value + __half2float(shift[col]);
    y[index] = finite_half(shifted);
}

__global__ void gate_kernel(
    const __half* residual, const __half* gate, const __half* branch, __half* y, int rows, int dim) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int count = rows * dim;
    if (index >= count) {
        return;
    }
    float base = __half2float(residual[index]);
    float extra = __half2float(branch[index]);
    if (gate != nullptr) {
        extra *= __half2float(gate[index % dim]);
    }
    y[index] = finite_half(base + extra);
}

__global__ void rope_text_kernel(
    const __half* x, const float* cos, const float* sin, __half* y, int seq, int heads, int dim) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int count = seq * heads;
    if (index >= count) {
        return;
    }
    int token = index / heads;
    int half = dim >> 1;
    const __half* in = x + static_cast<long long>(index) * dim;
    __half* out = y + static_cast<long long>(index) * dim;
    const float* cos_row = cos + static_cast<long long>(token) * dim;
    const float* sin_row = sin + static_cast<long long>(token) * dim;
    for (int col = 0; col < half; ++col) {
        float first = __half2float(in[col]);
        float second = __half2float(in[half + col]);
        out[col] = __float2half(first * cos_row[col] - second * sin_row[col]);
        out[half + col] = __float2half(second * cos_row[half + col] + first * sin_row[half + col]);
    }
}

__global__ void rope_dit_kernel(
    const __half* x, const float* cos, const float* sin, __half* y, int seq, int heads, int dim) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int count = seq * heads;
    if (index >= count) {
        return;
    }
    int token = index / heads;
    const __half* in = x + static_cast<long long>(index) * dim;
    __half* out = y + static_cast<long long>(index) * dim;
    const float* cos_row = cos + static_cast<long long>(token) * dim;
    const float* sin_row = sin + static_cast<long long>(token) * dim;
    for (int col = 0; col < dim; col += 2) {
        float first = __half2float(in[col]);
        float second = __half2float(in[col + 1]);
        float pair_cos = cos_row[col];
        float pair_sin = sin_row[col];
        out[col] = __float2half(first * pair_cos - second * pair_sin);
        out[col + 1] = __float2half(first * pair_sin + second * pair_cos);
    }
}

// Head dim 128. One block keeps a query tile, a key tile, and that pair's score tile.
__global__ void attend_mma(
    const __half* q,
    const __half* k,
    const __half* v,
    __half* y,
    int seq,
    int heads,
    int kv_heads,
    int causal,
    int valid,
    float scale) {
    constexpr int kRows = 64;
    constexpr int kKeys = 32;
    constexpr int kDim = 128;
    __shared__ __half q_s[kRows][kDim];
    __shared__ __half k_s[kKeys][kDim];
    __shared__ __half v_s[kDim][kKeys];
    __shared__ __half p_s[kRows][kKeys];
    __shared__ float s_s[kRows][kKeys];
    __shared__ float row_max[kRows];
    __shared__ float row_sum[kRows];
    __shared__ float row_alpha[kRows];

    int head = blockIdx.y;
    int kv_head = head / (heads / kv_heads);
    int q_base = blockIdx.x * kRows;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row_base = warp * 16;

    for (int item = threadIdx.x; item < kRows * kDim; item += blockDim.x) {
        int row = item / kDim;
        int col = item - row * kDim;
        int query = q_base + row;
        __half value = __float2half(0.f);
        if (query < seq) {
            value = q[(static_cast<long long>(query) * heads + head) * kDim + col];
        }
        q_s[row][col] = value;
    }
    if (threadIdx.x < kRows) {
        row_max[threadIdx.x] = -1.0e30f;
        row_sum[threadIdx.x] = 0.f;
    }
    float acc[16][4];
#pragma unroll
    for (int tile = 0; tile < 16; ++tile) {
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            acc[tile][slot] = 0.f;
        }
    }
    __syncthreads();

    for (int tile = 0; tile < seq; tile += kKeys) {
        for (int item = threadIdx.x; item < kKeys * kDim; item += blockDim.x) {
            int row = item / kDim;
            int col = item - row * kDim;
            int key = tile + row;
            __half key_value = __float2half(0.f);
            __half value_value = __float2half(0.f);
            if (key < seq) {
                long long base = (static_cast<long long>(key) * kv_heads + kv_head) * kDim + col;
                key_value = k[base];
                value_value = v[base];
            }
            k_s[row][col] = key_value;
            v_s[col][row] = value_value;
        }
        __syncthreads();

        float score[4][4];
#pragma unroll
        for (int ntile = 0; ntile < 4; ++ntile) {
#pragma unroll
            for (int slot = 0; slot < 4; ++slot) {
                score[ntile][slot] = 0.f;
            }
        }
#pragma unroll
        for (int kk = 0; kk < kDim; kk += 16) {
            int group = lane >> 2;
            int pair = (lane & 3) * 2;
            unsigned a0 = pack_half2(q_s[row_base + group][kk + pair], q_s[row_base + group][kk + pair + 1]);
            unsigned a1 = pack_half2(q_s[row_base + group][kk + pair + 8], q_s[row_base + group][kk + pair + 9]);
            unsigned a2 = pack_half2(q_s[row_base + group + 8][kk + pair], q_s[row_base + group + 8][kk + pair + 1]);
            unsigned a3 = pack_half2(
                q_s[row_base + group + 8][kk + pair + 8], q_s[row_base + group + 8][kk + pair + 9]);
#pragma unroll
            for (int ntile = 0; ntile < 4; ++ntile) {
                int n_base = ntile * 8;
                int n_local = lane >> 2;
                int k_local = (lane & 3) * 2;
                unsigned b0 = pack_half2(
                    k_s[n_base + n_local][kk + k_local], k_s[n_base + n_local][kk + k_local + 1]);
                unsigned b1 = pack_half2(
                    k_s[n_base + n_local][kk + k_local + 8], k_s[n_base + n_local][kk + k_local + 9]);
                mma_16x8x16(score[ntile], a0, a2, a1, a3, b0, b1);
            }
        }

        int row0 = row_base + (lane >> 2);
        int pair = (lane & 3) * 2;
#pragma unroll
        for (int ntile = 0; ntile < 4; ++ntile) {
            int col_base = ntile * 8;
            int rows_out[2] = {row0, row0 + 8};
            int cols_out[2] = {col_base + pair, col_base + pair + 1};
#pragma unroll
            for (int slot = 0; slot < 4; ++slot) {
                int row = rows_out[slot >> 1];
                int col = cols_out[slot & 1];
                int query = q_base + row;
                int key = tile + col;
                float value = score[ntile][slot] * scale;
                int live = query < seq && key < seq && key < valid && (!causal || key <= query);
                s_s[row][col] = live ? value : -1.0e30f;
            }
        }
        __syncthreads();

        if (threadIdx.x < kRows) {
            int row = threadIdx.x;
            float peak = -1.0e30f;
            for (int col = 0; col < kKeys; ++col) {
                peak = fmaxf(peak, s_s[row][col]);
            }
            if (peak < -1.0e20f) {
                row_alpha[row] = 1.f;
                for (int col = 0; col < kKeys; ++col) {
                    p_s[row][col] = __float2half(0.f);
                }
            } else {
                float next = fmaxf(row_max[row], peak);
                float alpha = __expf(row_max[row] - next);
                float sum = 0.f;
                for (int col = 0; col < kKeys; ++col) {
                    float weight = s_s[row][col] < -1.0e20f ? 0.f : __expf(s_s[row][col] - next);
                    p_s[row][col] = __float2half(weight);
                    sum += weight;
                }
                row_alpha[row] = alpha;
                row_sum[row] = row_sum[row] * alpha + sum;
                row_max[row] = next;
            }
        }
        __syncthreads();

#pragma unroll
        for (int ntile = 0; ntile < 16; ++ntile) {
            float top = row_alpha[row0];
            float bottom = row_alpha[row0 + 8];
            acc[ntile][0] *= top;
            acc[ntile][1] *= top;
            acc[ntile][2] *= bottom;
            acc[ntile][3] *= bottom;
        }
#pragma unroll
        for (int kk = 0; kk < kKeys; kk += 16) {
            int group = lane >> 2;
            int pair_k = (lane & 3) * 2;
            unsigned a0 = pack_half2(p_s[row_base + group][kk + pair_k], p_s[row_base + group][kk + pair_k + 1]);
            unsigned a1 = pack_half2(p_s[row_base + group][kk + pair_k + 8], p_s[row_base + group][kk + pair_k + 9]);
            unsigned a2 = pack_half2(p_s[row_base + group + 8][kk + pair_k], p_s[row_base + group + 8][kk + pair_k + 1]);
            unsigned a3 = pack_half2(
                p_s[row_base + group + 8][kk + pair_k + 8], p_s[row_base + group + 8][kk + pair_k + 9]);
#pragma unroll
            for (int ntile = 0; ntile < 16; ++ntile) {
                int n_base = ntile * 8;
                int n_local = lane >> 2;
                int k_local = (lane & 3) * 2;
                unsigned b0 = pack_half2(
                    v_s[n_base + n_local][kk + k_local], v_s[n_base + n_local][kk + k_local + 1]);
                unsigned b1 = pack_half2(
                    v_s[n_base + n_local][kk + k_local + 8], v_s[n_base + n_local][kk + k_local + 9]);
                mma_16x8x16(acc[ntile], a0, a2, a1, a3, b0, b1);
            }
        }
        __syncthreads();
    }

    int row0 = row_base + (lane >> 2);
    int pair = (lane & 3) * 2;
    float inv_top = row_sum[row0] > 0.f ? 1.f / row_sum[row0] : 0.f;
    float inv_bottom = row_sum[row0 + 8] > 0.f ? 1.f / row_sum[row0 + 8] : 0.f;
#pragma unroll
    for (int ntile = 0; ntile < 16; ++ntile) {
        int col_base = ntile * 8;
        int rows_out[2] = {row0, row0 + 8};
        int cols_out[2] = {col_base + pair, col_base + pair + 1};
        float inv[2] = {inv_top, inv_bottom};
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            int row = rows_out[slot >> 1];
            int col = cols_out[slot & 1];
            int query = q_base + row;
            if (query < seq && col < kDim) {
                __half* out = y + (static_cast<long long>(query) * heads + head) * kDim;
                out[col] = finite_half(acc[ntile][slot] * inv[slot >> 1]);
            }
        }
    }
}

template <int DIM, int BR, int BC, int PARTS>
__global__ void attend_fast(
    const __half* q,
    const __half* k,
    const __half* v,
    __half* y,
    int seq,
    int heads,
    int kv_heads,
    int causal,
    int valid,
    float scale) {
    constexpr int kSpan = DIM / PARTS;
    __shared__ __half q_s[DIM][BR];
    __shared__ __half k_s[DIM][BC];
    __shared__ __half v_s[DIM][BC];
    __shared__ float acc_s[DIM][BR];

    int head = blockIdx.y;
    int kv_head = head / (heads / kv_heads);
    int q_base = blockIdx.x * BR;
    int local = threadIdx.x / PARTS;
    int part = threadIdx.x - local * PARTS;
    int q_index = q_base + local;
    int d0 = part * kSpan;
    int lane = threadIdx.x & 31;
    int leader = lane & ~(PARTS - 1);

    for (int i = threadIdx.x; i < BR * DIM; i += blockDim.x) {
        int row = i / DIM;
        int col = i - row * DIM;
        int query = q_base + row;
        __half value = __float2half(0.f);
        if (query < seq) {
            value = q[(static_cast<long long>(query) * heads + head) * DIM + col];
        }
        q_s[col][row] = value;
        acc_s[col][row] = 0.f;
    }
    __syncthreads();

    float running_max = -1.0e30f;
    float running_sum = 0.f;
    for (int tile = 0; tile < seq; tile += BC) {
        for (int i = threadIdx.x; i < BC * DIM; i += blockDim.x) {
            int row = i / DIM;
            int col = i - row * DIM;
            int key = tile + row;
            __half key_value = __float2half(0.f);
            __half value_value = __float2half(0.f);
            if (key < seq) {
                long long base = (static_cast<long long>(key) * kv_heads + kv_head) * DIM + col;
                key_value = k[base];
                value_value = v[base];
            }
            k_s[col][row] = key_value;
            v_s[col][row] = value_value;
        }
        __syncthreads();
        int limit = seq - tile;
        if (limit > BC) {
            limit = BC;
        }
        for (int row = 0; row < limit; ++row) {
            int key = tile + row;
            int live = q_index < seq && key < valid && (!causal || key <= q_index);
            float partial = 0.f;
            if (live) {
#pragma unroll 8
                for (int col = 0; col < kSpan; ++col) {
                    partial = fmaf(
                        __half2float(q_s[d0 + col][local]), __half2float(k_s[d0 + col][row]), partial);
                }
            }
#pragma unroll
            for (int offset = PARTS / 2; offset > 0; offset >>= 1) {
                partial += __shfl_xor_sync(0xffffffff, partial, offset);
            }
            float alpha = 1.f;
            float prob = 0.f;
            if (part == 0 && live) {
                float score = partial * scale;
                float next = fmaxf(running_max, score);
                alpha = __expf(running_max - next);
                prob = __expf(score - next);
                running_sum = running_sum * alpha + prob;
                running_max = next;
            }
            alpha = __shfl_sync(0xffffffff, alpha, leader);
            prob = __shfl_sync(0xffffffff, prob, leader);
            if (live) {
#pragma unroll 8
                for (int col = 0; col < kSpan; ++col) {
                    acc_s[d0 + col][local] =
                        acc_s[d0 + col][local] * alpha + prob * __half2float(v_s[d0 + col][row]);
                }
            }
        }
        __syncthreads();
    }

    float inv = 0.f;
    if (part == 0 && running_sum > 0.f) {
        inv = 1.f / running_sum;
    }
    inv = __shfl_sync(0xffffffff, inv, leader);
    if (q_index < seq) {
        __half* row = y + (static_cast<long long>(q_index) * heads + head) * DIM;
        for (int col = 0; col < kSpan; ++col) {
            row[d0 + col] = __float2half(acc_s[d0 + col][local] * inv);
        }
    }
}

__global__ void slice_cols_kernel(const __half* x, __half* y, int rows, int cols, int start, int width) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int count = rows * width;
    if (index >= count) {
        return;
    }
    int row = index / width;
    int col = index - row * width;
    y[index] = x[static_cast<long long>(row) * cols + start + col];
}

__global__ void pad_k_kernel(const __half* x, __half* y, int rows, int cols, int stored) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int count = rows * stored;
    if (index >= count) {
        return;
    }
    int row = index / stored;
    int col = index - row * stored;
    __half value = __float2half(0.f);
    if (col < cols) {
        value = x[static_cast<long long>(row) * cols + col];
    }
    y[index] = value;
}

__global__ void embed_kernel(
    const int* ids, const uint8_t* codes, const float* scales, __half* y, int seq, int inner) {
    int token = blockIdx.x;
    int col = threadIdx.x + blockIdx.y * blockDim.x;
    if (token >= seq || col >= inner) {
        return;
    }
    int row = ids[token];
    int packed = inner >> 1;
    const uint8_t* weight = codes + static_cast<long long>(row) * packed;
    y[static_cast<long long>(token) * inner + col] = dequant_nibble(weight, col, scales[row]);
}

__global__ void euler_kernel(const __half* x, const __half* velocity, __half* y, int count, float delta) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    y[index] = __float2half(__half2float(x[index]) + delta * __half2float(velocity[index]));
}

__global__ void add_kernel(const __half* a, const __half* b, __half* y, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    y[index] = __float2half(__half2float(a[index]) + __half2float(b[index]));
}

__global__ void add_f32_kernel(const float* a, const float* b, float* y, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    y[index] = a[index] + b[index];
}

__global__ void cast_f16_f32_kernel(const __half* x, float* y, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    y[index] = __half2float(x[index]);
}

__global__ void cast_f32_f16_kernel(const float* x, __half* y, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    y[index] = __float2half(x[index]);
}

template <typename T>
__device__ float load_t(const T* pointer) {
    return static_cast<float>(*pointer);
}

template <>
__device__ float load_t<__half>(const __half* pointer) {
    return __half2float(*pointer);
}

template <typename T>
__device__ void store_t(T* pointer, float value) {
    *pointer = static_cast<T>(value);
}

template <>
__device__ void store_t<__half>(__half* pointer, float value) {
    *pointer = __float2half(value);
}

template <typename T>
__global__ void group_stats_kernel(const T* x, float* mean, float* inv, int channels, int groups, int spatial, float eps) {
    int group = blockIdx.x;
    int width = channels / groups;
    long long count = static_cast<long long>(width) * spatial;
    double sum = 0.0;
    double squares = 0.0;
    for (long long index = threadIdx.x; index < count; index += blockDim.x) {
        int channel = static_cast<int>(index / spatial);
        int pixel = static_cast<int>(index - static_cast<long long>(channel) * spatial);
        int global_c = group * width + channel;
        float value = load_t(x + (static_cast<long long>(global_c) * spatial + pixel));
        sum += value;
        squares += static_cast<double>(value) * value;
    }
    __shared__ double share[64];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
        squares += __shfl_down_sync(0xffffffff, squares, offset);
    }
    if (lane == 0) {
        share[warp] = sum;
        share[32 + warp] = squares;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        double total = 0.0;
        double total_sq = 0.0;
        int warps = (blockDim.x + 31) >> 5;
        for (int index = 0; index < warps; ++index) {
            total += share[index];
            total_sq += share[32 + index];
        }
        double avg = total / static_cast<double>(count);
        double var = total_sq / static_cast<double>(count) - avg * avg;
        mean[group] = static_cast<float>(avg);
        inv[group] = rsqrtf(static_cast<float>(var) + eps);
    }
}

template <typename T>
__global__ void group_apply_kernel(
    const T* x,
    const __half* weight,
    const __half* bias,
    T* y,
    const float* mean,
    const float* inv,
    int channels,
    int groups,
    int spatial,
    int silu) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    long long count = static_cast<long long>(channels) * spatial;
    if (index >= count) {
        return;
    }
    int channel = static_cast<int>(index / spatial);
    int group = channel / (channels / groups);
    float value = (load_t(x + index) - mean[group]) * inv[group];
    if (weight != nullptr) {
        value = value * __half2float(weight[channel]) + __half2float(bias[channel]);
    }
    if (silu) {
        value = value / (1.f + expf(-value));
    }
    store_t(y + index, value);
}

__global__ void im2col_kernel(
    const __half* x,
    __half* columns,
    int channels,
    int height,
    int width,
    int kernel,
    int stride,
    int pad,
    int out_h,
    int out_w,
    int stored_k) {
    int spatial = out_h * out_w;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int logical = channels * kernel * kernel;
    int count = spatial * stored_k;
    if (index >= count) {
        return;
    }
    int pixel = index / stored_k;
    int col = index - pixel * stored_k;
    __half value = __float2half(0.f);
    if (col < logical) {
        int window = kernel * kernel;
        int cin = col / window;
        int rest = col - cin * window;
        int ky = rest / kernel;
        int kx = rest - ky * kernel;
        int oh = pixel / out_w;
        int ow = pixel - oh * out_w;
        int ih = oh * stride + ky - pad;
        int iw = ow * stride + kx - pad;
        if (ih >= 0 && iw >= 0 && ih < height && iw < width) {
            value = x[(static_cast<long long>(cin) * height + ih) * width + iw];
        }
    }
    columns[index] = value;
}

__global__ void im2col_f32_kernel(
    const float* x,
    __half* columns,
    int channels,
    int height,
    int width,
    int kernel,
    int stride,
    int pad,
    int out_h,
    int out_w,
    int stored_k) {
    int spatial = out_h * out_w;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int logical = channels * kernel * kernel;
    int count = spatial * stored_k;
    if (index >= count) {
        return;
    }
    int pixel = index / stored_k;
    int col = index - pixel * stored_k;
    __half value = __float2half(0.f);
    if (col < logical) {
        int window = kernel * kernel;
        int cin = col / window;
        int rest = col - cin * window;
        int ky = rest / kernel;
        int kx = rest - ky * kernel;
        int oh = pixel / out_w;
        int ow = pixel - oh * out_w;
        int ih = oh * stride + ky - pad;
        int iw = ow * stride + kx - pad;
        if (ih >= 0 && iw >= 0 && ih < height && iw < width) {
            value = __float2half(x[(static_cast<long long>(cin) * height + ih) * width + iw]);
        }
    }
    columns[index] = value;
}

__global__ void col2im_kernel(const __half* columns, __half* y, int channels, int height, int width) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int spatial = height * width;
    int count = channels * spatial;
    if (index >= count) {
        return;
    }
    int channel = index / spatial;
    int pixel = index - channel * spatial;
    y[index] = columns[static_cast<long long>(pixel) * channels + channel];
}

__global__ void col2im_f32_kernel(const __half* columns, float* y, int channels, int height, int width) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int spatial = height * width;
    int count = channels * spatial;
    if (index >= count) {
        return;
    }
    int channel = index / spatial;
    int pixel = index - channel * spatial;
    y[index] = __half2float(columns[static_cast<long long>(pixel) * channels + channel]);
}

__global__ void pad_edge_kernel(
    const __half* x, __half* y, int channels, int height, int width, int pad_b, int pad_r) {
    int out_h = height + pad_b;
    int out_w = width + pad_r;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int count = channels * out_h * out_w;
    if (index >= count) {
        return;
    }
    int channel = index / (out_h * out_w);
    int rest = index - channel * out_h * out_w;
    int row = rest / out_w;
    int col = rest - row * out_w;
    __half value = __float2half(0.f);
    if (row < height && col < width) {
        value = x[(static_cast<long long>(channel) * height + row) * width + col];
    }
    y[index] = value;
}

__global__ void pad_edge_f32_kernel(
    const float* x, float* y, int channels, int height, int width, int pad_b, int pad_r) {
    int out_h = height + pad_b;
    int out_w = width + pad_r;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int count = channels * out_h * out_w;
    if (index >= count) {
        return;
    }
    int channel = index / (out_h * out_w);
    int rest = index - channel * out_h * out_w;
    int row = rest / out_w;
    int col = rest - row * out_w;
    float value = 0.f;
    if (row < height && col < width) {
        value = x[(static_cast<long long>(channel) * height + row) * width + col];
    }
    y[index] = value;
}

__global__ void nearest2_kernel(const __half* x, __half* y, int channels, int height, int width) {
    int out_h = height * 2;
    int out_w = width * 2;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int count = channels * out_h * out_w;
    if (index >= count) {
        return;
    }
    int channel = index / (out_h * out_w);
    int rest = index - channel * out_h * out_w;
    int row = rest / out_w;
    int col = rest - row * out_w;
    y[index] = x[(static_cast<long long>(channel) * height + (row >> 1)) * width + (col >> 1)];
}

__global__ void nearest2_f32_kernel(const float* x, float* y, int channels, int height, int width) {
    int out_h = height * 2;
    int out_w = width * 2;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int count = channels * out_h * out_w;
    if (index >= count) {
        return;
    }
    int channel = index / (out_h * out_w);
    int rest = index - channel * out_h * out_w;
    int row = rest / out_w;
    int col = rest - row * out_w;
    y[index] = x[(static_cast<long long>(channel) * height + (row >> 1)) * width + (col >> 1)];
}

__global__ void prefix_channels_kernel(const __half* x, __half* y, int keep, int channels, int spatial) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int count = keep * spatial;
    if (index >= count) {
        return;
    }
    int channel = index / spatial;
    int pixel = index - channel * spatial;
    y[index] = x[static_cast<long long>(channel) * spatial + pixel];
    (void)channels;
}

__global__ void pack_latent_kernel(
    const __half* mean, const float* running_mean, const float* running_var, __half* tokens, int grid_h, int grid_w, float eps) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int count = grid_h * grid_w * 128;
    if (index >= count) {
        return;
    }
    int token = index / 128;
    int channel = index - token * 128;
    int row = token / grid_w;
    int col = token - row * grid_w;
    int source_c = channel >> 2;
    int patch = channel & 3;
    int py = patch >> 1;
    int px = patch & 1;
    int height = grid_h * 2;
    int width = grid_w * 2;
    float value = __half2float(mean[(static_cast<long long>(source_c) * height + (row * 2 + py)) * width + (col * 2 + px)]);
    float whitened = (value - running_mean[channel]) * rsqrtf(running_var[channel] + eps);
    tokens[index] = __float2half(whitened);
}

__global__ void unpack_latent_kernel(
    const __half* tokens, const float* running_mean, const float* running_var, float* volume, int grid_h, int grid_w, float eps) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int height = grid_h * 2;
    int width = grid_w * 2;
    int count = 32 * height * width;
    if (index >= count) {
        return;
    }
    int channel = index / (height * width);
    int rest = index - channel * height * width;
    int row = rest / width;
    int col = rest - row * width;
    int token_row = row >> 1;
    int token_col = col >> 1;
    int py = row & 1;
    int px = col & 1;
    int packed = channel * 4 + py * 2 + px;
    int token = token_row * grid_w + token_col;
    float whitened = __half2float(tokens[token * 128 + packed]);
    volume[index] = whitened * sqrtf(running_var[packed] + eps) + running_mean[packed];
}

template <typename In, typename Out>
__global__ void conv_int4_kernel(
    const In* volume,
    const uint8_t* codes,
    const float* scales,
    const __half* bias,
    Out* out,
    int channels,
    int height,
    int width,
    int kernel,
    int stride,
    int pad,
    int out_h,
    int out_w,
    int out_c,
    int logical_k,
    int stored_k) {
    __shared__ __half tile_a[kBM][kBK];
    __shared__ __half tile_b[kBN][kBK];

    int spatial = out_h * out_w;
    int block_m = blockIdx.y * kBM;
    int block_n = blockIdx.x * kBN;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int tid = threadIdx.x;
    int window = kernel * kernel;
    int packed_stride = stored_k >> 1;

    float acc[kNTiles][4];
#pragma unroll
    for (int tile = 0; tile < kNTiles; ++tile) {
#pragma unroll
        for (int lane_out = 0; lane_out < 4; ++lane_out) {
            acc[tile][lane_out] = 0.f;
        }
    }

    for (int k0 = 0; k0 < stored_k; k0 += kBK) {
        for (int item = tid; item < kBM * kBK; item += blockDim.x) {
            int row = item / kBK;
            int col = item - row * kBK;
            int global_m = block_m + row;
            int global_k = k0 + col;
            __half value = __float2half(0.f);
            if (global_m < spatial && global_k < logical_k) {
                int oh = global_m / out_w;
                int ow = global_m - oh * out_w;
                int cin = global_k / window;
                int rest = global_k - cin * window;
                int ky = rest / kernel;
                int kx = rest - ky * kernel;
                int ih = oh * stride + ky - pad;
                int iw = ow * stride + kx - pad;
                if (ih >= 0 && iw >= 0 && ih < height && iw < width && cin < channels) {
                    value = __float2half(load_t(volume + (static_cast<long long>(cin) * height + ih) * width + iw));
                }
            }
            tile_a[row][col] = value;
        }
        for (int item = tid; item < kBN * kBK; item += blockDim.x) {
            int row = item / kBK;
            int col = item - row * kBK;
            int global_n = block_n + row;
            int global_k = k0 + col;
            __half value = __float2half(0.f);
            if (global_n < out_c && global_k < stored_k) {
                const uint8_t* weight = codes + static_cast<long long>(global_n) * packed_stride;
                value = dequant_nibble(weight, global_k, scales[global_n]);
            }
            tile_b[row][col] = value;
        }
        __syncthreads();

        int row_base = warp * 16;
#pragma unroll
        for (int kk = 0; kk < kBK; kk += 16) {
            int group = lane >> 2;
            int pair = (lane & 3) * 2;
            unsigned a0 = pack_half2(tile_a[row_base + group][kk + pair], tile_a[row_base + group][kk + pair + 1]);
            unsigned a1 = pack_half2(tile_a[row_base + group][kk + pair + 8], tile_a[row_base + group][kk + pair + 9]);
            unsigned a2 = pack_half2(tile_a[row_base + group + 8][kk + pair], tile_a[row_base + group + 8][kk + pair + 1]);
            unsigned a3 = pack_half2(
                tile_a[row_base + group + 8][kk + pair + 8], tile_a[row_base + group + 8][kk + pair + 9]);
#pragma unroll
            for (int tile = 0; tile < kNTiles; ++tile) {
                int n_base = tile * 8;
                int n_local = lane >> 2;
                int k_local = (lane & 3) * 2;
                unsigned b0 = pack_half2(
                    tile_b[n_base + n_local][kk + k_local], tile_b[n_base + n_local][kk + k_local + 1]);
                unsigned b1 = pack_half2(
                    tile_b[n_base + n_local][kk + k_local + 8], tile_b[n_base + n_local][kk + k_local + 9]);
                mma_16x8x16(acc[tile], a0, a2, a1, a3, b0, b1);
            }
        }
        __syncthreads();
    }

    int row0 = warp * 16 + (lane >> 2);
    int pair = (lane & 3) * 2;
#pragma unroll
    for (int tile = 0; tile < kNTiles; ++tile) {
        int col_base = block_n + tile * 8;
        int rows_out[2] = {row0, row0 + 8};
        int cols_out[2] = {col_base + pair, col_base + pair + 1};
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            int row = block_m + rows_out[slot >> 1];
            int col = cols_out[slot & 1];
            if (row < spatial && col < out_c) {
                float value = acc[tile][slot];
                if (bias != nullptr) {
                    value += __half2float(bias[col]);
                }
                int oh = row / out_w;
                int ow = row - oh * out_w;
                store_t(out + (static_cast<long long>(col) * out_h + oh) * out_w + ow, value);
            }
        }
    }
}

__global__ void concat_cols_kernel(
    const __half* left, const __half* right, __half* out, int rows, int left_cols, int right_cols) {
    int width = left_cols + right_cols;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int count = rows * width;
    if (index >= count) {
        return;
    }
    int row = index / width;
    int col = index - row * width;
    __half value;
    if (col < left_cols) {
        value = left[static_cast<long long>(row) * left_cols + col];
    } else {
        value = right[static_cast<long long>(row) * right_cols + (col - left_cols)];
    }
    out[index] = value;
}

template <typename In, typename Out>
__global__ void transpose_kernel(const In* x, Out* y, int channels, int spatial, int to_tokens) {
    long long count = static_cast<long long>(channels) * spatial;
    long long index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    int channel = static_cast<int>(index / spatial);
    int pixel = static_cast<int>(index - static_cast<long long>(channel) * spatial);
    long long token = static_cast<long long>(pixel) * channels + channel;
    if (to_tokens) {
        store_t(y + token, load_t(x + index));
    } else {
        store_t(y + index, load_t(x + token));
    }
}

int launch_ok(const char* what) {
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
        return static_cast<int>(status);
    }
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s sync: %s\n", what, cudaGetErrorString(status));
        return static_cast<int>(status);
    }
    return 0;
}

}  // namespace

extern "C" {

__declspec(dllexport) int klein_alloc(unsigned long long bytes, void** out) {
    cudaError_t status = cudaMalloc(out, static_cast<size_t>(bytes));
    check(status, "cudaMalloc");
    return static_cast<int>(status);
}

__declspec(dllexport) int klein_free(void* pointer) {
    if (pointer == nullptr) {
        return 0;
    }
    cudaError_t status = cudaFree(pointer);
    check(status, "cudaFree");
    return static_cast<int>(status);
}

__declspec(dllexport) int klein_copy_h2d(void* dst, const void* src, unsigned long long bytes) {
    cudaError_t status = cudaMemcpy(dst, src, static_cast<size_t>(bytes), cudaMemcpyHostToDevice);
    check(status, "h2d");
    return static_cast<int>(status);
}

__declspec(dllexport) int klein_copy_d2h(void* dst, const void* src, unsigned long long bytes) {
    cudaError_t status = cudaMemcpy(dst, src, static_cast<size_t>(bytes), cudaMemcpyDeviceToHost);
    check(status, "d2h");
    return static_cast<int>(status);
}

__declspec(dllexport) int klein_copy_d2d(void* dst, const void* src, unsigned long long bytes) {
    cudaError_t status = cudaMemcpy(dst, src, static_cast<size_t>(bytes), cudaMemcpyDeviceToDevice);
    check(status, "d2d");
    return static_cast<int>(status);
}

__declspec(dllexport) int klein_sync(void) {
    cudaError_t status = cudaDeviceSynchronize();
    check(status, "sync");
    return static_cast<int>(status);
}

__declspec(dllexport) int klein_zero(void* pointer, unsigned long long bytes) {
    cudaError_t status = cudaMemset(pointer, 0, static_cast<size_t>(bytes));
    check(status, "memset");
    return static_cast<int>(status);
}

__global__ void dequant_kernel(
    const uint8_t* codes, const float* scales, __half* y, int rows, int inner) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int count = rows * inner;
    if (index >= count) {
        return;
    }
    int row = index / inner;
    int col = index - row * inner;
    y[index] = dequant_nibble(codes + static_cast<long long>(row) * (inner >> 1), col, scales[row]);
}

__global__ void add_bias_rows(__half* out, const __half* bias, int count, int cols) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    float value = __half2float(out[index]) + __half2float(bias[index % cols]);
    out[index] = finite_half(value);
}

static cublasHandle_t g_blas = nullptr;
static __half* g_weight = nullptr;
static size_t g_weight_bytes = 0;

static int blas_product(
    const __half* act,
    const uint8_t* codes,
    const float* scales,
    const __half* bias,
    __half* out,
    int rows,
    int cols,
    int inner) {
    size_t bytes = static_cast<size_t>(cols) * static_cast<size_t>(inner) * sizeof(__half);
    if (bytes > g_weight_bytes) {
        if (g_weight != nullptr) {
            cudaFree(g_weight);
            g_weight = nullptr;
            g_weight_bytes = 0;
        }
        if (cudaMalloc(&g_weight, bytes) != cudaSuccess) {
            return 1;
        }
        g_weight_bytes = bytes;
    }
    int count = cols * inner;
    int threads = 256;
    dequant_kernel<<<(count + threads - 1) / threads, threads>>>(codes, scales, g_weight, cols, inner);
    if (g_blas == nullptr && cublasCreate(&g_blas) != CUBLAS_STATUS_SUCCESS) {
        return 1;
    }
    float alpha = 1.f;
    float beta = 0.f;
    cublasStatus_t status = cublasGemmEx(
        g_blas,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        cols,
        rows,
        inner,
        &alpha,
        g_weight,
        CUDA_R_16F,
        inner,
        act,
        CUDA_R_16F,
        inner,
        &beta,
        out,
        CUDA_R_16F,
        cols,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return static_cast<int>(status);
    }
    if (bias != nullptr) {
        int outputs = rows * cols;
        add_bias_rows<<<(outputs + threads - 1) / threads, threads>>>(out, bias, outputs, cols);
    }
    return 0;
}

__declspec(dllexport) int klein_gemm(
    const void* act,
    const void* codes,
    const void* scales,
    const void* bias,
    void* out,
    int rows,
    int cols,
    int inner) {
    if (rows < 1 || cols < 1 || inner < 1) {
        return 0;
    }
    int status = blas_product(
        static_cast<const __half*>(act),
        static_cast<const uint8_t*>(codes),
        static_cast<const float*>(scales),
        static_cast<const __half*>(bias),
        static_cast<__half*>(out),
        rows,
        cols,
        inner);
    if (status != 0) {
        return status;
    }
    return launch_ok("gemm");
}

__declspec(dllexport) int klein_rmsnorm(const void* x, const void* weight, void* y, int rows, int dim, float eps) {
    rmsnorm_kernel<<<rows, 128>>>(
        static_cast<const __half*>(x), static_cast<const __half*>(weight), static_cast<__half*>(y), rows, dim, eps);
    return launch_ok("rmsnorm");
}

__declspec(dllexport) int klein_layernorm(const void* x, void* y, int rows, int dim, float eps) {
    layernorm_kernel<<<rows, 128>>>(static_cast<const __half*>(x), static_cast<__half*>(y), rows, dim, eps);
    return launch_ok("layernorm");
}

__declspec(dllexport) int klein_silu(const void* x, void* y, int count) {
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    silu_kernel<<<blocks, threads>>>(static_cast<const __half*>(x), static_cast<__half*>(y), count);
    return launch_ok("silu");
}

__declspec(dllexport) int klein_swiglu(const void* x, void* y, int rows, int cols) {
    int count = rows * (cols >> 1);
    int threads = 256;
    int blocks = std::max(1, (count + threads - 1) / threads);
    swiglu_kernel<<<blocks, threads>>>(static_cast<const __half*>(x), static_cast<__half*>(y), rows, cols);
    return launch_ok("swiglu");
}

__declspec(dllexport) int klein_silu_mul(const void* gate, const void* up, void* y, int count) {
    int threads = 256;
    int blocks = std::max(1, (count + threads - 1) / threads);
    silu_mul_kernel<<<blocks, threads>>>(
        static_cast<const __half*>(gate), static_cast<const __half*>(up), static_cast<__half*>(y), count);
    return launch_ok("silu_mul");
}

__declspec(dllexport) int klein_modulate(const void* x, const void* shift, const void* scale, void* y, int rows, int dim) {
    int count = rows * dim;
    int threads = 256;
    int blocks = std::max(1, (count + threads - 1) / threads);
    modulate_kernel<<<blocks, threads>>>(
        static_cast<const __half*>(x),
        static_cast<const __half*>(shift),
        static_cast<const __half*>(scale),
        static_cast<__half*>(y),
        rows,
        dim);
    return launch_ok("modulate");
}

__declspec(dllexport) int klein_gate(const void* residual, const void* gate, const void* branch, void* y, int rows, int dim) {
    int count = rows * dim;
    int threads = 256;
    int blocks = std::max(1, (count + threads - 1) / threads);
    gate_kernel<<<blocks, threads>>>(
        static_cast<const __half*>(residual),
        static_cast<const __half*>(gate),
        static_cast<const __half*>(branch),
        static_cast<__half*>(y),
        rows,
        dim);
    return launch_ok("gate");
}

__declspec(dllexport) int klein_rope_text(const void* x, const void* cos, const void* sin, void* y, int seq, int heads, int dim) {
    int count = seq * heads;
    int threads = 128;
    int blocks = std::max(1, (count + threads - 1) / threads);
    rope_text_kernel<<<blocks, threads>>>(
        static_cast<const __half*>(x),
        static_cast<const float*>(cos),
        static_cast<const float*>(sin),
        static_cast<__half*>(y),
        seq,
        heads,
        dim);
    return launch_ok("rope_text");
}

__declspec(dllexport) int klein_rope_dit(const void* x, const void* cos, const void* sin, void* y, int seq, int heads, int dim) {
    int count = seq * heads;
    int threads = 128;
    int blocks = std::max(1, (count + threads - 1) / threads);
    rope_dit_kernel<<<blocks, threads>>>(
        static_cast<const __half*>(x),
        static_cast<const float*>(cos),
        static_cast<const float*>(sin),
        static_cast<__half*>(y),
        seq,
        heads,
        dim);
    return launch_ok("rope_dit");
}

__declspec(dllexport) int klein_attend(
    const void* q,
    const void* k,
    const void* v,
    void* y,
    int seq,
    int heads,
    int kv_heads,
    int dim,
    int causal,
    int valid,
    float scale) {
    if (dim == 128) {
        int blocks = std::max(1, (seq + 63) / 64);
        attend_mma<<<dim3(blocks, heads), 128>>>(
            static_cast<const __half*>(q),
            static_cast<const __half*>(k),
            static_cast<const __half*>(v),
            static_cast<__half*>(y),
            seq,
            heads,
            kv_heads,
            causal,
            valid,
            scale);
    } else if (dim == 512) {
        int blocks = std::max(1, (seq + 3) / 4);
        attend_fast<512, 4, 16, 32><<<dim3(blocks, heads), 128>>>(
            static_cast<const __half*>(q),
            static_cast<const __half*>(k),
            static_cast<const __half*>(v),
            static_cast<__half*>(y),
            seq,
            heads,
            kv_heads,
            causal,
            valid,
            scale);
    } else {
        std::fprintf(stderr, "attend dim %d is not 128 or 512\n", dim);
        return 2;
    }
    return launch_ok("attend");
}

__declspec(dllexport) int klein_slice_cols(const void* x, void* y, int rows, int cols, int start, int width) {
    int count = rows * width;
    int threads = 256;
    int blocks = std::max(1, (count + threads - 1) / threads);
    slice_cols_kernel<<<blocks, threads>>>(
        static_cast<const __half*>(x), static_cast<__half*>(y), rows, cols, start, width);
    return launch_ok("slice_cols");
}

__declspec(dllexport) int klein_pad_k(const void* x, void* y, int rows, int cols, int stored) {
    int count = rows * stored;
    int threads = 256;
    int blocks = std::max(1, (count + threads - 1) / threads);
    pad_k_kernel<<<blocks, threads>>>(static_cast<const __half*>(x), static_cast<__half*>(y), rows, cols, stored);
    return launch_ok("pad_k");
}

__declspec(dllexport) int klein_embed(const void* ids, const void* codes, const void* scales, void* y, int seq, int inner) {
    dim3 block(128);
    dim3 grid(seq, (inner + 127) / 128);
    embed_kernel<<<grid, block>>>(
        static_cast<const int*>(ids),
        static_cast<const uint8_t*>(codes),
        static_cast<const float*>(scales),
        static_cast<__half*>(y),
        seq,
        inner);
    return launch_ok("embed");
}

__declspec(dllexport) int klein_euler(const void* x, const void* velocity, void* y, int count, float delta) {
    int threads = 256;
    int blocks = std::max(1, (count + threads - 1) / threads);
    euler_kernel<<<blocks, threads>>>(
        static_cast<const __half*>(x), static_cast<const __half*>(velocity), static_cast<__half*>(y), count, delta);
    return launch_ok("euler");
}

__declspec(dllexport) int klein_add(const void* a, const void* b, void* y, int count) {
    int threads = 256;
    int blocks = std::max(1, (count + threads - 1) / threads);
    add_kernel<<<blocks, threads>>>(static_cast<const __half*>(a), static_cast<const __half*>(b), static_cast<__half*>(y), count);
    return launch_ok("add");
}

__declspec(dllexport) int klein_add_f32(const void* a, const void* b, void* y, int count) {
    int threads = 256;
    int blocks = std::max(1, (count + threads - 1) / threads);
    add_f32_kernel<<<blocks, threads>>>(
        static_cast<const float*>(a), static_cast<const float*>(b), static_cast<float*>(y), count);
    return launch_ok("add_f32");
}

__declspec(dllexport) int klein_cast_f16_f32(const void* x, void* y, int count) {
    int threads = 256;
    int blocks = std::max(1, (count + threads - 1) / threads);
    cast_f16_f32_kernel<<<blocks, threads>>>(static_cast<const __half*>(x), static_cast<float*>(y), count);
    return launch_ok("cast_f16");
}

__declspec(dllexport) int klein_cast_f32_f16(const void* x, void* y, int count) {
    int threads = 256;
    int blocks = std::max(1, (count + threads - 1) / threads);
    cast_f32_f16_kernel<<<blocks, threads>>>(static_cast<const float*>(x), static_cast<__half*>(y), count);
    return launch_ok("cast_f32");
}

__declspec(dllexport) int klein_groupnorm(
    const void* x,
    const void* weight,
    const void* bias,
    void* y,
    void* scratch,
    int channels,
    int spatial,
    int groups,
    float eps,
    int silu,
    int fp32) {
    float* mean = static_cast<float*>(scratch);
    float* inv = mean + groups;
    if (fp32) {
        group_stats_kernel<float><<<groups, 256>>>(
            static_cast<const float*>(x), mean, inv, channels, groups, spatial, eps);
        if (int code = launch_ok("group_stats")) {
            return code;
        }
        int count = channels * spatial;
        int threads = 256;
        int blocks = std::max(1, (count + threads - 1) / threads);
        group_apply_kernel<float><<<blocks, threads>>>(
            static_cast<const float*>(x),
            static_cast<const __half*>(weight),
            static_cast<const __half*>(bias),
            static_cast<float*>(y),
            mean,
            inv,
            channels,
            groups,
            spatial,
            silu);
    } else {
        group_stats_kernel<__half><<<groups, 256>>>(
            static_cast<const __half*>(x), mean, inv, channels, groups, spatial, eps);
        if (int code = launch_ok("group_stats")) {
            return code;
        }
        int count = channels * spatial;
        int threads = 256;
        int blocks = std::max(1, (count + threads - 1) / threads);
        group_apply_kernel<__half><<<blocks, threads>>>(
            static_cast<const __half*>(x),
            static_cast<const __half*>(weight),
            static_cast<const __half*>(bias),
            static_cast<__half*>(y),
            mean,
            inv,
            channels,
            groups,
            spatial,
            silu);
    }
    return launch_ok("group_apply");
}

__declspec(dllexport) int klein_im2col(
    const void* x,
    void* columns,
    int channels,
    int height,
    int width,
    int kernel,
    int stride,
    int pad,
    int out_h,
    int out_w,
    int stored_k,
    int fp32) {
    int count = out_h * out_w * stored_k;
    int threads = 256;
    int blocks = std::max(1, (count + threads - 1) / threads);
    if (fp32) {
        im2col_f32_kernel<<<blocks, threads>>>(
            static_cast<const float*>(x),
            static_cast<__half*>(columns),
            channels,
            height,
            width,
            kernel,
            stride,
            pad,
            out_h,
            out_w,
            stored_k);
    } else {
        im2col_kernel<<<blocks, threads>>>(
            static_cast<const __half*>(x),
            static_cast<__half*>(columns),
            channels,
            height,
            width,
            kernel,
            stride,
            pad,
            out_h,
            out_w,
            stored_k);
    }
    return launch_ok("im2col");
}

__declspec(dllexport) int klein_col2im(const void* columns, void* y, int channels, int height, int width, int fp32) {
    int count = channels * height * width;
    int threads = 256;
    int blocks = std::max(1, (count + threads - 1) / threads);
    if (fp32) {
        col2im_f32_kernel<<<blocks, threads>>>(
            static_cast<const __half*>(columns), static_cast<float*>(y), channels, height, width);
    } else {
        col2im_kernel<<<blocks, threads>>>(
            static_cast<const __half*>(columns), static_cast<__half*>(y), channels, height, width);
    }
    return launch_ok("col2im");
}

__declspec(dllexport) int klein_pad_edge(const void* x, void* y, int channels, int height, int width, int pad_b, int pad_r, int fp32) {
    int out_h = height + pad_b;
    int out_w = width + pad_r;
    int count = channels * out_h * out_w;
    int threads = 256;
    int blocks = std::max(1, (count + threads - 1) / threads);
    if (fp32) {
        pad_edge_f32_kernel<<<blocks, threads>>>(
            static_cast<const float*>(x), static_cast<float*>(y), channels, height, width, pad_b, pad_r);
    } else {
        pad_edge_kernel<<<blocks, threads>>>(
            static_cast<const __half*>(x), static_cast<__half*>(y), channels, height, width, pad_b, pad_r);
    }
    return launch_ok("pad_edge");
}

__declspec(dllexport) int klein_nearest2(const void* x, void* y, int channels, int height, int width, int fp32) {
    int count = channels * height * 2 * width * 2;
    int threads = 256;
    int blocks = std::max(1, (count + threads - 1) / threads);
    if (fp32) {
        nearest2_f32_kernel<<<blocks, threads>>>(
            static_cast<const float*>(x), static_cast<float*>(y), channels, height, width);
    } else {
        nearest2_kernel<<<blocks, threads>>>(
            static_cast<const __half*>(x), static_cast<__half*>(y), channels, height, width);
    }
    return launch_ok("nearest2");
}

__declspec(dllexport) int klein_prefix_channels(const void* x, void* y, int keep, int channels, int spatial) {
    int count = keep * spatial;
    int threads = 256;
    int blocks = std::max(1, (count + threads - 1) / threads);
    prefix_channels_kernel<<<blocks, threads>>>(
        static_cast<const __half*>(x), static_cast<__half*>(y), keep, channels, spatial);
    return launch_ok("prefix");
}

__declspec(dllexport) int klein_pack_latent(
    const void* mean, const void* running_mean, const void* running_var, void* tokens, int grid_h, int grid_w, float eps) {
    int count = grid_h * grid_w * 128;
    int threads = 256;
    int blocks = std::max(1, (count + threads - 1) / threads);
    pack_latent_kernel<<<blocks, threads>>>(
        static_cast<const __half*>(mean),
        static_cast<const float*>(running_mean),
        static_cast<const float*>(running_var),
        static_cast<__half*>(tokens),
        grid_h,
        grid_w,
        eps);
    return launch_ok("pack");
}

__declspec(dllexport) int klein_conv(
    const void* volume,
    const void* codes,
    const void* scales,
    const void* bias,
    void* out,
    int channels,
    int height,
    int width,
    int kernel,
    int stride,
    int pad,
    int out_h,
    int out_w,
    int out_c,
    int logical_k,
    int stored_k,
    int fp32) {
    if (out_h < 1 || out_w < 1 || out_c < 1 || stored_k < 1) {
        return 0;
    }
    dim3 block(kGemmThreads);
    dim3 grid((out_c + kBN - 1) / kBN, (out_h * out_w + kBM - 1) / kBM);
    if (fp32) {
        conv_int4_kernel<float, float><<<grid, block>>>(
            static_cast<const float*>(volume),
            static_cast<const uint8_t*>(codes),
            static_cast<const float*>(scales),
            static_cast<const __half*>(bias),
            static_cast<float*>(out),
            channels,
            height,
            width,
            kernel,
            stride,
            pad,
            out_h,
            out_w,
            out_c,
            logical_k,
            stored_k);
    } else {
        conv_int4_kernel<__half, __half><<<grid, block>>>(
            static_cast<const __half*>(volume),
            static_cast<const uint8_t*>(codes),
            static_cast<const float*>(scales),
            static_cast<const __half*>(bias),
            static_cast<__half*>(out),
            channels,
            height,
            width,
            kernel,
            stride,
            pad,
            out_h,
            out_w,
            out_c,
            logical_k,
            stored_k);
    }
    return launch_ok("conv");
}

__declspec(dllexport) int klein_transpose(const void* x, void* y, int channels, int spatial, int to_tokens, int fp32) {
    long long count = static_cast<long long>(channels) * spatial;
    int threads = 256;
    int blocks = static_cast<int>((count + threads - 1) / threads);
    if (fp32 && to_tokens) {
        transpose_kernel<float, __half><<<blocks, threads>>>(
            static_cast<const float*>(x), static_cast<__half*>(y), channels, spatial, 1);
    } else if (fp32) {
        transpose_kernel<__half, float><<<blocks, threads>>>(
            static_cast<const __half*>(x), static_cast<float*>(y), channels, spatial, 0);
    } else if (to_tokens) {
        transpose_kernel<__half, __half><<<blocks, threads>>>(
            static_cast<const __half*>(x), static_cast<__half*>(y), channels, spatial, 1);
    } else {
        transpose_kernel<__half, __half><<<blocks, threads>>>(
            static_cast<const __half*>(x), static_cast<__half*>(y), channels, spatial, 0);
    }
    return launch_ok("transpose");
}

__declspec(dllexport) int klein_concat_cols(
    const void* left, const void* right, void* out, int rows, int left_cols, int right_cols) {
    int count = rows * (left_cols + right_cols);
    int threads = 256;
    int blocks = std::max(1, (count + threads - 1) / threads);
    concat_cols_kernel<<<blocks, threads>>>(
        static_cast<const __half*>(left),
        static_cast<const __half*>(right),
        static_cast<__half*>(out),
        rows,
        left_cols,
        right_cols);
    return launch_ok("concat_cols");
}

__declspec(dllexport) int klein_unpack_latent(
    const void* tokens, const void* running_mean, const void* running_var, void* volume, int grid_h, int grid_w, float eps) {
    int height = grid_h * 2;
    int width = grid_w * 2;
    int count = 32 * height * width;
    int threads = 256;
    int blocks = std::max(1, (count + threads - 1) / threads);
    unpack_latent_kernel<<<blocks, threads>>>(
        static_cast<const __half*>(tokens),
        static_cast<const float*>(running_mean),
        static_cast<const float*>(running_var),
        static_cast<float*>(volume),
        grid_h,
        grid_w,
        eps);
    return launch_ok("unpack");
}

}  // extern "C"
