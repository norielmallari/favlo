// Wan2.2 TI2V-5B CUDA step. Stored tensors are fp16.
// An int4 code is multiplied by its fp16 row scale and the product enters
// mma.sync.aligned.m16n8k16, which accumulates in fp32 registers. The store is fp16.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API
#endif

static cudaStream_t g_stream = nullptr;

static int check(cudaError_t status) {
    if (status == cudaSuccess) {
        return 0;
    }
    std::fprintf(stderr, "cuda: %s\n", cudaGetErrorString(status));
    return 1;
}

extern "C" API int engine_open() {
    if (g_stream != nullptr) {
        return 0;
    }
    int device = 0;
    if (check(cudaGetDevice(&device))) {
        return 2;
    }
    cudaDeviceProp prop{};
    if (check(cudaGetDeviceProperties(&prop, device))) {
        return 2;
    }
    if (prop.major < 8) {
        std::fprintf(stderr, "compute capability %d.%d is below 8.0\n", prop.major, prop.minor);
        return 3;
    }
    return check(cudaStreamCreate(&g_stream));
}

extern "C" API void engine_close() {
    if (g_stream != nullptr) {
        cudaStreamDestroy(g_stream);
        g_stream = nullptr;
    }
}

extern "C" API void* engine_alloc(size_t nbytes) {
    void* pointer = nullptr;
    if (cudaMalloc(&pointer, nbytes == 0 ? 1 : nbytes) != cudaSuccess) {
        return nullptr;
    }
    return pointer;
}

extern "C" API void engine_free(void* pointer) {
    if (pointer != nullptr) {
        cudaFree(pointer);
    }
}

extern "C" API int engine_h2d(void* destination, const void* source, size_t nbytes) {
    return check(cudaMemcpyAsync(destination, source, nbytes, cudaMemcpyHostToDevice, g_stream));
}

extern "C" API int engine_d2h(void* destination, const void* source, size_t nbytes) {
    int status = check(cudaMemcpyAsync(destination, source, nbytes, cudaMemcpyDeviceToHost, g_stream));
    if (status) {
        return status;
    }
    return check(cudaStreamSynchronize(g_stream));
}

__device__ __forceinline__ unsigned pack_half(__half low, __half high) {
    unsigned out;
    asm volatile("mov.b32 %0, {%1, %2};" : "=r"(out) : "h"(__half_as_ushort(low)), "h"(__half_as_ushort(high)));
    return out;
}

__device__ __forceinline__ void mma_f16(
    float& d0, float& d1, float& d2, float& d3, unsigned a0, unsigned a1, unsigned a2, unsigned a3,
    unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(d0), "f"(d1), "f"(d2), "f"(d3));
}

__device__ __forceinline__ float code_at(const uint8_t* codes, const __half* scales, int row, int col, int inner) {
    unsigned byte = codes[(static_cast<long long>(row) * inner + col) >> 1];
    unsigned nibble = (col & 1) ? (byte >> 4) : (byte & 0xF);
    int code = static_cast<int>(nibble);
    if (code >= 8) {
        code -= 16;
    }
    return static_cast<float>(code) * __half2float(scales[row]);
}

__global__ void gemv_kernel(
    const uint8_t* codes, const __half* scales, const __half* source, __half* destination, int rows, int inner) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    float acc = 0.0f;
    for (int col = threadIdx.x; col < inner; col += blockDim.x) {
        acc += code_at(codes, scales, row, col, inner) * __half2float(source[col]);
    }
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(mask, acc, offset);
    }
    __shared__ float warp[4];
    if (threadIdx.x < 4) {
        warp[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    if (lane == 0) {
        warp[warp_id] = acc;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = warp[0] + warp[1] + warp[2] + warp[3];
        destination[row] = __float2half(sum);
    }
}

__global__ void gemm_kernel(
    const uint8_t* codes, const __half* scales, const __half* source, __half* destination, int rows, int inner,
    int tokens) {
    int tile_n = blockIdx.x;
    int tile_m = blockIdx.y;
    int lane = threadIdx.x & 31;
    int group = lane >> 2;
    int in_group = lane & 3;
    int row0 = tile_n * 8;
    int token0 = tile_m * 16;
    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;
    for (int k0 = 0; k0 < inner; k0 += 16) {
        __half a[8];
        __half b[4];
        for (int i = 0; i < 8; ++i) {
            int row = (i < 2 || (i >= 4 && i < 6)) ? group : group + 8;
            int col = (in_group * 2) + (i & 1);
            if (i >= 4) {
                col += 8;
            }
            int token = token0 + row;
            int k = k0 + col;
            float value = 0.0f;
            if (token < tokens && k < inner) {
                value = __half2float(source[static_cast<long long>(token) * inner + k]);
            }
            a[i] = __float2half(value);
        }
        for (int i = 0; i < 4; ++i) {
            int k_local = (in_group * 2) + (i & 1);
            if (i >= 2) {
                k_local += 8;
            }
            int n = row0 + group;
            int k = k0 + k_local;
            float value = 0.0f;
            if (n < rows && k < inner) {
                value = code_at(codes, scales, n, k, inner);
            }
            b[i] = __float2half(value);
        }
        mma_f16(
            d0, d1, d2, d3, pack_half(a[0], a[1]), pack_half(a[2], a[3]), pack_half(a[4], a[5]),
            pack_half(a[6], a[7]), pack_half(b[0], b[1]), pack_half(b[2], b[3]));
    }
    float out[4] = {d0, d1, d2, d3};
    for (int i = 0; i < 4; ++i) {
        int row = i < 2 ? group : group + 8;
        int col = (in_group * 2) + (i & 1);
        int token = token0 + row;
        int n = row0 + col;
        if (token < tokens && n < rows) {
            destination[static_cast<long long>(token) * rows + n] = __float2half(out[i]);
        }
    }
}

extern "C" API int engine_linear(
    const void* codes, const void* scales, const void* source, void* destination, int rows, int inner, int tokens) {
    if (tokens < 1 || rows < 1 || inner < 1 || (inner & 1)) {
        return 1;
    }
    if (tokens == 1) {
        gemv_kernel<<<rows, 128, 0, g_stream>>>(
            static_cast<const uint8_t*>(codes), static_cast<const __half*>(scales), static_cast<const __half*>(source),
            static_cast<__half*>(destination), rows, inner);
        return check(cudaGetLastError());
    }
    dim3 grid((rows + 7) / 8, (tokens + 15) / 16);
    gemm_kernel<<<grid, 32, 0, g_stream>>>(
        static_cast<const uint8_t*>(codes), static_cast<const __half*>(scales), static_cast<const __half*>(source),
        static_cast<__half*>(destination), rows, inner, tokens);
    return check(cudaGetLastError());
}

__global__ void layernorm_kernel(const __half* source, __half* destination, int width) {
    float peak = 0.0f;
    float mean = 0.0f;
    float square = 0.0f;
    for (int col = threadIdx.x; col < width; col += blockDim.x) {
        peak = fmaxf(peak, fabsf(__half2float(source[col])));
    }
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        peak = fmaxf(peak, __shfl_xor_sync(mask, peak, offset));
    }
    float denom = peak > 0.0f ? peak : 1.0f;
    for (int col = threadIdx.x; col < width; col += blockDim.x) {
        float scaled = __half2float(__float2half(__half2float(source[col]) / denom));
        mean += scaled;
        square += scaled * scaled;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        mean += __shfl_xor_sync(mask, mean, offset);
        square += __shfl_xor_sync(mask, square, offset);
    }
    float inv = 1.0f / static_cast<float>(width);
    float center = mean * inv;
    float var = square * inv - center * center;
    float scale = rsqrtf(var + 1.0e-6f / (denom * denom));
    for (int col = threadIdx.x; col < width; col += blockDim.x) {
        float scaled = __half2float(source[col]) / denom;
        destination[col] = __float2half((scaled - center) * scale);
    }
}

__global__ void attend_kernel(
    const __half* query, const __half* key, const __half* value, __half* destination, int keys, int dim) {
    float max_score = -1.0e30f;
    for (int index = 0; index < keys; ++index) {
        float dot = 0.0f;
        for (int col = 0; col < dim; ++col) {
            dot += __half2float(query[col]) * __half2float(key[index * dim + col]);
        }
        max_score = fmaxf(max_score, dot);
    }
    float sum = 0.0f;
    for (int index = 0; index < keys; ++index) {
        float dot = 0.0f;
        for (int col = 0; col < dim; ++col) {
            dot += __half2float(query[col]) * __half2float(key[index * dim + col]);
        }
        sum += __expf(dot - max_score);
    }
    for (int col = 0; col < dim; ++col) {
        float mixed = 0.0f;
        for (int index = 0; index < keys; ++index) {
            float dot = 0.0f;
            for (int inner = 0; inner < dim; ++inner) {
                dot += __half2float(query[inner]) * __half2float(key[index * dim + inner]);
            }
            float weight = __expf(dot - max_score) / sum;
            mixed += weight * __half2float(value[index * dim + col]);
        }
        destination[col] = __float2half(mixed);
    }
}

static bool near(float got, float want, float tol) {
    float gap = got - want;
    if (gap < 0.0f) {
        gap = -gap;
    }
    return gap <= tol;
}

extern "C" API int engine_selfcheck() {
    constexpr int kTokens = 16;
    constexpr int kRows = 8;
    constexpr int kInner = 16;
    uint8_t codes[kRows * (kInner / 2)] = {};
    __half scales[kRows];
    __half act[kTokens * kInner];
    for (int row = 0; row < kRows; ++row) {
        codes[row * (kInner / 2)] = 0x01;
        scales[row] = __float2half(1.0f);
    }
    for (int token = 0; token < kTokens; ++token) {
        for (int col = 0; col < kInner; ++col) {
            act[token * kInner + col] = __float2half(col == 0 ? 1.0f : 0.0f);
        }
    }
    uint8_t* device_codes = static_cast<uint8_t*>(engine_alloc(sizeof(codes)));
    __half* device_scales = static_cast<__half*>(engine_alloc(sizeof(scales)));
    __half* device_act = static_cast<__half*>(engine_alloc(sizeof(act)));
    __half* device_out = static_cast<__half*>(engine_alloc(kTokens * kRows * sizeof(__half)));
    if (device_codes == nullptr || device_scales == nullptr || device_act == nullptr || device_out == nullptr) {
        return 4;
    }
    engine_h2d(device_codes, codes, sizeof(codes));
    engine_h2d(device_scales, scales, sizeof(scales));
    engine_h2d(device_act, act, sizeof(act));
    if (engine_linear(device_codes, device_scales, device_act, device_out, kRows, kInner, kTokens)) {
        return 5;
    }
    __half host[kTokens * kRows];
    if (engine_d2h(host, device_out, sizeof(host))) {
        return 6;
    }
    for (int index = 0; index < kTokens * kRows; ++index) {
        float value = __half2float(host[index]);
        if (!near(value, 1.0f, 0.02f)) {
            std::fprintf(stderr, "fp16 mma %d was %f\n", index, value);
            return 7;
        }
    }
    uint8_t nibble = 0x21;
    __half one = __float2half(1.0f);
    __half pair[2] = {__float2half(1.0f), __float2half(1.0f)};
    uint8_t* device_nibble = static_cast<uint8_t*>(engine_alloc(1));
    __half* device_one = static_cast<__half*>(engine_alloc(sizeof(one)));
    __half* device_pair = static_cast<__half*>(engine_alloc(sizeof(pair)));
    __half* device_dot = static_cast<__half*>(engine_alloc(sizeof(__half)));
    engine_h2d(device_nibble, &nibble, 1);
    engine_h2d(device_one, &one, sizeof(one));
    engine_h2d(device_pair, pair, sizeof(pair));
    if (engine_linear(device_nibble, device_one, device_pair, device_dot, 1, 2, 1)) {
        return 8;
    }
    __half dot{};
    if (engine_d2h(&dot, device_dot, sizeof(dot))) {
        return 8;
    }
    if (!near(__half2float(dot), 3.0f, 0.02f)) {
        std::fprintf(stderr, "int4 nibble dot was %f\n", __half2float(dot));
        return 9;
    }
    __half row[4] = {__float2half(0.0f), __float2half(0.0f), __float2half(0.0f), __float2half(2.0f)};
    __half* device_row = static_cast<__half*>(engine_alloc(sizeof(row)));
    __half* device_norm = static_cast<__half*>(engine_alloc(sizeof(row)));
    engine_h2d(device_row, row, sizeof(row));
    layernorm_kernel<<<1, 32, 0, g_stream>>>(device_row, device_norm, 4);
    if (check(cudaGetLastError())) {
        return 10;
    }
    __half norm[4];
    if (engine_d2h(norm, device_norm, sizeof(norm))) {
        return 10;
    }
    float inv = rsqrtf(0.75f + 1.0e-6f);
    if (!near(__half2float(norm[0]), -0.5f * inv, 0.05f) || !near(__half2float(norm[3]), 1.5f * inv, 0.05f)) {
        std::fprintf(stderr, "layernorm was %f %f\n", __half2float(norm[0]), __half2float(norm[3]));
        return 11;
    }
    __half query[4] = {__float2half(1.0f), __float2half(0.0f), __float2half(0.0f), __float2half(0.0f)};
    __half key[8];
    __half value[8];
    for (int index = 0; index < 8; ++index) {
        key[index] = __float2half(index % 4 == 0 ? 1.0f : 0.0f);
        value[index] = __float2half(0.0f);
    }
    value[0] = __float2half(2.0f);
    value[4] = __float2half(4.0f);
    __half* device_q = static_cast<__half*>(engine_alloc(sizeof(query)));
    __half* device_k = static_cast<__half*>(engine_alloc(sizeof(key)));
    __half* device_v = static_cast<__half*>(engine_alloc(sizeof(value)));
    __half* device_mix = static_cast<__half*>(engine_alloc(sizeof(query)));
    engine_h2d(device_q, query, sizeof(query));
    engine_h2d(device_k, key, sizeof(key));
    engine_h2d(device_v, value, sizeof(value));
    attend_kernel<<<1, 1, 0, g_stream>>>(device_q, device_k, device_v, device_mix, 2, 4);
    if (check(cudaGetLastError())) {
        return 12;
    }
    __half mix[4];
    if (engine_d2h(mix, device_mix, sizeof(mix))) {
        return 12;
    }
    if (!near(__half2float(mix[0]), 3.0f, 0.02f)) {
        std::fprintf(stderr, "attention was %f\n", __half2float(mix[0]));
        return 13;
    }
    engine_free(device_codes);
    engine_free(device_scales);
    engine_free(device_act);
    engine_free(device_out);
    engine_free(device_nibble);
    engine_free(device_one);
    engine_free(device_pair);
    engine_free(device_dot);
    engine_free(device_row);
    engine_free(device_norm);
    engine_free(device_q);
    engine_free(device_k);
    engine_free(device_v);
    engine_free(device_mix);
    return 0;
}
