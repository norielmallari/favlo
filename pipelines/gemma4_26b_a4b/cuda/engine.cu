// Gemma 4 CUDA step. Activations, norms, attention, and the cache are fp16.
// int8 and int4 weights are dequantized inside the kernel and multiplied with
// mma.sync.aligned.m16n8k16, which accumulates in fp32.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <vector>

#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API
#endif

static cudaStream_t g_stream = nullptr;
static int g_sms = 1;
static cudaStream_t g_copy = nullptr;
constexpr int kEvents = 4096;
static cudaEvent_t g_events[kEvents];
static int g_event_ready = 0;
static int g_event_next = 0;
static cudaStream_t g_main = nullptr;
static int* g_step = nullptr;
constexpr int kAttendHeads = 16;
constexpr int kAttendDim = 512;
constexpr int kAttendSplits = 128;
static int g_splits = 1;
static float* g_partial_peak = nullptr;
static float* g_partial_sum = nullptr;
static float* g_partial_acc = nullptr;
__device__ int* g_step_ptr;
constexpr int kGraphs = 128;
static cudaGraphExec_t g_exec[kGraphs] = {};

__device__ int step_position() {
    return reinterpret_cast<volatile int*>(g_step_ptr)[0];
}

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
    g_sms = prop.multiProcessorCount > 0 ? prop.multiProcessorCount : 1;
    if (check(cudaStreamCreate(&g_stream))) {
        return 2;
    }
    g_main = g_stream;
    if (check(cudaStreamCreate(&g_copy))) {
        return 2;
    }
    for (int index = 0; index < kEvents; ++index) {
        if (check(cudaEventCreateWithFlags(&g_events[index], cudaEventDisableTiming))) {
            return 2;
        }
        g_event_ready = index + 1;
    }
    if (check(cudaMalloc(&g_step, sizeof(int)))) {
        return 2;
    }
    if (check(cudaMemset(g_step, 0, sizeof(int)))) {
        return 2;
    }
    g_splits = g_sms * 2;
    if (g_splits > kAttendSplits) {
        g_splits = kAttendSplits;
    }
    if (g_splits < 1) {
        g_splits = 1;
    }
    size_t parts = static_cast<size_t>(kAttendSplits) * kAttendHeads * 4;
    if (check(cudaMalloc(&g_partial_peak, parts * sizeof(float)))) {
        return 2;
    }
    if (check(cudaMalloc(&g_partial_sum, parts * sizeof(float)))) {
        return 2;
    }
    if (check(cudaMalloc(&g_partial_acc, parts * kAttendDim * sizeof(float)))) {
        return 2;
    }
    return check(cudaMemcpyToSymbol(g_step_ptr, &g_step, sizeof(g_step)));
}

extern "C" API void engine_close() {
    for (int slot = 0; slot < kGraphs; ++slot) {
        if (g_exec[slot] != nullptr) {
            cudaGraphExecDestroy(g_exec[slot]);
            g_exec[slot] = nullptr;
        }
    }
    if (g_partial_acc != nullptr) {
        cudaFree(g_partial_acc);
        g_partial_acc = nullptr;
    }
    if (g_partial_sum != nullptr) {
        cudaFree(g_partial_sum);
        g_partial_sum = nullptr;
    }
    if (g_partial_peak != nullptr) {
        cudaFree(g_partial_peak);
        g_partial_peak = nullptr;
    }
    if (g_step != nullptr) {
        cudaFree(g_step);
        g_step = nullptr;
    }
    for (int index = 0; index < g_event_ready; ++index) {
        cudaEventDestroy(g_events[index]);
    }
    g_event_ready = 0;
    g_event_next = 0;
    if (g_copy != nullptr) {
        cudaStreamDestroy(g_copy);
        g_copy = nullptr;
    }
    if (g_stream != nullptr) {
        cudaStreamDestroy(g_stream);
        g_stream = nullptr;
        g_main = nullptr;
    }
}

extern "C" API int engine_set_step(int position) {
    return check(cudaMemcpyAsync(g_step, &position, sizeof(position), cudaMemcpyHostToDevice, g_stream));
}

extern "C" API int engine_capture_begin(int slot) {
    if (slot < 0 || slot >= kGraphs) {
        return 1;
    }
    cudaStream_t created = nullptr;
    int status = check(cudaStreamCreate(&created));
    if (status != 0) {
        return status;
    }
    g_stream = created;
    status = check(cudaStreamBeginCapture(created, cudaStreamCaptureModeRelaxed));
    if (status != 0) {
        cudaStreamDestroy(created);
        g_stream = g_main;
    }
    return status;
}

extern "C" API int engine_capture_end(int slot) {
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
    if (g_exec[slot] != nullptr) {
        cudaGraphExecDestroy(g_exec[slot]);
        g_exec[slot] = nullptr;
    }
    status = check(cudaGraphInstantiate(&g_exec[slot], graph, 0));
    cudaGraphDestroy(graph);
    return status;
}

extern "C" API int engine_replay(int slot) {
    if (slot < 0 || slot >= kGraphs || g_exec[slot] == nullptr) {
        return 1;
    }
    return check(cudaGraphLaunch(g_exec[slot], g_stream));
}

extern "C" API void engine_reset_graphs() {
    for (int slot = 0; slot < kGraphs; ++slot) {
        if (g_exec[slot] != nullptr) {
            cudaGraphExecDestroy(g_exec[slot]);
            g_exec[slot] = nullptr;
        }
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

extern "C" API void* engine_host_alloc(size_t nbytes) {
    void* pointer = nullptr;
    unsigned int flags = cudaHostAllocWriteCombined;
    if (cudaHostAlloc(&pointer, nbytes == 0 ? 1 : nbytes, flags) != cudaSuccess) {
        return nullptr;
    }
    return pointer;
}

extern "C" API void engine_host_free(void* pointer) {
    if (pointer != nullptr) {
        cudaFreeHost(pointer);
    }
}

extern "C" API int engine_h2d(void* destination, const void* source, size_t nbytes) {
    int status = check(cudaMemcpyAsync(destination, source, nbytes, cudaMemcpyHostToDevice, g_stream));
    if (status) {
        return status;
    }
    // Pageable host memory is released by the caller when this returns.
    return check(cudaStreamSynchronize(g_stream));
}

extern "C" API int engine_d2h(void* destination, const void* source, size_t nbytes) {
    int status = check(cudaMemcpyAsync(destination, source, nbytes, cudaMemcpyDeviceToHost, g_stream));
    if (status) {
        return status;
    }
    return check(cudaStreamSynchronize(g_stream));
}

extern "C" API int engine_h2d_ptr(void* destination, const void* source, size_t nbytes) {
    return check(cudaMemcpyAsync(destination, source, nbytes, cudaMemcpyHostToDevice, g_stream));
}

extern "C" API int engine_copy_async(void* destination, const void* source, size_t nbytes, int* event_id) {
    if (destination == nullptr || source == nullptr || nbytes == 0 || event_id == nullptr || g_copy == nullptr) {
        return 1;
    }
    if (g_event_next >= g_event_ready) {
        int status = check(cudaStreamSynchronize(g_stream));
        if (status != 0) {
            return status;
        }
        g_event_next = 0;
    }
    int slot = g_event_next++;
    int status = check(cudaMemcpyAsync(destination, source, nbytes, cudaMemcpyHostToDevice, g_copy));
    if (status != 0) {
        return status;
    }
    status = check(cudaEventRecord(g_events[slot], g_copy));
    if (status != 0) {
        return status;
    }
    *event_id = slot;
    return 0;
}

extern "C" API int engine_wait_copy(int event_id) {
    if (event_id < 0 || event_id >= g_event_ready) {
        return 1;
    }
    return check(cudaStreamWaitEvent(g_stream, g_events[event_id], 0));
}

extern "C" API int engine_d2d(void* destination, const void* source, size_t nbytes) {
    return check(cudaMemcpyAsync(destination, source, nbytes, cudaMemcpyDeviceToDevice, g_stream));
}

extern "C" API int engine_register_ro(void* pointer, size_t nbytes) {
    return cudaHostRegister(pointer, nbytes, cudaHostRegisterReadOnly) == cudaSuccess ? 0 : 1;
}

extern "C" API int engine_register(void* pointer, size_t nbytes) {
    return cudaHostRegister(pointer, nbytes, cudaHostRegisterDefault) == cudaSuccess ? 0 : 1;
}

extern "C" API void engine_unregister(void* pointer) {
    cudaHostUnregister(pointer);
}

extern "C" API int engine_sync() {
    return check(cudaStreamSynchronize(g_stream));
}

extern "C" API int engine_mem_info(size_t* free_bytes, size_t* total_bytes) {
    return check(cudaMemGetInfo(free_bytes, total_bytes));
}

extern "C" API int engine_zero(void* destination, size_t nbytes) {
    return check(cudaMemsetAsync(destination, 0, nbytes, g_stream));
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

__device__ __forceinline__ float weight_at(const void* codes, const float* scales, int row, int col, int inner, int bits) {
    float scale = scales[row];
    int code;
    if (bits == 8) {
        code = static_cast<const int8_t*>(codes)[static_cast<long long>(row) * inner + col];
    } else {
        unsigned byte = static_cast<const uint8_t*>(codes)[(static_cast<long long>(row) * inner + col) >> 1];
        unsigned nibble = (col & 1) ? (byte >> 4) : (byte & 0xF);
        code = static_cast<int>(nibble);
        if (code >= 8) {
            code -= 16;
        }
    }
    return static_cast<float>(code) * scale;
}

__device__ __forceinline__ int code_at(const void* codes, int row, int col, int inner, int bits) {
    if (bits == 8) {
        return static_cast<const int8_t*>(codes)[static_cast<long long>(row) * inner + col];
    }
    unsigned byte = static_cast<const uint8_t*>(codes)[(static_cast<long long>(row) * inner + col) >> 1];
    unsigned nibble = (col & 1) ? (byte >> 4) : (byte & 0xF);
    int code = static_cast<int>(nibble);
    if (code >= 8) {
        code -= 16;
    }
    return code;
}

// One warp owns one row and walks the row in 16-byte loads. The activation is shared through the cache.
__global__ void gemv_kernel(
    const void* codes, const float* scales, const __half* source, __half* destination, int rows, int inner,
    int bits) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int stride = gridDim.x * 4;
    for (int row = blockIdx.x * 4 + warp; row < rows; row += stride) {
        float dot = 0.0f;
        if (bits == 8 && (inner & 15) == 0) {
            const int8_t* line = static_cast<const int8_t*>(codes) + static_cast<long long>(row) * inner;
            for (int col = lane * 16; col < inner; col += 512) {
                int4 raw = *reinterpret_cast<const int4*>(line + col);
                unsigned word[4] = {
                    static_cast<unsigned>(raw.x), static_cast<unsigned>(raw.y), static_cast<unsigned>(raw.z),
                    static_cast<unsigned>(raw.w)};
#pragma unroll
                for (int part = 0; part < 4; ++part) {
#pragma unroll
                    for (int byte = 0; byte < 4; ++byte) {
                        int code = static_cast<int8_t>((word[part] >> (byte * 8)) & 0xFFu);
                        int at = col + part * 4 + byte;
                        dot += static_cast<float>(code) * __half2float(__ldg(source + at));
                    }
                }
            }
        } else if (bits == 4 && (inner & 31) == 0) {
            const uint8_t* line = static_cast<const uint8_t*>(codes) + ((static_cast<long long>(row) * inner) >> 1);
            for (int col = lane * 32; col < inner; col += 1024) {
                int4 raw = *reinterpret_cast<const int4*>(line + (col >> 1));
                unsigned word[4] = {
                    static_cast<unsigned>(raw.x), static_cast<unsigned>(raw.y), static_cast<unsigned>(raw.z),
                    static_cast<unsigned>(raw.w)};
#pragma unroll
                for (int part = 0; part < 4; ++part) {
#pragma unroll
                    for (int byte = 0; byte < 4; ++byte) {
                        unsigned packed = (word[part] >> (byte * 8)) & 0xFFu;
                        int low = static_cast<int>(packed & 0xF);
                        int high = static_cast<int>(packed >> 4);
                        if (low >= 8) {
                            low -= 16;
                        }
                        if (high >= 8) {
                            high -= 16;
                        }
                        int at = col + (part * 4 + byte) * 2;
                        dot += static_cast<float>(low) * __half2float(__ldg(source + at));
                        dot += static_cast<float>(high) * __half2float(__ldg(source + at + 1));
                    }
                }
            }
        } else {
            for (int col = lane; col < inner; col += 32) {
                dot += static_cast<float>(code_at(codes, row, col, inner, bits)) * __half2float(__ldg(source + col));
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_xor_sync(0xffffffffu, dot, offset);
        }
        if (lane == 0) {
            destination[row] = __float2half(dot * __ldg(scales + row));
        }
    }
}

// The activation is converted once per block. Every row of that block reuses it from shared memory.
__global__ void gemv_cached_kernel(
    const void* codes, const float* scales, const __half* source, __half* destination, int rows, int inner,
    int bits) {
    extern __shared__ float mix[];
    for (int col = threadIdx.x; col < inner; col += blockDim.x) {
        mix[col] = __half2float(__ldg(source + col));
    }
    __syncthreads();
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps = blockDim.x >> 5;
    int stride = gridDim.x * warps;
    for (int row = blockIdx.x * warps + warp; row < rows; row += stride) {
        float dot = 0.0f;
        if (bits == 8 && (inner & 15) == 0) {
            const int8_t* line = static_cast<const int8_t*>(codes) + static_cast<long long>(row) * inner;
            for (int col = lane * 16; col < inner; col += 512) {
                int4 raw = *reinterpret_cast<const int4*>(line + col);
                unsigned word[4] = {
                    static_cast<unsigned>(raw.x), static_cast<unsigned>(raw.y), static_cast<unsigned>(raw.z),
                    static_cast<unsigned>(raw.w)};
#pragma unroll
                for (int part = 0; part < 4; ++part) {
#pragma unroll
                    for (int byte = 0; byte < 4; ++byte) {
                        int code = static_cast<int8_t>((word[part] >> (byte * 8)) & 0xFFu);
                        dot += static_cast<float>(code) * mix[col + part * 4 + byte];
                    }
                }
            }
        } else if (bits == 4 && (inner & 31) == 0) {
            const uint8_t* line = static_cast<const uint8_t*>(codes) + ((static_cast<long long>(row) * inner) >> 1);
            for (int col = lane * 32; col < inner; col += 1024) {
                int4 raw = *reinterpret_cast<const int4*>(line + (col >> 1));
                unsigned word[4] = {
                    static_cast<unsigned>(raw.x), static_cast<unsigned>(raw.y), static_cast<unsigned>(raw.z),
                    static_cast<unsigned>(raw.w)};
#pragma unroll
                for (int part = 0; part < 4; ++part) {
#pragma unroll
                    for (int byte = 0; byte < 4; ++byte) {
                        unsigned packed = (word[part] >> (byte * 8)) & 0xFFu;
                        int low = static_cast<int>(packed & 0xF);
                        int high = static_cast<int>(packed >> 4);
                        if (low >= 8) {
                            low -= 16;
                        }
                        if (high >= 8) {
                            high -= 16;
                        }
                        int at = col + (part * 4 + byte) * 2;
                        dot += static_cast<float>(low) * mix[at];
                        dot += static_cast<float>(high) * mix[at + 1];
                    }
                }
            }
        } else {
            for (int col = lane; col < inner; col += 32) {
                dot += static_cast<float>(code_at(codes, row, col, inner, bits)) * mix[col];
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_xor_sync(0xffffffffu, dot, offset);
        }
        if (lane == 0) {
            destination[row] = __float2half(dot * __ldg(scales + row));
        }
    }
}

// A 64 by 64 output tile. Four warps each own 16 token rows. The K step is dequantized into shared memory, then one
// m16n8k16 issues per 8 output rows.
template <int kStep>
__global__ void gemm_kernel(
    const void* codes, const float* scales, const __half* source, __half* destination, int rows, int inner,
    int tokens, int bits) {
    constexpr int kTokenTile = 64;
    constexpr int kRowTile = 64;
    int row0 = blockIdx.x * kRowTile;
    int token0 = blockIdx.y * kTokenTile;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int group = lane >> 2;
    int in_group = lane & 3;
    __shared__ __half act[kTokenTile * kStep];
    __shared__ __half weight[kRowTile * kStep];
    float acc[8][4];
#pragma unroll
    for (int tile = 0; tile < 8; ++tile) {
#pragma unroll
        for (int lane_out = 0; lane_out < 4; ++lane_out) {
            acc[tile][lane_out] = 0.0f;
        }
    }
    for (int k0 = 0; k0 < inner; k0 += kStep) {
        bool aligned = (k0 + kStep) <= inner && (k0 & 15) == 0;
        if (aligned && (inner & 7) == 0) {
            int local_token = threadIdx.x;
            if (local_token < kTokenTile) {
                int token = token0 + local_token;
                __half* slot = act + local_token * kStep;
                if (token < tokens) {
                    const __half* src = source + static_cast<long long>(token) * inner + k0;
#pragma unroll
                    for (int chunk = 0; chunk < kStep / 8; ++chunk) {
                        reinterpret_cast<int4*>(slot)[chunk] = reinterpret_cast<const int4*>(src)[chunk];
                    }
                } else {
#pragma unroll
                    for (int col = 0; col < kStep; ++col) {
                        slot[col] = __float2half(0.0f);
                    }
                }
            }
        } else {
            for (int item = threadIdx.x; item < kTokenTile * kStep; item += blockDim.x) {
                int local_token = item / kStep;
                int local_k = item - local_token * kStep;
                int token = token0 + local_token;
                int k = k0 + local_k;
                float value = 0.0f;
                if (token < tokens && k < inner) {
                    value = __half2float(source[static_cast<long long>(token) * inner + k]);
                }
                act[item] = __float2half(value);
            }
        }
        if (bits == 8 && aligned && (inner & 15) == 0) {
            int local_row = threadIdx.x;
            if (local_row < kRowTile) {
                int row = row0 + local_row;
                __half* slot = weight + local_row * kStep;
                if (row < rows) {
                    const int8_t* src = static_cast<const int8_t*>(codes) + static_cast<long long>(row) * inner + k0;
                    float scale = scales[row];
#pragma unroll
                    for (int pack = 0; pack < kStep / 16; ++pack) {
                        int4 raw = reinterpret_cast<const int4*>(src)[pack];
                        unsigned word[4] = {
                            static_cast<unsigned>(raw.x), static_cast<unsigned>(raw.y), static_cast<unsigned>(raw.z),
                            static_cast<unsigned>(raw.w)};
#pragma unroll
                        for (int part = 0; part < 4; ++part) {
#pragma unroll
                            for (int byte = 0; byte < 4; ++byte) {
                                int code = static_cast<int8_t>((word[part] >> (byte * 8)) & 0xFFu);
                                slot[pack * 16 + part * 4 + byte] = __float2half(static_cast<float>(code) * scale);
                            }
                        }
                    }
                } else {
#pragma unroll
                    for (int col = 0; col < kStep; ++col) {
                        slot[col] = __float2half(0.0f);
                    }
                }
            }
        } else if (bits == 4 && aligned && (inner & 31) == 0) {
            int local_row = threadIdx.x;
            if (local_row < kRowTile) {
                int row = row0 + local_row;
                __half* slot = weight + local_row * kStep;
                if (row < rows) {
                    const uint8_t* src =
                        static_cast<const uint8_t*>(codes) + ((static_cast<long long>(row) * inner + k0) >> 1);
                    float scale = scales[row];
#pragma unroll
                    for (int pack = 0; pack < kStep / 16; ++pack) {
                        unsigned word[2] = {
                            *reinterpret_cast<const unsigned*>(src + pack * 8),
                            *reinterpret_cast<const unsigned*>(src + pack * 8 + 4)};
#pragma unroll
                        for (int part = 0; part < 2; ++part) {
#pragma unroll
                            for (int byte = 0; byte < 4; ++byte) {
                                unsigned packed = (word[part] >> (byte * 8)) & 0xFFu;
                                int low = static_cast<int>(packed & 0xF);
                                int high = static_cast<int>(packed >> 4);
                                if (low >= 8) {
                                    low -= 16;
                                }
                                if (high >= 8) {
                                    high -= 16;
                                }
                                int at = pack * 16 + (part * 4 + byte) * 2;
                                slot[at] = __float2half(static_cast<float>(low) * scale);
                                slot[at + 1] = __float2half(static_cast<float>(high) * scale);
                            }
                        }
                    }
                } else {
#pragma unroll
                    for (int col = 0; col < kStep; ++col) {
                        slot[col] = __float2half(0.0f);
                    }
                }
            }
        } else {
            for (int item = threadIdx.x; item < kRowTile * kStep; item += blockDim.x) {
                int local_row = item / kStep;
                int local_k = item - local_row * kStep;
                int row = row0 + local_row;
                int k = k0 + local_k;
                float value = 0.0f;
                if (row < rows && k < inner) {
                    value = weight_at(codes, scales, row, k, inner, bits);
                }
                weight[item] = __float2half(value);
            }
        }
        __syncthreads();
#pragma unroll
        for (int tile = 0; tile < 8; ++tile) {
#pragma unroll
            for (int kk = 0; kk < kStep / 16; ++kk) {
                __half a[8];
                __half b[4];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    int row = (i < 2 || (i >= 4 && i < 6)) ? group : group + 8;
                    int col = (in_group * 2) + (i & 1);
                    if (i >= 4) {
                        col += 8;
                    }
                    a[i] = act[(warp * 16 + row) * kStep + kk * 16 + col];
                }
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    int k_local = (in_group * 2) + (i & 1);
                    if (i >= 2) {
                        k_local += 8;
                    }
                    b[i] = weight[(tile * 8 + group) * kStep + kk * 16 + k_local];
                }
                float d0 = acc[tile][0];
                float d1 = acc[tile][1];
                float d2 = acc[tile][2];
                float d3 = acc[tile][3];
                mma_f16(
                    d0, d1, d2, d3, pack_half(a[0], a[1]), pack_half(a[2], a[3]), pack_half(a[4], a[5]),
                    pack_half(a[6], a[7]), pack_half(b[0], b[1]), pack_half(b[2], b[3]));
                acc[tile][0] = d0;
                acc[tile][1] = d1;
                acc[tile][2] = d2;
                acc[tile][3] = d3;
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int tile = 0; tile < 8; ++tile) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int row = i < 2 ? group : group + 8;
            int col = (in_group * 2) + (i & 1);
            int token = token0 + warp * 16 + row;
            int n = row0 + tile * 8 + col;
            if (token < tokens && n < rows) {
                destination[static_cast<long long>(token) * rows + n] = __float2half(acc[tile][i]);
            }
        }
    }
}

extern "C" API int engine_linear(
    const void* codes, const float* scales, const void* source, void* destination, int rows, int inner, int tokens,
    int bits) {
    if (tokens < 1 || rows < 1 || inner < 1) {
        return 1;
    }
    if (tokens == 1) {
        size_t smem = static_cast<size_t>(inner) * sizeof(float);
        if (smem <= 48u * 1024u) {
            int threads = 256;
            int blocks = g_sms > 0 ? g_sms : 1;
            if (rows < blocks * 8) {
                blocks = (rows + 7) / 8;
            }
            gemv_cached_kernel<<<blocks, threads, smem, g_stream>>>(
                codes, scales, static_cast<const __half*>(source), static_cast<__half*>(destination), rows, inner,
                bits);
        } else {
            int blocks = rows < 2048 ? (rows + 3) / 4 : 512;
            gemv_kernel<<<blocks, 128, 0, g_stream>>>(
                codes, scales, static_cast<const __half*>(source), static_cast<__half*>(destination), rows, inner,
                bits);
        }
        return check(cudaGetLastError());
    }
    dim3 grid((rows + 63) / 64, (tokens + 63) / 64);
    if ((inner % 64) == 0) {
        gemm_kernel<64><<<grid, 128, 0, g_stream>>>(
            codes, scales, static_cast<const __half*>(source), static_cast<__half*>(destination), rows, inner, tokens,
            bits);
    } else {
        gemm_kernel<16><<<grid, 128, 0, g_stream>>>(
            codes, scales, static_cast<const __half*>(source), static_cast<__half*>(destination), rows, inner, tokens,
            bits);
    }
    return check(cudaGetLastError());
}

__global__ void rmsnorm_kernel(
    const __half* source, const __half* weight, __half* destination, int width, float eps, int tokens, int unit) {
    int token = blockIdx.x;
    if (token >= tokens) {
        return;
    }
    const __half* row = source + static_cast<long long>(token) * width;
    float sum = 0.0f;
    for (int col = threadIdx.x; col < width; col += blockDim.x) {
        float value = __half2float(row[col]);
        sum += value * value;
    }
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(mask, sum, offset);
    }
    __shared__ float warp[8];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    if (lane == 0) {
        warp[warp_id] = sum;
    }
    __syncthreads();
    __shared__ float scale_box;
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int warp_id = 0; warp_id < 8; ++warp_id) {
            total += warp[warp_id];
        }
        scale_box = rsqrtf(total / static_cast<float>(width) + eps);
    }
    __syncthreads();
    float scale = scale_box;
    __half* out = destination + static_cast<long long>(token) * width;
    for (int col = threadIdx.x; col < width; col += blockDim.x) {
        float gain = unit ? 1.0f : __half2float(weight[col]);
        out[col] = __float2half(__half2float(row[col]) * scale * gain);
    }
}

extern "C" API int engine_rmsnorm(
    const void* source, const void* weight, void* destination, int width, float eps, int tokens, int unit) {
    rmsnorm_kernel<<<tokens, 256, 0, g_stream>>>(
        static_cast<const __half*>(source), static_cast<const __half*>(weight), static_cast<__half*>(destination),
        width, eps, tokens, unit);
    return check(cudaGetLastError());
}

__global__ void head_norm_kernel(
    __half* values, const __half* weight, int heads, int dim, float eps, int unit) {
    int head = blockIdx.x;
    if (head >= heads) {
        return;
    }
    __half* row = values + static_cast<long long>(head) * dim;
    const __half* gain = unit ? nullptr : weight;
    float sum = 0.0f;
    for (int col = threadIdx.x; col < dim; col += blockDim.x) {
        float value = __half2float(row[col]);
        sum += value * value;
    }
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(mask, sum, offset);
    }
    __shared__ float warp[4];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    if (lane == 0) {
        warp[warp_id] = sum;
    }
    __syncthreads();
    __shared__ float reduced;
    if (threadIdx.x == 0) {
        float total = warp[0] + warp[1] + warp[2] + warp[3];
        reduced = rsqrtf(total / static_cast<float>(dim) + eps);
    }
    __syncthreads();
    for (int col = threadIdx.x; col < dim; col += blockDim.x) {
        float scale = gain == nullptr ? 1.0f : __half2float(gain[col]);
        row[col] = __float2half(__half2float(row[col]) * reduced * scale);
    }
}

extern "C" API int engine_head_norm(void* values, const void* weight, int heads, int dim, float eps, int unit) {
    head_norm_kernel<<<heads, 128, 0, g_stream>>>(
        static_cast<__half*>(values), static_cast<const __half*>(weight), heads, dim, eps, unit);
    return check(cudaGetLastError());
}

__global__ void rope_kernel(
    __half* query, __half* key, int query_heads, int key_heads, int dim, int rotating, float theta, int position) {
    int pairs = dim >> 1;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int span = query_heads * pairs;
    __half* plane = query;
    int local = index;
    if (index >= span) {
        local = index - span;
        if (local >= key_heads * pairs) {
            return;
        }
        plane = key;
    }
    int head = local / pairs;
    int pair = local - head * pairs;
    float cosine = 1.0f;
    float sine = 0.0f;
    int at_pos = position < 0 ? step_position() : position;
    if (pair < rotating) {
        float fraction = (2.0f * static_cast<float>(pair)) / static_cast<float>(dim);
        float angle = static_cast<float>(at_pos) / powf(theta, fraction);
        cosine = cosf(angle);
        sine = sinf(angle);
    }
    int base = head * dim;
    float left = __half2float(plane[base + pair]);
    float right = __half2float(plane[base + pairs + pair]);
    plane[base + pair] = __float2half(left * cosine - right * sine);
    plane[base + pairs + pair] = __float2half(right * cosine + left * sine);
}

extern "C" API int engine_rope(
    void* query, void* key, int query_heads, int key_heads, int dim, int rotating, float theta, int position) {
    int pairs = dim >> 1;
    int count = (query_heads + key_heads) * pairs;
    int threads = 128;
    rope_kernel<<<(count + threads - 1) / threads, threads, 0, g_stream>>>(
        static_cast<__half*>(query), static_cast<__half*>(key), query_heads, key_heads, dim, rotating, theta,
        position);
    return check(cudaGetLastError());
}

__global__ void store_kv_kernel(const __half* source, __half* cache, int heads, int dim, int position) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int count = heads * dim;
    if (index >= count) {
        return;
    }
    int pos = position < 0 ? step_position() : position;
    cache[static_cast<long long>(pos) * count + index] = source[index];
}

extern "C" API int engine_store_kv(const void* source, void* cache, int heads, int dim, int position) {
    int count = heads * dim;
    store_kv_kernel<<<(count + 127) / 128, 128, 0, g_stream>>>(
        static_cast<const __half*>(source), static_cast<__half*>(cache), heads, dim, position);
    return check(cudaGetLastError());
}

__global__ void attend_kernel(
    const __half* query, const __half* key, const __half* value, __half* destination, int q_heads, int kv_heads,
    int dim, int position, int window) {
    int head = blockIdx.x;
    if (head >= q_heads) {
        return;
    }
    int group = q_heads / kv_heads;
    int kv = head / group;
    const __half* q = query + static_cast<long long>(head) * dim;
    int begin = 0;
    int end = position + 1;
    if (window > 0 && end - begin > window) {
        begin = end - window;
    }
    unsigned mask = 0xffffffffu;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    float peak = -1.0e30f;
    for (int pos = begin + threadIdx.x; pos < end; pos += blockDim.x) {
        const __half* k = key + (static_cast<long long>(pos) * kv_heads + kv) * dim;
        float dot = 0.0f;
        for (int col = 0; col < dim; ++col) {
            dot += __half2float(q[col]) * __half2float(k[col]);
        }
        peak = fmaxf(peak, dot);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        peak = fmaxf(peak, __shfl_xor_sync(mask, peak, offset));
    }
    __shared__ float peaks[4];
    if (lane == 0) {
        peaks[warp_id] = peak;
    }
    __syncthreads();
    __shared__ float reduced;
    if (threadIdx.x == 0) {
        float best = peaks[0];
        best = fmaxf(best, peaks[1]);
        best = fmaxf(best, peaks[2]);
        best = fmaxf(best, peaks[3]);
        reduced = best;
    }
    __syncthreads();
    peak = reduced;
    float sum = 0.0f;
    extern __shared__ float mix[];
    for (int col = threadIdx.x; col < dim; col += blockDim.x) {
        mix[col] = 0.0f;
    }
    __syncthreads();
    for (int pos = begin; pos < end; ++pos) {
        const __half* k = key + (static_cast<long long>(pos) * kv_heads + kv) * dim;
        float dot = 0.0f;
        for (int col = threadIdx.x; col < dim; col += blockDim.x) {
            dot += __half2float(q[col]) * __half2float(k[col]);
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_xor_sync(mask, dot, offset);
        }
        __shared__ float dots[4];
        if (lane == 0) {
            dots[warp_id] = dot;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float total = dots[0] + dots[1] + dots[2] + dots[3];
            dots[0] = expf(total - peak);
        }
        __syncthreads();
        float weight = dots[0];
        if (threadIdx.x == 0) {
            sum += weight;
        }
        const __half* v = value + (static_cast<long long>(pos) * kv_heads + kv) * dim;
        for (int col = threadIdx.x; col < dim; col += blockDim.x) {
            mix[col] += weight * __half2float(v[col]);
        }
        __syncthreads();
    }
    __shared__ float denom;
    if (threadIdx.x == 0) {
        denom = sum > 0.0f ? 1.0f / sum : 0.0f;
    }
    __syncthreads();
    __half* out = destination + static_cast<long long>(head) * dim;
    for (int col = threadIdx.x; col < dim; col += blockDim.x) {
        out[col] = __float2half(mix[col] * denom);
    }
}

extern "C" API int engine_attend(
    const void* query, const void* key, const void* value, void* destination, int q_heads, int kv_heads, int dim,
    int position, int window) {
    int shared = dim * static_cast<int>(sizeof(float));
    attend_kernel<<<q_heads, 128, shared, g_stream>>>(
        static_cast<const __half*>(query), static_cast<const __half*>(key), static_cast<const __half*>(value),
        static_cast<__half*>(destination), q_heads, kv_heads, dim, position, window);
    return check(cudaGetLastError());
}

__global__ void rope_span_kernel(
    __half* query, __half* key, int query_heads, int key_heads, int dim, int rotating, float theta, int base,
    int tokens) {
    int token = blockIdx.y;
    if (token >= tokens) {
        return;
    }
    int pairs = dim >> 1;
    int span = query_heads * pairs;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    __half* plane = query + static_cast<long long>(token) * query_heads * dim;
    int local = index;
    if (index >= span) {
        local = index - span;
        if (local >= key_heads * pairs) {
            return;
        }
        plane = key + static_cast<long long>(token) * key_heads * dim;
    }
    int head = local / pairs;
    int pair = local - head * pairs;
    float cosine = 1.0f;
    float sine = 0.0f;
    int origin = base < 0 ? step_position() : base;
    if (pair < rotating) {
        float fraction = (2.0f * static_cast<float>(pair)) / static_cast<float>(dim);
        float angle = static_cast<float>(origin + token) / powf(theta, fraction);
        cosine = cosf(angle);
        sine = sinf(angle);
    }
    int at = head * dim;
    float left = __half2float(plane[at + pair]);
    float right = __half2float(plane[at + pairs + pair]);
    plane[at + pair] = __float2half(left * cosine - right * sine);
    plane[at + pairs + pair] = __float2half(right * cosine + left * sine);
}

extern "C" API int engine_rope_span(
    void* query, void* key, int query_heads, int key_heads, int dim, int rotating, float theta, int base, int tokens) {
    if (tokens < 1 || dim < 2) {
        return 1;
    }
    int pairs = dim >> 1;
    int count = (query_heads + key_heads) * pairs;
    dim3 grid((count + 127) / 128, tokens);
    rope_span_kernel<<<grid, 128, 0, g_stream>>>(
        static_cast<__half*>(query), static_cast<__half*>(key), query_heads, key_heads, dim, rotating, theta, base,
        tokens);
    return check(cudaGetLastError());
}

__global__ void store_span_kernel(
    const __half* source, __half* cache, int heads, int dim, int base, int tokens) {
    int token = blockIdx.y;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int count = heads * dim;
    if (token >= tokens || index >= count) {
        return;
    }
    int origin = base < 0 ? step_position() : base;
    cache[static_cast<long long>(origin + token) * count + index] =
        source[static_cast<long long>(token) * count + index];
}

extern "C" API int engine_store_span(
    const void* source, void* cache, int heads, int dim, int base, int tokens) {
    if (tokens < 1 || heads < 1 || dim < 1) {
        return 1;
    }
    int count = heads * dim;
    dim3 grid((count + 127) / 128, tokens);
    store_span_kernel<<<grid, 128, 0, g_stream>>>(
        static_cast<const __half*>(source), static_cast<__half*>(cache), heads, dim, base, tokens);
    return check(cudaGetLastError());
}

// One warp scores one query head. Keys are the positions this token is allowed to see.
__global__ void attend_span_kernel(
    const __half* query, const __half* key, const __half* value, __half* destination, int q_heads, int kv_heads,
    int dim, int base, int tokens, int window) {
    int token = blockIdx.x;
    int head = blockIdx.y;
    if (token >= tokens || head >= q_heads) {
        return;
    }
    int lane = threadIdx.x & 31;
    int group = q_heads / kv_heads;
    int kv = head / group;
    int origin = base < 0 ? step_position() : base;
    int position = origin + token;
    int end = position + 1;
    int begin = 0;
    if (window > 0 && end > window) {
        begin = end - window;
    }
    const __half2* q2 = reinterpret_cast<const __half2*>(query + (static_cast<long long>(token) * q_heads + head) * dim);
    int pairs = dim >> 1;
    float ox[8];
    float oy[8];
#pragma unroll
    for (int slot = 0; slot < 8; ++slot) {
        ox[slot] = 0.0f;
        oy[slot] = 0.0f;
    }
    float peak = -1.0e30f;
    float sum = 0.0f;
    for (int pos = begin; pos < end; ++pos) {
        const __half* krow = key + (static_cast<long long>(pos) * kv_heads + kv) * dim;
        const __half2* k2 = reinterpret_cast<const __half2*>(krow);
        float dot = 0.0f;
        for (int col = lane; col < pairs; col += 32) {
            __half2 left = q2[col];
            __half2 right = k2[col];
            dot = fmaf(__half2float(left.x), __half2float(right.x), dot);
            dot = fmaf(__half2float(left.y), __half2float(right.y), dot);
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_xor_sync(0xffffffffu, dot, offset);
        }
        float next = fmaxf(peak, dot);
        float alpha = __expf(peak - next);
        float prob = __expf(dot - next);
        sum = sum * alpha + prob;
        const __half2* v2 = reinterpret_cast<const __half2*>(value + (static_cast<long long>(pos) * kv_heads + kv) * dim);
        int slot = 0;
        for (int col = lane; col < pairs; col += 32, ++slot) {
            __half2 sample = v2[col];
            ox[slot] = fmaf(prob, __half2float(sample.x), ox[slot] * alpha);
            oy[slot] = fmaf(prob, __half2float(sample.y), oy[slot] * alpha);
        }
        peak = next;
    }
    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    __half2* out = reinterpret_cast<__half2*>(destination + (static_cast<long long>(token) * q_heads + head) * dim);
    int slot = 0;
    for (int col = lane; col < pairs; col += 32, ++slot) {
        out[col] = __halves2half2(__float2half(ox[slot] * inv), __float2half(oy[slot] * inv));
    }
}

template <int kDim>
__global__ void attend_wide_kernel(
    const __half* query, const __half* key, const __half* value, int q_heads, int kv_heads, int base, int window,
    int splits, float* peaks, float* sums, float* acc) {
    constexpr int kVec = kDim / 128;
    int head = blockIdx.x;
    int split = blockIdx.y;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int group = q_heads / kv_heads;
    int kv = head / group;
    int origin = base < 0 ? step_position() : base;
    int end = origin + 1;
    int begin = 0;
    if (window > 0 && end > window) {
        begin = end - window;
    }
    int length = end - begin;
    int chunk = (length + splits - 1) / splits;
    int pos0 = begin + split * chunk;
    int pos1 = pos0 + chunk;
    if (pos1 > end) {
        pos1 = end;
    }
    float mine[kVec * 4];
#pragma unroll
    for (int item = 0; item < kVec * 4; ++item) {
        mine[item] = 0.0f;
    }
    float peak = -1.0e30f;
    float sum = 0.0f;
    if (head < q_heads && pos0 < pos1) {
        const uint2* qv = reinterpret_cast<const uint2*>(query + static_cast<long long>(head) * kDim);
        uint2 qreg[kVec];
#pragma unroll
        for (int item = 0; item < kVec; ++item) {
            qreg[item] = qv[lane + item * 32];
        }
        for (int pos = pos0 + warp; pos < pos1; pos += 4) {
            const uint2* kvec = reinterpret_cast<const uint2*>(
                key + (static_cast<long long>(pos) * kv_heads + kv) * kDim);
            float dot = 0.0f;
#pragma unroll
            for (int item = 0; item < kVec; ++item) {
                uint2 packed = kvec[lane + item * 32];
                __half2 low = *reinterpret_cast<__half2*>(&packed.x);
                __half2 high = *reinterpret_cast<__half2*>(&packed.y);
                uint2 query_pack = qreg[item];
                __half2 qlow = *reinterpret_cast<__half2*>(&query_pack.x);
                __half2 qhigh = *reinterpret_cast<__half2*>(&query_pack.y);
                dot = fmaf(__half2float(qlow.x), __half2float(low.x), dot);
                dot = fmaf(__half2float(qlow.y), __half2float(low.y), dot);
                dot = fmaf(__half2float(qhigh.x), __half2float(high.x), dot);
                dot = fmaf(__half2float(qhigh.y), __half2float(high.y), dot);
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                dot += __shfl_xor_sync(0xffffffffu, dot, offset);
            }
            float next = fmaxf(peak, dot);
            float alpha = __expf(peak - next);
            float prob = __expf(dot - next);
            sum = sum * alpha + prob;
            const uint2* vvec = reinterpret_cast<const uint2*>(
                value + (static_cast<long long>(pos) * kv_heads + kv) * kDim);
#pragma unroll
            for (int item = 0; item < kVec; ++item) {
                uint2 packed = vvec[lane + item * 32];
                __half2 low = *reinterpret_cast<__half2*>(&packed.x);
                __half2 high = *reinterpret_cast<__half2*>(&packed.y);
                int base_item = item * 4;
                mine[base_item] = fmaf(prob, __half2float(low.x), mine[base_item] * alpha);
                mine[base_item + 1] = fmaf(prob, __half2float(low.y), mine[base_item + 1] * alpha);
                mine[base_item + 2] = fmaf(prob, __half2float(high.x), mine[base_item + 2] * alpha);
                mine[base_item + 3] = fmaf(prob, __half2float(high.y), mine[base_item + 3] * alpha);
            }
            peak = next;
        }
    }
    int part = ((split * q_heads + head) * 4) + warp;
    if (lane == 0) {
        peaks[part] = peak;
        sums[part] = sum;
    }
    float* slot = acc + static_cast<long long>(part) * kDim;
#pragma unroll
    for (int item = 0; item < kVec; ++item) {
        int col = (lane + item * 32) * 4;
        slot[col] = mine[item * 4];
        slot[col + 1] = mine[item * 4 + 1];
        slot[col + 2] = mine[item * 4 + 2];
        slot[col + 3] = mine[item * 4 + 3];
    }
}

__global__ void attend_join_kernel(
    __half* destination, int q_heads, int dim, int base, int window, int splits, const float* peaks,
    const float* sums, const float* acc) {
    int head = blockIdx.x;
    if (head >= q_heads) {
        return;
    }
    int origin = base < 0 ? step_position() : base;
    int end = origin + 1;
    int begin = 0;
    if (window > 0 && end > window) {
        begin = end - window;
    }
    int length = end - begin;
    int chunk = (length + splits - 1) / splits;
    int owned = dim >> 7;
    float mine[4];
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        mine[item] = 0.0f;
    }
    float peak = -1.0e30f;
    float sum = 0.0f;
    for (int split = 0; split < splits; ++split) {
        int pos0 = begin + split * chunk;
        if (pos0 >= end) {
            continue;
        }
        for (int warp = 0; warp < 4; ++warp) {
            int part = ((split * q_heads + head) * 4) + warp;
            float part_sum = sums[part];
            if (part_sum == 0.0f) {
                continue;
            }
            float part_peak = peaks[part];
            float next = fmaxf(peak, part_peak);
            float left = __expf(peak - next);
            float right = __expf(part_peak - next);
            sum = sum * left + part_sum * right;
            const float* slot = acc + static_cast<long long>(part) * dim;
            for (int item = 0; item < owned; ++item) {
                int col = threadIdx.x + item * 128;
                mine[item] = fmaf(right, slot[col], mine[item] * left);
            }
            peak = next;
        }
    }
    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    __half* out = destination + static_cast<long long>(head) * dim;
    for (int item = 0; item < owned; ++item) {
        out[threadIdx.x + item * 128] = __float2half(mine[item] * inv);
    }
}

extern "C" API int engine_attend_span(
    const void* query, const void* key, const void* value, void* destination, int q_heads, int kv_heads, int dim,
    int base, int tokens, int window) {
    if (tokens < 1 || q_heads < 1 || kv_heads < 1 || dim < 2 || (dim & 1) || dim > 512) {
        return 1;
    }
    if (tokens == 1 && q_heads <= kAttendHeads && (dim & 127) == 0 && g_partial_peak != nullptr) {
        int splits = g_splits;
        if (window > 0) {
            int cap = (window + 127) / 128;
            if (cap < splits) {
                splits = cap;
            }
        }
        if (splits < 1) {
            splits = 1;
        }
        dim3 grid(q_heads, splits);
        if (dim == 512) {
            attend_wide_kernel<512><<<grid, 128, 0, g_stream>>>(
                static_cast<const __half*>(query), static_cast<const __half*>(key), static_cast<const __half*>(value),
                q_heads, kv_heads, base, window, splits, g_partial_peak, g_partial_sum, g_partial_acc);
        } else {
            attend_wide_kernel<256><<<grid, 128, 0, g_stream>>>(
                static_cast<const __half*>(query), static_cast<const __half*>(key), static_cast<const __half*>(value),
                q_heads, kv_heads, base, window, splits, g_partial_peak, g_partial_sum, g_partial_acc);
        }
        attend_join_kernel<<<q_heads, 128, 0, g_stream>>>(
            static_cast<__half*>(destination), q_heads, dim, base, window, splits, g_partial_peak, g_partial_sum,
            g_partial_acc);
        return check(cudaGetLastError());
    }
    dim3 grid(tokens, q_heads);
    attend_span_kernel<<<grid, 32, 0, g_stream>>>(
        static_cast<const __half*>(query), static_cast<const __half*>(key), static_cast<const __half*>(value),
        static_cast<__half*>(destination), q_heads, kv_heads, dim, base, tokens, window);
    return check(cudaGetLastError());
}

__global__ void add_kernel(const __half* left, const __half* right, __half* destination, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        destination[index] = __float2half(__half2float(left[index]) + __half2float(right[index]));
    }
}

extern "C" API int engine_add(const void* left, const void* right, void* destination, int count) {
    add_kernel<<<(count + 127) / 128, 128, 0, g_stream>>>(
        static_cast<const __half*>(left), static_cast<const __half*>(right), static_cast<__half*>(destination), count);
    return check(cudaGetLastError());
}

__global__ void scale_kernel(__half* values, float factor, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        values[index] = __float2half(__half2float(values[index]) * factor);
    }
}

extern "C" API int engine_scale(void* values, float factor, int count) {
    scale_kernel<<<(count + 127) / 128, 128, 0, g_stream>>>(static_cast<__half*>(values), factor, count);
    return check(cudaGetLastError());
}

__global__ void copy_kernel(const __half* source, __half* destination, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        destination[index] = source[index];
    }
}

extern "C" API int engine_copy(const void* source, void* destination, int count) {
    copy_kernel<<<(count + 127) / 128, 128, 0, g_stream>>>(
        static_cast<const __half*>(source), static_cast<__half*>(destination), count);
    return check(cudaGetLastError());
}

__device__ float gelu_tanh(float value) {
    const float k = 0.7978845608028654f;
    const float cube = 0.044715f;
    float inner = k * (value + cube * value * value * value);
    return 0.5f * value * (1.0f + tanhf(inner));
}

__global__ void geglu_kernel(const __half* fused, __half* destination, int width) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= width) {
        return;
    }
    float gate = __half2float(fused[index]);
    float up = __half2float(fused[width + index]);
    destination[index] = __float2half(gelu_tanh(gate) * up);
}

extern "C" API int engine_geglu(const void* fused, void* destination, int width) {
    geglu_kernel<<<(width + 127) / 128, 128, 0, g_stream>>>(
        static_cast<const __half*>(fused), static_cast<__half*>(destination), width);
    return check(cudaGetLastError());
}

__global__ void geglu_rows_kernel(const __half* fused, __half* destination, int width, int rows) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int count = width * rows;
    if (index >= count) {
        return;
    }
    int row = index / width;
    int col = index - row * width;
    const __half* base = fused + static_cast<long long>(row) * (width * 2) + col;
    float gate = __half2float(base[0]);
    float up = __half2float(base[width]);
    destination[index] = __float2half(gelu_tanh(gate) * up);
}

extern "C" API int engine_geglu_rows(const void* fused, void* destination, int width, int rows) {
    int count = width * rows;
    geglu_rows_kernel<<<(count + 127) / 128, 128, 0, g_stream>>>(
        static_cast<const __half*>(fused), static_cast<__half*>(destination), width, rows);
    return check(cudaGetLastError());
}

__global__ void router_kernel(const __half* logits, const __half* expert_scale, int* ids, float* weights, int count, int keep) {
    int token = blockIdx.x;
    logits += static_cast<long long>(token) * count;
    ids += static_cast<long long>(token) * keep;
    weights += static_cast<long long>(token) * keep;
    __shared__ float scores[128];
    __shared__ int alive[128];
    int lane = threadIdx.x;
    scores[lane] = lane < count ? __half2float(logits[lane]) : -1.0e30f;
    alive[lane] = lane < count ? 1 : 0;
    __syncthreads();
    __shared__ int picked_i[8];
    __shared__ float picked_v[8];
    if (lane == 0) {
        for (int rank = 0; rank < keep; ++rank) {
            int winner = 0;
            float best = -1.0e30f;
            for (int expert = 0; expert < count; ++expert) {
                if (!alive[expert]) {
                    continue;
                }
                float value = scores[expert];
                if (value > best || (value == best && expert < winner)) {
                    best = value;
                    winner = expert;
                }
            }
            alive[winner] = 0;
            picked_i[rank] = winner;
            picked_v[rank] = scores[winner];
            ids[rank] = winner;
        }
        float peak = picked_v[0];
        for (int rank = 1; rank < keep; ++rank) {
            peak = fmaxf(peak, picked_v[rank]);
        }
        float sum = 0.0f;
        float chosen[8];
        for (int rank = 0; rank < keep; ++rank) {
            chosen[rank] = expf(picked_v[rank] - peak);
            sum += chosen[rank];
        }
        float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        for (int rank = 0; rank < keep; ++rank) {
            weights[rank] = chosen[rank] * inv * __half2float(expert_scale[picked_i[rank]]);
        }
    }
}

extern "C" API int engine_router(const void* logits, const void* expert_scale, void* ids, void* weights, int count, int keep) {
    router_kernel<<<1, 128, 0, g_stream>>>(
        static_cast<const __half*>(logits), static_cast<const __half*>(expert_scale), static_cast<int*>(ids),
        static_cast<float*>(weights), count, keep);
    return check(cudaGetLastError());
}

extern "C" API int engine_router_tokens(
    const void* logits, const void* expert_scale, void* ids, void* weights, int count, int keep, int tokens) {
    router_kernel<<<tokens, 128, 0, g_stream>>>(
        static_cast<const __half*>(logits), static_cast<const __half*>(expert_scale), static_cast<int*>(ids),
        static_cast<float*>(weights), count, keep);
    return check(cudaGetLastError());
}

__global__ void mix_kernel(__half* acc, const __half* row, const float* weight, int slot, int width) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= width) {
        return;
    }
    float value = __half2float(acc[index]) + __half2float(row[index]) * weight[slot];
    acc[index] = __float2half(value);
}

extern "C" API int engine_mix(void* acc, const void* row, const void* weight, int slot, int width) {
    mix_kernel<<<(width + 127) / 128, 128, 0, g_stream>>>(
        static_cast<__half*>(acc), static_cast<const __half*>(row), static_cast<const float*>(weight), slot, width);
    return check(cudaGetLastError());
}

__global__ void embed_kernel(const int8_t* table, const float* scale, int token, __half* destination, int inner, float gain) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= inner) {
        return;
    }
    float value = static_cast<float>(table[static_cast<long long>(token) * inner + col]) * scale[token] * gain;
    destination[col] = __float2half(value);
}

extern "C" API int engine_embed(const void* table, const void* scale, int token, void* destination, int inner, float gain) {
    embed_kernel<<<(inner + 127) / 128, 128, 0, g_stream>>>(
        static_cast<const int8_t*>(table), static_cast<const float*>(scale), token, static_cast<__half*>(destination),
        inner, gain);
    return check(cudaGetLastError());
}

__global__ void softcap_kernel(__half* logits, float cap, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        float value = __half2float(logits[index]);
        logits[index] = __float2half(cap * tanhf(value / cap));
    }
}

extern "C" API int engine_softcap(void* logits, float cap, int count) {
    softcap_kernel<<<(count + 127) / 128, 128, 0, g_stream>>>(static_cast<__half*>(logits), cap, count);
    return check(cudaGetLastError());
}

__global__ void argmax_kernel(const __half* logits, int* destination, int count) {
    __shared__ float best_v[32];
    __shared__ int best_i[32];
    float mine = -1.0e30f;
    int which = 0;
    for (int index = threadIdx.x; index < count; index += blockDim.x) {
        float value = __half2float(logits[index]);
        if (value > mine) {
            mine = value;
            which = index;
        }
    }
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(mask, mine, offset);
        int other_i = __shfl_xor_sync(mask, which, offset);
        if (other > mine) {
            mine = other;
            which = other_i;
        }
    }
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    if (lane == 0) {
        best_v[warp_id] = mine;
        best_i[warp_id] = which;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float best = best_v[0];
        int id = best_i[0];
        int warps = blockDim.x >> 5;
        for (int warp = 1; warp < warps; ++warp) {
            if (best_v[warp] > best) {
                best = best_v[warp];
                id = best_i[warp];
            }
        }
        destination[0] = id;
    }
}

extern "C" API int engine_argmax(const void* logits, void* destination, int count) {
    argmax_kernel<<<1, 256, 0, g_stream>>>(static_cast<const __half*>(logits), static_cast<int*>(destination), count);
    return check(cudaGetLastError());
}

__global__ void penalize_kernel(__half* logits, const int* counts, float presence, int vocab) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < vocab && counts[index] > 0) {
        logits[index] = __float2half(__half2float(logits[index]) - presence);
    }
}

extern "C" API int engine_penalize(void* logits, const void* counts, float presence, int vocab) {
    penalize_kernel<<<(vocab + 127) / 128, 128, 0, g_stream>>>(
        static_cast<__half*>(logits), static_cast<const int*>(counts), presence, vocab);
    return check(cudaGetLastError());
}

__global__ void count_kernel(int* counts, const int* token) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        counts[token[0]] += 1;
    }
}

extern "C" API int engine_count(void* counts, const void* token) {
    count_kernel<<<1, 1, 0, g_stream>>>(static_cast<int*>(counts), static_cast<const int*>(token));
    return check(cudaGetLastError());
}

__global__ void gemv_f16_kernel(
    const __half* weight, const __half* source, __half* destination, int rows, int inner) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const __half* line = weight + static_cast<long long>(row) * inner;
    float acc = 0.0f;
    for (int col = threadIdx.x; col < inner; col += blockDim.x) {
        acc += __half2float(line[col]) * __half2float(source[col]);
    }
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(mask, acc, offset);
    }
    __shared__ float warp[4];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    if (lane == 0) {
        warp[warp_id] = acc;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        destination[row] = __float2half(warp[0] + warp[1] + warp[2] + warp[3]);
    }
}

extern "C" API int engine_gemv_f16(const void* weight, const void* source, void* destination, int rows, int inner) {
    gemv_f16_kernel<<<rows, 128, 0, g_stream>>>(
        static_cast<const __half*>(weight), static_cast<const __half*>(source), static_cast<__half*>(destination),
        rows, inner);
    return check(cudaGetLastError());
}

__global__ void gemv_f16_tokens_kernel(
    const __half* weight, const __half* source, __half* destination, int rows, int inner, int tokens) {
    int row = blockIdx.x;
    int token = blockIdx.y;
    if (row >= rows || token >= tokens) {
        return;
    }
    const __half* line = weight + static_cast<long long>(row) * inner;
    const __half* act = source + static_cast<long long>(token) * inner;
    float acc = 0.0f;
    for (int col = threadIdx.x; col < inner; col += blockDim.x) {
        acc += __half2float(line[col]) * __half2float(act[col]);
    }
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(mask, acc, offset);
    }
    __shared__ float warp[4];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    if (lane == 0) {
        warp[warp_id] = acc;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        destination[static_cast<long long>(token) * rows + row] = __float2half(warp[0] + warp[1] + warp[2] + warp[3]);
    }
}

extern "C" API int engine_gemv_f16_tokens(
    const void* weight, const void* source, void* destination, int rows, int inner, int tokens) {
    dim3 grid(rows, tokens);
    gemv_f16_tokens_kernel<<<grid, 128, 0, g_stream>>>(
        static_cast<const __half*>(weight), static_cast<const __half*>(source), static_cast<__half*>(destination),
        rows, inner, tokens);
    return check(cudaGetLastError());
}

__global__ void gather_f16_kernel(const __half* source, __half* destination, const int* index, int rows, int width) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || col >= width) {
        return;
    }
    destination[static_cast<long long>(row) * width + col] =
        source[static_cast<long long>(index[row]) * width + col];
}

extern "C" API int engine_gather_f16(const void* source, void* destination, const void* index, int rows, int width) {
    dim3 grid((width + 127) / 128, rows);
    gather_f16_kernel<<<grid, 128, 0, g_stream>>>(
        static_cast<const __half*>(source), static_cast<__half*>(destination), static_cast<const int*>(index), rows,
        width);
    return check(cudaGetLastError());
}

__global__ void scatter_add_f16_kernel(
    __half* destination, const __half* source, const int* token, const int* slot, const float* route, int rows,
    int width, int topk) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || col >= width) {
        return;
    }
    int at = token[row];
    float scale = route[static_cast<long long>(at) * topk + slot[row]];
    __half* out = destination + static_cast<long long>(at) * width + col;
    out[0] = __float2half(__half2float(out[0]) + __half2float(source[static_cast<long long>(row) * width + col]) * scale);
}

extern "C" API int engine_scatter_add_f16(
    void* destination, const void* source, const void* token, const void* slot, const void* route, int rows, int width,
    int topk) {
    dim3 grid((width + 127) / 128, rows);
    scatter_add_f16_kernel<<<grid, 128, 0, g_stream>>>(
        static_cast<__half*>(destination), static_cast<const __half*>(source), static_cast<const int*>(token),
        static_cast<const int*>(slot), static_cast<const float*>(route), rows, width, topk);
    return check(cudaGetLastError());
}

__global__ void geglu_strided_kernel(const __half* fused, __half* destination, int width, int stride, int rows) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int count = width * rows;
    if (index >= count) {
        return;
    }
    int row = index / width;
    int col = index - row * width;
    const __half* base = fused + static_cast<long long>(row) * (width * 2) + col;
    float gate = __half2float(base[0]);
    float up = __half2float(base[width]);
    destination[static_cast<long long>(row) * stride + col] = __float2half(gelu_tanh(gate) * up);
}

extern "C" API int engine_geglu_strided(const void* fused, void* destination, int width, int stride, int rows) {
    int count = width * rows;
    geglu_strided_kernel<<<(count + 127) / 128, 128, 0, g_stream>>>(
        static_cast<const __half*>(fused), static_cast<__half*>(destination), width, stride, rows);
    return check(cudaGetLastError());
}

__global__ void geglu_pair_kernel(const __half* gate, const __half* up, __half* destination, int width, int rows) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int count = width * rows;
    if (index >= count) {
        return;
    }
    float gate_v = __half2float(gate[index]);
    float up_v = __half2float(up[index]);
    destination[index] = __float2half(gelu_tanh(gate_v) * up_v);
}

extern "C" API int engine_geglu_pair(const void* gate, const void* up, void* destination, int width, int rows) {
    int count = width * rows;
    geglu_pair_kernel<<<(count + 127) / 128, 128, 0, g_stream>>>(
        static_cast<const __half*>(gate), static_cast<const __half*>(up), static_cast<__half*>(destination), width,
        rows);
    return check(cudaGetLastError());
}

__device__ void insert_top(int* ids, float* vals, int* held, int keep, int index, float value) {
    if (*held < keep) {
        int slot = *held;
        while (slot > 0 && value > vals[slot - 1]) {
            ids[slot] = ids[slot - 1];
            vals[slot] = vals[slot - 1];
            slot -= 1;
        }
        ids[slot] = index;
        vals[slot] = value;
        *held += 1;
        return;
    }
    if (value <= vals[*held - 1]) {
        return;
    }
    int slot = *held - 1;
    while (slot > 0 && value > vals[slot - 1]) {
        ids[slot] = ids[slot - 1];
        vals[slot] = vals[slot - 1];
        slot -= 1;
    }
    ids[slot] = index;
    vals[slot] = value;
}

__global__ void sample_partial_kernel(
    const __half* logits, int* out_ids, float* out_vals, int vocab, float inv, int keep) {
    int local_ids[64];
    float local_vals[64];
    int held = 0;
    for (int index = blockIdx.x; index < vocab; index += gridDim.x) {
        insert_top(local_ids, local_vals, &held, keep, index, __half2float(logits[index]) * inv);
    }
    int* ids = out_ids + blockIdx.x * 64;
    float* vals = out_vals + blockIdx.x * 64;
    for (int slot = 0; slot < 64; ++slot) {
        ids[slot] = slot < held ? local_ids[slot] : 0;
        vals[slot] = slot < held ? local_vals[slot] : -1.0e30f;
    }
}

__global__ void sample_finish_kernel(
    const int* partial_ids, const float* partial_vals, int* destination, int blocks, int keep, float top_p,
    float uniform) {
    if (threadIdx.x != 0) {
        return;
    }
    int ids[64];
    float vals[64];
    int held = 0;
    int count = blocks * 64;
    for (int index = 0; index < count; ++index) {
        insert_top(ids, vals, &held, keep, partial_ids[index], partial_vals[index]);
    }
    float peak = vals[0];
    float sum = 0.0f;
    float probs[64];
    for (int slot = 0; slot < held; ++slot) {
        probs[slot] = expf(vals[slot] - peak);
        sum += probs[slot];
    }
    float mass = 0.0f;
    float limit = top_p > 0.0f && top_p < 1.0f ? top_p * sum : sum;
    int used = 0;
    for (int slot = 0; slot < held; ++slot) {
        mass += probs[slot];
        used = slot + 1;
        if (mass >= limit) {
            break;
        }
    }
    float chosen = 0.0f;
    for (int slot = 0; slot < used; ++slot) {
        chosen += probs[slot];
    }
    float cursor = uniform;
    if (cursor < 0.0f) {
        cursor = 0.0f;
    }
    if (cursor > 0.999999f) {
        cursor = 0.999999f;
    }
    cursor *= chosen > 0.0f ? chosen : 1.0f;
    float walked = 0.0f;
    int pick = held > 0 ? ids[used - 1] : 0;
    for (int slot = 0; slot < used; ++slot) {
        walked += probs[slot];
        if (walked >= cursor) {
            pick = ids[slot];
            break;
        }
    }
    destination[0] = pick;
}

extern "C" API int engine_sample(
    const void* logits, void* destination, int vocab, float temperature, int top_k, float top_p, float uniform) {
    if (temperature <= 1.0e-6f) {
        argmax_kernel<<<1, 256, 0, g_stream>>>(
            static_cast<const __half*>(logits), static_cast<int*>(destination), vocab);
        return check(cudaGetLastError());
    }
    constexpr int kBlocks = 128;
    int keep = top_k;
    if (keep < 1 || keep > 64) {
        keep = 64;
    }
    int* partial_ids = nullptr;
    float* partial_vals = nullptr;
    if (cudaMalloc(&partial_ids, kBlocks * 64 * sizeof(int)) != cudaSuccess) {
        return 3;
    }
    if (cudaMalloc(&partial_vals, kBlocks * 64 * sizeof(float)) != cudaSuccess) {
        cudaFree(partial_ids);
        return 3;
    }
    sample_partial_kernel<<<kBlocks, 1, 0, g_stream>>>(
        static_cast<const __half*>(logits), partial_ids, partial_vals, vocab, 1.0f / temperature, keep);
    sample_finish_kernel<<<1, 1, 0, g_stream>>>(
        partial_ids, partial_vals, static_cast<int*>(destination), kBlocks, keep, top_p, uniform);
    int status = check(cudaGetLastError());
    if (status == 0) {
        status = check(cudaStreamSynchronize(g_stream));
    }
    cudaFree(partial_ids);
    cudaFree(partial_vals);
    return status;
}

extern "C" API int engine_selfcheck() {
    constexpr int kTokens = 16;
    constexpr int kRows = 8;
    constexpr int kInner = 16;
    int8_t codes[kRows * kInner] = {};
    float scales[kRows];
    __half act[kTokens * kInner];
    for (int row = 0; row < kRows; ++row) {
        codes[row * kInner] = 1;
        scales[row] = 1.0f;
    }
    for (int token = 0; token < kTokens; ++token) {
        for (int col = 0; col < kInner; ++col) {
            act[token * kInner + col] = __float2half(col == 0 ? 1.0f : 0.0f);
        }
    }
    int8_t* device_codes = static_cast<int8_t*>(engine_alloc(sizeof(codes)));
    float* device_scales = static_cast<float*>(engine_alloc(sizeof(scales)));
    __half* device_act = static_cast<__half*>(engine_alloc(sizeof(act)));
    __half* device_out = static_cast<__half*>(engine_alloc(kTokens * kRows * sizeof(__half)));
    if (device_codes == nullptr || device_scales == nullptr || device_act == nullptr || device_out == nullptr) {
        return 4;
    }
    engine_h2d(device_codes, codes, sizeof(codes));
    engine_h2d(device_scales, scales, sizeof(scales));
    engine_h2d(device_act, act, sizeof(act));
    if (engine_linear(device_codes, device_scales, device_act, device_out, kRows, kInner, kTokens, 8)) {
        return 5;
    }
    __half host[kTokens * kRows];
    if (engine_d2h(host, device_out, sizeof(host))) {
        return 6;
    }
    for (int index = 0; index < kTokens * kRows; ++index) {
        float value = __half2float(host[index]);
        if (value < 0.99f || value > 1.01f) {
            std::fprintf(stderr, "fp16 mma %d was %f\n", index, value);
            return 7;
        }
    }
    uint8_t nibble = 0x21;
    float one = 1.0f;
    __half pair[2] = {__float2half(1.0f), __float2half(1.0f)};
    uint8_t* device_nibble = static_cast<uint8_t*>(engine_alloc(1));
    float* device_one = static_cast<float*>(engine_alloc(sizeof(float)));
    __half* device_pair = static_cast<__half*>(engine_alloc(sizeof(pair)));
    __half* device_dot = static_cast<__half*>(engine_alloc(sizeof(__half)));
    engine_h2d(device_nibble, &nibble, 1);
    engine_h2d(device_one, &one, sizeof(one));
    engine_h2d(device_pair, pair, sizeof(pair));
    if (engine_linear(device_nibble, device_one, device_pair, device_dot, 1, 2, 1, 4)) {
        return 8;
    }
    __half dot{};
    engine_d2h(&dot, device_dot, sizeof(dot));
    if (__half2float(dot) < 2.99f || __half2float(dot) > 3.01f) {
        std::fprintf(stderr, "int4 nibble dot was %f\n", __half2float(dot));
        return 9;
    }
    alignas(4096) unsigned char page[4096];
    page[0] = 0x5a;
    if (engine_register_ro(page, sizeof(page))) {
        return 10;
    }
    unsigned char* device_page = static_cast<unsigned char*>(engine_alloc(sizeof(page)));
    if (engine_h2d_ptr(device_page, page, sizeof(page))) {
        engine_unregister(page);
        return 11;
    }
    unsigned char back = 0;
    engine_d2h(&back, device_page, 1);
    engine_unregister(page);
    if (back != 0x5a) {
        std::fprintf(stderr, "pinned copy was %u\n", back);
        return 12;
    }
    engine_free(device_codes);
    engine_free(device_scales);
    engine_free(device_act);
    engine_free(device_out);
    engine_free(device_nibble);
    engine_free(device_one);
    engine_free(device_pair);
    engine_free(device_dot);
    engine_free(device_page);
    std::printf("fp16 mma, int4 nibble, pinned copy\n");
    return 0;
}
