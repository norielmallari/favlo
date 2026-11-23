// Dispatch check: one s8 tensor-core MMA, one int4 nibble dot, and a pinned copy.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdint>

__global__ void mma_check(int* out) {
    int d0 = 0;
    int d1 = 0;
    int d2 = 0;
    int d3 = 0;
    unsigned a = 0x01010101u;
    unsigned b = 0x01010101u;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(a), "r"(a), "r"(a), "r"(a), "r"(b), "r"(b), "r"(0), "r"(0), "r"(0), "r"(0));
    if (threadIdx.x == 0) {
        out[0] = d0 + d1 + d2 + d3;
    }
}

__global__ void nibble_check(int* out) {
    unsigned char packed = 0x21;
    int lo = packed & 0xF;
    int hi = packed >> 4;
    out[0] = lo * 3 + hi * 4;
}

int main() {
    if (cudaSetDevice(0) != cudaSuccess) {
        std::fprintf(stderr, "no CUDA device\n");
        return 1;
    }
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        return 1;
    }
    if (prop.major * 10 + prop.minor < 86) {
        std::fprintf(stderr, "compute capability %d.%d is below 8.6\n", prop.major, prop.minor);
        return 1;
    }
    int* device = nullptr;
    if (cudaMalloc(&device, sizeof(int)) != cudaSuccess) {
        return 1;
    }
    mma_check<<<1, 32>>>(device);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        return 1;
    }
    int host = 0;
    cudaMemcpy(&host, device, sizeof(int), cudaMemcpyDeviceToHost);
    if (host == 0) {
        std::fprintf(stderr, "int8 mma produced 0\n");
        return 1;
    }
    nibble_check<<<1, 1>>>(device);
    cudaDeviceSynchronize();
    cudaMemcpy(&host, device, sizeof(int), cudaMemcpyDeviceToHost);
    if (host != 11) {
        std::fprintf(stderr, "int4 nibble dot was %d\n", host);
        return 1;
    }
    void* pinned = nullptr;
    if (cudaHostAlloc(&pinned, 256, cudaHostAllocDefault) != cudaSuccess) {
        return 1;
    }
    static_cast<unsigned char*>(pinned)[0] = 7;
    unsigned char* stage = nullptr;
    cudaMalloc(&stage, 256);
    cudaMemcpy(stage, pinned, 256, cudaMemcpyHostToDevice);
    unsigned char back = 0;
    cudaMemcpy(&back, stage, 1, cudaMemcpyDeviceToHost);
    cudaFree(stage);
    cudaFreeHost(pinned);
    cudaFree(device);
    if (back != 7) {
        std::fprintf(stderr, "pinned copy returned %u\n", back);
        return 1;
    }
    std::puts("ok");
    return 0;
}
