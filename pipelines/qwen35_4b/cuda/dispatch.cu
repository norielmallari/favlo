// int8 GEMM through cuBLASLt, and one s4 tensor-op tile through the toolkit MMA.
// The s4 tile is the CUDA WMMA shape m8n8k32. Nibbles are packed low-first.

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <mma.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kM = 64;
constexpr int kN = 64;
constexpr int kK = 64;

void fail(const char* what, int code) {
    std::fprintf(stderr, "%s (%d)\n", what, code);
    std::exit(1);
}

void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        fail(what, static_cast<int>(status));
    }
}

void check_blas(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fail(what, static_cast<int>(status));
    }
}

int8_t value_at(int index) {
    return static_cast<int8_t>((index * 3 % 7) - 3);
}

void check_int8() {
    std::vector<int8_t> host_a(static_cast<size_t>(kM) * kK);
    std::vector<int8_t> host_b(static_cast<size_t>(kK) * kN);
    std::vector<int32_t> expect(static_cast<size_t>(kM) * kN);
    for (int col = 0; col < kK; ++col) {
        for (int row = 0; row < kM; ++row) {
            host_a[static_cast<size_t>(row + col * kM)] = value_at(row + col);
        }
    }
    for (int col = 0; col < kN; ++col) {
        for (int row = 0; row < kK; ++row) {
            host_b[static_cast<size_t>(row + col * kK)] = value_at(row + 2 * col + 1);
        }
    }
    for (int col = 0; col < kN; ++col) {
        for (int row = 0; row < kM; ++row) {
            int32_t acc = 0;
            for (int inner = 0; inner < kK; ++inner) {
                acc += static_cast<int32_t>(host_a[static_cast<size_t>(row + inner * kM)]) *
                       static_cast<int32_t>(host_b[static_cast<size_t>(inner + col * kK)]);
            }
            expect[static_cast<size_t>(row + col * kM)] = acc;
        }
    }

    int8_t* device_a = nullptr;
    int8_t* device_b = nullptr;
    int32_t* device_c = nullptr;
    void* workspace = nullptr;
    constexpr size_t kWorkspace = 32U << 20;
    check_cuda(cudaMalloc(&device_a, host_a.size()), "cudaMalloc A");
    check_cuda(cudaMalloc(&device_b, host_b.size()), "cudaMalloc B");
    check_cuda(cudaMalloc(&device_c, expect.size() * sizeof(int32_t)), "cudaMalloc C");
    check_cuda(cudaMalloc(&workspace, kWorkspace), "cudaMalloc workspace");
    check_cuda(cudaMemcpy(device_a, host_a.data(), host_a.size(), cudaMemcpyHostToDevice), "copy A");
    check_cuda(cudaMemcpy(device_b, host_b.data(), host_b.size(), cudaMemcpyHostToDevice), "copy B");
    check_cuda(cudaMemset(device_c, 0, expect.size() * sizeof(int32_t)), "clear C");

    cublasLtHandle_t handle = nullptr;
    check_blas(cublasLtCreate(&handle), "cublasLtCreate");
    cublasLtMatmulDesc_t operation = nullptr;
    check_blas(cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32I, CUDA_R_32I), "matmul desc");
    cublasOperation_t no_trans = CUBLAS_OP_N;
    check_blas(
        cublasLtMatmulDescSetAttribute(
            operation, CUBLASLT_MATMUL_DESC_TRANSA, &no_trans, sizeof(no_trans)),
        "transa");
    check_blas(
        cublasLtMatmulDescSetAttribute(
            operation, CUBLASLT_MATMUL_DESC_TRANSB, &no_trans, sizeof(no_trans)),
        "transb");

    cublasLtMatrixLayout_t layout_a = nullptr;
    cublasLtMatrixLayout_t layout_b = nullptr;
    cublasLtMatrixLayout_t layout_c = nullptr;
    check_blas(cublasLtMatrixLayoutCreate(&layout_a, CUDA_R_8I, kM, kK, kM), "layout A");
    check_blas(cublasLtMatrixLayoutCreate(&layout_b, CUDA_R_8I, kK, kN, kK), "layout B");
    check_blas(cublasLtMatrixLayoutCreate(&layout_c, CUDA_R_32I, kM, kN, kM), "layout C");

    cublasLtMatmulPreference_t preference = nullptr;
    check_blas(cublasLtMatmulPreferenceCreate(&preference), "preference");
    check_blas(
        cublasLtMatmulPreferenceSetAttribute(
            preference,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &kWorkspace,
            sizeof(kWorkspace)),
        "workspace");

    cublasLtMatmulHeuristicResult_t heuristic[8] = {};
    int found = 0;
    check_blas(
        cublasLtMatmulAlgoGetHeuristic(
            handle,
            operation,
            layout_a,
            layout_b,
            layout_c,
            layout_c,
            preference,
            8,
            heuristic,
            &found),
        "heuristic");
    int chosen = -1;
    for (int index = 0; index < found; ++index) {
        if (heuristic[index].state == CUBLAS_STATUS_SUCCESS) {
            chosen = index;
            break;
        }
    }
    if (chosen < 0) {
        fail("int8 heuristic returned no algorithm", found);
    }

    const int32_t alpha = 1;
    const int32_t beta = 0;
    check_blas(
        cublasLtMatmul(
            handle,
            operation,
            &alpha,
            device_a,
            layout_a,
            device_b,
            layout_b,
            &beta,
            device_c,
            layout_c,
            device_c,
            layout_c,
            &heuristic[chosen].algo,
            workspace,
            kWorkspace,
            nullptr),
        "cublasLtMatmul");
    check_cuda(cudaDeviceSynchronize(), "sync int8");

    std::vector<int32_t> got(expect.size());
    check_cuda(
        cudaMemcpy(got.data(), device_c, got.size() * sizeof(int32_t), cudaMemcpyDeviceToHost),
        "copy C");
    for (size_t index = 0; index < expect.size(); ++index) {
        if (got[index] != expect[index]) {
            std::fprintf(
                stderr,
                "int8 mismatch at %zu got %d expect %d\n",
                index,
                got[index],
                expect[index]);
            std::exit(1);
        }
    }

    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(layout_a);
    cublasLtMatrixLayoutDestroy(layout_b);
    cublasLtMatrixLayoutDestroy(layout_c);
    cublasLtMatmulDescDestroy(operation);
    cublasLtDestroy(handle);
    cudaFree(workspace);
    cudaFree(device_c);
    cudaFree(device_b);
    cudaFree(device_a);
    std::printf("int8 ok\n");
}

int nibble(int value) {
    return value & 0xf;
}

void pack_low_first(const int* values, int count, uint8_t* packed) {
    for (int index = 0; index < count; index += 2) {
        packed[index / 2] = static_cast<uint8_t>(nibble(values[index]) | (nibble(values[index + 1]) << 4));
    }
}

__global__ void s4_tile(const void* matrix_a, const void* matrix_b, int* matrix_c) {
    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_a,
        8,
        8,
        32,
        nvcuda::wmma::experimental::precision::s4,
        nvcuda::wmma::row_major>
        frag_a;
    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_b,
        8,
        8,
        32,
        nvcuda::wmma::experimental::precision::s4,
        nvcuda::wmma::col_major>
        frag_b;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 32, int> frag_c;
    nvcuda::wmma::fill_fragment(frag_c, 0);
    nvcuda::wmma::load_matrix_sync(frag_a, matrix_a, 32);
    nvcuda::wmma::load_matrix_sync(frag_b, matrix_b, 32);
    nvcuda::wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
    nvcuda::wmma::store_matrix_sync(matrix_c, frag_c, 8, nvcuda::wmma::mem_row_major);
}

void check_int4() {
    constexpr int kRows = 8;
    constexpr int kCols = 8;
    constexpr int kInner = 32;
    int host_a[kRows * kInner];
    int host_b[kInner * kCols];
    int expect[kRows * kCols];
    for (int row = 0; row < kRows; ++row) {
        for (int col = 0; col < kInner; ++col) {
            host_a[row * kInner + col] = ((row + col) % 7) - 3;
        }
    }
    for (int col = 0; col < kCols; ++col) {
        for (int row = 0; row < kInner; ++row) {
            host_b[col * kInner + row] = ((row + 2 * col) % 5) - 2;
        }
    }
    for (int row = 0; row < kRows; ++row) {
        for (int col = 0; col < kCols; ++col) {
            int acc = 0;
            for (int inner = 0; inner < kInner; ++inner) {
                acc += host_a[row * kInner + inner] * host_b[col * kInner + inner];
            }
            expect[row * kCols + col] = acc;
        }
    }

    uint8_t packed_a[(kRows * kInner) / 2];
    uint8_t packed_b[(kInner * kCols) / 2];
    pack_low_first(host_a, kRows * kInner, packed_a);
    pack_low_first(host_b, kInner * kCols, packed_b);

    void* device_a = nullptr;
    void* device_b = nullptr;
    int* device_c = nullptr;
    check_cuda(cudaMalloc(&device_a, sizeof(packed_a)), "cudaMalloc s4 A");
    check_cuda(cudaMalloc(&device_b, sizeof(packed_b)), "cudaMalloc s4 B");
    check_cuda(cudaMalloc(&device_c, sizeof(expect)), "cudaMalloc s4 C");
    check_cuda(cudaMemcpy(device_a, packed_a, sizeof(packed_a), cudaMemcpyHostToDevice), "copy s4 A");
    check_cuda(cudaMemcpy(device_b, packed_b, sizeof(packed_b), cudaMemcpyHostToDevice), "copy s4 B");
    s4_tile<<<1, 32>>>(device_a, device_b, device_c);
    check_cuda(cudaGetLastError(), "s4 launch");
    check_cuda(cudaDeviceSynchronize(), "sync s4");

    int got[kRows * kCols];
    check_cuda(cudaMemcpy(got, device_c, sizeof(got), cudaMemcpyDeviceToHost), "copy s4 C");
    for (int index = 0; index < kRows * kCols; ++index) {
        if (got[index] != expect[index]) {
            std::fprintf(stderr, "int4 mismatch at %d got %d expect %d\n", index, got[index], expect[index]);
            std::exit(1);
        }
    }
    cudaFree(device_c);
    cudaFree(device_b);
    cudaFree(device_a);
    std::printf("int4 ok\n");
}

}  // namespace

int main() {
    int device = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp prop{};
    check_cuda(cudaGetDeviceProperties(&prop, device), "device properties");
    int capability = prop.major * 10 + prop.minor;
    if (capability < 86) {
        fail("RTX 30 and later need compute capability 8.6", capability);
    }
    check_int8();
    check_int4();
    std::printf("ok\n");
    return 0;
}
