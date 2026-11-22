// Prefill tile and decode GEMV. int8 products stay int8. The activation scale
// is one fp32 value per 32 channels. Prefill attention is the FlashAttention
// score and value mix, Dao, Fu, Ermon, Rudra, and Ré, arXiv:2205.14135, and
// Dao, arXiv:2307.08691. The key record is this pipeline's 264-byte plane.

__device__ __forceinline__ int dot16(int4 left, int4 right) {
    int acc = __dp4a(left.x, right.x, 0);
    acc = __dp4a(left.y, right.y, acc);
    acc = __dp4a(left.z, right.z, acc);
    return __dp4a(left.w, right.w, acc);
}

__device__ __forceinline__ float warp_sum(float value) {
    float next;
    asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 31, 0xffffffff;" : "=f"(next) : "f"(value), "r"(16));
    value += next;
    asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 31, 0xffffffff;" : "=f"(next) : "f"(value), "r"(8));
    value += next;
    asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 31, 0xffffffff;" : "=f"(next) : "f"(value), "r"(4));
    value += next;
    asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 31, 0xffffffff;" : "=f"(next) : "f"(value), "r"(2));
    value += next;
    asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 31, 0xffffffff;" : "=f"(next) : "f"(value), "r"(1));
    return value + next;
}

__device__ void mma_s8(
    int& d0, int& d1, int& d2, int& d3, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    int z0 = 0;
    int z1 = 0;
    int z2 = 0;
    int z3 = 0;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(z0), "r"(z1), "r"(z2), "r"(z3));
}

__device__ void mma_f16(
    float& d0, float& d1, float& d2, float& d3, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
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

__device__ float e8m0_byte(unsigned byte) {
    unsigned bits = byte == 0u ? (1u << 22) : (byte << 23);
    return __uint_as_float(bits);
}

__device__ void cp_async16(void* shared, const void* global) {
    unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(shared));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(address), "l"(global));
}

__device__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }

__device__ void cp_async_wait() { asm volatile("cp.async.wait_group 0;\n" ::); }

__global__ void gemv_i8_warp(
    const int8_t* weight,
    const float* scale,
    const int8_t* codes,
    const float* code_scale,
    float* destination,
    int rows,
    int inner) {
    int row = blockIdx.x;
    int token = blockIdx.y;
    if (row >= rows) {
        return;
    }
    const int4* line = reinterpret_cast<const int4*>(weight + static_cast<long long>(row) * inner);
    const int4* act = reinterpret_cast<const int4*>(codes + static_cast<long long>(token) * inner);
    int vecs = inner >> 4;
    int blocks = inner >> 5;
    float acc = 0.0f;
    for (int vec = threadIdx.x; vec < vecs; vec += 32) {
        acc += static_cast<float>(dot16(line[vec], act[vec])) * code_scale[static_cast<long long>(token) * blocks + (vec >> 1)];
    }
    acc = warp_sum(acc);
    if (threadIdx.x == 0) {
        destination[static_cast<long long>(token) * rows + row] = acc * scale[row];
    }
}

// One block converts the activation once. Every warp in the block reuses it.
__global__ void gemv_i8_cached(
    const int8_t* weight,
    const float* scale,
    const int8_t* codes,
    const float* code_scale,
    float* destination,
    int rows,
    int inner) {
    extern __shared__ __align__(16) char shared_mix[];
    auto* act = reinterpret_cast<int8_t*>(shared_mix);
    int blocks = inner >> 5;
    auto* scales = reinterpret_cast<float*>(act + inner);
    for (int col = threadIdx.x; col < inner; col += blockDim.x) {
        act[col] = codes[col];
    }
    for (int block = threadIdx.x; block < blocks; block += blockDim.x) {
        scales[block] = code_scale[block];
    }
    __syncthreads();
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps = blockDim.x >> 5;
    int row = blockIdx.x * warps + warp;
    if (row >= rows) {
        return;
    }
    const int4* line = reinterpret_cast<const int4*>(weight + static_cast<long long>(row) * inner);
    const int4* avec = reinterpret_cast<const int4*>(act);
    int vecs = inner >> 4;
    float acc = 0.0f;
    for (int vec = lane; vec < vecs; vec += 32) {
        acc += static_cast<float>(dot16(line[vec], avec[vec])) * scales[vec >> 1];
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        destination[row] = acc * scale[row];
    }
}

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
    if (byte < b_bytes) {
        const int8_t* b_src = g_zero_ptr;
        if (tok0 + local < tokens) {
            b_src = codes + static_cast<long long>(tok0 + local) * inner + k0 + koff;
        }
        cp_async16(b_dst + byte, b_src);
    }
    cp_async_commit();
}

__device__ void stage_scales(
    float* tile_scale, const float* code_scale, int tok0, int tokens, int kblocks, int k0, int tid) {
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

__global__ void quant_batch(const float* source, int8_t* codes, float* scales, int inner, int tokens) {
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

// 64 rows by 64 tokens. Two blocks fit on one SM.
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
            stage_tile(As[next], Bs[next], weight, codes, row0, tok0, k0 + kK, rows, tokens, inner, b_bytes, tid);
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

constexpr int kPrefillTokens = 16;
constexpr int kPrefillKeys = 16;
constexpr int kQueryWarps = 8;

__device__ void stage_record_codes(
    int8_t* tile, const int8_t* cache, int base, int rows, int kv, int layer, int full, int tid, int threads) {
    constexpr int kDim = 256;
    constexpr int kVec = 16;
    int vecs = kDim / kVec;
    int items = rows * vecs;
    for (int item = tid; item < items; item += threads) {
        int row = item / vecs;
        int vec = item - row * vecs;
        long long index = (static_cast<long long>(base + row) * full + layer) * 2 + kv;
        const char* src = reinterpret_cast<const char*>(cache + index * 264) + vec * 16;
        unsigned long long* dst = reinterpret_cast<unsigned long long*>(tile) + static_cast<long long>(item) * 2;
        dst[0] = *reinterpret_cast<const unsigned long long*>(src);
        dst[1] = *reinterpret_cast<const unsigned long long*>(src + 8);
    }
}

__device__ void stage_record_scales(
    unsigned char* tile, const int8_t* cache, int base, int rows, int kv, int layer, int full, int tid, int threads) {
    constexpr int kBytes = 8;
    int items = rows * kBytes;
    for (int item = tid; item < items; item += threads) {
        int row = item / kBytes;
        int col = item - row * kBytes;
        long long index = (static_cast<long long>(base + row) * full + layer) * 2 + kv;
        tile[item] = static_cast<const unsigned char*>(static_cast<const void*>(cache + index * 264))[256 + col];
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

// Sixteen queries share one staged key tile. Eight warps cover the query
// heads that read one key/value head.
__global__ void attend_prefill(
    const float* query,
    const float* gate,
    const int8_t* k_cache,
    const int8_t* v_cache,
    float* output,
    int length,
    int layer,
    int full,
    float scale,
    int count) {
    constexpr int kDim = 256;
    constexpr int kGroups = kDim / 32;
    constexpr int kHeads = 16;
    __shared__ __align__(16) int8_t key_tile[kPrefillKeys * kDim];
    __shared__ __align__(16) int8_t value_tile[kPrefillKeys * kDim];
    __shared__ unsigned char key_bytes[kPrefillKeys * kGroups];
    __shared__ unsigned char value_bytes[kPrefillKeys * kGroups];
    __shared__ __align__(16) int8_t query_codes[kQueryWarps * kPrefillTokens * kDim];
    __shared__ float query_scales[kQueryWarps * kPrefillTokens * kGroups];
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int frag = lane >> 2;
    int inside = lane & 3;
    int kv = blockIdx.x % 2;
    int head = kv * kQueryWarps + warp;
    int token0 = (blockIdx.x / 2) * kPrefillTokens;
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
    if (lane < kPrefillTokens && head < kHeads) {
        int token = token0 + lane;
        for (int kgroup = 0; kgroup < kGroups; ++kgroup) {
            int slot = warp * kPrefillTokens + lane;
            int at = (slot * kGroups + kgroup) * 32;
            if (token < count) {
                const float* row = query + (static_cast<long long>(token) * kHeads + head) * kDim + kgroup * 32;
                quant_query_group(query_codes + at, query_scales + slot * kGroups + kgroup, row);
            } else {
                query_scales[slot * kGroups + kgroup] = 0.0f;
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
    bool live_lo = head < kHeads && token_lo < count;
    bool live_hi = head < kHeads && token_hi < count;
    int seen_lo = origin + token_lo;
    int seen_hi = origin + token_hi;
    int tid = threadIdx.x;
    int threads = blockDim.x;
    for (int base = 0; base < furthest; base += kPrefillKeys) {
        int rows = furthest - base;
        if (rows > kPrefillKeys) {
            rows = kPrefillKeys;
        }
        stage_record_codes(key_tile, k_cache, base, rows, kv, layer, full, tid, threads);
        stage_record_codes(value_tile, v_cache, base, rows, kv, layer, full, tid, threads);
        stage_record_scales(key_bytes, k_cache, base, rows, kv, layer, full, tid, threads);
        stage_record_scales(value_bytes, v_cache, base, rows, kv, layer, full, tid, threads);
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
                    key0 = e8m0_byte(key_bytes[col0 * kGroups + kgroup]);
                }
                if (col0 + 1 < rows) {
                    key1 = e8m0_byte(key_bytes[(col0 + 1) * kGroups + kgroup]);
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
                bool keep_lo = live_lo && pos < seen_lo && pos < base + rows;
                bool keep_hi = live_hi && pos < seen_hi && pos < base + rows;
                score_lo[oct][col] = keep_lo ? score_lo[oct][col] * scale : -1.0e30f;
                score_hi[oct][col] = keep_hi ? score_hi[oct][col] * scale : -1.0e30f;
            }
        }
        float local_lo = fmaxf(fmaxf(score_lo[0][0], score_lo[0][1]), fmaxf(score_lo[1][0], score_lo[1][1]));
        float local_hi = fmaxf(fmaxf(score_hi[0][0], score_hi[0][1]), fmaxf(score_hi[1][0], score_hi[1][1]));
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
                float factor = e8m0_byte(value_bytes[key * kGroups + scale_group]);
                return static_cast<float>(value_tile[key * kDim + channel]) * factor;
            };
            unsigned b0 = pack_half2(value_at(inside * 2), value_at(inside * 2 + 1));
            unsigned b1 = pack_half2(value_at(8 + inside * 2), value_at(8 + inside * 2 + 1));
            unsigned p0 = pack_half2(prob_lo[0][0], prob_lo[0][1]);
            unsigned p1 = pack_half2(prob_hi[0][0], prob_hi[0][1]);
            unsigned p2 = pack_half2(prob_lo[1][0], prob_lo[1][1]);
            unsigned p3 = pack_half2(prob_hi[1][0], prob_hi[1][1]);
            mma_f16(acc[tile][0], acc[tile][1], acc[tile][2], acc[tile][3], p0, p1, p2, p3, b0, b1);
        }
        __syncthreads();
    }
    float inv_lo = sum_lo == 0.0f ? 0.0f : 1.0f / sum_lo;
    float inv_hi = sum_hi == 0.0f ? 0.0f : 1.0f / sum_hi;
#pragma unroll
    for (int tile = 0; tile < kChannelTiles; ++tile) {
        int c0 = tile * 8 + inside * 2;
        int c1 = c0 + 1;
        if (live_lo) {
            float* out = output + (static_cast<long long>(token_lo) * kHeads + head) * kDim;
            const float* grow = gate + (static_cast<long long>(token_lo) * kHeads + head) * kDim;
            out[c0] = acc[tile][0] * inv_lo * (1.0f / (1.0f + __expf(-grow[c0])));
            out[c1] = acc[tile][1] * inv_lo * (1.0f / (1.0f + __expf(-grow[c1])));
        }
        if (live_hi) {
            float* out = output + (static_cast<long long>(token_hi) * kHeads + head) * kDim;
            const float* grow = gate + (static_cast<long long>(token_hi) * kHeads + head) * kDim;
            out[c0] = acc[tile][2] * inv_hi * (1.0f / (1.0f + __expf(-grow[c0])));
            out[c1] = acc[tile][3] * inv_hi * (1.0f / (1.0f + __expf(-grow[c1])));
        }
    }
}
