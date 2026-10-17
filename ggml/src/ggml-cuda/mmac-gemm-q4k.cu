// Hygon DCU MMAC matrix-core GEMM for Q4_K weights (gfx928 / gfx936).
//
// See mmac-gemm-q4k.cuh for the operand/layout contract. The key property: the Q4_K
// super-blocks are dequantized *inside the GEMM kernel* into the f16 shared-memory tile that
// the MMAC mainloop consumes. No f16 copy of the weights ever exists in HBM, and the model
// file is never touched -- this is a pure runtime compute path.
//
// Layout notes (matches dequantize.cuh::dequantize_q4_K):
//   block_q4_K holds QK_K = 256 elements. Element e of a block:
//     il   = e >> 6              (0..3, 64-element group)
//     half = (e >> 5) & 1        (0: low nibbles, 1: high nibbles)
//     byte = qs[32*il + (e & 31)]
//     value = d*sc(e>>5) * nibble - dmin*mn(e>>5)
//   so 16/32 K-contiguous elements map to 16/32 consecutive bytes, one nibble each, all inside
//   one 32-element sub-block (scales are fetched once per tile row).

#include "mmac-gemm-q4k.cuh"
#include "mmac-gemm.cuh"     // ggml_cuda_mmac_available
#include "dequantize.cuh"    // get_scale_min_k4

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <climits>

// The MMAC builtin is only visible during the device compilation pass (see mmac-gemm.cu).
#if defined(__has_builtin)
#  if __has_builtin(__builtin_amdgcn_mmac_f32_16x16x16f16)
#    define GGML_MMAC_QK_BUILTIN 1
#  endif
#endif
#ifndef GGML_MMAC_QK_BUILTIN
#  define GGML_MMAC_QK_BUILTIN 0
#endif

typedef _Float16 mmac_qk_f16x4 __attribute__((vector_size(8)));
typedef float    mmac_qk_f32x4 __attribute__((vector_size(16)));

// Dense tile: same geometry as mmac-gemm.cu (each warp owns a 64x64 subtile of 4x4 MMAC
// fragments; 8 warps per block).
#define QK4_BM 128                       // rows of A (tokens) per block
#define QK4_BN 256                       // rows of B (features) per block
#define QK4_KT 32                        // contraction step
#define QK4_PAD 8
#define QK4_LDS (QK4_KT + QK4_PAD)
#define QK4_NT  512

// Grouped (MoE) tile: only 64 A-rows per block. With e.g. 512 experts and 10 used per token a
// 2048-token ubatch gives ~40 rows per expert, so a 128-row tile would waste half the MMACs.
#define QK4_GBM 64
#define QK4_GBN 128
#define QK4_GNT (QK4_GBM/64 * QK4_GBN/64 * 64)   // 256

#define QK4_MIN_ROWS 256

// ---------------------------------------------------------------------------------------------
// Shared device helpers
// ---------------------------------------------------------------------------------------------

// Dequantize 4/8/... elements (multiple of 4) of one Q4_K row into f16 and store them as one
// f16x4 to the LDS tile. `w` holds the 4 source bytes (one nibble per element).
static __device__ __forceinline__ void mmac_qk_store4(__half * lds, uint32_t w, bool hi_nib,
                                                      float d1, float m1) {
    const uint32_t n = hi_nib ? (w >> 4) : w;
    mmac_qk_f16x4 h4;
    h4[0] = __float2half(d1 * ( n        & 0xF) - m1);
    h4[1] = __float2half(d1 * ((n >>  8) & 0xF) - m1);
    h4[2] = __float2half(d1 * ((n >> 16) & 0xF) - m1);
    h4[3] = __float2half(d1 * ((n >> 24) & 0xF) - m1);
    *reinterpret_cast<mmac_qk_f16x4 *>(lds) = h4;
}

// ---------------------------------------------------------------------------------------------
// Dense kernel: C[M, N] = A[M, K] * W[N, K]^T, W in Q4_K, A in F32.
// ---------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(QK4_NT, 1)
mmac_gemm_q4k_kernel(const char * __restrict__ A, const block_q4_K * __restrict__ Wq,
                     float * __restrict__ C, int M, int N, int K, int w_stride) {
#if GGML_MMAC_QK_BUILTIN
    __shared__ __half As[QK4_BM * QK4_LDS];
    __shared__ __half Bs[QK4_BN * QK4_LDS];

    const int t = threadIdx.x;
    const int l = t & 63;
    const int wm = (t / 64) % (QK4_BM/64);
    const int wn = (t / 64) / (QK4_BM/64);
    const int m0 = blockIdx.y * QK4_BM, n0 = blockIdx.x * QK4_BN;
    const int lr = l % 16, lg = l / 16;
    const int lc = (l % 4) * 4 + (l % 16) / 4;

    mmac_qk_f32x4 acc[4][4];
    #pragma unroll
    for (int r = 0; r < 4; ++r)
        #pragma unroll
        for (int c = 0; c < 4; ++c) acc[r][c] = {0, 0, 0, 0};

    constexpr int PER_A = (QK4_BM * QK4_KT) / QK4_NT;   // 8
    constexpr int PER_B = (QK4_BN * QK4_KT) / QK4_NT;   // 16
    const int arow = t / (QK4_KT / PER_A), acol = (t % (QK4_KT / PER_A)) * PER_A;
    const int brow = t / (QK4_KT / PER_B), bcol = (t % (QK4_KT / PER_B)) * PER_B;

    const bool m_full = (m0 + QK4_BM <= M);
    const bool n_full = (n0 + QK4_BN <= N);
    const int  m_last = M - 1, n_last = N - 1;

    for (int k0 = 0; k0 < K; k0 += QK4_KT) {
        #pragma unroll
        for (int v = 0; v < PER_A; v += 4) {
            int r = m0 + arow;
            if (!m_full) r = min(r, m_last);
            const int c = k0 + acol + v;
            const float4 a4 = *reinterpret_cast<const float4 *>(A + ((size_t) r * K + c) * sizeof(float));
            mmac_qk_f16x4 h4;
            h4[0] = __float2half(a4.x);
            h4[1] = __float2half(a4.y);
            h4[2] = __float2half(a4.z);
            h4[3] = __float2half(a4.w);
            *reinterpret_cast<mmac_qk_f16x4 *>(&As[arow * QK4_LDS + acol + v]) = h4;
        }
        {
            // One thread loads PER_B (16) K-contiguous elements of one weight row. All 16 lie in
            // one 32-element sub-block of one super-block: 16 bytes, one nibble per element.
            int r = n0 + brow;
            if (!n_full) r = min(r, n_last);
            const int c = k0 + bcol;
            const int o = c & 255;
            const block_q4_K * bx = Wq + (int64_t) r * w_stride + (c >> 8);
            uint8_t sc, mn;
            get_scale_min_k4(o >> 5, bx->scales, sc, mn);
            const float d1 = __low2float(bx->dm) * sc;
            const float m1 = __high2float(bx->dm) * mn;
            const bool hi_nib = (o >> 5) & 1;
            const uint4 qv = *reinterpret_cast<const uint4 *>(bx->qs + 32 * (o >> 6) + (o & 31));
            const uint32_t w32[4] = {qv.x, qv.y, qv.z, qv.w};
            #pragma unroll
            for (int v = 0; v < PER_B; v += 4) {
                mmac_qk_store4(&Bs[brow * QK4_LDS + bcol + v], w32[v >> 2], hi_nib, d1, m1);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < QK4_KT; kk += 16) {
            mmac_qk_f16x4 a[4], b[4];
            #pragma unroll
            for (int rg = 0; rg < 4; ++rg)
                a[rg] = *reinterpret_cast<const mmac_qk_f16x4 *>(
                    &As[(wm * 64 + rg * 16 + lr) * QK4_LDS + kk + lg * 4]);
            #pragma unroll
            for (int cg = 0; cg < 4; ++cg)
                b[cg] = *reinterpret_cast<const mmac_qk_f16x4 *>(
                    &Bs[(wn * 64 + cg * 16 + lc) * QK4_LDS + kk + lg * 4]);
            #pragma unroll
            for (int rg = 0; rg < 4; ++rg)
                #pragma unroll
                for (int cg = 0; cg < 4; ++cg)
                    acc[rg][cg] = __builtin_amdgcn_mmac_f32_16x16x16f16(a[rg], b[cg], acc[rg][cg]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int rg = 0; rg < 4; ++rg)
        #pragma unroll
        for (int cg = 0; cg < 4; ++cg) {
            const int cr = m0 + wm * 64 + rg * 16 + lr;
            const int cn = n0 + wn * 64 + cg * 16 + lg * 4;
            if (cr < M) {
                if (n_full) {
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) C[(size_t) cr * N + cn + i] = acc[rg][cg][i];
                } else {
                    #pragma unroll
                    for (int i = 0; i < 4; ++i)
                        if (cn + i < N) C[(size_t) cr * N + cn + i] = acc[rg][cg][i];
                }
            }
        }
#else
    (void) A; (void) Wq; (void) C; (void) M; (void) N; (void) K; (void) w_stride;
#endif
}

// ---------------------------------------------------------------------------------------------
// Grouped kernel for GGML_MUL_MAT_ID (MoE): rows sorted by expert.
//
// blockIdx.z selects the expert via expert_bounds; A rows are resolved through ids_src1 and
// results written through ids_dst. Rows of the tile beyond the expert's range are clamped for
// loads and discarded by the store guard.
// ---------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(QK4_GNT, 1)
mmac_gemm_q4k_grouped_kernel(const char * __restrict__ A, const block_q4_K * __restrict__ Wq,
                             const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
                             const int32_t * __restrict__ expert_bounds,
                             float * __restrict__ C, int N, int K,
                             int64_t act_stride, int w_stride, int64_t w_mat_stride) {
#if GGML_MMAC_QK_BUILTIN
    __shared__ __half As[QK4_GBM * QK4_LDS];
    __shared__ __half Bs[QK4_GBN * QK4_LDS];

    const int t = threadIdx.x;
    constexpr int WTM = QK4_GBM / 64;   // 1
    const int l = t & 63;
    const int wm = (t / 64) % WTM;      // 0
    const int wn = (t / 64) / WTM;      // 0..3
    const int n0 = blockIdx.x * QK4_GBN;
    const int e  = blockIdx.z;

    const int row_lo = expert_bounds[e];
    const int n_rows = expert_bounds[e + 1] - row_lo;

    const int lr = l % 16, lg = l / 16;
    const int lc = (l % 4) * 4 + (l % 16) / 4;

    const block_q4_K * We = Wq + e * w_mat_stride;

    constexpr int PER_A = (QK4_GBM * QK4_KT) / QK4_GNT;   // 8
    constexpr int PER_B = (QK4_GBN * QK4_KT) / QK4_GNT;   // 32
    const int arow = t / (QK4_KT / PER_A);                // 0..63
    const int acol = (t % (QK4_KT / PER_A)) * PER_A;
    const int brow = t;                                   // one full row of B per thread

    const bool n_full = (n0 + QK4_GBN <= N);
    const int  n_last = N - 1;
    const int  m_stride = gridDim.y * QK4_GBM;

    // Grid-stride over this expert's row blocks. The host sizes gridDim.y from the *average*
    // rows per expert (not from the global row count), so a grid of 320 mostly-empty blocks per
    // expert collapses to 1-2; an unusually heavy expert simply loops here. Correctness does not
    // depend on the grid size.
    for (int m0 = blockIdx.y * QK4_GBM; m0 < n_rows; m0 += m_stride) {
    mmac_qk_f32x4 acc[4][4];
    #pragma unroll
    for (int r = 0; r < 4; ++r)
        #pragma unroll
        for (int c = 0; c < 4; ++c) acc[r][c] = {0, 0, 0, 0};

    // Resolve this thread's A row once (fixed across the k loop).
    int sr = row_lo + m0 + arow;
    if (sr > row_lo + n_rows - 1) sr = row_lo + n_rows - 1;
    const char * Arow = A + (int64_t) ids_src1[sr] * act_stride * sizeof(float);

    for (int k0 = 0; k0 < K; k0 += QK4_KT) {
        #pragma unroll
        for (int v = 0; v < PER_A; v += 4) {
            const int c = k0 + acol + v;
            const float4 a4 = *reinterpret_cast<const float4 *>(Arow + (size_t) c * sizeof(float));
            mmac_qk_f16x4 h4;
            h4[0] = __float2half(a4.x);
            h4[1] = __float2half(a4.y);
            h4[2] = __float2half(a4.z);
            h4[3] = __float2half(a4.w);
            *reinterpret_cast<mmac_qk_f16x4 *>(&As[arow * QK4_LDS + acol + v]) = h4;
        }
        {
            // One thread loads PER_B (32) elements of one weight row = exactly one 32-element
            // sub-block: 32 bytes (2x uint4), one nibble per element, a single scale pair.
            int r = n0 + brow;
            if (!n_full) r = min(r, n_last);
            const int o = k0 & 255;
            const block_q4_K * bx = We + (int64_t) r * w_stride + (k0 >> 8);
            uint8_t sc, mn;
            get_scale_min_k4(o >> 5, bx->scales, sc, mn);
            const float d1 = __low2float(bx->dm) * sc;
            const float m1 = __high2float(bx->dm) * mn;
            const bool hi_nib = (o >> 5) & 1;
            const uint4 qa = *reinterpret_cast<const uint4 *>(bx->qs + 32 * (o >> 6) + (o & 31));
            const uint4 qb = *reinterpret_cast<const uint4 *>(bx->qs + 32 * (o >> 6) + (o & 31) + 16);
            const uint32_t w32[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};
            #pragma unroll
            for (int v = 0; v < PER_B; v += 4) {
                mmac_qk_store4(&Bs[brow * QK4_LDS + v], w32[v >> 2], hi_nib, d1, m1);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < QK4_KT; kk += 16) {
            mmac_qk_f16x4 a[4], b[4];
            #pragma unroll
            for (int rg = 0; rg < 4; ++rg)
                a[rg] = *reinterpret_cast<const mmac_qk_f16x4 *>(
                    &As[(wm * 64 + rg * 16 + lr) * QK4_LDS + kk + lg * 4]);
            #pragma unroll
            for (int cg = 0; cg < 4; ++cg)
                b[cg] = *reinterpret_cast<const mmac_qk_f16x4 *>(
                    &Bs[(wn * 64 + cg * 16 + lc) * QK4_LDS + kk + lg * 4]);
            #pragma unroll
            for (int rg = 0; rg < 4; ++rg)
                #pragma unroll
                for (int cg = 0; cg < 4; ++cg)
                    acc[rg][cg] = __builtin_amdgcn_mmac_f32_16x16x16f16(a[rg], b[cg], acc[rg][cg]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int rg = 0; rg < 4; ++rg)
        #pragma unroll
        for (int cg = 0; cg < 4; ++cg) {
            const int row = m0 + wm * 64 + rg * 16 + lr;   // within the expert's range
            if (row < n_rows) {
                const int dst_row = ids_dst[row_lo + row];
                const int cn = n0 + wn * 64 + cg * 16 + lg * 4;
                if (n_full) {
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) C[(size_t) dst_row * N + cn + i] = acc[rg][cg][i];
                } else {
                    #pragma unroll
                    for (int i = 0; i < 4; ++i)
                        if (cn + i < N) C[(size_t) dst_row * N + cn + i] = acc[rg][cg][i];
                }
            }
        }
    }   // m0 grid-stride loop
#else
    (void) A; (void) Wq; (void) ids_src1; (void) ids_dst; (void) expert_bounds; (void) C;
    (void) N; (void) K; (void) act_stride; (void) w_stride; (void) w_mat_stride;
#endif
}

// ---------------------------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------------------------

static bool mmac_qk_env_enabled() {
    const char * env = getenv("GGML_CUDA_MMAC_Q4K");
    return env == nullptr || atoi(env) != 0;
}

// Separate switch for the grouped (MUL_MAT_ID / MoE) path so the two can be A/B tested
// independently. Unset -> follow GGML_CUDA_MMAC_Q4K.
static bool mmac_qk_id_env_enabled() {
    const char * env = getenv("GGML_CUDA_MMAC_Q4K_ID");
    if (env == nullptr) {
        return mmac_qk_env_enabled();
    }
    return atoi(env) != 0;
}

static bool mmac_qk_shape_ok(int64_t K) {
    return K % 256 == 0 && K <= INT_MAX;   // K % 32 == 0 is implied by K % 256 == 0
}

bool ggml_cuda_mmac_gemm_q4_K(cudaStream_t stream,
                              const void * act, bool act_is_f32, const void * weight, float * dst,
                              int64_t tokens, int64_t features, int64_t K,
                              int64_t w_row_stride) {
    if (!act_is_f32 || !mmac_qk_env_enabled() || !ggml_cuda_mmac_available()) {
        return false;
    }
    if (tokens < QK4_MIN_ROWS || features < QK4_BN) {
        return false;
    }
    if (!mmac_qk_shape_ok(K) || w_row_stride < K / 256 || w_row_stride > INT_MAX) {
        return false;
    }
    if ((tokens + QK4_BM - 1) / QK4_BM > 65535 || features > INT_MAX) {
        return false;
    }

    const dim3 grid((unsigned) ((features + QK4_BN - 1) / QK4_BN),
                    (unsigned) ((tokens   + QK4_BM - 1) / QK4_BM));

    static bool logged = false;
    if (!logged) {
        logged = true;
        GGML_LOG_INFO("%s: MMAC Q4_K GEMM in use, first shape tokens=%lld features=%lld K=%lld\n",
                      __func__, (long long) tokens, (long long) features, (long long) K);
    }

    mmac_gemm_q4k_kernel<<<grid, QK4_NT, 0, stream>>>(
        (const char *) act, (const block_q4_K *) weight, dst,
        (int) tokens, (int) features, (int) K, (int) w_row_stride);
    return true;
}

bool ggml_cuda_mmac_gemm_q4_K_id_usable(const ggml_tensor * src0, const ggml_tensor * src1,
                                        const ggml_tensor * ids, const ggml_tensor * dst) {
    if (!src0 || !src1 || !ids || !dst) {
        return false;
    }
    if (src0->type != GGML_TYPE_Q4_K || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            ids->type != GGML_TYPE_I32) {
        return false;
    }
    if (!mmac_qk_id_env_enabled()) {
        return false;
    }
    if (!ggml_cuda_mmac_available()) {
        return false;
    }

    GGML_TENSOR_BINARY_OP_LOCALS;

    const int64_t n_expert_used = ids->ne[0];
    const int64_t rows = ne12 * n_expert_used;

    if (rows < QK4_MIN_ROWS || ne01 < QK4_GBN || ne13 != 1) {
        return false;
    }
    if (!mmac_qk_shape_ok(ne00) ||
            src0->nb[0] != ggml_type_size(src0->type) ||
            src0->nb[1] % ggml_type_size(src0->type) != 0 ||
            src0->nb[2] % ggml_type_size(src0->type) != 0 ||
            src1->nb[0] != sizeof(float) ||
            src1->nb[1] % (4 * sizeof(float)) != 0 ||
            dst->nb[0] != sizeof(float) ||
            dst->nb[1] != ne0 * sizeof(float) ||
            ids->nb[0] != ggml_element_size(ids) ||
            ne02 > 65535) {
        return false;
    }
    return true;
}

bool ggml_cuda_mmac_gemm_q4_K_id(cudaStream_t stream,
                                 const void * act, const void * weight, float * dst,
                                 const int32_t * ids_src1, const int32_t * ids_dst, const int32_t * expert_bounds,
                                 int64_t n_expert, int64_t rows, int64_t features, int64_t K,
                                 int64_t act_stride, int64_t w_row_stride, int64_t w_mat_stride) {
    if (!mmac_qk_id_env_enabled() || !ggml_cuda_mmac_available()) {
        return false;
    }
    if (rows < QK4_MIN_ROWS || features < QK4_GBN || !mmac_qk_shape_ok(K)) {
        return false;
    }
    if (act_stride % 4 != 0 || w_row_stride > INT_MAX || w_mat_stride > INT_MAX || n_expert > 65535 ||
            features > INT_MAX || (rows + QK4_GBM - 1) / QK4_GBM > 65535) {
        return false;
    }

    const dim3 grid((unsigned) ((features + QK4_GBN - 1) / QK4_GBN),
                    (unsigned) ((rows     + QK4_GBM - 1) / QK4_GBM),
                    (unsigned) n_expert);

    static bool logged = false;
    if (!logged) {
        logged = true;
        GGML_LOG_INFO("%s: MMAC Q4_K grouped GEMM in use, first shape rows=%lld features=%lld K=%lld experts=%lld\n",
                      __func__, (long long) rows, (long long) features, (long long) K, (long long) n_expert);
    }

    mmac_gemm_q4k_grouped_kernel<<<grid, QK4_GNT, 0, stream>>>(
        (const char *) act, (const block_q4_K *) weight, ids_src1, ids_dst, expert_bounds, dst,
        (int) features, (int) K, act_stride, (int) w_row_stride, w_mat_stride);
    return true;
}
