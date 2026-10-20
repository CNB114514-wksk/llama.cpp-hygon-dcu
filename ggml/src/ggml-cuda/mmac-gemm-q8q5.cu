// Hygon DCU MMAC matrix-core GEMM for Q8_0 / Q5_1 weights (gfx928 / gfx936).
//
// Same structure as mmac-gemm-q4k.cu, but the 32-element sub-block is the natural unit for
// these two types (QK8_0 == QK5_1 == 32), so K only needs to be a multiple of 32.
//
// Alignment notes:
//   block_q8_0 = { half d; int8 qs[32]; }   34 bytes -> qs is only 2-byte aligned. A 16-byte
//              vector load would be misaligned, so the tile is filled with uint16 loads.
//   block_q5_1 = { half2 dm; uint8 qh[4]; uint8 qs[16]; } 24 bytes -> qs at offset 8 is
//              4-byte aligned, qh at offset 4 is 4-byte aligned. qs[j] holds element j in
//              its low nibble and element j+16 in its high nibble, so 4 consecutive elements
//              come from one uint32 load.

#include "mmac-gemm-q8q5.cuh"
#include "mmac-gemm.cuh"     // ggml_cuda_mmac_available

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <climits>

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

#define QKQ_BM 128
#define QKQ_BN 256
#define QKQ_KT 32
#define QKQ_PAD 8
#define QKQ_LDS (QKQ_KT + QKQ_PAD)
#define QKQ_NT  512

#define QKQ_GBM 64
#define QKQ_GBN 128
#define QKQ_GNT 128

#define QKQ_MIN_ROWS 256

enum { QKQ_W_Q8_0 = 0, QKQ_W_Q5_1 = 1 };

// Dequantize `n` (16 or 32) K-contiguous weights of row `r` starting at absolute element `c`
// into the f16 LDS tile. (c & 31) + n <= 32 is guaranteed by the callers.
template <int WK>
static __device__ __forceinline__ void mmac_w_load(__half * dst, const void * W,
                                                   int r, int c, int n, int w_stride) {
    if (WK == QKQ_W_Q8_0) {
        const block_q8_0 * bx = (const block_q8_0 *) W + (int64_t) r * w_stride + (c >> 5);
        const float d1 = __half2float(bx->d);
        const uint16_t * q = reinterpret_cast<const uint16_t *>(bx->qs + (c & 31));
        #pragma unroll
        for (int v = 0; v < n; v += 4) {
            const uint32_t w = (uint32_t) q[v >> 1] | ((uint32_t) q[(v >> 1) + 1] << 16);
            mmac_qk_f16x4 h4;
            h4[0] = __float2half(d1 * (float) (int8_t) ( w        & 0xFF));
            h4[1] = __float2half(d1 * (float) (int8_t) ((w >>  8) & 0xFF));
            h4[2] = __float2half(d1 * (float) (int8_t) ((w >> 16) & 0xFF));
            h4[3] = __float2half(d1 * (float) (int8_t) ((w >> 24) & 0xFF));
            *reinterpret_cast<mmac_qk_f16x4 *>(dst + v) = h4;
        }
    } else {
        const block_q5_1 * bx = (const block_q5_1 *) W + (int64_t) r * w_stride + (c >> 5);
        const float d1 = __low2float (bx->dm);
        const float m1 = __high2float(bx->dm);
        uint32_t qh;
        memcpy(&qh, bx->qh, sizeof(qh));
        // ggml's Q5_1 layout is NOT interleaved pairs: qs[j] holds element j in its low nibble
        // and element j+16 in its high nibble (see dequantize_row_q5_1). The 5th bit of element
        // e is qh bit e in both halves.
        const int o  = c & 31;
        const uint32_t * q32 = reinterpret_cast<const uint32_t *>(bx->qs);
        #pragma unroll
        for (int v = 0; v < n; v += 4) {
            // Elements 16..31 reuse bytes 0..15 (their high nibbles), so the byte index wraps
            // at 16 rather than running to 31.
            const int e0 = o + v;
            const bool hi = e0 >= 16;
            const uint32_t w = q32[(e0 & 15) >> 2];
            mmac_qk_f16x4 h4;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int e  = e0 + j;
                const int nb = hi ? ((w >> (8*j + 4)) & 0xF) : ((w >> (8*j)) & 0xF);
                const int q5 = nb | (((qh >> e) & 1) << 4);
                h4[j] = __float2half(d1 * (float) q5 + m1);
            }
            *reinterpret_cast<mmac_qk_f16x4 *>(dst + v) = h4;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Dense: C[M, N] = A[M, K] * W[N, K]^T, W in Q8_0, A in F32.
// ---------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(QKQ_NT, 1)
mmac_gemm_q80_kernel(const char * __restrict__ A, const void * __restrict__ Wq,
                     float * __restrict__ C, int M, int N, int K, int w_stride) {
#if GGML_MMAC_QK_BUILTIN
    __shared__ __half As[QKQ_BM * QKQ_LDS];
    __shared__ __half Bs[QKQ_BN * QKQ_LDS];

    const int t = threadIdx.x;
    const int l = t & 63;
    const int wm = (t / 64) % (QKQ_BM/64);
    const int wn = (t / 64) / (QKQ_BM/64);
    const int m0 = blockIdx.y * QKQ_BM, n0 = blockIdx.x * QKQ_BN;
    const int lr = l % 16, lg = l / 16;
    const int lc = (l % 4) * 4 + (l % 16) / 4;

    mmac_qk_f32x4 acc[4][4];
    #pragma unroll
    for (int r = 0; r < 4; ++r)
        #pragma unroll
        for (int c = 0; c < 4; ++c) acc[r][c] = {0, 0, 0, 0};

    constexpr int PER_A = (QKQ_BM * QKQ_KT) / QKQ_NT;   // 8
    constexpr int PER_B = (QKQ_BN * QKQ_KT) / QKQ_NT;   // 16
    const int arow = t / (QKQ_KT / PER_A), acol = (t % (QKQ_KT / PER_A)) * PER_A;
    const int brow = t / (QKQ_KT / PER_B), bcol = (t % (QKQ_KT / PER_B)) * PER_B;

    const bool m_full = (m0 + QKQ_BM <= M);
    const bool n_full = (n0 + QKQ_BN <= N);
    const int  m_last = M - 1, n_last = N - 1;

    for (int k0 = 0; k0 < K; k0 += QKQ_KT) {
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
            *reinterpret_cast<mmac_qk_f16x4 *>(&As[arow * QKQ_LDS + acol + v]) = h4;
        }
        {
            int r = n0 + brow;
            if (!n_full) r = min(r, n_last);
            mmac_w_load<QKQ_W_Q8_0>(&Bs[brow * QKQ_LDS + bcol], Wq, r, k0 + bcol, PER_B, w_stride);
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < QKQ_KT; kk += 16) {
            mmac_qk_f16x4 a[4], b[4];
            #pragma unroll
            for (int rg = 0; rg < 4; ++rg)
                a[rg] = *reinterpret_cast<const mmac_qk_f16x4 *>(
                    &As[(wm * 64 + rg * 16 + lr) * QKQ_LDS + kk + lg * 4]);
            #pragma unroll
            for (int cg = 0; cg < 4; ++cg)
                b[cg] = *reinterpret_cast<const mmac_qk_f16x4 *>(
                    &Bs[(wn * 64 + cg * 16 + lc) * QKQ_LDS + kk + lg * 4]);
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
// Grouped for GGML_MUL_MAT_ID (MoE down projection), rows sorted by expert.
// ---------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(QKQ_GNT, 1)
mmac_gemm_q51_grouped_kernel(const char * __restrict__ A, const void * __restrict__ Wq,
                             const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
                             const int32_t * __restrict__ expert_bounds,
                             float * __restrict__ C, int N, int K,
                             int64_t act_stride, int w_stride, int64_t w_mat_stride) {
#if GGML_MMAC_QK_BUILTIN
    __shared__ __half As[QKQ_GBM * QKQ_LDS];
    __shared__ __half Bs[QKQ_GBN * QKQ_LDS];

    const int t = threadIdx.x;
    constexpr int WTM = QKQ_GBM / 64;   // 1
    const int l = t & 63;
    const int wm = (t / 64) % WTM;
    const int wn = (t / 64) / WTM;      // 0..1
    const int n0 = blockIdx.x * QKQ_GBN;
    const int e  = blockIdx.z;

    const int row_lo = expert_bounds[e];
    const int n_rows = expert_bounds[e + 1] - row_lo;

    const int lr = l % 16, lg = l / 16;
    const int lc = (l % 4) * 4 + (l % 16) / 4;

    const block_q5_1 * We = (const block_q5_1 *) Wq + (int64_t) e * w_mat_stride;

    constexpr int PER_A = (QKQ_GBM * QKQ_KT) / QKQ_GNT;   // 16
    constexpr int PER_B = (QKQ_GBN * QKQ_KT) / QKQ_GNT;   // 32
    const int arow = t / (QKQ_KT / PER_A);
    const int acol = (t % (QKQ_KT / PER_A)) * PER_A;
    const int brow = t;

    const bool n_full = (n0 + QKQ_GBN <= N);
    const int  n_last = N - 1;
    const int  m_stride = gridDim.y * QKQ_GBM;

    for (int m0 = blockIdx.y * QKQ_GBM; m0 < n_rows; m0 += m_stride) {
    mmac_qk_f32x4 acc[4][4];
    #pragma unroll
    for (int r = 0; r < 4; ++r)
        #pragma unroll
        for (int c = 0; c < 4; ++c) acc[r][c] = {0, 0, 0, 0};

    int sr = row_lo + m0 + arow;
    if (sr > row_lo + n_rows - 1) sr = row_lo + n_rows - 1;
    const char * Arow = A + (int64_t) ids_src1[sr] * act_stride * sizeof(float);

    for (int k0 = 0; k0 < K; k0 += QKQ_KT) {
        #pragma unroll
        for (int v = 0; v < PER_A; v += 4) {
            const int c = k0 + acol + v;
            const float4 a4 = *reinterpret_cast<const float4 *>(Arow + (size_t) c * sizeof(float));
            mmac_qk_f16x4 h4;
            h4[0] = __float2half(a4.x);
            h4[1] = __float2half(a4.y);
            h4[2] = __float2half(a4.z);
            h4[3] = __float2half(a4.w);
            *reinterpret_cast<mmac_qk_f16x4 *>(&As[arow * QKQ_LDS + acol + v]) = h4;
        }
        {
            int r = n0 + brow;
            if (!n_full) r = min(r, n_last);
            mmac_w_load<QKQ_W_Q5_1>(&Bs[brow * QKQ_LDS], We, r, k0, PER_B, w_stride);
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < QKQ_KT; kk += 16) {
            mmac_qk_f16x4 a[4], b[4];
            #pragma unroll
            for (int rg = 0; rg < 4; ++rg)
                a[rg] = *reinterpret_cast<const mmac_qk_f16x4 *>(
                    &As[(wm * 64 + rg * 16 + lr) * QKQ_LDS + kk + lg * 4]);
            #pragma unroll
            for (int cg = 0; cg < 4; ++cg)
                b[cg] = *reinterpret_cast<const mmac_qk_f16x4 *>(
                    &Bs[(wn * 64 + cg * 16 + lc) * QKQ_LDS + kk + lg * 4]);
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
            const int row = m0 + wm * 64 + rg * 16 + lr;
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

static bool mmac_q8_env_enabled() {
    const char * env = getenv("GGML_CUDA_MMAC_Q8");
    return env == nullptr || atoi(env) != 0;
}

static bool mmac_q51_env_enabled() {
    const char * env = getenv("GGML_CUDA_MMAC_Q51");
    return env == nullptr || atoi(env) != 0;
}

static bool mmac_q8q5_shape_ok(int64_t K) {
    return K % 32 == 0 && K <= INT_MAX;
}

bool ggml_cuda_mmac_gemm_q8_0(cudaStream_t stream,
                              const void * act, const void * weight, float * dst,
                              int64_t tokens, int64_t features, int64_t K,
                              int64_t w_row_stride) {
    if (!mmac_q8_env_enabled() || !ggml_cuda_mmac_available()) {
        return false;
    }
    if (tokens < QKQ_MIN_ROWS || features < QKQ_BN) {
        return false;
    }
    if (!mmac_q8q5_shape_ok(K) || w_row_stride < K / 32 || w_row_stride > INT_MAX) {
        return false;
    }
    if ((tokens + QKQ_BM - 1) / QKQ_BM > 65535 || features > INT_MAX) {
        return false;
    }

    const dim3 grid((unsigned) ((features + QKQ_BN - 1) / QKQ_BN),
                    (unsigned) ((tokens   + QKQ_BM - 1) / QKQ_BM));

    static bool logged = false;
    if (!logged) {
        logged = true;
        fprintf(stderr, "%s: MMAC Q8_0 GEMM in use, first shape tokens=%lld features=%lld K=%lld\n",
                __func__, (long long) tokens, (long long) features, (long long) K);
    }

    mmac_gemm_q80_kernel<<<grid, QKQ_NT, 0, stream>>>(
        (const char *) act, weight, dst,
        (int) tokens, (int) features, (int) K, (int) w_row_stride);
    return true;
}

bool ggml_cuda_mmac_gemm_q8_0_usable(const ggml_tensor * src0, const ggml_tensor * src1,
                                     const ggml_tensor * dst) {
    if (!src0 || !src1 || !dst) {
        return false;
    }
    if (src0->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!mmac_q8_env_enabled() || !ggml_cuda_mmac_available()) {
        return false;
    }

    GGML_TENSOR_BINARY_OP_LOCALS;

    if (ne11 < QKQ_MIN_ROWS || ne01 < QKQ_BN || ne12 != 1 || ne13 != 1 ||
            src0->ne[2] != 1 || src0->ne[3] != 1) {
        return false;
    }
    if (!mmac_q8q5_shape_ok(ne00) ||
            src0->nb[0] != ggml_type_size(src0->type) ||
            src0->nb[1] % ggml_type_size(src0->type) != 0 ||
            src1->nb[0] != sizeof(float) ||
            src1->nb[1] == 0 ||
            dst->nb[0] != sizeof(float) ||
            dst->nb[1] != ne0 * sizeof(float)) {
        return false;
    }
    return true;
}

bool ggml_cuda_mmac_gemm_q5_1_id_usable(const ggml_tensor * src0, const ggml_tensor * src1,
                                        const ggml_tensor * ids, const ggml_tensor * dst) {
    if (!src0 || !src1 || !ids || !dst) {
        return false;
    }
    if (src0->type != GGML_TYPE_Q5_1 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            ids->type != GGML_TYPE_I32) {
        return false;
    }
    if (!mmac_q51_env_enabled() || !ggml_cuda_mmac_available()) {
        return false;
    }

    GGML_TENSOR_BINARY_OP_LOCALS;

    const int64_t n_expert_used = ids->ne[0];
    const int64_t rows = ne12 * n_expert_used;

    if (rows < QKQ_MIN_ROWS || ne01 < QKQ_GBN || ne13 != 1) {
        return false;
    }
    if (!mmac_q8q5_shape_ok(ne00) ||
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

bool ggml_cuda_mmac_gemm_q5_1_id(cudaStream_t stream,
                                 const void * act, const void * weight, float * dst,
                                 const int32_t * ids_src1, const int32_t * ids_dst, const int32_t * expert_bounds,
                                 int64_t n_expert, int64_t rows, int64_t features, int64_t K,
                                 int64_t act_stride, int64_t w_row_stride, int64_t w_mat_stride) {
    if (!mmac_q51_env_enabled() || !ggml_cuda_mmac_available()) {
        return false;
    }
    if (rows < QKQ_MIN_ROWS || features < QKQ_GBN || !mmac_q8q5_shape_ok(K)) {
        return false;
    }
    if (act_stride % 4 != 0 || w_row_stride > INT_MAX || w_mat_stride > INT_MAX || n_expert > 65535 ||
            features > INT_MAX || (rows + QKQ_GBM - 1) / QKQ_GBM > 65535) {
        return false;
    }

    const dim3 grid((unsigned) ((features + QKQ_GBN - 1) / QKQ_GBN),
                    (unsigned) ((rows     + QKQ_GBM - 1) / QKQ_GBM),
                    (unsigned) n_expert);

    static bool logged = false;
    if (!logged) {
        logged = true;
        fprintf(stderr, "%s: MMAC Q5_1 grouped GEMM in use, first shape rows=%lld features=%lld K=%lld experts=%lld\n",
                __func__, (long long) rows, (long long) features, (long long) K, (long long) n_expert);
    }

    mmac_gemm_q51_grouped_kernel<<<grid, QKQ_GNT, 0, stream>>>(
        (const char *) act, weight, ids_src1, ids_dst, expert_bounds, dst,
        (int) features, (int) K, act_stride, (int) w_row_stride, w_mat_stride);
    return true;
}
