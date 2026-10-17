// Hygon DCU MMAC matrix-core F16 GEMM (gfx928 / gfx936).
//
// Replaces the cuBLAS/rocBLAS F16 dense GEMM for ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F16>.
// On K100_AI the shipped rocBLAS has only gfx906 Tensile kernels for F16, so the default path
// runs a non-matrix-core kernel at ~4-11 TFLOPS; this kernel reaches ~74-98 TFLOPS with the
// Hygon matrix core (MMAC, `v_mmac_f32_16x16x16_f16`).
//
// See mmac-gemm.cuh for the operand/layout contract.

#include "mmac-gemm.cuh"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <climits>

// The MMAC builtin is only visible during the device compilation pass, so the availability macro
// below is 1 in the device pass and 0 in the host pass. It is therefore only used *inside* kernel
// bodies -- never to guard a kernel definition or a host function (that would make the two passes
// disagree). The host side learns availability at runtime from the probe kernel.
#if defined(__has_builtin)
#  if __has_builtin(__builtin_amdgcn_mmac_f32_16x16x16f16)
#    define GGML_MMAC_BUILTIN 1
#  endif
#endif
#ifndef GGML_MMAC_BUILTIN
#  define GGML_MMAC_BUILTIN 0
#endif

// Tile configuration. Each wave computes a 64x64 tile out of 4x4 MMAC 16x16x16 accumulators.
#define MMAC_BM 128                       // rows of A (tokens) per block
#define MMAC_BN 256                       // rows of B (features) per block
#define MMAC_KT 32                        // contraction step; K must be a multiple of this
#define MMAC_PAD 8                        // LDS bank-conflict padding
#define MMAC_LDS_STRIDE (MMAC_KT + MMAC_PAD)
#define MMAC_WTM (MMAC_BM/64)
#define MMAC_WTN (MMAC_BN/64)
#define MMAC_NT  ((MMAC_WTM)*(MMAC_WTN)*64)   // 512 threads

// Below this many tokens the rocBLAS path is competitive and the small grid leaves the device
// under-utilized, so we fall back.
#define MMAC_MIN_TOKENS 256

typedef _Float16 mmac_f16x4 __attribute__((vector_size(8)));
typedef float    mmac_f32x4 __attribute__((vector_size(16)));

// ---------------------------------------------------------------------------------------------
// Probe: reports 1 when the MMAC builtin was actually available for this target.
// ---------------------------------------------------------------------------------------------
__global__ void mmac_probe_kernel(int * out) {
    if (threadIdx.x == 0) {
#if GGML_MMAC_BUILTIN
        *out = 1;
#else
        *out = 0;
#endif
    }
}

// ---------------------------------------------------------------------------------------------
// C[tokens][features] = A[tokens, K] * B[features, K]^T   (A = activation, B = weight)
//
// Requires K % MMAC_KT == 0 (so the k-index of a vector load is always in bounds).
// tokens/features may be arbitrary: an out-of-range *row* is clamped to the last valid row, which
// keeps every vector load in bounds, and the resulting garbage rows are dropped by the store
// guard. ALIGN=true (both dims are exact multiples of the tile) removes the clamps and guards
// entirely -- that instantiation is bit-for-bit the plain fast kernel.
// ---------------------------------------------------------------------------------------------
template <bool ALIGN>
__global__ void __launch_bounds__(MMAC_NT, 1)
mmac_gemm_f16_kernel(const __half * __restrict__ A, const __half * __restrict__ B,
                     float * __restrict__ C, int M, int N, int K) {
#if GGML_MMAC_BUILTIN
    __shared__ __half As[MMAC_BM * MMAC_LDS_STRIDE];
    __shared__ __half Bs[MMAC_BN * MMAC_LDS_STRIDE];

    const int t = threadIdx.x;
    const int l = t & 63;
    const int wm = (t / 64) % MMAC_WTM;
    const int wn = (t / 64) / MMAC_WTM;
    const int m0 = blockIdx.y * MMAC_BM, n0 = blockIdx.x * MMAC_BN;
    const int lr = l % 16, lg = l / 16;
    const int lc = (l % 4) * 4 + (l % 16) / 4;

    mmac_f32x4 acc[4][4];
    #pragma unroll
    for (int r = 0; r < 4; ++r)
        #pragma unroll
        for (int c = 0; c < 4; ++c) acc[r][c] = {0, 0, 0, 0};

    constexpr int PER_A = (MMAC_BM * MMAC_KT) / MMAC_NT;   // 8
    constexpr int PER_B = (MMAC_BN * MMAC_KT) / MMAC_NT;   // 16
    const int arow = t / (MMAC_KT / PER_A), acol = (t % (MMAC_KT / PER_A)) * PER_A;
    const int brow = t / (MMAC_KT / PER_B), bcol = (t % (MMAC_KT / PER_B)) * PER_B;

    // Uniform per block, so interior blocks never diverge.
    const bool m_full = ALIGN || (m0 + MMAC_BM <= M);
    const bool n_full = ALIGN || (n0 + MMAC_BN <= N);
    const int  m_last = M - 1, n_last = N - 1;

    for (int k0 = 0; k0 < K; k0 += MMAC_KT) {
        #pragma unroll
        for (int v = 0; v < PER_A; v += 4) {
            int r = m0 + arow;
            if constexpr (!ALIGN) { if (!m_full) r = min(r, m_last); }
            const int c = k0 + acol + v;
            *reinterpret_cast<mmac_f16x4 *>(&As[arow * MMAC_LDS_STRIDE + acol + v]) =
                *reinterpret_cast<const mmac_f16x4 *>(A + (size_t) r * K + c);
        }
        #pragma unroll
        for (int v = 0; v < PER_B; v += 4) {
            int r = n0 + brow;
            if constexpr (!ALIGN) { if (!n_full) r = min(r, n_last); }
            const int c = k0 + bcol + v;
            *reinterpret_cast<mmac_f16x4 *>(&Bs[brow * MMAC_LDS_STRIDE + bcol + v]) =
                *reinterpret_cast<const mmac_f16x4 *>(B + (size_t) r * K + c);
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < MMAC_KT; kk += 16) {
            mmac_f16x4 a[4], b[4];
            #pragma unroll
            for (int rg = 0; rg < 4; ++rg)
                a[rg] = *reinterpret_cast<const mmac_f16x4 *>(
                    &As[(wm * 64 + rg * 16 + lr) * MMAC_LDS_STRIDE + kk + lg * 4]);
            #pragma unroll
            for (int cg = 0; cg < 4; ++cg)
                b[cg] = *reinterpret_cast<const mmac_f16x4 *>(
                    &Bs[(wn * 64 + cg * 16 + lc) * MMAC_LDS_STRIDE + kk + lg * 4]);
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
            if constexpr (ALIGN) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) C[(size_t) cr * N + cn + i] = acc[rg][cg][i];
            } else {
                if (m_full && n_full) {
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) C[(size_t) cr * N + cn + i] = acc[rg][cg][i];
                } else if (cr < M) {
                    #pragma unroll
                    for (int i = 0; i < 4; ++i)
                        if (cn + i < N) C[(size_t) cr * N + cn + i] = acc[rg][cg][i];
                }
            }
        }
#else
    (void) A; (void) B; (void) C; (void) M; (void) N; (void) K;
#endif
}

// ---------------------------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------------------------

static bool mmac_env_enabled() {
    const char * env = getenv("GGML_CUDA_MMAC_GEMM");
    if (env == nullptr) {
        return true;   // enabled by default
    }
    return atoi(env) != 0;
}

// Cheap host-side gate: only Hygon DCU parts whose ISA includes MMAC. This avoids running the
// device probe (which allocates and synchronizes) on hardware that can never use the kernel.
static bool mmac_arch_capable() {
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) {
        return false;
    }
    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, dev) != hipSuccess) {
        return false;
    }
    const char * name = prop.gcnArchName;   // e.g. "gfx928:sramecc+:xnack-"
    return strncmp(name, "gfx928", 6) == 0 || strncmp(name, "gfx936", 6) == 0;
}

// Cached one-time device probe. Returns true when the target really has the MMAC builtin.
static bool mmac_available() {
    static int cached = -1;
    if (cached >= 0) {
        return cached != 0;
    }
    cached = 0;
    if (!mmac_env_enabled() || !mmac_arch_capable()) {
        return false;
    }

    int * d_flag = nullptr;
    if (hipMalloc(&d_flag, sizeof(int)) != hipSuccess) {
        GGML_LOG_WARN("%s: hipMalloc failed, disabling MMAC GEMM\n", __func__);
        return false;
    }
    mmac_probe_kernel<<<1, 1>>>(d_flag);
    const hipError_t err = hipDeviceSynchronize();
    int h_flag = 0;
    if (err == hipSuccess) {
        if (hipMemcpy(&h_flag, d_flag, sizeof(int), hipMemcpyDeviceToHost) != hipSuccess) {
            h_flag = 0;
        }
    }
    hipFree(d_flag);

    cached = (err == hipSuccess && h_flag == 1) ? 1 : 0;
    GGML_LOG_INFO("%s: MMAC matrix-core F16 GEMM %s\n", __func__, cached ? "enabled" : "unavailable");
    return cached != 0;
}

// Exported for the sibling Q4_K MMAC kernels (mmac-gemm-q4k.cu): cheap after the first call.
bool ggml_cuda_mmac_available() {
    return mmac_available();
}

bool ggml_cuda_mmac_gemm_f16(cudaStream_t stream,
                             const void * act, const void * weight, float * dst,
                             int64_t tokens, int64_t features, int64_t K) {
    if (!mmac_available()) {
        return false;
    }
    if (tokens < MMAC_MIN_TOKENS || features < MMAC_BN || K < MMAC_KT) {
        return false;
    }
    // K must be a multiple of the contraction step: a k-vector load must never cross into the
    // next row. tokens/features may be arbitrary (row clamping + store guard handles them).
    if (K % MMAC_KT != 0) {
        return false;
    }
    if ((tokens + MMAC_BM - 1) / MMAC_BM > 65535 || features > INT_MAX || K > INT_MAX) {
        return false;   // grid.y limit / 32-bit index overflow
    }

    const dim3 grid((unsigned) ((features + MMAC_BN - 1) / MMAC_BN),
                    (unsigned) ((tokens   + MMAC_BM - 1) / MMAC_BM));
    const bool aligned = (tokens % MMAC_BM == 0) && (features % MMAC_BN == 0);

    static bool logged = false;
    if (!logged) {
        logged = true;
        GGML_LOG_INFO("%s: MMAC F16 GEMM in use (%s), first shape tokens=%lld features=%lld K=%lld\n",
                      __func__, aligned ? "aligned" : "clamped",
                      (long long) tokens, (long long) features, (long long) K);
    }

    if (aligned) {
        mmac_gemm_f16_kernel<true><<<grid, MMAC_NT, 0, stream>>>(
            (const __half *) act, (const __half *) weight, dst,
            (int) tokens, (int) features, (int) K);
    } else {
        mmac_gemm_f16_kernel<false><<<grid, MMAC_NT, 0, stream>>>(
            (const __half *) act, (const __half *) weight, dst,
            (int) tokens, (int) features, (int) K);
    }
    return true;
}
