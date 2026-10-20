// Hygon DCU: CPU half of the host-expert split. See moe-host-split.cuh for the rationale.

#include "moe-host-split.cuh"
#include "mmac-gemm.cuh"   // ggml-impl.h -> ggml-common.h (block_q4_K / block_q5_1)

#include <immintrin.h>
#include <omp.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// The host half runs on the CPU while the GPU kernel is in flight, so it is compiled for AVX2
// explicitly instead of relying on the target flags of the rest of the backend.
#pragma clang attribute push(__attribute__((target("avx2,fma"))), apply_to = function)

namespace {

// block_q4_K holds QK_K = 256 elements, block_q5_1 holds QK5_1 = 32. Both are declared in
// ggml-common.h, which mmac-gemm.cuh pulls in.

static inline float h2f(uint16_t h) {
    return ggml_fp16_to_fp32(*(ggml_fp16_t *) &h);
}

// Same 6-bit packing as dequantize.cuh::get_scale_min_k4 (which is __device__ only).
static inline void host_get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j]     & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

// sum(q[i] * a[i]) for 8 elements. `hi` selects the high nibbles of q.
static inline __m256 dot_nib8(const uint8_t * q, const float * a, bool hi) {
    const __m128i b  = _mm_loadl_epi64((const __m128i *) q);
    __m256i       qi = _mm256_cvtepu8_epi32(b);
    if (hi) {
        qi = _mm256_srli_epi32(qi, 4);
    } else {
        qi = _mm256_and_si256(qi, _mm256_set1_epi32(0x0F));
    }
    return _mm256_mul_ps(_mm256_cvtepi32_ps(qi), _mm256_loadu_ps(a));
}

static inline float hsum256(__m256 v) {
    __m128 t = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    t = _mm_add_ps(t, _mm_movehl_ps(t, t));
    t = _mm_add_ss(t, _mm_shuffle_ps(t, t, 1));
    return _mm_cvtss_f32(t);
}

// a_blk[i] = sum of activation elements [32*i, 32*i+32). Shared by all output rows.
static float dot_q4_K(const void * wp, const float * a, const float * a_blk, int64_t K) {
    const block_q4_K * x = (const block_q4_K *) wp;
    __m256 acc  = _mm256_setzero_ps();
    float  bias = 0.0f;   // the -dmin*m_j*sum(a) terms; must not go through the 8-lane accumulator
    for (int64_t i = 0; i < K/QK_K; ++i) {
        // The scale/min header is a union whose member name depends on the declaration mode
        // (anonymous for C, `data` for C++), so overlay it with a plain 2x half struct.
        const uint16_t * hdr = (const uint16_t *) &x[i];
        const float d  = h2f(hdr[0]);
        const float mn = h2f(hdr[1]);
        const uint8_t * q = x[i].qs;
        for (int g = 0; g < 4; ++g) {
            uint8_t s0, m0, s1, m1;
            host_get_scale_min_k4(2*g + 0, x[i].scales, &s0, &m0);
            host_get_scale_min_k4(2*g + 1, x[i].scales, &s1, &m1);
            const int64_t b = 8*i + 2*g;
            __m256 lo = _mm256_setzero_ps();
            __m256 hi = _mm256_setzero_ps();
            for (int l = 0; l < 32; l += 8) {
                lo = _mm256_add_ps(lo, dot_nib8(q + l, a + 32*b       + l, false));
                hi = _mm256_add_ps(hi, dot_nib8(q + l, a + 32*(b + 1) + l, true));
            }
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d*s0), lo, acc);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d*s1), hi, acc);
            bias -= mn*(m0*a_blk[b] + m1*a_blk[b + 1]);
            q += 32;
        }
    }
    return hsum256(acc) + bias;
}

static float dot_q5_1(const void * wp, const float * a, const float * a_blk, int64_t K) {
    const block_q5_1 * x = (const block_q5_1 *) wp;
    const __m256i lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256 acc  = _mm256_setzero_ps();
    float  bias = 0.0f;
    for (int64_t i = 0; i < K/QK5_1; ++i) {
        const uint16_t * hdr = (const uint16_t *) &x[i];
        const float d = h2f(hdr[0]);
        const float m = h2f(hdr[1]);
        uint32_t qh;
        memcpy(&qh, x[i].qh, sizeof(qh));
        const float * ap = a + 32*i;
        __m256 lo = _mm256_setzero_ps();
        __m256 hi = _mm256_setzero_ps();
        for (int j = 0; j < 16; j += 8) {
            const __m128i b  = _mm_loadl_epi64((const __m128i *) (x[i].qs + j));
            const __m256i qi = _mm256_cvtepu8_epi32(b);
            // Element j keeps qs[j]'s low nibble and qh bit j; element j+16 keeps the high
            // nibble of the same byte and qh bit j+16.
            // The 5th bit of element e is bit e of qh. srli+mask would leave it at 1<<lane,
            // so shift per lane and keep only bit 0 before moving it into position 4.
            const __m256i qhb = _mm256_set1_epi32((int) qh);
            const __m256i b0 = _mm256_slli_epi32(_mm256_and_si256(
                    _mm256_srlv_epi32(qhb, _mm256_add_epi32(lane, _mm256_set1_epi32(j))),
                    _mm256_set1_epi32(1)), 4);
            const __m256i b1 = _mm256_slli_epi32(_mm256_and_si256(
                    _mm256_srlv_epi32(qhb, _mm256_add_epi32(lane, _mm256_set1_epi32(j + 16))),
                    _mm256_set1_epi32(1)), 4);
            const __m256i q0 = _mm256_or_si256(_mm256_and_si256(qi, _mm256_set1_epi32(0x0F)), b0);
            const __m256i q1 = _mm256_or_si256(_mm256_srli_epi32(qi, 4), b1);
            lo = _mm256_add_ps(lo, _mm256_mul_ps(_mm256_cvtepi32_ps(q0), _mm256_loadu_ps(ap + j)));
            hi = _mm256_add_ps(hi, _mm256_mul_ps(_mm256_cvtepi32_ps(q1), _mm256_loadu_ps(ap + j + 16)));
        }
        acc  = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_add_ps(lo, hi), acc);
        bias += m*a_blk[i];
    }
    return hsum256(acc) + bias;
}

// a_blk[i] = sum of activation elements [32*i, 32*i+32), one row at a time.
static void act_block_sums(const float * act, int64_t act_stride, int n_rows, int64_t n32, float * out) {
    for (int e = 0; e < n_rows; ++e) {
        const float * a = act + e*act_stride;
        for (int64_t i = 0; i < n32; ++i) {
            __m256 s = _mm256_setzero_ps();
            for (int j = 0; j < 32; j += 8) {
                s = _mm256_add_ps(s, _mm256_loadu_ps(a + 32*i + j));
            }
            out[(size_t) e*n32 + i] = hsum256(s);
        }
    }
}

} // namespace

#pragma clang attribute pop

bool ggml_cuda_moe_host_split_supported(enum ggml_type type, int64_t K) {
    switch (type) {
        case GGML_TYPE_Q4_K: return K % QK_K  == 0;
        case GGML_TYPE_Q5_1: return K % QK5_1 == 0;
        default:             return false;
    }
}

void ggml_cuda_moe_host_split_rows(
        enum ggml_type type,
        const char * w, int64_t w_row_stride, int64_t w_mat_stride,
        const float * act, int64_t act_stride, int64_t K,
        const int32_t * experts, int n_rows, int64_t M,
        float * out, int n_threads) {
    const int64_t n32 = K/32;

    // Block sums of the activations: one pass per row instead of once per output row.
    static std::vector<float> a_blk;
    a_blk.resize((size_t) n_rows * n32);
    act_block_sums(act, act_stride, n_rows, n32, a_blk.data());

    const int64_t total = (int64_t) n_rows * M;
    #pragma omp parallel for num_threads(n_threads) schedule(static)
    for (int64_t i = 0; i < total; ++i) {
        const int     e = (int) (i / M);
        const int64_t m = i % M;
        const char *  p = w + (int64_t) experts[e]*w_mat_stride + m*w_row_stride;
        const float * a = act + (int64_t) e*act_stride;
        const float * ab = a_blk.data() + (size_t) e*n32;
        out[i] = type == GGML_TYPE_Q4_K ? dot_q4_K(p, a, ab, K) : dot_q5_1(p, a, ab, K);
    }
}
