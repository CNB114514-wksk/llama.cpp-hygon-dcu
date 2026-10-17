#pragma once

#include "common.cuh"

// Hygon DCU MMAC matrix-core GEMM for Q4_K weights (gfx928 / gfx936).
//
// Extends mmac-gemm.cu to quantized weights: Q4_K super-blocks are dequantized on the fly into
// f16 shared-memory tiles and fed to `v_mmac_f32_16x16x16_f16`. The weights are NEVER
// materialized in f16 -- only the 256x32 tile currently being consumed lives in LDS, so the
// HBM traffic stays at the Q4_K rate (4.5 bits/weight), same as the MMQ path it replaces.
//
// Both entry points compute:
//
//     dst[tokens][features] = act[tokens, K] * weight[features, K]^T
//
//   act    : row-major F32 (or F16 for the dense variant), row stride == K
//   weight : Q4_K super-blocks along K (K % 256 == 0), row stride `w_row_stride` blocks,
//            expert e matrix base offset `w_mat_stride` blocks (grouped variant)
//   dst    : row-major F32, row stride == features
//
// Requires K % 256 == 0 (GGUF Q4_K tensors always are) and K % 32 == 0 (MMAC_KT).
// Returns true when the kernel handled the op; false = caller must fall back (MMQ).
//
// Toggle with the environment variable GGML_CUDA_MMAC_Q4K (default: enabled).

// Dense GGML_MUL_MAT (attention projections, shared expert, lm_head, ...).
bool ggml_cuda_mmac_gemm_q4_K(cudaStream_t stream,
                              const void * act, bool act_is_f32, const void * weight, float * dst,
                              int64_t tokens, int64_t features, int64_t K,
                              int64_t w_row_stride);

// Host-side gate for the grouped variant (mirrors the shape checks below so the caller can
// decide before allocating ids buffers; also used for the CUDA-graph capability check).
bool ggml_cuda_mmac_gemm_q4_K_id_usable(const ggml_tensor * src0, const ggml_tensor * src1,
                                        const ggml_tensor * ids, const ggml_tensor * dst);

// Grouped GGML_MUL_MAT_ID (routed MoE experts). The activation/output rows are sorted by
// expert, exactly like the MMQ ids path:
//   ids_src1[j]    : activation row for sorted row j (units of `act_stride` f32 elements)
//   ids_dst[j]     : output row for sorted row j (units of `features` floats)
//   expert_bounds  : [n_expert+1] int32; sorted rows of expert e = [eb[e], eb[e+1])
bool ggml_cuda_mmac_gemm_q4_K_id(cudaStream_t stream,
                                 const void * act, const void * weight, float * dst,
                                 const int32_t * ids_src1, const int32_t * ids_dst, const int32_t * expert_bounds,
                                 int64_t n_expert, int64_t rows, int64_t features, int64_t K,
                                 int64_t act_stride, int64_t w_row_stride, int64_t w_mat_stride);
