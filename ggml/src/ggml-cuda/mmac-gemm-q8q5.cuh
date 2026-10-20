#pragma once

#include "common.cuh"

// Hygon DCU MMAC matrix-core GEMM for Q8_0 (dense projections) and Q5_1 (routed expert
// down-projections). Same idea as mmac-gemm-q4k.cuh: dequantize the 32-element sub-block into
// the f16 LDS tile that the MMAC mainloop consumes, so no f16 copy of the weights exists.
//
// Why these two types: in Qwen3.8-Flash-Next the whole dense stack (attn_qkv, attn_gate,
// ssm_out, hc_*, ple_*) is Q8_0 and every ffn_down_exps is Q5_1. Neither was covered by the
// Q4_K path, so ~70% of prefill FLOPs still ran on the SIMT MMQ kernels.
//
// Contract is identical to the Q4_K entry points:
//     dst[tokens][features] = act[tokens, K] * weight[features, K]^T
// with K % 32 == 0 (Q8_0 / Q5_1 blocks are 32 elements, unlike Q4_K's 256).
//
// Toggle with GGML_CUDA_MMAC_Q8 / GGML_CUDA_MMAC_Q51 (default: enabled).

bool ggml_cuda_mmac_gemm_q8_0(cudaStream_t stream,
                              const void * act, const void * weight, float * dst,
                              int64_t tokens, int64_t features, int64_t K,
                              int64_t w_row_stride);

bool ggml_cuda_mmac_gemm_q8_0_usable(const ggml_tensor * src0, const ggml_tensor * src1,
                                     const ggml_tensor * dst);

bool ggml_cuda_mmac_gemm_q5_1_id_usable(const ggml_tensor * src0, const ggml_tensor * src1,
                                        const ggml_tensor * ids, const ggml_tensor * dst);

bool ggml_cuda_mmac_gemm_q5_1_id(cudaStream_t stream,
                                 const void * act, const void * weight, float * dst,
                                 const int32_t * ids_src1, const int32_t * ids_dst, const int32_t * expert_bounds,
                                 int64_t n_expert, int64_t rows, int64_t features, int64_t K,
                                 int64_t act_stride, int64_t w_row_stride, int64_t w_mat_stride);
