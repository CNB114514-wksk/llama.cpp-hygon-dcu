#pragma once

#include "common.cuh"

// Hygon DCU MMAC matrix-core F16 GEMM.
//
// Computes (in ggml terms, for the cuBLAS CUBLAS_OP_T/CUBLAS_OP_N dense path):
//
//     dst[tokens][features] = act[tokens, K] * weight[features, K]^T
//
//   act    : row-major F16, k contiguous, row stride == K   (ggml src1, the activation)
//   weight : row-major F16, k contiguous, row stride == K   (ggml src0, the weight)
//   dst    : row-major F32, row stride == features          (ggml dst, dst->ne[0] == features)
//
// Note the argument order: the activation is passed as the "A" operand and the weight as the
// "B" operand so that the natural [features]-contiguous output layout matches ggml's dst.
//
// Returns true when the MMAC kernel handled the operation. Returns false when the shape/arch is
// unsupported or the kernel is disabled, in which case the caller must fall back to cuBLAS.
//
// The result is bit-exact against an F32 reference for the supported shapes; the accumulation is
// performed in F32 by the MMAC instruction itself.
//
// Fragment layout of `v_mmac_f32_16x16x16_f16` (verified on gfx928 and gfx936; NOT the AMD MFMA
// layout -- MFMA experience does not carry over). With l = lane % 64, h = l % 16, g = l / 16,
// bitswap(h) = (h % 4) * 4 + h / 4, and i = fragment element (0..3):
//
//   A[l, i] = A_mem[ h         ][ 4*g + i ]     <- K contiguous, one f16x4 load
//   B[l, i] = B_mem[ bitswap(h)][ 4*g + i ]     <- K contiguous, one f16x4 load
//   C[l, i] = C_mem[ h         ][ 4*g + i ]
//
// The hardware computes C = A * B^T, i.e. B is stored as (N, K).
//
// There is a second, equally correct convention that uses a strided K mapping:
//   A[l, i] = A_mem[h][4*i + g],  B[l, i] = B_mem[h][4*i + g],  C[l, i] = C_mem[h][4*i + g]
// It costs 4 scalar 2-byte loads per fragment instead of one 8-byte vector load, which is worth
// roughly 3x in throughput -- use the contiguous form above. The two conventions are a matched
// pair: mixing the contiguous-K A/B with a non-bitswapped B yields wrong results (192/256
// mismatches), so always take one pair as a whole. See mmac_dev/16_layout_crosscheck2.hip.
//
// Toggle with the environment variable GGML_CUDA_MMAC_GEMM (default: enabled).
bool ggml_cuda_mmac_gemm_f16(cudaStream_t stream,
                             const void * act, const void * weight, float * dst,
                             int64_t tokens, int64_t features, int64_t K);

// Runtime availability of the MMAC matrix core on the current device (arch + one-time probe).
bool ggml_cuda_mmac_available();
