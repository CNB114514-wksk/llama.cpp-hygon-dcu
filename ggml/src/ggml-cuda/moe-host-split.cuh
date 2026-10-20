// Hygon DCU: CPU half of the host-expert split (see ggml_cuda_moe_host_split in ggml-cuda.cu).
//
// The host-resident expert layers of this model are read over PCIe by the GPU at ~15.4 GB/s
// (PCIe 3.0 x16 ceiling) while the 8 cores the server is pinned to stream the same weights out
// of DRAM at ~16 GB/s. The two resources are independent, so giving each path half of the
// expert slots and running them concurrently halves the per-layer cost (measured: 1.32 ms and
// 1.27 ms per layer for the two paths separately).
//
// This file implements the CPU side: out[e][m] = dot(W[experts[e]] + m*w_row_stride, act[e]).
// The dot is fused with on-the-fly dequantization, using q8_0 activations so that 32 elements
// are consumed per vpmaddubsw (the same trick ggml-cpu uses).

#pragma once

#include "ggml.h"

#include <cstdint>

// True if the CPU kernels below understand this quantized type at the given K.
bool ggml_cuda_moe_host_split_supported(enum ggml_type type, int64_t K);

// act:      n_rows activation rows, each K floats, row stride act_stride floats
// experts:  n_rows expert indices into w (w_mat_stride bytes apart)
// out:      n_rows * M floats, laid out as out[e*M + m]
void ggml_cuda_moe_host_split_rows(
        enum ggml_type type,
        const char * w, int64_t w_row_stride, int64_t w_mat_stride,
        const float * act, int64_t act_stride, int64_t K,
        const int32_t * experts, int n_rows, int64_t M,
        float * out, int n_threads);
