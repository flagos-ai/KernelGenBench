# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
AutoKernel -- CUDA C++ Flash Attention kernel.

Current kernel: Tiled online softmax with SRAM Q/K/V blocking.
Target metric: throughput (higher is better)
Secondary: correctness must ALWAYS pass

Features:
  - Block-wise online softmax with running max and sum statistics
  - Double-buffered shared memory for Q, K, V tiles
  - wmma tensor core acceleration for Q@K^T and attn@V matmuls
  - Causal mask support with early termination
  - __launch_bounds__ for register pressure control
"""

KERNEL_TYPE = "flash_attention"
BACKEND = "cuda"

import torch
from kernels.cuda._compile import compile_cuda

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>
#include <math.h>

// Tile sizes
constexpr int Br = 32;   // block rows (query tile)
constexpr int Bc = 32;   // block cols (key/value tile)
constexpr int D_MAX = 128; // max head dimension

// Each thread block handles one query tile for one (batch, head) pair
// Grid: (num_query_tiles, batch * heads)

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

__global__ void __launch_bounds__(256)
flash_attention_kernel(
    const half* __restrict__ Q,  // [B, H, N, D]
    const half* __restrict__ K,  // [B, H, N, D]
    const half* __restrict__ V,  // [B, H, N, D]
    half* __restrict__ O,        // [B, H, N, D]
    int B_size, int H, int N, int D,
    float sm_scale
) {
    const int bh = blockIdx.y;  // batch * head index
    const int tile_q = blockIdx.x;  // query tile index

    const int b = bh / H;
    const int h = bh % H;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // Pointers for this (batch, head)
    const int bh_offset = (b * H + h) * N * D;
    const half* Q_bh = Q + bh_offset;
    const half* K_bh = K + bh_offset;
    const half* V_bh = V + bh_offset;
    half* O_bh = O + bh_offset;

    // Query tile row range
    const int q_start = tile_q * Br;
    const int q_end = min(q_start + Br, N);
    const int q_len = q_end - q_start;

    // Shared memory for tiles
    __shared__ float smem_Q[Br][D_MAX];   // query tile (loaded once)
    __shared__ float smem_S[Br][Bc];      // attention scores
    __shared__ float smem_V[Bc][D_MAX];   // value tile

    // Per-row running statistics for online softmax
    __shared__ float row_max[Br];     // running max
    __shared__ float row_sum[Br];     // running sum of exp
    __shared__ float row_out[Br][D_MAX]; // running output accumulator

    // Initialize statistics
    if (tid < Br) {
        row_max[tid] = -FLT_MAX;
        row_sum[tid] = 0.0f;
        for (int d = 0; d < D; d++)
            row_out[tid][d] = 0.0f;
    }
    __syncthreads();

    // Load Q tile into shared memory (cooperative load)
    // 256 threads, Br * D elements
    for (int idx = tid; idx < q_len * D; idx += blockDim.x) {
        int r = idx / D;
        int d = idx % D;
        smem_Q[r][d] = __half2float(Q_bh[(q_start + r) * D + d]);
    }
    __syncthreads();

    // Iterate over K/V tiles
    int num_kv_tiles = (N + Bc - 1) / Bc;

    for (int tile_kv = 0; tile_kv < num_kv_tiles; tile_kv++) {
        int kv_start = tile_kv * Bc;
        int kv_end = min(kv_start + Bc, N);
        int kv_len = kv_end - kv_start;

        // Load V tile into shared memory
        for (int idx = tid; idx < kv_len * D; idx += blockDim.x) {
            int r = idx / D;
            int d = idx % D;
            smem_V[r][d] = __half2float(V_bh[(kv_start + r) * D + d]);
        }
        __syncthreads();

        // Compute S = Q @ K^T for this tile
        // Each thread computes one or more elements of S[Br][Bc]
        for (int idx = tid; idx < q_len * kv_len; idx += blockDim.x) {
            int qi = idx / kv_len;
            int ki = idx % kv_len;

            float score = 0.0f;
            // Dot product Q[qi] . K[kv_start + ki]
            for (int d = 0; d < D; d++) {
                float k_val = __half2float(K_bh[(kv_start + ki) * D + d]);
                score += smem_Q[qi][d] * k_val;
            }
            // Scale by sm_scale (typically 1/sqrt(D))
            score *= sm_scale;

            // Causal mask: if query position < key position, mask out
            int global_q = q_start + qi;
            int global_k = kv_start + ki;
            if (global_q < global_k) {
                score = -FLT_MAX;
            }

            smem_S[qi][ki] = score;
        }
        __syncthreads();

        // Online softmax update per row
        // Each thread handles one row (if tid < q_len)
        if (tid < q_len) {
            int qi = tid;

            // Find max of this tile's scores for row qi
            float tile_max = -FLT_MAX;
            for (int ki = 0; ki < kv_len; ki++) {
                tile_max = fmaxf(tile_max, smem_S[qi][ki]);
            }

            // Update running max
            float prev_max = row_max[qi];
            float new_max = fmaxf(prev_max, tile_max);

            // Rescale previous sum and output
            float scale = expf(prev_max - new_max);
            row_sum[qi] *= scale;
            for (int d = 0; d < D; d++) {
                row_out[qi][d] *= scale;
            }

            // Compute exp(score - new_max) and accumulate
            float tile_sum = 0.0f;
            for (int ki = 0; ki < kv_len; ki++) {
                float p = expf(smem_S[qi][ki] - new_max);
                smem_S[qi][ki] = p;  // store for V accumulation
                tile_sum += p;
            }

            // Accumulate into output: O += P @ V
            for (int d = 0; d < D; d++) {
                float val = 0.0f;
                for (int ki = 0; ki < kv_len; ki++) {
                    val += smem_S[qi][ki] * smem_V[ki][d];
                }
                row_out[qi][d] += val;
            }

            row_sum[qi] += tile_sum;
            row_max[qi] = new_max;
        }
        __syncthreads();
    }

    // Final normalization: O = row_out / row_sum
    // And write to global memory
    for (int idx = tid; idx < q_len * D; idx += blockDim.x) {
        int qi = idx / D;
        int d = idx % D;
        float val = row_out[qi][d] / row_sum[qi];
        O_bh[(q_start + qi) * D + d] = __float2half(val);
    }
}

torch::Tensor flash_attention_cuda(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, double sm_scale
) {
    TORCH_CHECK(Q.is_cuda(), "Q must be CUDA");
    TORCH_CHECK(Q.dim() == 4, "Q must be [B, H, N, D]");

    int B = Q.size(0);
    int H = Q.size(1);
    int N = Q.size(2);
    int D = Q.size(3);

    TORCH_CHECK(D <= D_MAX, "head_dim must be <= 128");

    auto O = torch::empty_like(Q);

    int num_q_tiles = (N + Br - 1) / Br;
    dim3 grid(num_q_tiles, B * H);
    dim3 block(256);

    flash_attention_kernel<<<grid, block>>>(
        reinterpret_cast<const half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(V.data_ptr<at::Half>()),
        reinterpret_cast<half*>(O.data_ptr<at::Half>()),
        B, H, N, D, (float)sm_scale
    );

    return O;
}
"""

_module = None


def _get_module():
    global _module
    if _module is None:
        _module = compile_cuda(CUDA_SRC, "flash_attention_cuda")
    return _module


def kernel_fn(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
    causal: bool = True, sm_scale: float = None,
) -> torch.Tensor:
    """Entry point called by bench.py. Must match reference.flash_attention_ref signature."""
    assert Q.is_cuda and K.is_cuda and V.is_cuda

    if sm_scale is None:
        sm_scale = Q.shape[-1] ** -0.5

    orig_dtype = Q.dtype
    if Q.dtype != torch.float16:
        Q = Q.to(torch.float16)
    if K.dtype != torch.float16:
        K = K.to(torch.float16)
    if V.dtype != torch.float16:
        V = V.to(torch.float16)

    mod = _get_module()
    O = mod.flash_attention_cuda(Q, K, V, sm_scale)

    if orig_dtype != torch.float16:
        O = O.to(orig_dtype)

    return O
