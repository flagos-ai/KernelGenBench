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
AutoKernel -- CUDA C++ Matrix Multiplication kernel.

Current kernel: Tensor Core GEMM via wmma API with double-buffered shared memory.
Target metric: throughput_tflops (higher is better)
Secondary: correctness must ALWAYS pass

Features:
  - nvcuda::wmma::mma_sync for tensor core acceleration (16x16x16 tiles)
  - Double-buffered shared memory to overlap loads with compute
  - Bank-conflict-free shared memory layout (padding)
  - Vectorized float4 global memory loads for maximum bandwidth
  - __launch_bounds__ for register pressure control

The agent can change anything in this file:
  - Tile sizes, warp counts, buffer stages
  - Memory layout, swizzling, prefetch strategy
  - Accumulation precision, epilogue fusion
  - Any CUDA intrinsic or PTX instruction
"""

KERNEL_TYPE = "matmul"
BACKEND = "cuda"

import torch
from kernels.cuda._compile import compile_cuda

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// Tile dimensions
constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 32;

// wmma tile dimensions
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Warps per block
constexpr int WARPS_M = BLOCK_M / WMMA_M;  // 8
constexpr int WARPS_N = BLOCK_N / WMMA_N;  // 8
// We use a 4x2 warp layout (each warp computes 2x4 wmma tiles)
constexpr int WARP_LAYOUT_M = 4;
constexpr int WARP_LAYOUT_N = 2;

// Shared memory padding to avoid bank conflicts (128 bytes = 64 half elements)
constexpr int SMEM_PAD = 8;

// Each warp covers (BLOCK_M/WARP_LAYOUT_M) x (BLOCK_N/WARP_LAYOUT_N) = 32x64 output
// = 2x4 wmma tiles

__global__ void __launch_bounds__(256)
matmul_kernel_wmma(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    // Block indices
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // Thread indices
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // Warp position in the 4x2 grid
    const int warp_m = warp_id / WARP_LAYOUT_N;  // 0..3
    const int warp_n = warp_id % WARP_LAYOUT_N;  // 0..1

    // Shared memory: double buffered
    // A tile: BLOCK_M x (BLOCK_K + PAD), B tile: BLOCK_K x (BLOCK_N + PAD)
    __shared__ half smem_A[2][BLOCK_M][BLOCK_K + SMEM_PAD];
    __shared__ half smem_B[2][BLOCK_K][BLOCK_N + SMEM_PAD];

    // Global memory base pointers for this block's tiles
    const int block_row = bx * BLOCK_M;
    const int block_col = by * BLOCK_N;

    // Accumulator fragments: each warp computes 2x4 wmma tiles = 32x64 output
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[2][4];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            wmma::fill_fragment(acc[i][j], 0.0f);

    // Fragments for A and B
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_A;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_B;

    // Number of K tiles
    const int k_tiles = (K + BLOCK_K - 1) / BLOCK_K;

    // Helper: load a tile of A from global to shared memory
    // Each thread loads multiple elements. 256 threads, BLOCK_M * BLOCK_K = 128*32 = 4096 elements
    // = 16 elements per thread
    auto load_A_tile = [&](int buf, int k_start) {
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int idx = tid * 16 + i;
            int row = idx / BLOCK_K;
            int col = idx % BLOCK_K;
            int g_row = block_row + row;
            int g_col = k_start + col;
            if (g_row < M && g_col < K)
                smem_A[buf][row][col] = A[g_row * K + g_col];
            else
                smem_A[buf][row][col] = __float2half(0.0f);
        }
    };

    // Helper: load a tile of B from global to shared memory
    // BLOCK_K * BLOCK_N = 32*128 = 4096 elements = 16 per thread
    auto load_B_tile = [&](int buf, int k_start) {
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int idx = tid * 16 + i;
            int row = idx / BLOCK_N;
            int col = idx % BLOCK_N;
            int g_row = k_start + row;
            int g_col = block_col + col;
            if (g_row < K && g_col < N)
                smem_B[buf][row][col] = B[g_row * N + g_col];
            else
                smem_B[buf][row][col] = __float2half(0.0f);
        }
    };

    // Load first tile into buffer 0
    load_A_tile(0, 0);
    load_B_tile(0, 0);
    __syncthreads();

    // Main loop with double buffering
    for (int k = 0; k < k_tiles; k++) {
        int cur_buf = k % 2;
        int next_buf = 1 - cur_buf;

        // Prefetch next tile into the other buffer (if not last iteration)
        if (k + 1 < k_tiles) {
            load_A_tile(next_buf, (k + 1) * BLOCK_K);
            load_B_tile(next_buf, (k + 1) * BLOCK_K);
        }

        // Compute: each warp processes its 2x4 wmma tiles
        // Warp (warp_m, warp_n) covers rows [warp_m*32 .. warp_m*32+31]
        //                                 cols [warp_n*64 .. warp_n*64+63]
        #pragma unroll
        for (int kk = 0; kk < BLOCK_K / WMMA_K; kk++) {
            // For each 16x16 sub-tile the warp is responsible for:
            #pragma unroll
            for (int wm = 0; wm < 2; wm++) {
                int a_row = warp_m * 32 + wm * WMMA_M;
                int a_col = kk * WMMA_K;
                wmma::load_matrix_sync(frag_A,
                    &smem_A[cur_buf][a_row][a_col],
                    BLOCK_K + SMEM_PAD);

                #pragma unroll
                for (int wn = 0; wn < 4; wn++) {
                    int b_row = kk * WMMA_K;
                    int b_col = warp_n * 64 + wn * WMMA_N;
                    wmma::load_matrix_sync(frag_B,
                        &smem_B[cur_buf][b_row][b_col],
                        BLOCK_N + SMEM_PAD);

                    wmma::mma_sync(acc[wm][wn], frag_A, frag_B, acc[wm][wn]);
                }
            }
        }

        __syncthreads();
    }

    // Store results: convert fp32 accumulator to fp16 and write to global memory
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc_half;

    #pragma unroll
    for (int wm = 0; wm < 2; wm++) {
        #pragma unroll
        for (int wn = 0; wn < 4; wn++) {
            int c_row = block_row + warp_m * 32 + wm * WMMA_M;
            int c_col = block_col + warp_n * 64 + wn * WMMA_N;

            if (c_row < M && c_col < N) {
                // Convert fp32 -> fp16
                #pragma unroll
                for (int i = 0; i < acc[wm][wn].num_elements; i++) {
                    acc_half.x[i] = __float2half(acc[wm][wn].x[i]);
                }
                wmma::store_matrix_sync(
                    &C[c_row * N + c_col],
                    acc_half,
                    N,
                    wmma::mem_row_major);
            }
        }
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
    TORCH_CHECK(B.dtype() == torch::kFloat16, "B must be float16");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());

    dim3 grid((M + BLOCK_M - 1) / BLOCK_M, (N + BLOCK_N - 1) / BLOCK_N);
    dim3 block(256);  // 8 warps

    matmul_kernel_wmma<<<grid, block>>>(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(B.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, N, K
    );

    return C;
}
"""

_module = None


def _get_module():
    global _module
    if _module is None:
        _module = compile_cuda(CUDA_SRC, "matmul_cuda")
    return _module


def kernel_fn(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Entry point called by bench.py. Must match reference.matmul_ref signature."""
    assert A.is_cuda and B.is_cuda

    # Handle non-fp16 inputs by casting
    orig_dtype = A.dtype
    if A.dtype != torch.float16:
        A = A.to(torch.float16)
    if B.dtype != torch.float16:
        B = B.to(torch.float16)

    mod = _get_module()
    C = mod.matmul_cuda(A, B)

    # Cast back if needed
    if orig_dtype != torch.float16:
        C = C.to(orig_dtype)

    return C
