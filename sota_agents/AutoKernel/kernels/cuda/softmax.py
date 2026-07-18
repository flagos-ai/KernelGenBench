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
AutoKernel -- CUDA C++ Softmax kernel.

Current kernel: Row-parallel numerically stable softmax with warp-level reductions.
Target metric: throughput (higher is better)
Secondary: correctness must ALWAYS pass

Features:
  - Numerically stable softmax (subtract row max before exp)
  - Warp-level tree reductions via __shfl_xor_sync for max and sum
  - Grid-stride loop over rows for large row counts
  - half2 vectorized global memory loads for fp16 inputs
  - One warp per row; supports arbitrary (non-power-of-2) row lengths
  - __expf / __fdividef for fast math intrinsics

The agent can change anything in this file:
  - Block size, warps per block, vectorization strategy
  - Reduction algorithm, memory access pattern
  - Any CUDA intrinsic or PTX instruction
"""

KERNEL_TYPE = "softmax"
BACKEND = "cuda"

import torch
from kernels.cuda._compile import compile_cuda

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>

// =========================================================================
// Warp-level reduction helpers
// =========================================================================

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xffffffff, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// =========================================================================
// Softmax kernel: one warp per row, grid-stride over rows
// =========================================================================

// WARPS_PER_BLOCK controls occupancy; each warp handles one row at a time
constexpr int WARPS_PER_BLOCK = 8;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;

__global__ void __launch_bounds__(THREADS_PER_BLOCK)
softmax_kernel(
    const float* __restrict__ input,
    float*       __restrict__ output,
    int n_rows,
    int n_cols
) {
    // Identify which warp we are
    const int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane        = threadIdx.x % 32;
    const int total_warps = (gridDim.x * blockDim.x) / 32;

    // Grid-stride loop over rows
    for (int row = warp_global; row < n_rows; row += total_warps) {
        const float* row_in  = input  + (long long)row * n_cols;
        float*       row_out = output + (long long)row * n_cols;

        // ---- Pass 1: find row max (for numerical stability) ----
        float thread_max = -FLT_MAX;
        for (int col = lane; col < n_cols; col += 32) {
            thread_max = fmaxf(thread_max, row_in[col]);
        }
        float row_max = warp_reduce_max(thread_max);

        // ---- Pass 2: compute exp(x - max) and accumulate sum ----
        float thread_sum = 0.0f;
        for (int col = lane; col < n_cols; col += 32) {
            float val = __expf(row_in[col] - row_max);
            row_out[col] = val;           // stash numerator
            thread_sum += val;
        }
        float row_sum = warp_reduce_sum(thread_sum);

        // ---- Pass 3: divide by sum ----
        float inv_sum = __fdividef(1.0f, row_sum);
        for (int col = lane; col < n_cols; col += 32) {
            row_out[col] *= inv_sum;
        }
    }
}

// =========================================================================
// Softmax kernel -- fp16 path with half2 vectorized loads
// =========================================================================

__global__ void __launch_bounds__(THREADS_PER_BLOCK)
softmax_kernel_fp16(
    const half* __restrict__ input,
    half*       __restrict__ output,
    int n_rows,
    int n_cols
) {
    const int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane        = threadIdx.x % 32;
    const int total_warps = (gridDim.x * blockDim.x) / 32;

    // Number of half2 pairs (for vectorized path)
    const int n_pairs = n_cols / 2;
    const int n_tail  = n_cols % 2;  // 0 or 1

    for (int row = warp_global; row < n_rows; row += total_warps) {
        const half* row_in  = input  + (long long)row * n_cols;
        half*       row_out = output + (long long)row * n_cols;

        // Reinterpret as half2 for vectorized access
        const half2* row_in_v  = reinterpret_cast<const half2*>(row_in);
        half2*       row_out_v = reinterpret_cast<half2*>(row_out);

        // ---- Pass 1: find row max ----
        float thread_max = -FLT_MAX;

        // Vectorized (half2) portion
        for (int i = lane; i < n_pairs; i += 32) {
            half2 v = row_in_v[i];
            float lo = __half2float(v.x);
            float hi = __half2float(v.y);
            thread_max = fmaxf(thread_max, fmaxf(lo, hi));
        }
        // Handle the odd trailing element (only lane 0 when n_cols is odd)
        if (n_tail && lane == 0) {
            float last = __half2float(row_in[n_cols - 1]);
            thread_max = fmaxf(thread_max, last);
        }

        float row_max = warp_reduce_max(thread_max);

        // ---- Pass 2: exp(x - max) and accumulate sum ----
        float thread_sum = 0.0f;

        for (int i = lane; i < n_pairs; i += 32) {
            half2 v = row_in_v[i];
            float lo = __expf(__half2float(v.x) - row_max);
            float hi = __expf(__half2float(v.y) - row_max);
            row_out_v[i] = __halves2half2(__float2half(lo), __float2half(hi));
            thread_sum += lo + hi;
        }
        if (n_tail && lane == 0) {
            float val = __expf(__half2float(row_in[n_cols - 1]) - row_max);
            row_out[n_cols - 1] = __float2half(val);
            thread_sum += val;
        }

        float row_sum = warp_reduce_sum(thread_sum);

        // ---- Pass 3: divide by sum ----
        float inv_sum = __fdividef(1.0f, row_sum);

        for (int i = lane; i < n_pairs; i += 32) {
            half2 v = row_out_v[i];
            float lo = __half2float(v.x) * inv_sum;
            float hi = __half2float(v.y) * inv_sum;
            row_out_v[i] = __halves2half2(__float2half(lo), __float2half(hi));
        }
        if (n_tail && lane == 0) {
            float val = __half2float(row_out[n_cols - 1]) * inv_sum;
            row_out[n_cols - 1] = __float2half(val);
        }
    }
}

// =========================================================================
// C++ launcher
// =========================================================================

torch::Tensor softmax_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    const int ndim   = input.dim();
    TORCH_CHECK(ndim >= 1, "input must have at least 1 dimension");

    // Flatten to 2-D: (n_rows, n_cols) where softmax is along the last dim
    const int n_cols = input.size(ndim - 1);
    const int n_rows = input.numel() / n_cols;

    auto output = torch::empty_like(input);

    // Choose grid size: one warp per row, WARPS_PER_BLOCK warps per block
    const int total_warps_needed = n_rows;
    const int blocks = (total_warps_needed + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    // Cap to reasonable grid size; grid-stride handles the rest
    const int grid = min(blocks, 65535);

    if (input.dtype() == torch::kFloat16) {
        softmax_kernel_fp16<<<grid, THREADS_PER_BLOCK>>>(
            reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
            reinterpret_cast<half*>(output.data_ptr<at::Half>()),
            n_rows, n_cols
        );
    } else if (input.dtype() == torch::kBFloat16) {
        // BFloat16: cast to fp32, run fp32 kernel, cast back
        auto input_f32  = input.to(torch::kFloat32);
        auto output_f32 = torch::empty_like(input_f32);
        softmax_kernel<<<grid, THREADS_PER_BLOCK>>>(
            input_f32.data_ptr<float>(),
            output_f32.data_ptr<float>(),
            n_rows, n_cols
        );
        output = output_f32.to(torch::kBFloat16);
    } else {
        // fp32 path
        softmax_kernel<<<grid, THREADS_PER_BLOCK>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            n_rows, n_cols
        );
    }

    return output;
}
"""

_module = None


def _get_module():
    global _module
    if _module is None:
        _module = compile_cuda(CUDA_SRC, "softmax_cuda")
    return _module


def kernel_fn(x: torch.Tensor) -> torch.Tensor:
    """Entry point called by bench.py. Must match reference.softmax_ref signature."""
    assert x.is_cuda

    # Preserve original shape and dtype
    orig_shape = x.shape
    orig_dtype = x.dtype

    # Ensure contiguous layout
    x = x.contiguous()

    # Flatten to 2-D: (n_rows, n_cols) -- softmax along last dim
    if x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim > 2:
        x = x.view(-1, x.shape[-1])

    # The CUDA launcher handles fp16/bf16/fp32 internally.
    # For unsupported dtypes, cast to fp32 round-trip.
    needs_cast = orig_dtype not in (torch.float16, torch.bfloat16, torch.float32)
    if needs_cast:
        x = x.to(torch.float32)

    mod = _get_module()
    out = mod.softmax_cuda(x)

    # Cast back if we had to promote
    if needs_cast:
        out = out.to(orig_dtype)

    return out.view(orig_shape)
