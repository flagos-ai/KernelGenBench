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
AutoKernel -- CUDA C++ RMSNorm kernel.

Current kernel: Fused RMSNorm with warp-shuffle reduction and vectorized loads.
Target metric: throughput (higher is better)
Secondary: correctness must ALWAYS pass

Features:
  - Single-pass RMS computation: rms = sqrt(mean(x^2) + eps)
  - Warp shuffle cascade (__shfl_down_sync) for fast intra-warp sum reduction
  - Block-level reduction via shared memory across warps
  - rsqrtf() for fast inverse square root
  - Vectorized half2 loads for maximum memory bandwidth
  - Fused normalize + scale in one output pass

The agent can change anything in this file:
  - Block sizes, thread counts, vectorization width
  - Reduction strategy, shared memory layout
  - Any CUDA intrinsic or PTX instruction
"""

KERNEL_TYPE = "rmsnorm"
BACKEND = "cuda"

import torch
from kernels.cuda._compile import compile_cuda

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// One block per row. Each thread handles multiple elements along N.
// Reduction: warp shuffle + shared memory across warps.

constexpr float EPS = 1e-6f;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void rmsnorm_kernel(
    const half* __restrict__ X,
    const half* __restrict__ W,
    half* __restrict__ OUT,
    int M, int N
) {
    // One block per row
    const int row = blockIdx.x;
    if (row >= M) return;

    const int tid = threadIdx.x;
    const int blockSize = blockDim.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = blockSize / 32;

    const half* x_row = X + row * N;
    half* out_row = OUT + row * N;

    // Phase 1: Compute sum of squares using vectorized loads
    float local_sum_sq = 0.0f;

    // Vectorized path: use half2 loads when possible
    int idx = tid * 2;
    for (; idx + 1 < N; idx += blockSize * 2) {
        half2 val = *reinterpret_cast<const half2*>(x_row + idx);
        float lo = __half2float(val.x);
        float hi = __half2float(val.y);
        local_sum_sq += lo * lo + hi * hi;
    }
    // Handle remainder (odd N or leftover element)
    if (idx < N) {
        float val = __half2float(x_row[idx]);
        local_sum_sq += val * val;
    }

    // Warp-level reduction
    local_sum_sq = warp_reduce_sum(local_sum_sq);

    // Block-level reduction via shared memory
    __shared__ float shared_sums[32];  // max 32 warps per block (1024 threads)

    if (lane_id == 0) {
        shared_sums[warp_id] = local_sum_sq;
    }
    __syncthreads();

    // First warp reduces across all warps
    float block_sum = 0.0f;
    if (warp_id == 0) {
        block_sum = (lane_id < num_warps) ? shared_sums[lane_id] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
    }

    // Broadcast the final rms_inv to all threads via shared memory
    __shared__ float s_rms_inv;
    if (tid == 0) {
        float mean_sq = block_sum / static_cast<float>(N);
        s_rms_inv = rsqrtf(mean_sq + EPS);
    }
    __syncthreads();

    float rms_inv = s_rms_inv;

    // Phase 2: Fused normalize + scale, vectorized stores
    idx = tid * 2;
    for (; idx + 1 < N; idx += blockSize * 2) {
        half2 xval = *reinterpret_cast<const half2*>(x_row + idx);
        half2 wval = *reinterpret_cast<const half2*>(W + idx);

        float x0 = __half2float(xval.x) * rms_inv * __half2float(wval.x);
        float x1 = __half2float(xval.y) * rms_inv * __half2float(wval.y);

        half2 result;
        result.x = __float2half(x0);
        result.y = __float2half(x1);
        *reinterpret_cast<half2*>(out_row + idx) = result;
    }
    // Handle remainder
    if (idx < N) {
        float xv = __half2float(x_row[idx]);
        float wv = __half2float(W[idx]);
        out_row[idx] = __float2half(xv * rms_inv * wv);
    }
}

torch::Tensor rmsnorm_cuda(torch::Tensor x, torch::Tensor weight) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat16, "x must be float16");
    TORCH_CHECK(weight.dtype() == torch::kFloat16, "weight must be float16");

    int M = x.size(0);
    int N = x.size(1);

    auto out = torch::empty_like(x);

    // Choose block size: enough threads to cover N with vectorized loads
    // Each thread handles 2 elements, so we need N/2 threads minimum, up to 1024
    int threads = min(1024, max(32, ((N + 1) / 2 + 31) / 32 * 32));

    dim3 grid(M);
    dim3 block(threads);

    rmsnorm_kernel<<<grid, block>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        M, N
    );

    return out;
}
"""

_module = None


def _get_module():
    global _module
    if _module is None:
        _module = compile_cuda(CUDA_SRC, "rmsnorm_cuda")
    return _module


def kernel_fn(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Entry point called by bench.py. Must match reference.rmsnorm_ref signature."""
    assert x.is_cuda and weight.is_cuda

    # Handle non-fp16 inputs by casting
    orig_dtype = x.dtype
    if x.dtype != torch.float16:
        x = x.to(torch.float16)
    if weight.dtype != torch.float16:
        weight = weight.to(torch.float16)

    mod = _get_module()
    out = mod.rmsnorm_cuda(x, weight)

    # Cast back if needed
    if orig_dtype != torch.float16:
        out = out.to(orig_dtype)

    return out
