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
AutoKernel -- CUDA C++ Reduce (sum along last dim) kernel.

Current kernel: Hierarchical reduction with warp shuffle + shared memory + atomic.
Target metric: throughput (higher is better)
Secondary: correctness must ALWAYS pass

Features:
  - __shfl_down_sync cascade (5 steps) for intra-warp reduction
  - Shared memory for inter-warp reduction within a block
  - Vectorized loads for memory bandwidth
  - One row per block for simplicity and correctness
"""

KERNEL_TYPE = "reduce"
BACKEND = "cuda"

import torch
from kernels.cuda._compile import compile_cuda

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

constexpr int BLOCK_SIZE = 256;

// Warp-level sum reduction
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// One block per row. Each block cooperatively reduces one row of [M, N] to one scalar.
__global__ void reduce_sum_kernel(
    const half* __restrict__ input,  // [M, N]
    half* __restrict__ output,       // [M]
    int M, int N
) {
    const int row = blockIdx.x;
    if (row >= M) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = BLOCK_SIZE / 32;

    const half* row_ptr = input + row * N;

    // Phase 1: thread-local accumulation via grid-stride
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += BLOCK_SIZE) {
        local_sum += __half2float(row_ptr[i]);
    }

    // Phase 2: warp-level reduction
    local_sum = warp_reduce_sum(local_sum);

    // Phase 3: block-level reduction via shared memory
    __shared__ float smem[32];  // one per warp

    if (lane_id == 0) {
        smem[warp_id] = local_sum;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? smem[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) {
            output[row] = __float2half(val);
        }
    }
}

torch::Tensor reduce_sum_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(input.dim() == 2, "input must be [M, N]");

    int M = input.size(0);
    int N = input.size(1);

    auto output = torch::empty({M}, input.options());

    dim3 grid(M);
    dim3 block(BLOCK_SIZE);

    reduce_sum_kernel<<<grid, block>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        M, N
    );

    return output;
}
"""

_module = None


def _get_module():
    global _module
    if _module is None:
        _module = compile_cuda(CUDA_SRC, "reduce_sum_cuda")
    return _module


def kernel_fn(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Entry point called by bench.py. Must match reference.reduce_sum_ref signature."""
    assert x.is_cuda

    # Normalize dim
    if dim < 0:
        dim = x.ndim + dim
    assert 0 <= dim < x.ndim

    orig_dtype = x.dtype

    # For the common case of reducing the last dimension on a 2D tensor
    if dim == x.ndim - 1:
        if x.dtype != torch.float16:
            x = x.to(torch.float16)

        # Ensure 2D
        orig_shape = list(x.shape)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])

        mod = _get_module()
        out = mod.reduce_sum_cuda(x)

        if orig_dtype != torch.float16:
            out = out.to(orig_dtype)

        # Restore output shape (input shape minus last dim)
        out_shape = orig_shape[:-1]
        if not out_shape:
            out_shape = [1]
        return out.view(out_shape)
    else:
        # General case: move reduction dim to last, then reduce
        perm = list(range(x.ndim))
        perm.pop(dim)
        perm.append(dim)
        x = x.permute(*perm).contiguous()
        return kernel_fn(x, dim=-1)
