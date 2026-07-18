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
AutoKernel -- CUDA C++ Layer Normalization kernel.

Current kernel: LayerNorm with Welford's online algorithm and warp shuffle reductions.
Target metric: throughput (higher is better)
Secondary: correctness must ALWAYS pass

Features:
  - Welford's online algorithm for numerically stable single-pass mean/variance
  - Warp shuffle (__shfl_down_sync) reductions for partial statistics merging
  - Block-level cooperative reduction via shared memory across warps
  - Vectorized float4 global memory loads for maximum bandwidth
  - Fused scale+bias epilogue with rsqrtf fast inverse square root
  - One block per row for full row-parallel processing

The agent can change anything in this file:
  - Block sizes, warp counts, vectorization width
  - Reduction strategy, shared memory layout
  - Precision handling, epilogue fusion
  - Any CUDA intrinsic or PTX instruction
"""

KERNEL_TYPE = "layernorm"
BACKEND = "cuda"

import torch
from kernels.cuda._compile import compile_cuda

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// -----------------------------------------------------------------------
// Welford's online algorithm helpers
// -----------------------------------------------------------------------

struct WelfordState {
    float mean;
    float m2;
    float count;
};

// Merge two Welford partial aggregates
__device__ __forceinline__ WelfordState welford_merge(
    WelfordState a, WelfordState b
) {
    WelfordState out;
    float count = a.count + b.count;
    if (count == 0.0f) {
        out.mean = 0.0f;
        out.m2 = 0.0f;
        out.count = 0.0f;
        return out;
    }
    float delta = b.mean - a.mean;
    float new_mean = a.mean + delta * (b.count / count);
    float new_m2 = a.m2 + b.m2 + delta * delta * a.count * b.count / count;
    out.mean = new_mean;
    out.m2 = new_m2;
    out.count = count;
    return out;
}

// Update Welford state with a single new observation
__device__ __forceinline__ WelfordState welford_update(
    WelfordState state, float val
) {
    state.count += 1.0f;
    float delta = val - state.mean;
    state.mean += delta / state.count;
    float delta2 = val - state.mean;
    state.m2 += delta * delta2;
    return state;
}

// -----------------------------------------------------------------------
// Warp-level Welford reduction via __shfl_down_sync
// -----------------------------------------------------------------------

__device__ __forceinline__ WelfordState welford_warp_reduce(WelfordState state) {
    #pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
        WelfordState other;
        other.mean  = __shfl_down_sync(0xffffffff, state.mean,  offset);
        other.m2    = __shfl_down_sync(0xffffffff, state.m2,    offset);
        other.count = __shfl_down_sync(0xffffffff, state.count, offset);
        state = welford_merge(state, other);
    }
    return state;
}

// -----------------------------------------------------------------------
// Block-level Welford reduction via shared memory
// -----------------------------------------------------------------------

// Maximum warps per block (1024 threads / 32 = 32 warps max)
constexpr int MAX_WARPS = 32;

__device__ WelfordState welford_block_reduce(
    WelfordState state,
    float* __restrict__ smem_mean,
    float* __restrict__ smem_m2,
    float* __restrict__ smem_count,
    int warp_id,
    int lane_id,
    int num_warps
) {
    // First reduce within each warp
    state = welford_warp_reduce(state);

    // Lane 0 of each warp writes its partial result to shared memory
    if (lane_id == 0) {
        smem_mean[warp_id]  = state.mean;
        smem_m2[warp_id]    = state.m2;
        smem_count[warp_id] = state.count;
    }
    __syncthreads();

    // First warp reads all partial results and reduces them
    if (warp_id == 0) {
        if (lane_id < num_warps) {
            state.mean  = smem_mean[lane_id];
            state.m2    = smem_m2[lane_id];
            state.count = smem_count[lane_id];
        } else {
            state.mean  = 0.0f;
            state.m2    = 0.0f;
            state.count = 0.0f;
        }
        state = welford_warp_reduce(state);
    }

    return state;
}

// -----------------------------------------------------------------------
// Main LayerNorm kernel
//   - One block per row
//   - Vectorized float4 loads where possible
//   - Welford's algorithm for single-pass mean + variance
//   - Fused scale + bias epilogue
// -----------------------------------------------------------------------

__global__ void __launch_bounds__(1024)
layernorm_kernel(
    const half* __restrict__ X,
    const half* __restrict__ W,
    const half* __restrict__ B,
    half* __restrict__ Y,
    int N          // number of columns (hidden dim)
) {
    // Each block handles one row
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int blockSize = blockDim.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = blockSize / 32;

    const half* row_in = X + row * N;
    half* row_out = Y + row * N;

    // ---- Phase 1: Welford accumulation over the row ----

    WelfordState local_state;
    local_state.mean  = 0.0f;
    local_state.m2    = 0.0f;
    local_state.count = 0.0f;

    // Vectorized float4 path: process 8 halfs (= 4 half2 = 1 float4) at a time
    // float4 is 16 bytes = 8 half elements
    const int vec_size = 8;  // halfs per float4 load
    int n_vec = N / vec_size;
    int n_tail = N - n_vec * vec_size;

    // Cast input row pointer to float4* for vectorized access
    const float4* row_in_vec = reinterpret_cast<const float4*>(row_in);

    for (int i = tid; i < n_vec; i += blockSize) {
        float4 v = row_in_vec[i];
        // Reinterpret as 8 half values
        const half* hp = reinterpret_cast<const half*>(&v);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            local_state = welford_update(local_state, __half2float(hp[j]));
        }
    }

    // Handle tail elements (non-vectorized)
    int tail_start = n_vec * vec_size;
    for (int i = tail_start + tid; i < N; i += blockSize) {
        local_state = welford_update(local_state, __half2float(row_in[i]));
    }

    // ---- Phase 2: Block-level reduction ----

    __shared__ float smem_mean[MAX_WARPS];
    __shared__ float smem_m2[MAX_WARPS];
    __shared__ float smem_count[MAX_WARPS];

    local_state = welford_block_reduce(
        local_state,
        smem_mean, smem_m2, smem_count,
        warp_id, lane_id, num_warps
    );

    // Broadcast final mean and inv_std from lane 0 of warp 0
    __shared__ float s_mean;
    __shared__ float s_inv_std;

    if (tid == 0) {
        s_mean = local_state.mean;
        float variance = local_state.m2 / (float)N;
        constexpr float eps = 1e-5f;
        s_inv_std = rsqrtf(variance + eps);
    }
    __syncthreads();

    float row_mean = s_mean;
    float row_inv_std = s_inv_std;

    // ---- Phase 3: Normalize with fused scale + bias (vectorized writes) ----

    // Vectorized path for weight and bias
    const float4* W_vec = reinterpret_cast<const float4*>(W);
    const float4* B_vec = reinterpret_cast<const float4*>(B);
    float4* row_out_vec = reinterpret_cast<float4*>(row_out);

    for (int i = tid; i < n_vec; i += blockSize) {
        float4 x_vec = row_in_vec[i];
        float4 w_vec = W_vec[i];
        float4 b_vec = B_vec[i];

        const half* xh = reinterpret_cast<const half*>(&x_vec);
        const half* wh = reinterpret_cast<const half*>(&w_vec);
        const half* bh = reinterpret_cast<const half*>(&b_vec);

        float4 out_vec;
        half* oh = reinterpret_cast<half*>(&out_vec);

        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float x_val = __half2float(xh[j]);
            float w_val = __half2float(wh[j]);
            float b_val = __half2float(bh[j]);
            float norm_val = (x_val - row_mean) * row_inv_std;
            oh[j] = __float2half(norm_val * w_val + b_val);
        }

        row_out_vec[i] = out_vec;
    }

    // Handle tail elements
    for (int i = tail_start + tid; i < N; i += blockSize) {
        float x_val = __half2float(row_in[i]);
        float w_val = __half2float(W[i]);
        float b_val = __half2float(B[i]);
        float norm_val = (x_val - row_mean) * row_inv_std;
        row_out[i] = __float2half(norm_val * w_val + b_val);
    }
}

// -----------------------------------------------------------------------
// C++ launcher
// -----------------------------------------------------------------------

torch::Tensor layernorm_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat16, "x must be float16");
    TORCH_CHECK(weight.dtype() == torch::kFloat16, "weight must be float16");
    TORCH_CHECK(bias.dtype() == torch::kFloat16, "bias must be float16");

    TORCH_CHECK(x.dim() == 2, "x must be 2D [batch, dim]");
    int batch = x.size(0);
    int dim = x.size(1);

    TORCH_CHECK(weight.size(0) == dim, "weight dim must match x dim");
    TORCH_CHECK(bias.size(0) == dim, "bias dim must match x dim");

    auto y = torch::empty_like(x);

    // Choose thread count: use enough threads to saturate vectorized loads
    // Each thread handles ceil(dim / blockSize) elements
    int threads = 256;
    if (dim >= 2048) threads = 512;
    if (dim >= 8192) threads = 1024;

    dim3 grid(batch);
    dim3 block(threads);

    layernorm_kernel<<<grid, block>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(bias.data_ptr<at::Half>()),
        reinterpret_cast<half*>(y.data_ptr<at::Half>()),
        dim
    );

    return y;
}
"""

_module = None


def _get_module():
    global _module
    if _module is None:
        _module = compile_cuda(CUDA_SRC, "layernorm_cuda")
    return _module


def kernel_fn(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Entry point called by bench.py. Must match reference.layernorm_ref signature."""
    assert x.is_cuda

    # Flatten to 2D for row-parallel processing
    orig_shape = x.shape
    if x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim > 2:
        x = x.view(-1, x.shape[-1])

    n_rows, n_cols = x.shape
    assert weight.shape[0] == n_cols
    assert bias.shape[0] == n_cols

    # Handle non-fp16 inputs by casting to fp16 for the CUDA kernel
    orig_dtype = x.dtype
    if x.dtype != torch.float16:
        x = x.to(torch.float16)
    if weight.dtype != torch.float16:
        weight = weight.to(torch.float16)
    if bias.dtype != torch.float16:
        bias = bias.to(torch.float16)

    mod = _get_module()
    y = mod.layernorm_cuda(x, weight, bias)

    # Cast back to original dtype if needed
    if orig_dtype != torch.float16:
        y = y.to(orig_dtype)

    return y.view(orig_shape)
