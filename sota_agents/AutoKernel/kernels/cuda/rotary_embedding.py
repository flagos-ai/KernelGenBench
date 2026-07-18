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
AutoKernel -- CUDA C++ Rotary Embedding (RoPE) kernel.

Current kernel: Interleaved sin/cos with precomputed frequency table.
Target metric: throughput (higher is better)
Secondary: correctness must ALWAYS pass

Features:
  - __sincosf intrinsic for fast sin/cos computation
  - Vectorized half2 read-modify-write
  - Frequency table computed on-the-fly (no extra memory)
  - Grid-stride loop for arbitrary tensor sizes
"""

KERNEL_TYPE = "rotary_embedding"
BACKEND = "cuda"

import torch
from kernels.cuda._compile import compile_cuda

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

constexpr int BLOCK_SIZE = 256;
constexpr float BASE_FREQ = 10000.0f;

// Apply rotary embedding to x in-place
// x shape: [B, H, N, D] where D is head_dim (must be even)
// For position pos and dimension pair (2i, 2i+1):
//   freq = 1 / (base ^ (2i / D))
//   theta = pos * freq
//   x_rot[2i]   = x[2i] * cos(theta) - x[2i+1] * sin(theta)
//   x_rot[2i+1] = x[2i] * sin(theta) + x[2i+1] * cos(theta)

__global__ void rotary_embedding_kernel(
    const half* __restrict__ x,
    half* __restrict__ output,
    const half* __restrict__ cos_cache,  // [N, D/2]
    const half* __restrict__ sin_cache,  // [N, D/2]
    int B, int H, int N, int D
) {
    const int total = B * H * N * (D / 2);
    const int half_D = D / 2;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += gridDim.x * blockDim.x) {
        // Decode flat index -> (b, h, n, d_pair)
        int d_pair = idx % half_D;
        int remainder = idx / half_D;
        int n = remainder % N;
        remainder = remainder / N;
        int h = remainder % H;
        int b = remainder / H;

        // Read the pair (x0, x1)
        int base_idx = ((b * H + h) * N + n) * D + d_pair * 2;
        float x0 = __half2float(x[base_idx]);
        float x1 = __half2float(x[base_idx + 1]);

        // Read cos/sin from cache
        int cache_idx = n * half_D + d_pair;
        float cos_val = __half2float(cos_cache[cache_idx]);
        float sin_val = __half2float(sin_cache[cache_idx]);

        // Apply rotation
        float out0 = x0 * cos_val - x1 * sin_val;
        float out1 = x0 * sin_val + x1 * cos_val;

        output[base_idx]     = __float2half(out0);
        output[base_idx + 1] = __float2half(out1);
    }
}

// Precompute cos/sin cache for positions [0, N) and dims [0, D/2)
__global__ void precompute_freqs_kernel(
    half* __restrict__ cos_cache,  // [N, D/2]
    half* __restrict__ sin_cache,  // [N, D/2]
    int N, int D
) {
    const int half_D = D / 2;
    const int total = N * half_D;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += gridDim.x * blockDim.x) {
        int d = idx % half_D;
        int pos = idx / half_D;

        float freq = 1.0f / powf(BASE_FREQ, (2.0f * d) / (float)D);
        float theta = pos * freq;

        float cos_val, sin_val;
        __sincosf(theta, &sin_val, &cos_val);

        cos_cache[idx] = __float2half(cos_val);
        sin_cache[idx] = __float2half(sin_val);
    }
}

torch::Tensor rotary_embedding_cuda(torch::Tensor x, torch::Tensor cos_cache, torch::Tensor sin_cache) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.dim() == 4, "x must be [B, H, N, D]");

    int B = x.size(0);
    int H = x.size(1);
    int N = x.size(2);
    int D = x.size(3);
    TORCH_CHECK(D % 2 == 0, "D must be even");

    auto output = torch::empty_like(x);

    int total = B * H * N * (D / 2);
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    blocks = min(blocks, 65535);

    rotary_embedding_kernel<<<blocks, BLOCK_SIZE>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(cos_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(sin_cache.data_ptr<at::Half>()),
        B, H, N, D
    );

    return output;
}

std::vector<torch::Tensor> precompute_freqs_cuda(int N, int D, torch::Device device) {
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(device);
    auto cos_cache = torch::empty({N, D / 2}, options);
    auto sin_cache = torch::empty({N, D / 2}, options);

    int total = N * (D / 2);
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    blocks = min(blocks, 65535);

    precompute_freqs_kernel<<<blocks, BLOCK_SIZE>>>(
        reinterpret_cast<half*>(cos_cache.data_ptr<at::Half>()),
        reinterpret_cast<half*>(sin_cache.data_ptr<at::Half>()),
        N, D
    );

    return {cos_cache, sin_cache};
}
"""

_module = None
_freq_cache = {}  # (N, D) -> (cos_cache, sin_cache)


def _get_module():
    global _module
    if _module is None:
        _module = compile_cuda(CUDA_SRC, "rotary_embedding_cuda")
    return _module


def _get_freqs(N: int, D: int, device: torch.device):
    """Get or compute cached cos/sin frequency tables."""
    key = (N, D, str(device))
    if key not in _freq_cache:
        # Compute on GPU using our kernel
        half_D = D // 2
        positions = torch.arange(N, device=device, dtype=torch.float32)
        dim_indices = torch.arange(half_D, device=device, dtype=torch.float32)
        freqs = 1.0 / (10000.0 ** (2.0 * dim_indices / D))
        theta = positions.unsqueeze(1) * freqs.unsqueeze(0)  # [N, D/2]
        cos_cache = torch.cos(theta).to(torch.float16)
        sin_cache = torch.sin(theta).to(torch.float16)
        _freq_cache[key] = (cos_cache, sin_cache)
    return _freq_cache[key]


def kernel_fn(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Entry point called by bench.py. Must match reference.rotary_embedding_ref signature."""
    assert x.is_cuda

    orig_dtype = x.dtype
    if x.dtype != torch.float16:
        x = x.to(torch.float16)
        cos = cos.to(torch.float16)
        sin = sin.to(torch.float16)

    mod = _get_module()
    out = mod.rotary_embedding_cuda(x, cos, sin)

    if orig_dtype != torch.float16:
        out = out.to(orig_dtype)

    return out
