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
AutoKernel starter -- RMS Normalization
Basic Triton kernel. The agent improves this.
"""

KERNEL_TYPE = "rmsnorm"

import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    X_ptr, W_ptr, OUT_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_om, stride_on,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Row-parallel RMS normalization."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Load row into float32 for numerical stability
    x = tl.load(X_ptr + row * stride_xm + offs * stride_xn, mask=mask, other=0.0).to(tl.float32)

    # Compute RMS
    sq_mean = tl.sum(x * x, axis=0) / N
    rms = tl.sqrt(sq_mean + eps)

    # Normalize
    x_norm = x / rms

    # Scale by weight
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = x_norm * w

    # Store (cast back to input dtype via the output tensor's dtype)
    tl.store(OUT_ptr + row * stride_om + offs * stride_on, out, mask=mask)


def kernel_fn(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Entry point called by bench.py. Must match reference.rmsnorm_ref signature."""
    assert x.is_cuda
    M, N = x.shape
    out = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(N)

    rmsnorm_kernel[(M,)](
        x, weight, out,
        M, N,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
