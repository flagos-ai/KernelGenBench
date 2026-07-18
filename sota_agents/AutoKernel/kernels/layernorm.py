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
AutoKernel -- Layer Normalization kernel.

Current kernel: LayerNorm (row-parallel)
Target metric: throughput (higher is better)
Secondary: correctness must ALWAYS pass

Each program instance handles one row of the input tensor.
Two-pass approach: compute mean, then variance, then normalize.
"""

KERNEL_TYPE = "layernorm"

import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_kernel(
    X_ptr,
    Y_ptr,
    W_ptr,
    B_ptr,
    stride_x_row,
    stride_y_row,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Row-parallel layer normalization. One program per row."""
    row_idx = tl.program_id(0)

    row_start_x = X_ptr + row_idx * stride_x_row
    row_start_y = Y_ptr + row_idx * stride_y_row

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    # Load row into float32 for numerical stability
    x = tl.load(row_start_x + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Pass 1: compute mean
    mean = tl.sum(x, axis=0) / N

    # Pass 2: compute variance
    x_centered = tl.where(mask, x - mean, 0.0)
    variance = tl.sum(x_centered * x_centered, axis=0) / N

    # Normalize
    inv_std = 1.0 / tl.sqrt(variance + eps)
    x_norm = x_centered * inv_std

    # Load weight and bias
    w = tl.load(W_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Apply affine transform
    y = x_norm * w + b

    # Store (cast back to input dtype via the store)
    tl.store(row_start_y + col_offsets, y, mask=mask)


def kernel_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
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

    y = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    grid = (n_rows,)
    layernorm_kernel[grid](
        x, y,
        weight, bias,
        x.stride(0),
        y.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return y.view(orig_shape)
