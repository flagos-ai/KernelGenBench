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
AutoKernel -- Parallel Reduction (sum) kernel.

Current kernel: Row-parallel sum reduction
Target metric: throughput (higher is better)
Secondary: correctness must ALWAYS pass

Performs sum reduction along a specified dimension.
Each program handles one output element (one "row" after flattening
along the reduction dimension).
"""

KERNEL_TYPE = "reduce"

import torch
import triton
import triton.language as tl


@triton.jit
def reduce_sum_kernel(
    X_ptr,
    OUT_ptr,
    reduce_size,
    stride_x_row,
    stride_x_col,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Parallel sum reduction. One program per output element.
    Reduces over `reduce_size` elements with stride `stride_x_col`.
    """
    row_idx = tl.program_id(0)

    # Base pointer for this row
    row_start = X_ptr + row_idx * stride_x_row

    # Accumulate in float32 for stability
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for offset in range(0, reduce_size, BLOCK_SIZE):
        col_offsets = offset + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < reduce_size
        x = tl.load(row_start + col_offsets * stride_x_col, mask=mask, other=0.0).to(tl.float32)
        acc += x

    result = tl.sum(acc, axis=0)
    tl.store(OUT_ptr + row_idx, result)


def kernel_fn(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Entry point called by bench.py. Must match reference.reduce_sum_ref signature.

    Args:
        x: Input tensor of any shape
        dim: Dimension to reduce over (default: -1, last dim)

    Returns:
        Tensor with the specified dimension reduced (summed).
    """
    assert x.is_cuda

    # Normalize dim
    if dim < 0:
        dim = x.ndim + dim
    assert 0 <= dim < x.ndim, f"dim {dim} out of range for tensor with {x.ndim} dims"

    # Compute shapes
    # We want to reshape to [outer, reduce_size, inner] then reduce the middle dim
    outer_size = 1
    for i in range(dim):
        outer_size *= x.size(i)

    reduce_size = x.size(dim)

    inner_size = 1
    for i in range(dim + 1, x.ndim):
        inner_size *= x.size(i)

    # Make contiguous and reshape
    x_contig = x.contiguous()

    # Output shape: same as input but with dim removed
    out_shape = list(x.shape)
    out_shape.pop(dim)
    if len(out_shape) == 0:
        out_shape = [1]

    # Total number of output elements
    n_output = outer_size * inner_size

    # For the simple case where we reduce over the last dimension
    # and inner_size == 1, we can use a straightforward approach
    if inner_size == 1:
        x_2d = x_contig.view(outer_size, reduce_size)
        out_flat = torch.empty(n_output, device=x.device, dtype=torch.float32)

        BLOCK_SIZE = triton.next_power_of_2(min(reduce_size, 8192))

        grid = (n_output,)
        reduce_sum_kernel[grid](
            x_2d,
            out_flat,
            reduce_size,
            x_2d.stride(0),
            x_2d.stride(1),
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return out_flat.to(x.dtype).view(out_shape)
    else:
        # General case: transpose so reduce dim is last, then reduce
        # Move reduce dim to last position
        perm = list(range(x.ndim))
        perm.pop(dim)
        perm.append(dim)
        x_transposed = x.permute(*perm).contiguous()

        # Now reduce over last dim
        total_rows = outer_size * inner_size
        x_2d = x_transposed.view(total_rows, reduce_size)
        out_flat = torch.empty(total_rows, device=x.device, dtype=torch.float32)

        BLOCK_SIZE = triton.next_power_of_2(min(reduce_size, 8192))

        grid = (total_rows,)
        reduce_sum_kernel[grid](
            x_2d,
            out_flat,
            reduce_size,
            x_2d.stride(0),
            x_2d.stride(1),
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # After permutation, the output follows permuted dim order (without the
        # reduce dim). Build the correct shape from the permuted order.
        permuted_out_shape = [x.shape[d] for d in perm[:-1]]
        return out_flat.to(x.dtype).view(permuted_out_shape)
