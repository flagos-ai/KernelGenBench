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
AutoKernel -- Rotary Position Embedding (RoPE) kernel.

Current kernel: RoPE application
Target metric: throughput (higher is better)
Secondary: correctness must ALWAYS pass

Applies precomputed cos/sin rotary embeddings to input tensor.
The rotation is applied to pairs of elements along the last dimension
using interleaved (even/odd) decomposition:
  x1 = x[..., 0], x[..., 2], x[..., 4], ...   (even indices)
  x2 = x[..., 1], x[..., 3], x[..., 5], ...   (odd indices)
  x_rot[..., 2i]   = x1[i] * cos[i] - x2[i] * sin[i]
  x_rot[..., 2i+1] = x1[i] * sin[i] + x2[i] * cos[i]
"""

KERNEL_TYPE = "rotary_embedding"

import torch
import triton
import triton.language as tl


@triton.jit
def rotary_embedding_kernel(
    X_ptr,
    COS_ptr,
    SIN_ptr,
    OUT_ptr,
    seq_len,
    head_dim,
    stride_x_row,
    stride_cos_row,
    stride_sin_row,
    stride_out_row,
    half_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Apply rotary embeddings using interleaved (even/odd) decomposition.

    One program per row. Each row has head_dim elements.
    x1 = even-indexed elements (0, 2, 4, ...)
    x2 = odd-indexed elements  (1, 3, 5, ...)

    out[2i]   = x1[i] * cos[i] - x2[i] * sin[i]
    out[2i+1] = x1[i] * sin[i] + x2[i] * cos[i]
    """
    row_idx = tl.program_id(0)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask_half = col_offsets < half_dim

    # Compute pointers to even-indexed (x1) and odd-indexed (x2) elements
    # even indices: 0, 2, 4, ... => col_offsets * 2
    # odd indices:  1, 3, 5, ... => col_offsets * 2 + 1
    x_row_base = X_ptr + row_idx * stride_x_row
    x1 = tl.load(x_row_base + col_offsets * 2, mask=mask_half, other=0.0).to(tl.float32)
    x2 = tl.load(x_row_base + col_offsets * 2 + 1, mask=mask_half, other=0.0).to(tl.float32)

    # Load cos and sin (shape [n_rows, half_dim])
    cos = tl.load(COS_ptr + row_idx * stride_cos_row + col_offsets, mask=mask_half, other=1.0).to(tl.float32)
    sin = tl.load(SIN_ptr + row_idx * stride_sin_row + col_offsets, mask=mask_half, other=0.0).to(tl.float32)

    # Apply rotation
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos

    # Store results interleaved: out[2i] = rx1[i], out[2i+1] = rx2[i]
    out_row_base = OUT_ptr + row_idx * stride_out_row
    tl.store(out_row_base + col_offsets * 2, rx1, mask=mask_half)
    tl.store(out_row_base + col_offsets * 2 + 1, rx2, mask=mask_half)


def kernel_fn(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Entry point called by bench.py. Must match reference.rotary_embedding_ref signature.

    Args:
        x: [..., head_dim] tensor to apply rotary embeddings to
        cos: [..., head_dim // 2] precomputed cosines
        sin: [..., head_dim // 2] precomputed sines

    Returns:
        Tensor of same shape as x with rotary embeddings applied.
    """
    assert x.is_cuda

    orig_shape = x.shape
    head_dim = x.shape[-1]
    half_dim = head_dim // 2

    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    assert cos.shape[-1] == half_dim
    assert sin.shape[-1] == half_dim

    # Flatten to 2D: [n_rows, head_dim]
    x_flat = x.contiguous().view(-1, head_dim)
    n_rows = x_flat.shape[0]

    # Flatten cos/sin and broadcast to match x rows
    cos_flat = cos.contiguous().view(-1, half_dim)
    sin_flat = sin.contiguous().view(-1, half_dim)

    # Handle broadcasting: if cos/sin have fewer rows, expand to match
    if cos_flat.shape[0] < n_rows:
        repeat_factor = (n_rows + cos_flat.shape[0] - 1) // cos_flat.shape[0]
        cos_flat = cos_flat.repeat(repeat_factor, 1)[:n_rows]
        sin_flat = sin_flat.repeat(repeat_factor, 1)[:n_rows]

    out = torch.empty_like(x_flat)

    BLOCK_SIZE = triton.next_power_of_2(half_dim)

    grid = (n_rows,)
    rotary_embedding_kernel[grid](
        x_flat,
        cos_flat,
        sin_flat,
        out,
        n_rows,
        head_dim,
        x_flat.stride(0),
        cos_flat.stride(0),
        sin_flat.stride(0),
        out.stride(0),
        half_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out.view(orig_shape)
