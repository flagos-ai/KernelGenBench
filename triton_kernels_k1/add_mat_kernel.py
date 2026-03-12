#!/usr/bin/env python3
"""
Triton kernel for add_mat operation
Matrix addition with scalar multiplier (in-place): dst = alpha * src + dst
"""

import torch
import triton
import triton.language as tl


@triton.jit
def add_mat_kernel(
    dst_ptr,  # Pointer to destination matrix
    src_ptr,  # Pointer to source matrix
    alpha,  # Scalar multiplier
    M,  # Number of rows
    N,  # Number of columns
    dst_stride_row,  # Destination stride along rows
    dst_stride_col,  # Destination stride along columns
    src_stride_row,  # Source stride along rows
    src_stride_col,  # Source stride along columns
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Elementwise operation: dst[i,j] = alpha * src[i,j] + dst[i,j]
    
    Each thread block processes BLOCK_SIZE_M x BLOCK_SIZE_N elements.
    """
    # Get the program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Calculate the starting row and column for this block
    row_start = pid_m * BLOCK_SIZE_M
    col_start = pid_n * BLOCK_SIZE_N
    
    # Create row and column offsets
    rows = row_start + tl.arange(0, BLOCK_SIZE_M)
    cols = col_start + tl.arange(0, BLOCK_SIZE_N)
    
    # Create mask for valid elements
    row_mask = rows < M
    col_mask = cols < N
    
    # Expand to 2D mask
    mask = row_mask[:, None] & col_mask[None, :]
    
    # Calculate pointer offsets for this block
    dst_offsets = rows[:, None] * dst_stride_row + cols[None, :] * dst_stride_col
    src_offsets = rows[:, None] * src_stride_row + cols[None, :] * src_stride_col
    
    # Load source and destination values
    src_vals = tl.load(src_ptr + src_offsets, mask=mask, other=0.0)
    dst_vals = tl.load(dst_ptr + dst_offsets, mask=mask, other=0.0)
    
    # Compute: dst = alpha * src + dst
    result = alpha * src_vals + dst_vals
    
    # Store result
    tl.store(dst_ptr + dst_offsets, result, mask=mask)


def add_mat(dst: torch.Tensor, src: torch.Tensor, alpha: float) -> None:
    """
    Matrix addition with scalar multiplier (in-place): dst = alpha * src + dst
    
    Args:
        dst: Destination matrix, must be on CUDA device, modified in-place
        src: Source matrix (same shape as dst), must be on CUDA device
        alpha: Scalar multiplier for source matrix
    """
    assert dst.is_cuda and src.is_cuda, "Tensors must be on CUDA device"
    assert dst.shape == src.shape, "Tensors must have same shape"
    assert dst.ndim == 2, "Tensors must be 2D"
    assert dst.is_contiguous() and src.is_contiguous(), "Tensors must be contiguous"
    
    M, N = dst.shape
    
    # Launch configuration
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    
    # Launch kernel
    add_mat_kernel[grid](
        dst, src, alpha,
        M, N,
        dst.stride(0), dst.stride(1),
        src.stride(0), src.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )


if __name__ == "__main__":
    print("Testing Triton add_mat kernel")
    print("=" * 60)
    
    # Test 1: Simple 2x3 matrix
    src = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ], device='cuda', dtype=torch.float32)
    
    dst = torch.tensor([
        [0.5, 1.0, 1.5],
        [2.0, 2.5, 3.0]
    ], device='cuda', dtype=torch.float32)
    
    print("Source:")
    print(src)
    print("\nDestination (before):")
    print(dst)
    
    alpha = 2.0
    print(f"\nalpha = {alpha}")
    
    add_mat(dst, src, alpha)
    
    print("\nDestination (after):")
    print(dst)
    
    expected = torch.tensor([
        [2.5, 5.0, 7.5],
        [10.0, 12.5, 15.0]
    ], device='cuda', dtype=torch.float32)
    
    if torch.allclose(dst, expected):
        print("\n✓ Test PASSED!")
    else:
        print("\n✗ Test FAILED!")
        print("Expected:")
        print(expected)
