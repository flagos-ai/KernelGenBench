#!/usr/bin/env python3
"""
Triton kernel for copy_upp_low operation
Copies upper triangle to lower triangle of a square matrix (in-place)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def copy_upp_low_kernel(
    A_ptr,  # Pointer to matrix A
    N,  # Matrix dimension (N x N)
    stride_row,  # Stride along rows
    stride_col,  # Stride along columns
    BLOCK_SIZE: tl.constexpr,
):
    """
    Copy upper triangle to lower triangle: For j > i: A[j,i] = A[i,j]
    
    Each thread block processes BLOCK_SIZE elements in the lower triangle.
    """
    # Get the global thread ID
    pid = tl.program_id(0)
    
    # Calculate which element this thread is processing
    elem_id = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Total elements in lower triangle (excluding diagonal)
    total_elems = (N * (N - 1)) // 2
    
    # Mask for valid elements
    mask = elem_id < total_elems
    
    # Convert elem_id to (row, col) coordinates for lower triangle
    # For lower triangle (row > col), we enumerate: (1,0), (2,0), (2,1), (3,0), ..., (N-1,N-2)
    
    # Use float for sqrt calculation
    elem_id_float = elem_id.to(tl.float32)
    discriminant = (2 * N - 1) * (2 * N - 1) - 8 * elem_id_float
    row = ((2 * N - 1) - tl.sqrt(discriminant)) / 2
    row = row.to(tl.int32)
    
    # Calculate column from row and elem_id
    elements_before_row = row * N - row * (row + 1) // 2
    col_offset = elem_id - elements_before_row
    col = row + 1 + col_offset
    
    # For lower triangle, we want row > col, so swap them
    tmp = row
    row = col
    col = tmp
    
    # Clamp row and col to valid range
    row = tl.where(mask, row, 0)
    col = tl.where(mask, col, 0)
    
    # Ensure row > col (lower triangle)
    valid_elem = mask & (row > col) & (row >= 0) & (row < N) & (col >= 0) & (col < N)
    
    # Calculate pointer offsets
    # For lower triangle element at (row, col): read from (col, row) in upper triangle
    upper_offset = col * stride_row + row * stride_col  # A[col, row]
    lower_offset = row * stride_row + col * stride_col  # A[row, col]
    
    # Load from upper triangle
    upper_val = tl.load(A_ptr + upper_offset, mask=valid_elem, other=0.0)
    
    # Store to lower triangle
    tl.store(A_ptr + lower_offset, upper_val, mask=valid_elem)


def copy_upp_low(A: torch.Tensor) -> None:
    """
    Copy upper triangle to lower triangle of a square matrix (in-place).
    
    Args:
        A: Square matrix (N x N), must be on CUDA device, modified in-place
    """
    assert A.is_cuda, "Tensor must be on CUDA device"
    assert A.ndim == 2, "Tensor must be 2D"
    assert A.size(0) == A.size(1), "Tensor must be square"
    assert A.is_contiguous(), "Tensor must be contiguous"
    
    N = A.size(0)
    
    # Number of elements in lower triangle (excluding diagonal)
    total_elems = (N * (N - 1)) // 2
    
    if total_elems == 0:
        return  # Nothing to copy for 1x1 matrix
    
    # Launch configuration
    BLOCK_SIZE = 256
    grid = (triton.cdiv(total_elems, BLOCK_SIZE),)
    
    # Launch kernel
    copy_upp_low_kernel[grid](
        A,
        N,
        A.stride(0),
        A.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )


if __name__ == "__main__":
    print("Testing Triton copy_upp_low kernel")
    print("=" * 60)
    
    # Test 1: 4x4 matrix
    A = torch.tensor([
        [1.0, 2.0, 4.0, 7.0],
        [0.0, 3.0, 5.0, 8.0],
        [0.0, 0.0, 6.0, 9.0],
        [0.0, 0.0, 0.0, 10.0]
    ], device='cuda', dtype=torch.float32)
    
    print("Before:")
    print(A)
    
    copy_upp_low(A)
    
    print("\nAfter:")
    print(A)
    
    expected = torch.tensor([
        [1.0, 2.0, 4.0, 7.0],
        [2.0, 3.0, 5.0, 8.0],
        [4.0, 5.0, 6.0, 9.0],
        [7.0, 8.0, 9.0, 10.0]
    ], device='cuda', dtype=torch.float32)
    
    if torch.allclose(A, expected):
        print("\n✓ Test PASSED!")
    else:
        print("\n✗ Test FAILED!")
        print("Expected:")
        print(expected)
