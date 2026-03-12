from sandbox.register import register
from flagbench.dataset import Autograd
import torch
import triton
import triton.language as tl

@triton.jit
def dsbmv_kernel(
    A_ptr, x_ptr, y_ptr,
    n, beta,
    k, lda,
    stride_Ak, stride_An,
    stride_x, stride_y,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    
    # Compute the banded matrix-vector product
    y_val = 0.0
    for col in range(tl.max(0, row - k), tl.min(n, row + k + 1)):
        # Check if we're in the upper or lower band
        if row <= col + k and col <= row + k:
            # Compute the actual index in the packed storage
            if row >= col:  # lower triangular part
                a_row = row - col
                a_col = col
            else:  # upper triangular part
                a_row = k + col - row
                a_col = row
            
            # Load A value from packed storage
            a_offset = a_row + a_col * lda
            a_val = tl.load(A_ptr + a_offset)
            
            # Load x value
            x_val = tl.load(x_ptr + col * stride_x)
            
            y_val += a_val * x_val
    
    # Load current y value and apply beta
    current_y = tl.load(y_ptr + row * stride_y)
    y_val = alpha * y_val + beta * current_y
    
    # Store result
    tl.store(y_ptr + row * stride_y, y_val)

@register("CUDA", "dsbmv", has_backward=Autograd.disable, namespace="triton")
def dsbmv(
    uplo: int,
    n: int,
    k: int,
    alpha: float,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
    beta: float,
    y: torch.Tensor,
    incy: int,
):
    # Validate inputs
    assert A.dtype == torch.float64, "Only float64 is supported"
    assert x.dtype == torch.float64, "Only float64 is supported"
    assert y.dtype == torch.float64, "Only float64 is supported"
    assert A.is_cuda and x.is_cuda and y.is_cuda, "All tensors must be on CUDA"
    
    # Compute grid size
    grid = lambda META: (n,)
    
    # Launch kernel
    dsbmv_kernel[grid](
        A, x, y,
        alpha, beta,
        k, lda,
        A.stride(0), A.stride(1),
        incx, incy,
        BLOCK_SIZE=128,
    )
    
    return y