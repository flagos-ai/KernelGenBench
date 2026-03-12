from sandbox.register import register
from flagbench.dataset import Autograd
import torch
import triton
import triton.language as tl

@triton.jit
def ssbmv_kernel(
    A_ptr, x_ptr, y_ptr,
    n, k,
    alpha, beta,
    lda, incx, incy,
    uplo: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    
    # Compute the range of columns to process
    col_start = tl.max(0, row - k)
    col_end = tl.min(n, row + k + 1)
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over the relevant band of the matrix
    for col in range(col_start, col_end):
        # Compute the actual index in the packed band storage
        if uplo == 0:  # lower triangular
            band_idx = row - col
            if band_idx >= 0 and band_idx <= k:
                a_idx = row * lda + band_idx
        else:  # upper triangular
            band_idx = col - row
            if band_idx >= 0 and band_idx <= k:
                a_idx = row * lda + band_idx
        
        # Load elements
        a = tl.load(A_ptr + a_idx)
        x_val = tl.load(x_ptr + col * incx)
        acc += a * x_val
    
    # Load current y value
    y_val = tl.load(y_ptr + row * incy)
    
    # Compute new y value
    new_y = alpha * acc + beta * y_val
    
    # Store result
    tl.store(y_ptr + row * incy, new_y)

@register("CUDA", "ssbmv", has_backward=Autograd.disable, namespace="triton")
def ssbmv(
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
    assert A.is_cuda and x.is_cuda and y.is_cuda
    assert A.dtype == torch.float32 and x.dtype == torch.float32 and y.dtype == torch.float32
    
    # Launch kernel
    grid = lambda META: (n,)
    ssbmv_kernel[grid](
        A, x, y,
        n, k,
        alpha, beta,
        lda, incx, incy,
        uplo,
        BLOCK_SIZE=128,
    )
    return y