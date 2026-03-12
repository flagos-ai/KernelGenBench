from sandbox.register import register
from flagbench.dataset import Autograd
import torch
import triton
import triton.language as tl

@triton.jit
def dger_kernel(
    x_ptr, y_ptr, A_ptr,
    m, n,
    stride_x, stride_y,
    stride_A_m, stride_A_n,
    alpha,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create masks for the last block
    mask_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) < m
    mask_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) < n
    
    # Load x and y
    x_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    y_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    x = tl.load(x_ptr + x_offsets * stride_x, mask=mask_m)
    y = tl.load(y_ptr + y_offsets * stride_y, mask=mask_n)
    
    # Compute outer product
    a = tl.reshape(x, (BLOCK_SIZE_M, 1)) * tl.reshape(y, (1, BLOCK_SIZE_N))
    a = a * alpha
    
    # Store result
    A_offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    A_offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    A_offsets = A_offsets_m[:, None] * stride_A_m + A_offsets_n[None, :] * stride_A_n
    
    current_A = tl.load(A_ptr + A_offsets, mask=mask_m[:, None] & mask_n[None, :])
    updated_A = current_A + a
    tl.store(A_ptr + A_offsets, updated_A, mask=mask_m[:, None] & mask_n[None, :])

@register("CUDA", "dger", has_backward=Autograd.disable, namespace="triton")
def dger(
    m: int,
    n: int,
    alpha: float,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    A: torch.Tensor,
    lda: int,
):
    # Check tensor dimensions and strides
    assert x.is_cuda and y.is_cuda and A.is_cuda
    assert x.dtype == torch.float64 and y.dtype == torch.float64 and A.dtype == torch.float64
    
    # Compute grid
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    
    grid = (
        triton.cdiv(m, BLOCK_SIZE_M),
        triton.cdiv(n, BLOCK_SIZE_N),
    )
    
    # Launch kernel
    dger_kernel[grid](
        x, y, A,
        m, n,
        incx, incy,
        lda, 1,
        alpha,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return A