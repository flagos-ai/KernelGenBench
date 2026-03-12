from sandbox.register import register
from flagbench.dataset import Autograd
import torch
import triton
import triton.language as tl

@triton.jit
def cgeru_kernel(
    x_ptr, y_ptr, A_ptr,
    m, n,
    stride_x, stride_y,
    stride_A_row, stride_A_col,
    alpha_real, alpha_imag,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create masks for boundary checks
    mask_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) < m
    mask_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) < n
    
    # Load x and y vectors
    x_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    y_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    x_real = tl.load(x_ptr + 2 * x_offsets, mask=mask_m, other=0.0)
    x_imag = tl.load(x_ptr + 2 * x_offsets + 1, mask=mask_m, other=0.0)
    y_real = tl.load(y_ptr + 2 * y_offsets, mask=mask_n, other=0.0)
    y_imag = tl.load(y_ptr + 2 * y_offsets + 1, mask=mask_n, other=0.0)
    
    # Compute outer product with alpha
    # A += alpha * x * y^T
    # Complex multiplication: (x_real + i*x_imag) * (y_real - i*y_imag)
    # But since it's cgeru (not cgerc), we don't conjugate y
    # So: (x_real + i*x_imag) * (y_real + i*y_imag)
    real_part = alpha_real * (x_real[:, None] * y_real[None, :]) - alpha_imag * (x_imag[:, None] * y_real[None, :]) + \
                alpha_real * (x_imag[:, None] * y_imag[None, :]) + alpha_imag * (x_real[:, None] * y_imag[None, :])
    imag_part = alpha_real * (x_imag[:, None] * y_real[None, :]) + alpha_imag * (x_real[:, None] * y_real[None, :]) + \
                alpha_real * (x_real[:, None] * y_imag[None, :]) - alpha_imag * (x_imag[:, None] * y_imag[None, :])
    
    # Compute A offsets
    A_offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]
    A_offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]
    A_offsets = 2 * (A_offsets_m * stride_A_row + A_offsets_n * stride_A_col)
    
    # Atomic add to A
    tl.atomic_add(A_ptr + A_offsets, real_part, mask=mask_m[:, None] & mask_n[None, :])
    tl.atomic_add(A_ptr + A_offsets + 1, imag_part, mask=mask_m[:, None] & mask_n[None, :])

@register("CUDA", "cgeru", has_backward=Autograd.disable, namespace="triton")
def cgeru(
    m: int,
    n: int,
    alpha: complex,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    A: torch.Tensor,
    lda: int,
):
    # Check dimensions and strides
    assert x.is_cuda and y.is_cuda and A.is_cuda
    assert x.dtype == torch.complex64 and y.dtype == torch.complex64 and A.dtype == torch.complex64
    
    # Extract real and imaginary parts of alpha
    alpha_real = alpha.real
    alpha_imag = alpha.imag
    
    # Compute grid and block sizes
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    
    grid = lambda META: (
        triton.cdiv(m, META['BLOCK_SIZE_M']),
        triton.cdiv(n, META['BLOCK_SIZE_N']),
    )
    
    # Launch kernel
    cgeru_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        A_ptr=A,
        m=m,
        n=n,
        stride_x=incx,
        stride_y=incy,
        stride_A_row=lda,
        stride_A_col=1,
        alpha_real=alpha_real,
        alpha_imag=alpha_imag,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return A