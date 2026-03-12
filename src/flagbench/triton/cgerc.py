from sandbox.register import register
from flagbench.dataset import Autograd
import torch
import triton
import triton.language as tl

@triton.jit
def cgerc_kernel(
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
    
    # Load x and y with boundary checks
    x_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    x_real = tl.load(x_ptr + 2 * x_offsets, mask=mask_m, other=0.0)
    x_imag = tl.load(x_ptr + 2 * x_offsets + 1, mask=mask_m, other=0.0)
    
    y_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    y_real = tl.load(y_ptr + 2 * y_offsets, mask=mask_n, other=0.0)
    y_imag = tl.load(y_ptr + 2 * y_offsets + 1, mask=mask_n, other=0.0)
    
    # Compute outer product with complex multiplication
    # A += alpha * x * y^H
    # y^H is conjugate of y
    y_conj_real = y_real
    y_conj_imag = -y_imag
    
    # Complex multiply: (x_real + i*x_imag) * (y_real - i*y_imag)
    # = (x_real*y_real + x_imag*y_imag) + i*(x_imag*y_real - x_real*y_imag)
    real_part = x_real[:, None] * y_conj_real[None, :] - x_imag[:, None] * y_conj_imag[None, :]
    imag_part = x_real[:, None] * y_conj_imag[None, :] + x_imag[:, None] * y_conj_real[None, :]
    
    # Scale by alpha
    real_part = alpha_real * real_part - alpha_imag * imag_part
    imag_part = alpha_real * imag_part + alpha_imag * real_part
    
    # Compute A offsets
    A_offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]
    A_offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]
    A_offsets = 2 * (A_offsets_m * stride_A_row + A_offsets_n * stride_A_col)
    
    # Atomic add to A
    tl.atomic_add(A_ptr + A_offsets, real_part, mask=mask_m[:, None] & mask_n[None, :])
    tl.atomic_add(A_ptr + A_offsets + 1, imag_part, mask=mask_m[:, None] & mask_n[None, :])

@register("CUDA", "cgerc", has_backward=Autograd.disable, namespace="triton")
def cgerc(
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
    # Extract real and imaginary parts of alpha
    alpha_real = alpha.real
    alpha_imag = alpha.imag
    
    # Adjust for strides
    if incx != 1:
        x = x.view(-1, incx)[:, 0]
    if incy != 1:
        y = y.view(-1, incy)[:, 0]
    
    # Define block sizes
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    
    # Compute grid
    grid = lambda META: (
        triton.cdiv(m, META['BLOCK_SIZE_M']),
        triton.cdiv(n, META['BLOCK_SIZE_N']),
    )
    
    # Launch kernel
    cgerc_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        A_ptr=A,
        m=m,
        n=n,
        stride_x=x.stride(0),
        stride_y=y.stride(0),
        stride_A_row=A.stride(0),
        stride_A_col=A.stride(1),
        alpha_real=alpha_real,
        alpha_imag=alpha_imag,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return A