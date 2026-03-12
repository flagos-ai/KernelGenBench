from sandbox.register import register
from flagbench.dataset import Autograd
import torch
import triton
import triton.language as tl

@triton.jit
def zgeru_kernel(
    x_ptr, y_ptr, A_ptr,
    m, n,
    stride_x, stride_y,
    stride_Am, stride_An,
    alpha_real, alpha_imag,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create masks for boundary checks
    mask_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) < m
    mask_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) < n
    
    # Load x and y
    x_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    y_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    x_real = x_ptr + x_offsets * stride_x
    x_imag = x_ptr + x_offsets * stride_x + 1
    y_real = y_ptr + y_offsets * stride_y
    y_imag = y_ptr + y_offsets * stride_y + 1
    
    x_val_real = tl.load(x_real, mask=mask_m, other=0.0)
    x_val_imag = tl.load(x_imag, mask=mask_m, other=0.0)
    y_val_real = tl.load(y_real, mask=mask_n, other=0.0)
    y_val_imag = tl.load(y_imag, mask=mask_n, other=0.0)
    
    # Compute outer product with alpha
    prod_real = (x_val_real * y_val_real - x_val_imag * y_val_imag) * alpha_real - \
                (x_val_real * y_val_imag + x_val_imag * y_val_real) * alpha_imag
    prod_imag = (x_val_real * y_val_imag + x_val_imag * y_val_real) * alpha_real + \
                (x_val_real * y_val_real - x_val_imag * y_val_imag) * alpha_imag
    
    # Store results
    A_offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    A_offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    A_ptr_real = A_ptr + A_offsets_m[:, None] * stride_Am + A_offsets_n[None, :] * stride_An
    A_ptr_imag = A_ptr + A_offsets_m[:, None] * stride_Am + A_offsets_n[None, :] * stride_An + 1
    
    # Atomic add to handle potential overlaps
    tl.atomic_add(A_ptr_real, prod_real, mask=mask_m[:, None] & mask_n[None, :])
    tl.atomic_add(A_ptr_imag, prod_imag, mask=mask_m[:, None] & mask_n[None, :])

@register("CUDA", "zgeru", has_backward=Autograd.disable, namespace="triton")
def zgeru(
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
    
    # Compute grid and block sizes
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    
    grid = lambda META: (
        triton.cdiv(m, META['BLOCK_SIZE_M']),
        triton.cdiv(n, META['BLOCK_SIZE_N']),
    )
    
    # Launch kernel
    zgeru_kernel[grid](
        x, y, A,
        m, n,
        incx, incy,
        lda * 2, 2,  # Strides account for complex numbers (2 elements per complex)
        alpha_real, alpha_imag,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return A