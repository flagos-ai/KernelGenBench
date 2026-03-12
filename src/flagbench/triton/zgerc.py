from sandbox.register import register
from flagbench.dataset import Autograd
import torch
import triton
import triton.language as tl

@triton.jit
def zgerc_kernel(
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
    # y^H means conjugate of y
    y_conj_real = y_real
    y_conj_imag = -y_imag
    
    # Complex multiplication: (x_real + i*x_imag) * (y_conj_real + i*y_conj_imag)
    real_part = x_real * y_conj_real - x_imag * y_conj_imag
    imag_part = x_real * y_conj_imag + x_imag * y_conj_real
    
    # Scale by alpha
    scaled_real = alpha_real * real_part - alpha_imag * imag_part
    scaled_imag = alpha_real * imag_part + alpha_imag * real_part
    
    # Prepare A offsets
    A_offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    A_offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    A_offsets = A_offsets_m[:, None] * stride_A_row + A_offsets_n[None, :] * stride_A_col
    
    # Atomic add to A
    tl.atomic_add(A_ptr + 2 * A_offsets, scaled_real, mask=mask_m[:, None] & mask_n[None, :])
    tl.atomic_add(A_ptr + 2 * A_offsets + 1, scaled_imag, mask=mask_m[:, None] & mask_n[None, :])

@register("CUDA", "zgerc", has_backward=Autograd.disable, namespace="triton")
def zgerc(
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
    
    # Compute grid sizes
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    
    grid = (
        triton.cdiv(m, BLOCK_SIZE_M),
        triton.cdiv(n, BLOCK_SIZE_N),
    )
    
    # Launch kernel
    zgerc_kernel[grid](
        x, y, A,
        m, n,
        incx, incy,
        lda, 1,  # A is column-major (lda is leading dimension)
        alpha_real, alpha_imag,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return A