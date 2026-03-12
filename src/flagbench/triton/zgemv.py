from sandbox.register import register
from flagbench.dataset import Autograd
import torch
import triton
import triton.language as tl

@triton.jit
def zgemv_kernel(
    A_ptr, x_ptr, y_ptr,
    alpha_real, alpha_imag,
    beta_real, beta_imag,
    M, N,
    stride_am, stride_an,
    stride_x, stride_y,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_m = pid_m < M

    # Initialize accumulator
    acc_real = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float64)
    acc_imag = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float64)

    for n in range(0, N):
        # Load A and x
        a_real = tl.load(A_ptr + pid_m * stride_am + n * stride_an, mask=mask_m, other=0.0).to(tl.float64)
        a_imag = tl.load(A_ptr + pid_m * stride_am + n * stride_an + 1, mask=mask_m, other=0.0).to(tl.float64)
        x_real = tl.load(x_ptr + n * stride_x).to(tl.float64)
        x_imag = tl.load(x_ptr + n * stride_x + 1).to(tl.float64)

        # Complex multiplication: A * x
        real_part = a_real * x_real - a_imag * x_imag
        imag_part = a_real * x_imag + a_imag * x_real

        # Accumulate
        acc_real += real_part
        acc_imag += imag_part

    # Multiply by alpha
    acc_real = acc_real * alpha_real - acc_imag * alpha_imag
    acc_imag = acc_real * alpha_imag + acc_imag * alpha_real

    # Load y and multiply by beta
    y_real = tl.load(y_ptr + pid_m * stride_y, mask=mask_m, other=0.0).to(tl.float64)
    y_imag = tl.load(y_ptr + pid_m * stride_y + 1, mask=mask_m, other=0.0).to(tl.float64)
    y_real = y_real * beta_real - y_imag * beta_imag
    y_imag = y_real * beta_imag + y_imag * beta_real

    # Add alpha*A*x + beta*y
    y_real += acc_real
    y_imag += acc_imag

    # Store result
    tl.store(y_ptr + pid_m * stride_y, y_real, mask=mask_m)
    tl.store(y_ptr + pid_m * stride_y + 1, y_imag, mask=mask_m)

@register("CUDA", "zgemv", has_backward=Autograd.disable, namespace="triton")
def zgemv(
    trans: int,
    m: int,
    n: int,
    alpha: complex,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
    beta: complex,
    y: torch.Tensor,
    incy: int,
):
    # Convert complex scalars to real/imag parts
    alpha_real, alpha_imag = alpha.real, alpha.imag
    beta_real, beta_imag = beta.real, beta.imag

    # Determine matrix dimensions based on transpose flag
    M = m if trans == 0 else n
    N = n if trans == 0 else m

    # Grid and block configuration
    BLOCK_SIZE_M = 64
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']),)

    # Launch kernel
    zgemv_kernel[grid](
        A, x, y,
        alpha_real, alpha_imag,
        beta_real, beta_imag,
        M, N,
        lda, 1,  # A strides
        incx, incy,  # x and y strides
        BLOCK_SIZE_M=BLOCK_SIZE_M,
    )

    return y