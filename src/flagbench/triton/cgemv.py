from sandbox.register import register
from flagbench.dataset import Autograd
import torch
import triton
import triton.language as tl

@triton.jit
def cgemv_kernel(
    A_ptr, x_ptr, y_ptr,
    alpha_real, alpha_imag,
    beta_real, beta_imag,
    M, N,
    stride_am, stride_ak,
    stride_x, stride_y,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask = pid_m < M

    # Initialize accumulator
    acc_real = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    acc_imag = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    for k in range(0, N):
        x_real = tl.load(x_ptr + k * stride_x + 0, mask=mask)
        x_imag = tl.load(x_ptr + k * stride_x + 1, mask=mask)
        a_real = tl.load(A_ptr + pid_m * stride_am + k * stride_ak + 0, mask=mask)
        a_imag = tl.load(A_ptr + pid_m * stride_am + k * stride_ak + 1, mask=mask)

        # Complex multiplication: A * x
        acc_real += a_real * x_real - a_imag * x_imag
        acc_imag += a_real * x_imag + a_imag * x_real

    # Apply alpha
    acc_real = acc_real * alpha_real - acc_imag * alpha_imag
    acc_imag = acc_real * alpha_imag + acc_imag * alpha_real

    # Load y and apply beta
    y_real = tl.load(y_ptr + pid_m * stride_y + 0, mask=mask)
    y_imag = tl.load(y_ptr + pid_m * stride_y + 1, mask=mask)
    y_real = y_real * beta_real - y_imag * beta_imag + acc_real
    y_imag = y_real * beta_imag + y_imag * beta_real + acc_imag

    # Store result
    tl.store(y_ptr + pid_m * stride_y + 0, y_real, mask=mask)
    tl.store(y_ptr + pid_m * stride_y + 1, y_imag, mask=mask)

@register("CUDA", "cgemv", has_backward=Autograd.disable, namespace="triton")
def cgemv(
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

    # Determine matrix dimensions
    M, N = (m, n) if trans == 0 else (n, m)
    BLOCK_SIZE_M = 64

    # Compute grid
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']),)

    # Launch kernel
    cgemv_kernel[grid](
        A, x, y,
        alpha_real, alpha_imag,
        beta_real, beta_imag,
        M, N,
        A.stride(0), A.stride(1),
        x.stride(0), y.stride(0),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
    )
    return y