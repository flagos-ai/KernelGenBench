from sandbox.register import register
from flagbench.dataset import Autograd
import torch
import triton
import triton.language as tl

@triton.jit
def zaxpy_kernel(
    x_ptr,
    y_ptr,
    alpha_real,
    alpha_imag,
    n,
    incx,
    incy,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x_real_ptr = x_ptr + 0
    x_imag_ptr = x_ptr + n * incx
    y_real_ptr = y_ptr + 0
    y_imag_ptr = y_ptr + n * incy

    x_real = tl.load(x_real_ptr + offsets * incx, mask=mask)
    x_imag = tl.load(x_imag_ptr + offsets * incx, mask=mask)
    y_real = tl.load(y_real_ptr + offsets * incy, mask=mask)
    y_imag = tl.load(y_imag_ptr + offsets * incy, mask=mask)

    # Complex multiplication: alpha * x
    ax_real = alpha_real * x_real - alpha_imag * x_imag
    ax_imag = alpha_real * x_imag + alpha_imag * x_real

    # Complex addition: y += alpha * x
    y_real_new = y_real + ax_real
    y_imag_new = y_imag + ax_imag

    tl.store(y_real_ptr + offsets * incy, y_real_new, mask=mask)
    tl.store(y_imag_ptr + offsets * incy, y_imag_new, mask=mask)

@register("CUDA", "zaxpy", has_backward=Autograd.disable, namespace="triton")
def zaxpy(
    n: int,
    alpha: torch.Tensor,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
):
    # Extract real and imaginary parts from alpha
    alpha_real = alpha.real.item()
    alpha_imag = alpha.imag.item()

    # Compute grid size
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)

    # Launch kernel
    zaxpy_kernel[grid](
        x,
        y,
        alpha_real,
        alpha_imag,
        n,
        incx,
        incy,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y