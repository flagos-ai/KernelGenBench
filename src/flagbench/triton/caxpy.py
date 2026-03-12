from sandbox.register import register
from flagbench.dataset import Autograd
import torch
import triton
import triton.language as tl

@triton.jit
def caxpy_kernel(
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

    x_real = tl.load(x_ptr + 2 * offsets * incx, mask=mask)
    x_imag = tl.load(x_ptr + 2 * offsets * incx + 1, mask=mask)
    y_real = tl.load(y_ptr + 2 * offsets * incy, mask=mask)
    y_imag = tl.load(y_ptr + 2 * offsets * incy + 1, mask=mask)

    # Compute alpha * x + y
    out_real = alpha_real * x_real - alpha_imag * x_imag + y_real
    out_imag = alpha_real * x_imag + alpha_imag * x_real + y_imag

    tl.store(y_ptr + 2 * offsets * incy, out_real, mask=mask)
    tl.store(y_ptr + 2 * offsets * incy + 1, out_imag, mask=mask)

@register("CUDA", "caxpy", has_backward=Autograd.disable, namespace="triton")
def caxpy(
    n: int,
    alpha: complex,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
):
    # Extract real and imaginary parts of alpha
    alpha_real = alpha.real
    alpha_imag = alpha.imag

    # Launch kernel
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    caxpy_kernel[grid](
        x,
        y,
        alpha_real,
        alpha_imag,
        n,
        incx,
        incy,
        BLOCK_SIZE=1024,
    )
    return y