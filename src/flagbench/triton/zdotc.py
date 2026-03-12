from sandbox.register import register
from flagbench.dataset import Autograd
import torch
import triton
import triton.language as tl

@triton.jit
def zdotc_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n,
    incx,
    incy,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x_offsets = offsets * incx
    y_offsets = offsets * incy

    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + y_offsets, mask=mask, other=0.0)

    # Compute complex dot product (conjugating x)
    x_real = tl.view(x, tl.float64)[::2]
    x_imag = tl.view(x, tl.float64)[1::2]
    y_real = tl.view(y, tl.float64)[::2]
    y_imag = tl.view(y, tl.float64)[1::2]

    real_part = x_real * y_real + x_imag * y_imag
    imag_part = x_real * y_imag - x_imag * y_real

    # Sum reductions
    sum_real = tl.sum(real_part, axis=0)
    sum_imag = tl.sum(imag_part, axis=0)

    # Store result
    if pid == 0:
        output = tl.zeros((2,), dtype=tl.float64)
        output = output.to(tl.uint64)  # reinterpret as complex
        output_ptr[0] = output

@register("CUDA", "zdotc", has_backward=Autograd.disable, namespace="triton")
def zdotc(
    n: int,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    result: torch.Tensor,
):
    # Ensure tensors are on CUDA and correct dtype
    assert x.is_cuda and y.is_cuda
    assert x.dtype == torch.complex128 and y.dtype == torch.complex128

    # Allocate output if needed
    if result is None:
        result = torch.empty(1, dtype=torch.complex128, device=x.device)

    # Grid and block configuration
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)

    # Launch kernel
    zdotc_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        output_ptr=result,
        n=n,
        incx=incx,
        incy=incy,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return result