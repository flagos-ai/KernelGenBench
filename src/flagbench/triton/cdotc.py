from sandbox.register import register
from flagbench.dataset import Autograd
import torch
import triton
import triton.language as tl

@triton.jit
def cdotc_kernel(
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
    x_real = tl.view(x, tl.float32)[::2]
    x_imag = tl.view(x, tl.float32)[1::2]
    y_real = tl.view(y, tl.float32)[::2]
    y_imag = tl.view(y, tl.float32)[1::2]

    real_part = x_real * y_real + x_imag * y_imag
    imag_part = x_real * y_imag - x_imag * y_real

    # Sum reduction within block
    real_sum = tl.sum(real_part, axis=0)
    imag_sum = tl.sum(imag_part, axis=0)

    # Atomic add to output
    output = tl.load(output_ptr)
    output_real = tl.view(output, tl.float32)[0] + real_sum
    output_imag = tl.view(output, tl.float32)[1] + imag_sum
    tl.store(output_ptr, tl.view(torch.tensor([output_real, output_imag], dtype=torch.float32), tl.float32))

@register("CUDA", "cdotc", has_backward=Autograd.disable, namespace="triton")
def cdotc(
    n: int,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    result: torch.Tensor,
):
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    
    # Initialize output to zero
    result.zero_()
    
    cdotc_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        output_ptr=result,
        n=n,
        incx=incx,
        incy=incy,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return result