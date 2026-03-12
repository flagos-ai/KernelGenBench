from sandbox.register import register
from flagbench.dataset import Autograd
import torch
import triton
import triton.language as tl

@triton.jit
def cdotu_kernel(
    x_ptr, y_ptr, output_ptr,
    n, incx, incy,
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

    product = x * y
    partial_sum = tl.sum(product, axis=0)

    if pid == 0:
        output = tl.zeros((1,), dtype=tl.float32)  # Initialize as float32 for complex64
        output += partial_sum
        tl.store(output_ptr, output)

@register("CUDA", "cdotu", has_backward=Autograd.disable, namespace="triton")
def cdotu(
    n: int,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    result: torch.Tensor = None,
):
    if result is None:
        result = torch.empty(1, dtype=x.dtype, device=x.device)

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    
    cdotu_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        output_ptr=result,
        n=n,
        incx=incx,
        incy=incy,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return result