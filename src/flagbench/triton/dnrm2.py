from sandbox.register import register
from flagbench.dataset import Autograd
import torch
import triton
import triton.language as tl

@triton.jit
def dnrm2_kernel(
    x_ptr,
    output_ptr,
    n,
    incx,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(x_ptr + offsets * incx, mask=mask, other=0.0)
    x_square = x * x

    # Parallel reduction within the block
    partial_sum = tl.sum(x_square, axis=0)

    # Atomically add the partial sum to the output
    tl.atomic_add(output_ptr, partial_sum)

@register("CUDA", "dnrm2", has_backward=Autograd.disable, namespace="triton")
def dnrm2(
    n: int,
    x: torch.Tensor,
    incx: int,
    result: torch.Tensor = None,
):
    if result is None:
        result = torch.empty(1, device=x.device, dtype=x.dtype)
    else:
        result.zero_()

    BLOCK_SIZE = 1024  # Optimal block size for reduction
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    
    dnrm2_kernel[grid](
        x_ptr=x,
        output_ptr=result,
        n=n,
        incx=incx,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Square root is computed on the host side
    result.sqrt_()
    return result