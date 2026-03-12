from sandbox.register import register
from flagbench.dataset import Autograd
import torch
import triton
import triton.language as tl

@triton.jit
def dasum_kernel(
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
    abs_x = tl.abs(x)
    partial_sum = tl.sum(abs_x, axis=0)
    
    # Use atomic add to accumulate the partial sums
    tl.atomic_add(output_ptr, partial_sum)

@register("CUDA", "dasum", has_backward=Autograd.disable, namespace="triton")
def dasum(
    n: int,
    x: torch.Tensor,
    incx: int,
    result: torch.Tensor,
):
    # Allocate output tensor if not provided
    if result is None:
        result = torch.zeros(1, device=x.device, dtype=x.dtype)
    else:
        result.zero_()
    
    # Compute number of elements
    num_elements = n
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
    
    dasum_kernel[grid](
        x_ptr=x,
        output_ptr=result,
        n=num_elements,
        incx=incx,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return result