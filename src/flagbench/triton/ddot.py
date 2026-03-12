from sandbox.register import register
from flagbench.dataset import Autograd
import torch
import triton
import triton.language as tl

@triton.jit
def ddot_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    incx, incy,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets * incx, mask=mask)
    y = tl.load(y_ptr + offsets * incy, mask=mask)
    
    product = x * y
    partial_sum = tl.sum(product, axis=0)
    
    if pid == 0:
        tl.atomic_add(output_ptr, partial_sum)

@register("CUDA", "ddot", has_backward=Autograd.disable, namespace="triton")
def ddot(
    n: int,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    result: torch.Tensor,
):
    assert x.dtype == torch.float64 and y.dtype == torch.float64
    assert x.is_cuda and y.is_cuda
    
    if n <= 0:
        return result.fill_(0.0)
    
    output = torch.zeros(1, device='cuda', dtype=torch.float64)
    
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    ddot_kernel[grid](
        x_ptr=x, y_ptr=y, output_ptr=output,
        n_elements=n,
        incx=incx, incy=incy,
        BLOCK_SIZE=1024,
    )
    
    result.copy_(output)
    return result