from sandbox.register import register
from flagbench.dataset import Autograd
import torch
import triton
import triton.language as tl

@triton.jit
def daxpy_kernel(
    x_ptr,
    y_ptr,
    alpha,
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
    
    x = tl.load(x_ptr + x_offsets, mask=mask)
    y = tl.load(y_ptr + y_offsets, mask=mask)
    
    y = y + alpha * x
    
    tl.store(y_ptr + y_offsets, y, mask=mask)

@register("CUDA", "daxpy", has_backward=Autograd.disable, namespace="triton")
def daxpy(
    n: int,
    alpha: float,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
):
    assert x.dtype == torch.float64 and y.dtype == torch.float64, "Input tensors must be float64"
    assert x.is_cuda and y.is_cuda, "Input tensors must be on CUDA device"
    
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    
    daxpy_kernel[grid](
        x,
        y,
        alpha,
        n,
        incx,
        incy,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return y