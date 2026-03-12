from sandbox.register import register
from flagbench.dataset import Autograd
import torch
import triton
import triton.language as tl

@triton.jit
def sscal_kernel(
    x_ptr,
    alpha,
    n,
    incx,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    x = tl.load(x_ptr + offsets * incx, mask=mask)
    x = x * alpha
    tl.store(x_ptr + offsets * incx, x, mask=mask)

@register("CUDA", "sscal", has_backward=Autograd.disable, namespace="triton")
def sscal(
    n: int,
    alpha: float,
    x: torch.Tensor,
    incx: int,
):
    assert x.is_cuda, "Input tensor must be on CUDA device"
    assert x.dtype == torch.float32, "Input tensor must be float32"
    
    grid = lambda META: (triton.cdiv(n, META['BLOCK_SIZE']),)
    sscal_kernel[grid](x, alpha, n, incx, BLOCK_SIZE=1024)
    return x