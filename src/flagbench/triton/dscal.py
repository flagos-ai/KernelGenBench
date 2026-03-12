from sandbox.register import register
from flagbench.dataset import Autograd
import torch
import triton
import triton.language as tl

@triton.jit
def dscal_kernel(
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

@register("CUDA", "dscal", has_backward=Autograd.disable, namespace="triton")
def dscal(
    n: int,
    alpha: float,
    x: torch.Tensor,
    incx: int,
):
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    dscal_kernel[grid](x, alpha, n, incx, BLOCK_SIZE=1024)
    return x