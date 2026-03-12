from sandbox.register import register
from flagbench.dataset import Autograd
import torch
import triton
import triton.language as tl

@triton.jit
def cscal_kernel(
    x_ptr,
    alpha_real,
    alpha_imag,
    n,
    incx,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    x_ptrs = x_ptr + offsets * incx * 2  # *2 for complex numbers (real+imag)
    
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    x_real = x[::2]
    x_imag = x[1::2]
    
    # Compute alpha * x
    out_real = alpha_real * x_real - alpha_imag * x_imag
    out_imag = alpha_real * x_imag + alpha_imag * x_real
    
    out = tl.zeros_like(x)
    out = out.at[::2].set(out_real)
    out = out.at[1::2].set(out_imag)
    
    tl.store(x_ptrs, out, mask=mask)

@register("CUDA", "cscal", has_backward=Autograd.disable, namespace="triton")
def cscal(
    n: int,
    alpha: torch.Tensor,
    x: torch.Tensor,
    incx: int,
):
    # Extract real and imaginary parts from alpha
    alpha_real = alpha.real.item()
    alpha_imag = alpha.imag.item()
    
    # Compute grid size
    grid = lambda META: (triton.cdiv(n, META['BLOCK_SIZE']),)
    
    # Launch kernel
    cscal_kernel[grid](
        x_ptr=x.data_ptr(),
        alpha_real=alpha_real,
        alpha_imag=alpha_imag,
        n=n,
        incx=incx,
        BLOCK_SIZE=1024,
    )
    
    return x