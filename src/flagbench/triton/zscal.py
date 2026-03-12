from sandbox.register import register
from flagbench.dataset import Autograd
import torch
import triton
import triton.language as tl

@triton.jit
def zscal_kernel(
    x_ptr,
    alpha_real,
    alpha_imag,
    n,
    incx,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    x_ptrs = x_ptr + offsets * incx * 2  # *2 for complex numbers (real+imag)
    
    # Load complex numbers
    x_real = tl.load(x_ptrs, mask=mask, other=0.0)
    x_imag = tl.load(x_ptrs + 1, mask=mask, other=0.0)
    
    # Perform complex multiplication: x = alpha * x
    out_real = alpha_real * x_real - alpha_imag * x_imag
    out_imag = alpha_real * x_imag + alpha_imag * x_real
    
    # Store results
    tl.store(x_ptrs, out_real, mask=mask)
    tl.store(x_ptrs + 1, out_imag, mask=mask)

@register("CUDA", "zscal", has_backward=Autograd.disable, namespace="triton")
def zscal(
    n: int,
    alpha: torch.Tensor,
    x: torch.Tensor,
    incx: int,
):
    # Extract real and imaginary parts from alpha
    alpha_real = alpha.real.item()
    alpha_imag = alpha.imag.item()
    
    # Compute grid size
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    zscal_kernel[grid](
        x,
        alpha_real,
        alpha_imag,
        n,
        incx,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return x