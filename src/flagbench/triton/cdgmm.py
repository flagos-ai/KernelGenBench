from sandbox.register import register
from flagbench.dataset import Autograd
import torch
import triton
import triton.language as tl

@triton.jit
def cdgmm_kernel(
    a_ptr, x_ptr, c_ptr,
    m, n,
    stride_am, stride_an,
    stride_x,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    a_offsets = rm[:, None] * stride_am + rn[None, :] * stride_an
    x_offsets = rn * stride_x if mode == 0 else rm * stride_x
    
    mask = (rm < m)[:, None] & (rn < n)[None, :]
    
    a = tl.load(a_ptr + a_offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + x_offsets, mask=rn < n if mode == 0 else rm < m, other=0.0)
    
    if mode == 0:  # right side mode
        x = tl.broadcast_to(x[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N))
    else:  # left side mode
        x = tl.broadcast_to(x[:, None], (BLOCK_SIZE_M, BLOCK_SIZE_N))
    
    c = a * x
    c_offsets = rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptr + c_offsets, c, mask=mask)

@register("CUDA", "cdgmm", has_backward=Autograd.disable, namespace="triton")
def cdgmm(
    mode: int,
    m: int,
    n: int,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
    C: torch.Tensor,
    ldc: int,
):
    # Ensure tensors are complex64 and on CUDA
    assert A.dtype == torch.complex64
    assert x.dtype == torch.complex64
    assert C.dtype == torch.complex64
    assert A.is_cuda and x.is_cuda and C.is_cuda
    
    # Determine grid and block sizes
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    
    grid = lambda META: (
        triton.cdiv(m, META['BLOCK_SIZE_M']),
        triton.cdiv(n, META['BLOCK_SIZE_N']),
    )
    
    # Launch kernel
    cdgmm_kernel[grid](
        A, x, C,
        m, n,
        A.stride(0), A.stride(1),
        x.stride(0) * incx,
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return C