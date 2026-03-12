from sandbox.register import register
from flagbench.dataset import Autograd
import torch
import triton
import triton.language as tl

@triton.jit
def zdgmm_kernel(
    a_ptr, x_ptr, c_ptr,
    m, n,
    stride_am, stride_an,
    stride_x,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    mode: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    a_offs = rm[:, None] * stride_am + rn[None, :] * stride_an
    x_offs = (rm if mode == 0 else rn) * stride_x
    
    mask = (rm < m)[:, None] & (rn < n)[None, :]
    
    a = tl.load(a_ptr + a_offs, mask=mask, other=0.0)
    x = tl.load(x_ptr + x_offs, mask=(rm if mode == 0 else rn) < (m if mode == 0 else n), other=0.0)
    
    if mode == 0:
        x = x[:, None]
    else:
        x = x[None, :]
    
    c = a * x
    tl.store(c_ptr + a_offs, c, mask=mask)

@register("CUDA", "zdgmm", has_backward=Autograd.disable, namespace="triton")
def zdgmm(
    handle,  # ignored
    mode,  # cublasSideMode_t
    m,  # int
    n,  # int
    A,  # torch.Tensor
    lda,  # int
    x,  # torch.Tensor
    incx,  # int
    C,  # torch.Tensor
    ldc,  # int
):
    assert A.dtype == torch.complex128
    assert x.dtype == torch.complex128
    assert C.dtype == torch.complex128
    
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    
    grid = lambda META: (
        triton.cdiv(m, META['BLOCK_SIZE_M']),
        triton.cdiv(n, META['BLOCK_SIZE_N']),
    )
    
    zdgmm_kernel[grid](
        A, x, C,
        m, n,
        lda, 1,
        incx,
        ldc, 1,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        mode=mode,
    )
    return C