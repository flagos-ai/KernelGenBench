from sandbox.register import register
from flagbench.dataset import Autograd
import torch
import triton
import triton.language as tl

@triton.jit
def sdgmm_kernel(
    a_ptr, x_ptr, c_ptr,
    m, n,
    stride_am, stride_an,
    stride_x,
    stride_cm, stride_cn,
    mode: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    a_offs = rm[:, None] * stride_am + rn[None, :] * stride_an
    x_offs = (rm if mode == 0 else rn) * stride_x
    c_offs = rm[:, None] * stride_cm + rn[None, :] * stride_cn
    
    mask = (rm < m)[:, None] & (rn < n)[None, :]
    
    a = tl.load(a_ptr + a_offs, mask=mask, other=0.0)
    x = tl.load(x_ptr + x_offs, mask=(rm if mode == 0 else rn) < (m if mode == 0 else n), other=0.0)
    
    if mode == 0:
        x = x[:, None]
    else:
        x = x[None, :]
    
    c = a * x
    tl.store(c_ptr + c_offs, c, mask=mask)

@register("CUDA", "sdgmm", has_backward=Autograd.disable, namespace="triton")
def sdgmm(
    handle,  # ignored
    mode,  # cublasSideMode_t: 0 (left) or 1 (right)
    m,  # rows of A
    n,  # cols of A
    A: torch.Tensor,  # input matrix
    lda,  # leading dim of A (>=m)
    x: torch.Tensor,  # input vector
    incx,  # stride of x
    C: torch.Tensor,  # output matrix
    ldc,  # leading dim of C (>=m)
):
    # Grid and block sizes
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    
    grid = lambda META: (
        triton.cdiv(m, META['BLOCK_SIZE_M']),
        triton.cdiv(n, META['BLOCK_SIZE_N']),
    )
    
    # Launch kernel
    sdgmm_kernel[grid](
        A, x, C,
        m, n,
        lda, 1,  # strides for A
        incx,  # stride for x
        ldc, 1,  # strides for C
        mode,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    return C