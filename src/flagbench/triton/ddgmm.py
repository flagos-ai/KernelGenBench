from sandbox.register import register
from flagbench.dataset import Autograd
import torch
import triton
import triton.language as tl

@triton.jit
def ddgmm_kernel(
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
    x_offs = rm * stride_x if mode == 0 else rn * stride_x
    
    mask = (rm < m)[:, None] & (rn < n)[None, :]
    a = tl.load(a_ptr + a_offs, mask=mask, other=0.0)
    x = tl.load(x_ptr + x_offs, mask=(rm < m) if mode == 0 else (rn < n), other=0.0)
    
    x = x[:, None] if mode == 0 else x[None, :]
    c = a * x
    
    c_offs = rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptr + c_offs, c, mask=mask)

@register("CUDA", "ddgmm", has_backward=Autograd.disable, namespace="triton")
def ddgmm(
    handle,  # ignored
    mode,  # cublasSideMode_t
    m,  # dimension
    n,  # dimension
    A,  # input tensor
    lda,  # stride
    x,  # input tensor
    incx,  # stride
    C,  # output tensor
    ldc,  # stride
):
    # Convert mode to triton-compatible flag
    # 0 = left (multiply rows), 1 = right (multiply columns)
    mode = 0 if mode == 0 else 1  # assuming cublasSideMode_t.LEFT = 0, RIGHT = 1
    
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    
    grid = lambda META: (
        triton.cdiv(m, META['BLOCK_SIZE_M']),
        triton.cdiv(n, META['BLOCK_SIZE_N']),
    )
    
    ddgmm_kernel[grid](
        A, x, C,
        m, n,
        lda, 1,  # A strides
        incx,  # x stride
        ldc, 1,  # C strides
        mode,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    return C