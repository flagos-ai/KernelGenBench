from sandbox.register import register
from flagbench.dataset import Autograd
import torch
import triton
import triton.language as tl

@triton.jit
def cgeam_kernel(
    a_ptr, b_ptr, c_ptr,
    m, n,
    alpha_real, alpha_imag,
    beta_real, beta_imag,
    stride_am, stride_an,
    stride_bm, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    a_mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)
    b_mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)
    
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an
    b_ptrs = b_ptr + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    
    a = tl.load(a_ptrs, mask=a_mask, other=0.0)
    b = tl.load(b_ptrs, mask=b_mask, other=0.0)
    
    alpha = tl.complex(alpha_real, alpha_imag)
    beta = tl.complex(beta_real, beta_imag)
    
    c = alpha * a + beta * b
    tl.store(c_ptrs, c, mask=a_mask & b_mask)

@register("CUDA", "cgeam", has_backward=Autograd.disable, namespace="triton")
def cgeam(
    transa: int,
    transb: int,
    m: int,
    n: int,
    alpha: complex,
    A: torch.Tensor,
    lda: int,
    beta: complex,
    B: torch.Tensor,
    ldb: int,
    C: torch.Tensor,
    ldc: int,
):
    # Handle transposition flags (simplified - actual implementation should handle properly)
    if transa != 0:
        A = A.t().contiguous()
    if transb != 0:
        B = B.t().contiguous()
    
    # Extract real and imaginary parts of scalars
    alpha_real = alpha.real
    alpha_imag = alpha.imag
    beta_real = beta.real
    beta_imag = beta.imag
    
    # Ensure output tensor is properly sized
    if C is None:
        C = torch.empty((m, n), device=A.device, dtype=A.dtype)
    
    # Define grid and launch kernel
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    
    grid = lambda META: (
        triton.cdiv(m, META['BLOCK_SIZE_M']),
        triton.cdiv(n, META['BLOCK_SIZE_N']),
    )
    
    cgeam_kernel[grid](
        A, B, C,
        m, n,
        alpha_real, alpha_imag,
        beta_real, beta_imag,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return C