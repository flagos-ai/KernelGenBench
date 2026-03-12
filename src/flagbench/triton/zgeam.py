from sandbox.register import register
from flagbench.dataset import Autograd
import torch
import triton
import triton.language as tl

@triton.jit
def zgeam_kernel(
    a_ptr, b_ptr, c_ptr,
    m, n,
    alpha_real, alpha_imag,
    beta_real, beta_imag,
    stride_am, stride_aan,
    stride_abm, stride_abn,
    stride_cm, stride_cn,
    transa: tl.constexpr,
    transb: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    mask_m = rm < m
    mask_n = rn < n
    mask = mask_m[:, None] & mask_n[None, :]
    
    # Compute offsets based on transpose flags
    if transa:
        a_offsets = rn[:, None] * stride_aan + rm[None, :] * stride_aam
    else:
        a_offsets = rm[:, None] * stride_aam + rn[None, :] * stride_aan
    
    if transb:
        b_offsets = rn[:, None] * stride_abn + rm[None, :] * stride_abm
    else:
        b_offsets = rm[:, None] * stride_abm + rn[None, :] * stride_abn
    
    c_offsets = rm[:, None] * stride_cm + rn[None, :] * stride_cn
    
    # Load A and B
    a = tl.load(a_ptr + a_offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + b_offsets, mask=mask, other=0.0)
    
    # Apply alpha and beta
    alpha = tl.complex(alpha_real, alpha_imag)
    beta = tl.complex(beta_real, beta_imag)
    
    # Compute C = alpha * A + beta * B
    c = alpha * a + beta * b
    
    # Store result
    tl.store(c_ptr + c_offsets, c, mask=mask)

@register("CUDA", "zgeam", has_backward=Autograd.disable, namespace="triton")
def zgeam(
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
    # Check dimensions
    assert A.shape[0] == m if transa == 0 else A.shape[1] == m
    assert A.shape[1] == n if transa == 0 else A.shape[0] == n
    assert B.shape[0] == m if transb == 0 else B.shape[1] == m
    assert B.shape[1] == n if transb == 0 else B.shape[0] == n
    
    # Extract real and imaginary parts of scalars
    alpha_real = alpha.real
    alpha_imag = alpha.imag
    beta_real = beta.real
    beta_imag = beta.imag
    
    # Get strides
    stride_am = A.stride(0) if transa == 0 else A.stride(1)
    stride_aan = A.stride(1) if transa == 0 else A.stride(0)
    stride_abm = B.stride(0) if transb == 0 else B.stride(1)
    stride_abn = B.stride(1) if transb == 0 else B.stride(0)
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)
    
    # Define grid and launch kernel
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    grid = lambda META: (
        triton.cdiv(m, META['BLOCK_SIZE_M']),
        triton.cdiv(n, META['BLOCK_SIZE_N']),
    )
    
    zgeam_kernel[grid](
        A, B, C,
        m, n,
        alpha_real, alpha_imag,
        beta_real, beta_imag,
        stride_am=stride_am,
        stride_an=stride_an,
        stride_bm=stride_bm,
        stride_bn=stride_bn,
        stride_cm=stride_cm,
        stride_cn=stride_cn,
        transa=transa,
        transb=transb,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return C