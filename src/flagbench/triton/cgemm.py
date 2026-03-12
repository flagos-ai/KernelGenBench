from sandbox.register import register
from flagbench.dataset import Autograd
import torch
import triton
import triton.language as tl

@triton.jit
def cgemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    alpha_real, alpha_imag,
    beta_real, beta_imag,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    TRANS_A: tl.constexpr,
    TRANS_B: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)
    
    if TRANS_A:
        a_ptrs = a_ptr + (rk[:, None] * stride_am + rm[None, :] * stride_ak)
    else:
        a_ptrs = a_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    
    if TRANS_B:
        b_ptrs = b_ptr + (rn[:, None] * stride_bk + rk[None, :] * stride_bn)
    else:
        b_ptrs = b_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)
    
    acc_real = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_imag = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_real = tl.load(a_ptrs, mask=rk[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        a_imag = tl.load(a_ptrs + 1, mask=rk[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b_real = tl.load(b_ptrs, mask=rk[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        b_imag = tl.load(b_ptrs + 1, mask=rk[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        
        # Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        acc_real += a_real * b_real - a_imag * b_imag
        acc_imag += a_real * b_imag + a_imag * b_real
        
        a_ptrs += BLOCK_SIZE_K * (stride_ak if not TRANS_A else stride_am)
        b_ptrs += BLOCK_SIZE_K * (stride_bn if not TRANS_B else stride_bk)
    
    # Apply alpha scaling
    acc_real = acc_real * alpha_real - acc_imag * alpha_imag
    acc_imag = acc_real * alpha_imag + acc_imag * alpha_real
    
    # Write back with beta scaling
    c_ptrs = c_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    c_real = tl.load(c_ptrs, mask=(rm[:, None] < M) & (rn[None, :] < N), other=0.0)
    c_imag = tl.load(c_ptrs + 1, mask=(rm[:, None] < M) & (rn[None, :] < N), other=0.0)
    
    c_real = acc_real + beta_real * c_real - beta_imag * c_imag
    c_imag = acc_imag + beta_real * c_imag + beta_imag * c_real
    
    tl.store(c_ptrs, c_real, mask=(rm[:, None] < M) & (rn[None, :] < N))
    tl.store(c_ptrs + 1, c_imag, mask=(rm[:, None] < M) & (rn[None, :] < N))

@register("CUDA", "cgemm", has_backward=Autograd.disable, namespace="triton")
def cgemm(
    transa: int,
    transb: int,
    m: int,
    n: int,
    k: int,
    alpha: complex,
    A: torch.Tensor,
    lda: int,
    B: torch.Tensor,
    ldb: int,
    beta: complex,
    C: torch.Tensor,
    ldc: int,
):
    # Convert complex scalars to real/imaginary parts
    alpha_real, alpha_imag = alpha.real, alpha.imag
    beta_real, beta_imag = beta.real, beta.imag
    
    # Determine matrix dimensions after transpose
    M = m
    N = n
    K = k
    
    # Grid and block sizes
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']),
        triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    cgemm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        alpha_real, alpha_imag,
        beta_real, beta_imag,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        TRANS_A=transa == 1 or transa == 2,
        TRANS_B=transb == 1 or transb == 2,
    )
    return C