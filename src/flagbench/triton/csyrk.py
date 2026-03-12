from sandbox.register import register
from flagbench.dataset import Autograd
import torch
import triton
import triton.language as tl

@triton.jit
def csyrk_kernel(
    A_ptr, C_ptr,
    n, k,
    alpha_re, alpha_im,
    beta_re, beta_im,
    stride_am, stride_ak,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Determine if we're in upper or lower triangular part
    if UPLO == 0:  # CUBLAS_FILL_MODE_LOWER
        if pid_m < pid_n:
            return
    else:  # CUBLAS_FILL_MODE_UPPER
        if pid_m > pid_n:
            return

    # Offsets for the block
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Load A matrix
    if TRANS == 0:  # CUBLAS_OP_N
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=offs_m[:, None] < n, other=0.0)
        a_re = a.real
        a_im = a.imag
        b_ptrs = A_ptr + offs_k[:, None] * stride_am + offs_n[None, :] * stride_ak
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k, other=0.0)
        b_re = b.real
        b_im = b.imag
    else:  # CUBLAS_OP_T or CUBLAS_OP_C
        a_ptrs = A_ptr + offs_k[:, None] * stride_am + offs_m[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=offs_k[:, None] < k, other=0.0)
        a_re = a.real
        a_im = a.imag
        b_ptrs = A_ptr + offs_n[:, None] * stride_am + offs_k[None, :] * stride_ak
        b = tl.load(b_ptrs, mask=offs_n[:, None] < n, other=0.0)
        b_re = b.real
        b_im = b.imag
    
    # Compute dot product
    for k_block in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
        a_block_re = tl.load(a_ptrs, mask=(offs_m[:, None] < n) & (k_block * BLOCK_SIZE_K + offs_k[None, :] < k), other=0.0).real
        a_block_im = tl.load(a_ptrs, mask=(offs_m[:, None] < n) & (k_block * BLOCK_SIZE_K + offs_k[None, :] < k), other=0.0).imag
        b_block_re = tl.load(b_ptrs, mask=(k_block * BLOCK_SIZE_K + offs_k[:, None] < k) & (offs_n[None, :] < n), other=0.0).real
        b_block_im = tl.load(b_ptrs, mask=(k_block * BLOCK_SIZE_K + offs_k[:, None] < k) & (offs_n[None, :] < n), other=0.0).imag
        
        if TRANS == 2:  # CUBLAS_OP_C
            b_block_im = -b_block_im
        
        acc += (a_block_re @ b_block_re - a_block_im @ b_block_im) * alpha_re
        acc += (a_block_re @ b_block_im + a_block_im @ b_block_re) * alpha_im
        
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_am
    
    # Load C matrix
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c = tl.load(c_ptrs, mask=(offs_m[:, None] < n) & (offs_n[None, :] < n), other=0.0)
    c_re = c.real * beta_re - c.imag * beta_im
    c_im = c.real * beta_im + c.imag * beta_re
    
    # Store result
    result = tl.complex(acc + c_re, c_im)
    tl.store(c_ptrs, result, mask=(offs_m[:, None] < n) & (offs_n[None, :] < n))

@register("CUDA", "csyrk", has_backward=Autograd.disable, namespace="triton")
def csyrk(
    uplo: int,
    trans: int,
    n: int,
    k: int,
    alpha: complex,
    A: torch.Tensor,
    lda: int,
    beta: complex,
    C: torch.Tensor,
    ldc: int,
):
    # Extract real and imaginary parts of alpha and beta
    alpha_re, alpha_im = alpha.real, alpha.imag
    beta_re, beta_im = beta.real, beta.imag
    
    # Grid and block sizes
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    
    grid = lambda META: (triton.cdiv(n, META['BLOCK_SIZE_M']), triton.cdiv(n, META['BLOCK_SIZE_N']))
    
    # Launch kernel
    csyrk_kernel[grid](
        A, C,
        n, k,
        alpha_re, alpha_im,
        beta_re, beta_im,
        A.stride(0), A.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        uplo, trans,
    )
    
    return C