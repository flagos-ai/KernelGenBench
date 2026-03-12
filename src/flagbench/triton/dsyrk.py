from sandbox.register import register
from flagbench.dataset import Autograd
import torch
import triton
import triton.language as tl

@triton.jit
def dsyrk_kernel(
    A_ptr, C_ptr,
    n, k,
    stride_am, stride_ak,
    stride_cm, stride_cn,
    alpha: tl.constexpr,
    beta: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Adjust for upper/lower triangular
    if UPLO == 0:  # CUBLAS_FILL_MODE_LOWER
        if pid_m < pid_n:
            return
    else:  # CUBLAS_FILL_MODE_UPPER
        if pid_m > pid_n:
            return
    
    # Offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Pointers
    A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float64)
    
    # Compute
    for k_block in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
        k_offs = k_block * BLOCK_SIZE_K
        a = tl.load(A_ptrs, mask=(offs_m[:, None] < n) & (k_offs + offs_k[None, :] < k), other=0.0)
        b = tl.load(A_ptrs, mask=(offs_n[None, :] < n) & (k_offs + offs_k[:, None] < k), other=0.0)
        if TRANS == 0:  # CUBLAS_OP_N
            b = tl.trans(b)
        acc += tl.dot(a, b)
    
    # Load current C value
    c = tl.load(C_ptrs, mask=(offs_m[:, None] < n) & (offs_n[None, :] < n), other=0.0)
    
    # Update C
    c = alpha * acc + beta * c
    tl.store(C_ptrs, c, mask=(offs_m[:, None] < n) & (offs_n[None, :] < n))

@register("CUDA", "dsyrk", has_backward=Autograd.disable, namespace="triton")
def dsyrk(
    uplo: int,
    trans: int,
    n: int,
    k: int,
    alpha: float,
    A: torch.Tensor,
    lda: int,
    beta: float,
    C: torch.Tensor,
    ldc: int,
):
    # Grid and block sizes
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    
    # Launch kernel
    grid = lambda META: (triton.cdiv(n, META['BLOCK_SIZE_M']), triton.cdiv(n, META['BLOCK_SIZE_N']))
    
    dsyrk_kernel[grid](
        A, C,
        n, k,
        A.stride(0), A.stride(1),
        C.stride(0), C.stride(1),
        alpha, beta,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        uplo, trans,
    )
    return C