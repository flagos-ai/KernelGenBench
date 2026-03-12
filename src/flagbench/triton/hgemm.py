from sandbox.register import register
from flagbench.dataset import Autograd
import torch
import triton
import triton.language as tl

@triton.jit
def hgemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    alpha,
    beta,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c = tl.load(c_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)
    c = c * beta + accumulator * alpha
    tl.store(c_ptrs, c.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

@register("CUDA", "hgemm", has_backward=Autograd.disable, namespace="triton")
def hgemm(
    handle,
    transa,
    transb,
    m,
    n,
    k,
    alpha,
    A,
    lda,
    B,
    ldb,
    beta,
    C,
    ldc,
):
    if transa == 0:  # No transpose
        stride_am = lda
        stride_ak = 1
    else:  # Transpose
        stride_am = 1
        stride_ak = lda
    
    if transb == 0:  # No transpose
        stride_bk = ldb
        stride_bn = 1
    else:  # Transpose
        stride_bk = 1
        stride_bn = ldb
    
    stride_cm = ldc
    stride_cn = 1
    
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    grid = lambda META: (
        triton.cdiv(m, META['BLOCK_SIZE_M']),
        triton.cdiv(n, META['BLOCK_SIZE_N']),
    )
    
    hgemm_kernel[grid](
        A, B, C,
        m, n, k,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        alpha, beta,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return C