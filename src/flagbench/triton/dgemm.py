from sandbox.register import register
from flagbench.dataset import Autograd
import torch
import triton
import triton.language as tl

@triton.jit
def dgemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    alpha: tl.constexpr,
    beta: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)
    
    A = tl.load(a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak, mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
    B = tl.load(b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn, mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float64)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptr + rm[:, None] * stride_am + (k * BLOCK_SIZE_K + rk[None, :]) * stride_ak, 
                   mask=(rm[:, None] < M) & ((k * BLOCK_SIZE_K + rk[None, :]) < K), other=0.0)
        b = tl.load(b_ptr + (k * BLOCK_SIZE_K + rk[:, None]) * stride_bk + rn[None, :] * stride_bn, 
                   mask=((k * BLOCK_SIZE_K + rk[:, None]) < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
    
    c = tl.load(c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn, 
               mask=(rm[:, None] < M) & (rn[None, :] < N), other=0.0)
    c = alpha * acc + beta * c
    tl.store(c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn, c, 
            mask=(rm[:, None] < M) & (rn[None, :] < N))

@register("CUDA", "dgemm", has_backward=Autograd.disable, namespace="triton")
def dgemm(
    handle,  # ignored
    transa, transb,
    m, n, k,
    alpha,
    A, lda,
    B, ldb,
    beta,
    C, ldc,
):
    if transa == 0:  # CUBLAS_OP_N
        stride_am = lda
        stride_ak = 1
        M, K_A = m, k
    else:  # CUBLAS_OP_T
        stride_am = 1
        stride_ak = lda
        M, K_A = k, m
    
    if transb == 0:  # CUBLAS_OP_N
        stride_bk = ldb
        stride_bn = 1
        K_B, N = k, n
    else:  # CUBLAS_OP_T
        stride_bk = 1
        stride_bn = ldb
        K_B, N = n, k
    
    assert K_A == K_B, "Matrix dimensions must match for multiplication"
    
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']),
        triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    dgemm_kernel[grid](
        A, B, C,
        M, N, K_A,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        ldc, 1,
        alpha, beta,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return C