from sandbox.register import register
from flagbench.dataset import Autograd
import torch
import triton
import triton.language as tl

@triton.jit
def sgemv_kernel(
    A_ptr, x_ptr, y_ptr,
    m, n,
    stride_am, stride_an,
    stride_x, stride_y,
    alpha, beta,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(m, BLOCK_SIZE_M)
    pid_m = pid // num_pid_m
    pid_n = pid % num_pid_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an
    x_ptrs = x_ptr + offs_n * stride_x
    
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for k in range(0, tl.cdiv(n, BLOCK_SIZE_N)):
        a = tl.load(A_ptrs, mask=offs_n[None, :] < n - k * BLOCK_SIZE_N, other=0.0)
        b = tl.load(x_ptrs, mask=offs_n < n - k * BLOCK_SIZE_N, other=0.0)
        acc += tl.sum(a * b, axis=1)
        A_ptrs += BLOCK_SIZE_N * stride_an
        x_ptrs += BLOCK_SIZE_N * stride_x
    
    y_ptrs = y_ptr + offs_m * stride_y
    y = tl.load(y_ptrs, mask=offs_m < m, other=0.0)
    y = alpha * acc + beta * y
    tl.store(y_ptrs, y, mask=offs_m < m)

@register("CUDA", "sgemv", has_backward=Autograd.disable, namespace="triton")
def sgemv(
    trans: int,
    m: int,
    n: int,
    alpha: float,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
    beta: float,
    y: torch.Tensor,
    incy: int,
):
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    
    grid = lambda META: (m // META['BLOCK_SIZE_M'] * n // META['BLOCK_SIZE_N'],)
    
    sgemv_kernel[grid](
        A, x, y,
        m, n,
        A.stride(0), A.stride(1),
        x.stride(0), y.stride(0),
        alpha, beta,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    return y