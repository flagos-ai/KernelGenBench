from sandbox.register import register
from flagbench.dataset import Autograd
import torch
import triton
import triton.language as tl

@triton.jit
def dgeam_kernel(
    a_ptr, b_ptr, c_ptr,
    m, n,
    alpha,
    lda,
    beta,
    ldb,
    ldc,
    transa: tl.constexpr,
    transb: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create masks for the current block
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Handle transposition for A
    if transa:
        a_offs = rn[:, None] * lda + rm[None, :]
    else:
        a_offs = rm[:, None] * lda + rn[None, :]
    
    # Handle transposition for B
    if transb:
        b_offs = rn[:, None] * ldb + rm[None, :]
    else:
        b_offs = rm[:, None] * ldb + rn[None, :]
    
    # Load A and B
    a = tl.load(a_ptr + a_offs, mask=(rm[:, None] < m) & (rn[None, :] < n), other=0.0)
    b = tl.load(b_ptr + b_offs, mask=(rm[:, None] < m) & (rn[None, :] < n), other=0.0)
    
    # Compute C = alpha * A + beta * B
    c = alpha * a + beta * b
    
    # Store result
    c_offs = rm[:, None] * ldc + rn[None, :]
    tl.store(c_ptr + c_offs, c, mask=(rm[:, None] < m) & (rn[None, :] < n))

@register("CUDA", "dgeam", has_backward=Autograd.disable, namespace="triton")
def dgeam(
    transa: int,
    transb: int,
    m: int,
    n: int,
    alpha: float,
    A: torch.Tensor,
    lda: int,
    beta: float,
    B: torch.Tensor,
    ldb: int,
    C: torch.Tensor,
    ldc: int,
):
    # Convert trans flags to boolean
    transa = transa != 0
    transb = transb != 0
    
    # Determine grid size
    grid = lambda META: (
        triton.cdiv(m, META['BLOCK_SIZE_M']),
        triton.cdiv(n, META['BLOCK_SIZE_N']),
    )
    
    # Launch kernel
    dgeam_kernel[grid](
        A, B, C,
        m, n,
        alpha,
        lda,
        beta,
        ldb,
        ldc,
        transa,
        transb,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=32,
    )
    return C