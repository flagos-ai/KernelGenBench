from sandbox.register import register
from flagbench.dataset import Autograd
import torch
import triton
import triton.language as tl

@triton.jit
def ssyrk_kernel(
    A_ptr, C_ptr,
    n, k,
    alpha, beta,
    stride_am, stride_ak,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    UPLO: tl.constexpr, TRANS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Check if we're in the lower or upper triangle based on UPLO
    if UPLO == 0:  # CUBLAS_FILL_MODE_LOWER
        if pid_m < pid_n:
            return
    else:  # CUBLAS_FILL_MODE_UPPER
        if pid_m > pid_n:
            return
    
    # Compute block offsets
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Compute matrix multiplication
    for ki in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
        rk = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        rk_mask = rk < k
        
        if TRANS == 0:  # CUBLAS_OP_N
            a = tl.load(A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak, 
                        mask=rk_mask[None, :] & (rm[:, None] < n), other=0.0)
            b = tl.load(A_ptr + rn[None, :] * stride_am + rk[:, None] * stride_ak, 
                        mask=rk_mask[:, None] & (rn[None, :] < n), other=0.0)
        else:  # CUBLAS_OP_T
            a = tl.load(A_ptr + rk[:, None] * stride_am + rm[None, :] * stride_ak, 
                        mask=rk_mask[:, None] & (rm[None, :] < n), other=0.0)
            b = tl.load(A_ptr + rk[:, None] * stride_am + rn[None, :] * stride_ak, 
                        mask=rk_mask[:, None] & (rn[None, :] < n), other=0.0)
        
        acc += tl.dot(a, b, allow_tf32=True)
    
    # Load current C value
    c = tl.load(C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn, 
                mask=(rm[:, None] < n) & (rn[None, :] < n), other=0.0)
    
    # Update C = alpha * A @ A^T + beta * C
    c = alpha * acc + beta * c
    
    # Store result
    tl.store(C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn, c, 
             mask=(rm[:, None] < n) & (rn[None, :] < n))

@register("CUDA", "ssyrk", has_backward=Autograd.disable, namespace="triton")
def ssyrk(
    uplo: int,  # 0 for lower, 1 for upper
    trans: int,  # 0 for no transpose, 1 for transpose
    n: int,
    k: int,
    alpha: float,
    A: torch.Tensor,
    lda: int,
    beta: float,
    C: torch.Tensor,
    ldc: int,
):
    # Validate inputs
    assert A.is_cuda and C.is_cuda, "Inputs must be on CUDA device"
    assert A.dtype == torch.float32 and C.dtype == torch.float32, "Only float32 supported"
    
    # Determine matrix dimensions based on transpose
    if trans == 0:  # CUBLAS_OP_N
        assert A.shape == (n, k), f"Expected A shape ({n}, {k}), got {A.shape}"
    else:  # CUBLAS_OP_T
        assert A.shape == (k, n), f"Expected A shape ({k}, {n}), got {A.shape}"
    
    assert C.shape == (n, n), f"Expected C shape ({n}, {n}), got {C.shape}"
    
    # Launch kernel
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    grid = lambda META: (triton.cdiv(n, META['BLOCK_SIZE_M']), triton.cdiv(n, META['BLOCK_SIZE_N']))
    
    ssyrk_kernel[grid](
        A, C,
        n, k,
        alpha, beta,
        A.stride(0), A.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        uplo, trans,
    )
    
    return C