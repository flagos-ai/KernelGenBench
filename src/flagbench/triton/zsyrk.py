from sandbox.register import register
from flagbench.dataset import Autograd
import torch
import triton
import triton.language as tl

@triton.jit
def zsyrk_kernel(
    A_ptr, C_ptr,
    n, k,
    stride_am, stride_ak,
    stride_cm, stride_cn,
    alpha_real, alpha_imag,
    beta_real, beta_imag,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create block pointers
    A_block_ptr = tl.make_block_ptr(
        base=A_ptr,
        shape=(n, k) if TRANS == 0 else (k, n),
        strides=(stride_am, stride_ak) if TRANS == 0 else (stride_ak, stride_am),
        offsets=(pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0)
    )
    
    C_block_ptr = tl.make_block_ptr(
        base=C_ptr,
        shape=(n, n),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0)
    )
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float64)
    
    # Compute matrix multiplication
    for i in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
        a = tl.load(A_block_ptr, boundary_check=(0, 1))
        if TRANS == 0:
            b = tl.load(A_block_ptr, boundary_check=(0, 1))
        else:
            b = tl.trans(tl.load(A_block_ptr, boundary_check=(0, 1)))
        
        # Complex multiplication
        ar = a.real
        ai = a.imag
        br = b.real
        bi = b.imag
        acc += (ar * br - ai * bi) + (ar * bi + ai * br) * 1j
        
        A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_SIZE_K))
    
    # Load current C value
    c = tl.load(C_block_ptr, boundary_check=(0, 1))
    
    # Apply alpha and beta
    alpha = tl.complex(alpha_real, alpha_imag)
    beta = tl.complex(beta_real, beta_imag)
    c = alpha * acc + beta * c
    
    # Symmetric update
    if UPLO == 0:  # CUBLAS_FILL_MODE_LOWER
        if pid_m >= pid_n:
            tl.store(C_block_ptr, c, boundary_check=(0, 1))
    else:  # CUBLAS_FILL_MODE_UPPER
        if pid_m <= pid_n:
            tl.store(C_block_ptr, c, boundary_check=(0, 1))

@register("CUDA", "zsyrk", has_backward=Autograd.disable, namespace="triton")
def zsyrk(
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
    # Convert complex to real/imag parts
    alpha_real, alpha_imag = alpha.real, alpha.imag
    beta_real, beta_imag = beta.real, beta.imag
    
    # Grid and block sizes
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    grid = lambda META: (
        triton.cdiv(n, META['BLOCK_SIZE_M']),
        triton.cdiv(n, META['BLOCK_SIZE_N']),
    )
    
    # Launch kernel
    zsyrk_kernel[grid](
        A, C,
        n, k,
        A.stride(0), A.stride(1),
        C.stride(0), C.stride(1),
        alpha_real, alpha_imag,
        beta_real, beta_imag,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        uplo, trans,
    )
    
    return C