from sandbox.register import register
from flagbench.dataset import Autograd
import torch
import triton
import triton.language as tl

@triton.jit
def zgemm_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is the stride of the A matrix
    # in the m dimension (outer dimension of the matrix multiplication)
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    alpha_real: float, alpha_imag: float,
    beta_real: float, beta_imag: float,
    # Whether to transpose A and B
    TRANSPOSE_A: tl.constexpr, TRANSPOSE_B: tl.constexpr,
):
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(K, M) if TRANSPOSE_A else (M, K),
        strides=(stride_ak, stride_am) if TRANSPOSE_A else (stride_am, stride_ak),
        offsets=(0, pid_m * BLOCK_SIZE_M),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_M),
        order=(1, 0) if TRANSPOSE_A else (0, 1)
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(N, K) if TRANSPOSE_B else (K, N),
        strides=(stride_bn, stride_bk) if TRANSPOSE_B else (stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K),
        order=(1, 0) if TRANSPOSE_B else (0, 1)
    )
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(0, 1)
    )

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b, out_dtype=tl.float32)
        # Advance the block pointers
        a_block_ptr = tl.advance(a_block_ptr, (BLOCK_SIZE_K, 0))
        b_block_ptr = tl.advance(b_block_ptr, (0, BLOCK_SIZE_K))

    # Convert accumulator to complex64 for final operations
    accumulator = accumulator.to(tl.complex64)
    alpha = tl.complex(alpha_real, alpha_imag)
    beta = tl.complex(beta_real, beta_imag)
    
    # Load current C block
    c = tl.load(c_block_ptr, boundary_check=(0, 1))
    # C = alpha * (A @ B) + beta * C
    c = alpha * accumulator + beta * c
    # Store the result
    tl.store(c_block_ptr, c.to(c_ptr.dtype.element_ty), boundary_check=(0, 1))

@register("CUDA", "zgemm", has_backward=Autograd.disable, namespace="triton")
def zgemm(
    transa: int,  # dimension/flag parameter
    transb: int,  # dimension/flag parameter
    m: int,  # dimension/flag parameter
    n: int,  # dimension/flag parameter
    k: int,  # dimension/flag parameter
    alpha: complex,  # scalar parameter
    A: torch.Tensor,  # input tensor
    lda: int,  # dimension/flag parameter
    B: torch.Tensor,  # input tensor
    ldb: int,  # dimension/flag parameter
    beta: complex,  # scalar parameter
    C: torch.Tensor,  # input/output tensor (in-place)
    ldc: int,  # dimension/flag parameter
):
    # Check matrix dimensions
    assert A.shape == (m, k) if transa == 0 else (k, m)
    assert B.shape == (k, n) if transb == 0 else (n, k)
    assert C.shape == (m, n)
    
    # Convert alpha and beta to real/imag parts
    alpha_real, alpha_imag = alpha.real, alpha.imag
    beta_real, beta_imag = beta.real, beta.imag
    
    # Grid and kernel configuration
    grid = lambda META: (triton.cdiv(m, META['BLOCK_SIZE_M']) * triton.cdiv(n, META['BLOCK_SIZE_N']),)
    
    # Launch kernel
    zgemm_kernel[grid](
        A, B, C,
        m, n, k,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32,
        alpha_real=alpha_real, alpha_imag=alpha_imag,
        beta_real=beta_real, beta_imag=beta_imag,
        TRANSPOSE_A=transa != 0, TRANSPOSE_B=transb != 0,
    )
    return C