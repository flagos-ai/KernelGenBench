import torch
import triton
import triton.language as tl

# cuBLAS operation enums for reference
CUBLAS_OP_N = 0
CUBLAS_OP_T = 1
CUBLAS_OP_C = 2

@triton.jit
def _zgemm_strided_batched_kernel(
    A_ptr, B_ptr, C_ptr,
    m, n, k,
    lda, ldb, ldc,
    strideA, strideB, strideC,
    alpha_r, alpha_i, beta_r, beta_i,
    transa, transb,
    batchCount,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc_r = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    acc_i = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)

    # Batch base offsets in complex elements, later scaled by 2 for float64 memory
    batch_off_A = pid_b * strideA
    batch_off_B = pid_b * strideB

    # Loop over k dimension
    for t in range(0, k, BLOCK_K):
        offs_k = t + tl.arange(0, BLOCK_K)

        # Masks for valid ranges
        mask_A = (offs_m[:, None] < m) & (offs_k[None, :] < k)
        mask_B = (offs_k[:, None] < k) & (offs_n[None, :] < n)

        # Compute complex-index offsets for A
        is_transa = transa != 0
        # If transa == N: idxA = i + p*lda
        # If transa == T/C: idxA = p + i*lda
        idxA_N = offs_m[:, None] + offs_k[None, :] * lda
        idxA_T = offs_k[None, :] + offs_m[:, None] * lda
        idxA = tl.where(is_transa, idxA_T, idxA_N) + batch_off_A

        # Compute complex-index offsets for B
        is_transb = transb != 0
        # If transb == N: idxB = p + j*ldb
        # If transb == T/C: idxB = p*ldb + j
        idxB_N = offs_k[:, None] + offs_n[None, :] * ldb
        idxB_T = offs_k[:, None] * ldb + offs_n[None, :]
        idxB = tl.where(is_transb, idxB_T, idxB_N) + batch_off_B

        # Load A (real/imag) and apply conjugation if needed
        A_r = tl.load(A_ptr + 2 * idxA + 0, mask=mask_A, other=0.0)
        A_i = tl.load(A_ptr + 2 * idxA + 1, mask=mask_A, other=0.0)
        conj_a = transa == 2
        A_i = tl.where(conj_a, -A_i, A_i)

        # Load B (real/imag) and apply conjugation if needed
        B_r = tl.load(B_ptr + 2 * idxB + 0, mask=mask_B, other=0.0)
        B_i = tl.load(B_ptr + 2 * idxB + 1, mask=mask_B, other=0.0)
        conj_b = transb == 2
        B_i = tl.where(conj_b, -B_i, B_i)

        # Accumulate over BLOCK_K
        for kk in range(0, BLOCK_K):
            a_r = A_r[:, kk]
            a_i = A_i[:, kk]
            b_r = B_r[kk, :]
            b_i = B_i[kk, :]
            acc_r += a_r[:, None] * b_r[None, :] - a_i[:, None] * b_i[None, :]
            acc_i += a_r[:, None] * b_i[None, :] + a_i[:, None] * b_r[None, :]

    # Compute final output with alpha and beta
    # C indices in complex elements
    idxC = offs_m[:, None] + offs_n[None, :] * ldc + pid_b * strideC
    mask_C = (offs_m[:, None] < m) & (offs_n[None, :] < n)

    # Apply alpha
    out_r = alpha_r * acc_r - alpha_i * acc_i
    out_i = alpha_r * acc_i + alpha_i * acc_r

    # Load existing C and apply beta
    C_r = tl.load(C_ptr + 2 * idxC + 0, mask=mask_C, other=0.0)
    C_i = tl.load(C_ptr + 2 * idxC + 1, mask=mask_C, other=0.0)

    out_r = out_r + beta_r * C_r - beta_i * C_i
    out_i = out_i + beta_r * C_i + beta_i * C_r

    # Store result
    tl.store(C_ptr + 2 * idxC + 0, out_r, mask=mask_C)
    tl.store(C_ptr + 2 * idxC + 1, out_i, mask=mask_C)


def cublasZgemmStridedBatched(transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount):
    # Validate device and dtype
    assert A.is_cuda and B.is_cuda and C.is_cuda, "Inputs must be CUDA tensors."
    assert A.dtype == torch.complex128 and B.dtype == torch.complex128 and C.dtype == torch.complex128, "Inputs must be complex128."

    # Views as float64 to access real/imag parts
    A_fp64 = A.view(torch.float64)
    B_fp64 = B.view(torch.float64)
    C_fp64 = C.view(torch.float64)

    # Kernel launch configuration
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    # Convert string trans to int if needed (N->0, T->1)
    transa_int = 0 if transa == 'N' else 1 if isinstance(transa, str) else transa
    transb_int = 0 if transb == 'N' else 1 if isinstance(transb, str) else transb

    grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(n, BLOCK_N), batchCount)

    alpha_r = float(alpha.real)
    alpha_i = float(alpha.imag)
    beta_r = float(beta.real)
    beta_i = float(beta.imag)

    _zgemm_strided_batched_kernel[grid](
        A_fp64, B_fp64, C_fp64,
        m, n, k,
        lda, ldb, ldc,
        strideA, strideB, strideC,
        alpha_r, alpha_i, beta_r, beta_i,
        transa_int, transb_int,
        batchCount,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2
    )

    return C