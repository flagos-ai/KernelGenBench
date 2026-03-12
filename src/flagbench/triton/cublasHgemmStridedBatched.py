import torch
import triton
import triton.language as tl


@triton.jit
def _hgemm_strided_batched_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    lda, ldb, ldc,
    strideA, strideB, strideC,
    alpha, beta,
    TA: tl.constexpr, TB: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_b = tl.program_id(axis=2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Batch base pointers
    A_batch = A_ptr + pid_b * strideA
    B_batch = B_ptr + pid_b * strideB
    C_batch = C_ptr + pid_b * strideC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_iter = 0
    while k_iter < K:
        if TA == 0:
            # A(i, p) at i + p*lda
            a_ptrs = A_batch + (offs_m[:, None] + (k_iter + offs_k[None, :]) * lda)
        else:
            # opA(i, p) = A^T(i, p) = A(p, i) => p + i*lda
            a_ptrs = A_batch + ((k_iter + offs_k)[None, :] + offs_m[:, None] * lda)
        if TB == 0:
            # B(p, j) at p + j*ldb
            b_ptrs = B_batch + ((k_iter + offs_k)[:, None] + offs_n[None, :] * ldb)
        else:
            # opB(p, j) = B^T(p, j) = B(j, p) => j + p*ldb
            b_ptrs = B_batch + (offs_n[None, :] + (k_iter + offs_k)[:, None] * ldb)

        a_mask = (offs_m[:, None] < M) & (k_iter + offs_k[None, :] < K)
        b_mask = (k_iter + offs_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float16)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float16)

        acc += tl.dot(a, b)

        k_iter += BLOCK_K

    c_ptrs = C_batch + (offs_m[:, None] + offs_n[None, :] * ldc)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_prev = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)

    c_out = alpha * acc + beta * c_prev
    tl.store(c_ptrs, c_out.to(tl.float16), mask=c_mask)


def cublasHgemmStridedBatched(transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount):
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("A, B, C must be CUDA tensors")
    if A.dtype != torch.float16 or B.dtype != torch.float16 or C.dtype != torch.float16:
        raise ValueError("A, B, C must be torch.float16 (half) tensors")

    # Early exit for empty computations
    if m == 0 or n == 0 or batchCount == 0:
        return C

    # Convert string trans to int if needed (N->0, T->1)
    transa_int = 0 if transa == 'N' else 1 if isinstance(transa, str) else transa
    transb_int = 0 if transb == 'N' else 1 if isinstance(transb, str) else transb

    # Map transposition flags: 0 = no transpose, 1 or 2 = transpose
    TA = 0 if transa_int == 0 else 1
    TB = 0 if transb_int == 0 else 1

    # Convert scalars to float32 for accumulation
    alpha_f = float(alpha)
    beta_f = float(beta)

    # Choose block sizes
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (
        triton.cdiv(m, BLOCK_M),
        triton.cdiv(n, BLOCK_N),
        batchCount
    )

    _hgemm_strided_batched_kernel[grid](
        A, B, C,
        m, n, k,
        lda, ldb, ldc,
        strideA, strideB, strideC,
        alpha_f, beta_f,
        TA=TA, TB=TB,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2
    )
    return C