import torch
import triton
import triton.language as tl

@triton.jit
def _dgemm_strided_batched_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    lda, ldb, ldc,
    strideA, strideB, strideC,
    transa, transb,
    alpha, beta,
    batch_count,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_b = tl.program_id(2)

    # Offsets for C tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Base pointers for this batch
    A_batch_ptr = A_ptr + pid_b * strideA
    B_batch_ptr = B_ptr + pid_b * strideB
    C_batch_ptr = C_ptr + pid_b * strideC

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)

    # Loop over K dimension
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k0 * BLOCK_K + tl.arange(0, BLOCK_K)

        # Compute A tile offsets based on transa
        if transa == 0:  # CUBLAS_OP_N
            # A[r, c] with r in M, c in K -> offset = r + c * lda
            a_offsets = (offs_m[:, None]) + (k_offs[None, :] * lda)
        else:  # CUBLAS_OP_T (or others treated as transpose)
            # A^T[r, c] = A[c, r] -> offset = c + r * lda
            a_offsets = (offs_m[:, None] * lda) + (k_offs[None, :])

        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        A_tile = tl.load(A_batch_ptr + a_offsets, mask=a_mask, other=0.0)

        # Compute B tile offsets based on transb
        if transb == 0:  # CUBLAS_OP_N
            # B[r, c] with r in K, c in N -> offset = r + c * ldb
            b_offsets = (k_offs[:, None]) + (offs_n[None, :] * ldb)
        else:  # CUBLAS_OP_T (or others treated as transpose)
            # B^T[r, c] = B[c, r] -> offset = c + r * ldb
            b_offsets = (k_offs[:, None] * ldb) + (offs_n[None, :])

        b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        B_tile = tl.load(B_batch_ptr + b_offsets, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(A_tile, B_tile)

    # Write back to C: C = alpha * acc + beta * C
    c_offsets = (offs_m[:, None]) + (offs_n[None, :] * ldc)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    C_old = tl.load(C_batch_ptr + c_offsets, mask=c_mask, other=0.0)
    C_res = alpha * acc + beta * C_old
    tl.store(C_batch_ptr + c_offsets, C_res, mask=c_mask)


def cublasDgemmStridedBatched(transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount):
    # Validate inputs
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("A, B, and C must be CUDA tensors")
    if not (A.dtype == torch.float64 and B.dtype == torch.float64 and C.dtype == torch.float64):
        raise ValueError("A, B, and C must be torch.float64 tensors")
    if batchCount == 0 or m == 0 or n == 0:
        return C

    # Tile sizes (tuned conservatively for fp64)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    # Convert string trans to int if needed (N->0, T->1)
    transa_int = 0 if transa == 'N' else 1 if isinstance(transa, str) else transa
    transb_int = 0 if transb == 'N' else 1 if isinstance(transb, str) else transb

    grid = (triton.cdiv(n, BLOCK_N), triton.cdiv(m, BLOCK_M), batchCount)

    # Choose num_warps based on tile sizes
    num_warps = 4
    num_stages = 2

    _dgemm_strided_batched_kernel[grid](
        A, B, C,
        m, n, k,
        lda, ldb, ldc,
        strideA, strideB, strideC,
        transa_int, transb_int,
        float(alpha), float(beta),
        batchCount,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return C