import torch
import triton
import triton.language as tl


@triton.jit
def _sgemm_strided_batched_kernel(
    A_ptr, B_ptr, C_ptr,
    alpha, beta,
    m, n, k,
    lda, ldb, ldc,
    strideA, strideB, strideC,
    batchCount, transa, transb,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Base pointers for this batch
    A_batch = A_ptr + pid_b * strideA
    B_batch = B_ptr + pid_b * strideB
    C_batch = C_ptr + pid_b * strideC

    # Determine effective strides based on transposition flags (0: N, 1: T)
    # For A:
    #   if transa == 0: opA(i,k) = A(i,k) => row stride = 1, k stride = lda
    #   if transa == 1: opA(i,k) = A(k,i) => row stride = lda, k stride = 1
    a_rs = tl.where(transa == 0, 1, lda)
    a_ks = tl.where(transa == 0, lda, 1)

    # For B:
    #   if transb == 0: opB(k,j) = B(k,j) => k stride = 1, col stride = ldb
    #   if transb == 1: opB(k,j) = B(j,k) => k stride = ldb, col stride = 1
    b_ks = tl.where(transb == 0, 1, ldb)
    b_ns = tl.where(transb == 0, ldb, 1)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    k0 = 0
    while k0 < k:
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Compute pointers for tiles
        A_tile_ptr = A_batch + (offs_m[:, None] * a_rs) + (offs_k[None, :] * a_ks)
        B_tile_ptr = B_batch + (offs_k[:, None] * b_ks) + (offs_n[None, :] * b_ns)

        # Masks for loads
        a_mask = (offs_m[:, None] < m) & (offs_k[None, :] < k)
        b_mask = (offs_k[:, None] < k) & (offs_n[None, :] < n)

        # Loads
        A_tile = tl.load(A_tile_ptr, mask=a_mask, other=0.0)
        B_tile = tl.load(B_tile_ptr, mask=b_mask, other=0.0)

        # FMA accumulate
        acc += tl.dot(A_tile, B_tile)

        k0 += BLOCK_K

    # Compute C pointers and store with alpha/beta scaling
    C_tile_ptr = C_batch + (offs_m[:, None] * 1) + (offs_n[None, :] * ldc)
    c_mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)

    if beta != 0.0:
        C_prev = tl.load(C_tile_ptr, mask=c_mask, other=0.0)
        C_out = acc * alpha + C_prev * beta
    else:
        C_out = acc * alpha

    tl.store(C_tile_ptr, C_out, mask=c_mask)


def cublasSgemmStridedBatched(transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount):
    # Basic checks to ensure inputs are valid
    assert isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor) and isinstance(C, torch.Tensor), "A, B, C must be torch.Tensors"
    assert A.is_cuda and B.is_cuda and C.is_cuda, "A, B, C must be CUDA tensors"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, "Only float32 supported"
    assert m >= 0 and n >= 0 and k >= 0 and batchCount >= 0, "Dimensions must be non-negative integers"

    if batchCount == 0 or m == 0 or n == 0:
        return C
    if k == 0:
        if beta == 0.0:
            C.zero_()
        else:
            C.mul_(beta)
        return C

    # Tile sizes: can be tuned
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    # Convert string trans to int if needed (N->0, T->1)
    transa_int = 0 if transa == 'N' else 1 if isinstance(transa, str) else transa
    transb_int = 0 if transb == 'N' else 1 if isinstance(transb, str) else transb

    grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(n, BLOCK_N), batchCount)

    _sgemm_strided_batched_kernel[grid](
        A, B, C,
        float(alpha), float(beta),
        m, n, k,
        lda, ldb, ldc,
        strideA, strideB, strideC,
        batchCount, transa_int, transb_int,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3
    )
    return C