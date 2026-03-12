import math
import torch
import triton
import triton.language as tl


@triton.jit
def sgemvstridedbatched(
    A_ptr, x_ptr, y_ptr,
    m, n, lda,
    strideA, stridex, stridey,
    incx, incy,
    alpha, beta,
    trans, batchCount,
    NUM_K_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)

    # Determine output length (L) and reduction length (K)
    L = tl.where(trans == 0, m, n)
    K = tl.where(trans == 0, n, m)

    # Batch base pointers (element-wise indexing)
    A_b = A_ptr + pid_b * strideA
    x_b = x_ptr + pid_b * stridex
    y_b = y_ptr + pid_b * stridey

    # Row indices for this program
    row_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_row = row_idx < L

    # Initialize accumulator
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Loop over K dimension in tiles
    for t in range(0, NUM_K_TILES):
        k_start = t * BLOCK_K
        k_idx = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_idx < K

        # Compute A tile offsets based on trans
        # trans == 0: y[L=m] = A[m x n] @ x[n]
        # A element offset (r, k) -> r + k*lda
        # trans == 1: y[L=n] = A^T[n x m] @ x[m]
        # A element offset (k, r) -> k + r*lda
        a_mask = mask_row[:, None] & mask_k[None, :]
        a_offsets_trans0 = row_idx[:, None] + k_idx[None, :] * lda
        a_offsets_trans1 = k_idx[None, :] + row_idx[:, None] * lda
        a_offsets = tl.where(trans == 0, a_offsets_trans0, a_offsets_trans1)

        # Load A tile
        A_tile = tl.load(A_b + a_offsets, mask=a_mask, other=0.0)

        # Load x chunk
        x_offsets = k_idx * incx
        x_vals = tl.load(x_b + x_offsets, mask=mask_k, other=0.0)

        # FMA reduce across K
        acc += tl.sum(A_tile * x_vals[None, :], axis=1)

    # Load y, apply alpha and beta, and store
    y_offsets = row_idx * incy
    y_old = tl.load(y_b + y_offsets, mask=mask_row, other=0.0)
    y_new = alpha * acc + beta * y_old
    tl.store(y_b + y_offsets, y_new, mask=mask_row)


def cublasSgemvStridedBatched(trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount):
    if not A.is_cuda or not x.is_cuda or not y.is_cuda:
        raise ValueError("A, x, and y must be CUDA tensors")
    if A.dtype != torch.float32 or x.dtype != torch.float32 or y.dtype != torch.float32:
        raise TypeError("A, x, and y must be torch.float32 tensors")
    if not A.is_contiguous() or not x.is_contiguous() or not y.is_contiguous():
        raise ValueError("A, x, and y must be contiguous tensors")
    if A.dim() != 3 or x.dim() != 2 or y.dim() != 2:
        raise ValueError("Expected A: (batch, ?, ?), x: (batch, ?), y: (batch, ?)")
    if A.size(0) < batchCount or x.size(0) < batchCount or y.size(0) < batchCount:
        raise ValueError("Batch sizes of A, x, y must be >= batchCount")
    if trans not in (0, 1):
        raise ValueError("Only CUBLAS_OP_N (0) and CUBLAS_OP_T (1) are supported")
    if incx <= 0 or incy <= 0:
        raise ValueError("Only positive increments incx and incy are supported")

    device = A.device
    # Determine output length (L) and reduction length (K)
    L = m if trans == 0 else n
    K = n if trans == 0 else m

    # Triton block sizes
    BLOCK_M = 128
    BLOCK_K = 128
    num_k_tiles = (K + BLOCK_K - 1) // BLOCK_K

    # Convert string trans to int if needed (N->0, T->1)
    trans_int = 0 if trans == 'N' else 1 if isinstance(trans, str) else trans

    grid = (triton.cdiv(L, BLOCK_M), batchCount)
    sgemvstridedbatched[grid](
        A, x, y,
        m, n, lda,
        strideA, stridex, stridey,
        incx, incy,
        float(alpha), float(beta),
        trans, batchCount,
        NUM_K_TILES=num_k_tiles,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2
    )
    return y