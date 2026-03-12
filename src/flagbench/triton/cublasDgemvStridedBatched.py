import torch
import triton
import triton.language as tl
import math


@triton.jit
def _dgemvstridedbatched_kernel(
    A_ptr, x_ptr, y_ptr,
    alpha, beta,
    m, n,
    lda,
    strideA,
    stridex,
    stridey,
    incx,
    incy,
    len_x,
    len_y,
    x_start,
    y_start,
    TRANS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TILES_K: tl.constexpr,
):
    batch_id = tl.program_id(1)
    row_id = tl.program_id(0)

    # Scalars as fp64
    alpha_val = tl.full((), alpha, tl.float64)
    beta_val = tl.full((), beta, tl.float64)

    # Base offsets per batch
    baseA = batch_id * strideA
    baseX = batch_id * stridex + x_start
    baseY = batch_id * stridey + y_start

    # Compute dot-product for this output element
    acc = tl.full((), 0.0, tl.float64)

    for t in range(TILES_K):
        k = t * BLOCK_K + tl.arange(0, BLOCK_K)
        mask = k < len_x

        if TRANS == 0:
            # y[row] = sum_{c=0..n-1} A[row + c*lda] * x[c]
            a_ptrs = A_ptr + baseA + row_id + k * lda
        else:
            # y[col] = sum_{r=0..m-1} A[r + col*lda] * x[r]
            a_ptrs = A_ptr + baseA + k + row_id * lda

        x_ptrs = x_ptr + baseX + k * incx

        a_vals = tl.load(a_ptrs, mask=mask, other=0.0)
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)

        acc += tl.sum(a_vals * x_vals, axis=0)

    # Write result: y[row] = alpha * acc + beta * y[row]
    y_ptrs = y_ptr + baseY + row_id * incy
    y_old = tl.load(y_ptrs)
    y_new = alpha_val * acc + beta_val * y_old
    tl.store(y_ptrs, y_new)


def cublasDgemvStridedBatched(trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount):
    # Validate tensors
    if not (A.is_cuda and x.is_cuda and y.is_cuda):
        raise ValueError("A, x, and y must be CUDA tensors")
    if not (A.dtype == torch.float64 and x.dtype == torch.float64 and y.dtype == torch.float64):
        raise ValueError("A, x, and y must be dtype torch.float64")

    # Flatten to 1D views to work with raw pointers and element strides
    A_flat = A.view(-1)
    x_flat = x.view(-1)
    y_flat = y.view(-1)

    # Interpret trans (0: N, 1: T, 2: C)
    TRANS = 0 if trans == 0 else 1

    # Compute vector lengths under column-major semantics
    len_x = n if TRANS == 0 else m
    len_y = m if TRANS == 0 else n

    # Handle negative increments by adjusting starting offsets
    x_start = 0
    if incx < 0:
        x_start = (-incx) * (len_x - 1)
    y_start = 0
    if incy < 0:
        y_start = (-incy) * (len_y - 1)

    # Heuristic block size along reduction dimension
    BLOCK_K = 128
    TILES_K = math.ceil(len_x / BLOCK_K)

    # Launch grid: one program per output element per batch
    # Convert string trans to int if needed (N->0, T->1)
    trans_int = 0 if trans == 'N' else 1 if isinstance(trans, str) else trans

    grid = (len_y, batchCount)

    _dgemvstridedbatched_kernel[grid](
        A_flat, x_flat, y_flat,
        float(alpha), float(beta),
        m, n,
        lda,
        strideA,
        stridex,
        stridey,
        incx,
        incy,
        len_x,
        len_y,
        x_start,
        y_start,
        TRANS=TRANS,
        BLOCK_K=BLOCK_K,
        TILES_K=TILES_K,
        num_warps=4,
        num_stages=2,
    )

    return y