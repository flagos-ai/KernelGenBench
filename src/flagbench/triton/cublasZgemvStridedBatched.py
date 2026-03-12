import torch
import triton
import triton.language as tl

@triton.jit
def _zgemvstridedbatched_kernel(
    A_ptr, x_ptr, y_ptr,
    alpha_r, alpha_i, beta_r, beta_i,
    m, n, lda,
    strideA, incx, stridex,
    incy, stridey,
    batchCount, trans,
    M_out, K_len,
    x_start_offset, y_start_offset,
    conj_sign,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)

    # Base pointers per batch (convert strides from complex units to float64 elements)
    A_base = A_ptr + 2 * (pid_b * strideA)
    x_base = x_ptr + 2 * (pid_b * stridex) + 2 * x_start_offset
    y_base = y_ptr + 2 * (pid_b * stridey) + 2 * y_start_offset

    # Output index range
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = rows < M_out

    acc_r = tl.zeros([BLOCK_M], dtype=tl.float64)
    acc_i = tl.zeros([BLOCK_M], dtype=tl.float64)

    # Loop over K dimension in tiles
    for k0 in range(0, K_len, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        mask_k = ks < K_len

        # Compute A offsets in complex elements depending on trans:
        # N: A(i, k) with i=rows, k=ks  => offset = i + k*lda
        # T/C: A(k, i) with k=ks, i=rows => offset = k + i*lda
        off_N = rows[:, None] + ks[None, :] * lda
        off_T = ks[:, None] + rows[None, :] * lda
        A_off = tl.where(trans == 0, off_N, off_T)

        # Load A (real & imag)
        A_ptrs_r = A_base + 2 * A_off
        A_ptrs_i = A_ptrs_r + 1
        A_r = tl.load(A_ptrs_r, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        A_i = tl.load(A_ptrs_i, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Conjugate for 'C' (conj_sign = -1.0 for C, 1.0 otherwise)
        A_i = A_i * conj_sign

        # Load x (broadcast along rows)
        x_off = ks * incx
        x_ptrs_r = x_base + 2 * x_off
        x_ptrs_i = x_ptrs_r + 1
        x_r = tl.load(x_ptrs_r, mask=mask_k, other=0.0)[None, :]
        x_i = tl.load(x_ptrs_i, mask=mask_k, other=0.0)[None, :]

        # Complex multiply A * x
        prod_r = A_r * x_r - A_i * x_i
        prod_i = A_r * x_i + A_i * x_r

        # Reduce across K tile
        acc_r += tl.sum(prod_r, axis=1)
        acc_i += tl.sum(prod_i, axis=1)

    # Compute alpha * acc
    tmp_r = alpha_r * acc_r - alpha_i * acc_i
    tmp_i = alpha_r * acc_i + alpha_i * acc_r

    # Load y_old
    y_off = rows * incy
    y_ptrs_r = y_base + 2 * y_off
    y_ptrs_i = y_ptrs_r + 1
    y_r_old = tl.load(y_ptrs_r, mask=mask_m, other=0.0)
    y_i_old = tl.load(y_ptrs_i, mask=mask_m, other=0.0)

    # Add beta * y_old
    out_r = tmp_r + (beta_r * y_r_old - beta_i * y_i_old)
    out_i = tmp_i + (beta_r * y_i_old + beta_i * y_r_old)

    # Store result
    tl.store(y_ptrs_r, out_r, mask=mask_m)
    tl.store(y_ptrs_i, out_i, mask=mask_m)


def cublasZgemvStridedBatched(trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount):
    # Map trans to cublasOperation_t
    def _to_cublas_op(t):
        if isinstance(t, int):
            return t
        t = str(t).upper()
        if t == 'N':
            return 0
        elif t == 'T':
            return 1
        elif t == 'C':
            return 2
        else:
            raise ValueError("Invalid trans value. Use 'N', 'T', or 'C'.")

    trans_op = _to_cublas_op(trans)

    # Determine output length and K length
    M_out = m if trans_op == 0 else n
    K_len = n if trans_op == 0 else m

    # Prepare views as real-imag float64 for Triton
    A_ri = torch.view_as_real(A).contiguous()
    x_ri = torch.view_as_real(x).contiguous()
    y_ri = torch.view_as_real(y).contiguous()

    # Scalars
    alpha_r = float(alpha.real)
    alpha_i = float(alpha.imag)
    beta_r = float(beta.real)
    beta_i = float(beta.imag)

    # Handle negative increments like BLAS semantics
    x_start_offset = 0
    y_start_offset = 0
    if incx < 0:
        x_start_offset = (K_len - 1) * (-incx)
    if incy < 0:
        y_start_offset = (M_out - 1) * (-incy)

    conj_sign = -1.0 if trans_op == 2 else 1.0

    BLOCK_M = 128
    BLOCK_K = 64
    # Convert string trans to int if needed (N->0, T->1)
    trans_int = 0 if trans == 'N' else 1 if isinstance(trans, str) else trans

    grid = (batchCount, triton.cdiv(M_out, BLOCK_M))

    _zgemvstridedbatched_kernel[grid](
        A_ri, x_ri, y_ri,
        alpha_r, alpha_i, beta_r, beta_i,
        m, n, lda,
        strideA, incx, stridex,
        incy, stridey,
        batchCount, trans_op,
        M_out, K_len,
        x_start_offset, y_start_offset,
        conj_sign,
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K
    )

    return y