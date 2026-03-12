import torch
import triton
import triton.language as tl
import math


@triton.jit
def _cgemv_strided_batched_kernel(
    A_ptr, x_ptr, y_ptr,
    m, n, lda,
    strideA, incx, stridex,
    incy, stridey,
    batchCount,
    trans_op,  # 0: N, 1: T, 2: C
    out_dim, K,  # computed on host: out_dim = m if trans=='N' else n; K = n if trans=='N' else m
    alpha_re, alpha_im,
    beta_re, beta_im,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_block = tl.program_id(1)

    # Offsets within output vector
    i_offsets = pid_block * BLOCK_M + tl.arange(0, BLOCK_M)
    i_mask = i_offsets < out_dim

    # Base offsets (in complex elements)
    baseA_cplx = pid_batch * strideA
    baseX_cplx = pid_batch * stridex
    baseY_cplx = pid_batch * stridey

    # Convert to float32 element offsets (2 floats per complex)
    baseA = 2 * baseA_cplx
    baseX = 2 * baseX_cplx
    baseY = 2 * baseY_cplx

    # Accumulators for complex result per output index
    acc_re = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_im = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Reduction loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K
        mask_mk = i_mask[:, None] & k_mask[None, :]

        # Compute A indices in column-major addressing:
        # 'N': A[row=i, col=k] -> offset = i + k*lda
        # 'T'/'C': A[row=k, col=i] -> offset = k + i*lda
        offA_N_cplx = baseA_cplx + i_offsets[:, None] + k_offsets[None, :] * lda
        offA_T_cplx = baseA_cplx + k_offsets[None, :] + i_offsets[:, None] * lda
        use_T = trans_op != 0
        offA_cplx = tl.where(use_T, offA_T_cplx, offA_N_cplx)
        offA = 2 * offA_cplx

        # Load A tile (complex)
        A_re = tl.load(A_ptr + offA + 0, mask=mask_mk, other=0.0)
        A_im = tl.load(A_ptr + offA + 1, mask=mask_mk, other=0.0)

        # Load x chunk (complex), shared across i
        offX_cplx = baseX_cplx + k_offsets * incx
        offX = 2 * offX_cplx
        x_re = tl.load(x_ptr + offX + 0, mask=k_mask, other=0.0)[None, :]
        x_im = tl.load(x_ptr + offX + 1, mask=k_mask, other=0.0)[None, :]

        # Complex multiply and accumulate
        # For 'C': conj(A) * x; else: A * x
        # A*x:
        re_nt = A_re * x_re - A_im * x_im
        im_nt = A_re * x_im + A_im * x_re
        # conj(A)*x:
        re_c = A_re * x_re + A_im * x_im
        im_c = A_re * x_im - A_im * x_re

        is_conj = trans_op == 2
        re_mk = tl.where(is_conj, re_c, re_nt)
        im_mk = tl.where(is_conj, im_c, im_nt)

        # Reduce over K
        s_re = tl.sum(re_mk, axis=1)
        s_im = tl.sum(im_mk, axis=1)

        acc_re += s_re
        acc_im += s_im

    # Scale by alpha
    acc_re2 = alpha_re * acc_re - alpha_im * acc_im
    acc_im2 = alpha_re * acc_im + alpha_im * acc_re

    # Load y, apply beta and store
    offY_cplx = baseY_cplx + i_offsets * incy
    offY = 2 * offY_cplx
    y_re = tl.load(y_ptr + offY + 0, mask=i_mask, other=0.0)
    y_im = tl.load(y_ptr + offY + 1, mask=i_mask, other=0.0)

    out_re = acc_re2 + (beta_re * y_re - beta_im * y_im)
    out_im = acc_im2 + (beta_re * y_im + beta_im * y_re)

    tl.store(y_ptr + offY + 0, out_re, mask=i_mask)
    tl.store(y_ptr + offY + 1, out_im, mask=i_mask)


def cublasCgemvStridedBatched(trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount):
    # Map trans to operation code
    def _to_op(t):
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

    trans_op = _to_op(trans)

    # Compute output length and reduction length
    out_dim = m if trans_op == 0 else n
    K = n if trans_op == 0 else m

    # Scalars
    alpha_re = float(alpha.real)
    alpha_im = float(alpha.imag)
    beta_re = float(beta.real)
    beta_im = float(beta.imag)

    # Launch configuration
    BLOCK_M = 128
    BLOCK_K = 128
    # Convert string trans to int if needed (N->0, T->1)
    trans_int = 0 if trans == 'N' else 1 if isinstance(trans, str) else trans

    grid = (batchCount, triton.cdiv(out_dim, BLOCK_M))

    _cgemv_strided_batched_kernel[grid](
        A, x, y,
        m, n, lda,
        strideA, incx, stridex,
        incy, stridey,
        batchCount,
        trans_op,
        out_dim, K,
        alpha_re, alpha_im,
        beta_re, beta_im,
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2
    )
    return y