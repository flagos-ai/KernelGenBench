import torch
import triton
import triton.language as tl

@triton.jit
def _cgemm_strided_batched_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    lda, ldb, ldc,
    strideA, strideB, strideC,
    alpha_r, alpha_i, beta_r, beta_i,
    transa, transb,
    NK: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Convert leading dimensions and strides from complex elements to float32 elements (interleaved real/imag)
    lda_f = lda * 2
    ldb_f = ldb * 2
    ldc_f = ldc * 2
    strideA_f = strideA * 2
    strideB_f = strideB * 2
    strideC_f = strideC * 2

    # Base pointers for this batch (in float32 elements)
    A_batch_ptr = A_ptr + pid_b * strideA_f
    B_batch_ptr = B_ptr + pid_b * strideB_f
    C_batch_ptr = C_ptr + pid_b * strideC_f

    acc_r = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_i = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Masks for output bounds
    omask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    for tile_k in range(0, NK):
        k0 = tile_k * BLOCK_K
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Masks for A and B tiles
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # Compute complex-element indices for A and B according to column-major with possible transpose
        # A index (complex elements)
        if transa == 0:
            # op(A) = A: index = k*lda + m
            a_idx_c = offs_k[None, :] * lda + offs_m[:, None]
        else:
            # op(A) = A^T: index = m*lda + k
            a_idx_c = offs_m[:, None] * lda + offs_k[None, :]
        # B index (complex elements)
        if transb == 0:
            # op(B) = B: index = n*ldb + k
            b_idx_c = offs_n[None, :] * ldb + offs_k[:, None]
        else:
            # op(B) = B^T: index = k*ldb + n
            b_idx_c = offs_k[:, None] * ldb + offs_n[None, :]

        # Convert complex-element indices to float32-element indices (interleaved real/imag)
        a_idx_f = a_idx_c * 2
        b_idx_f = b_idx_c * 2

        # Load A tile (real and imag)
        A_r = tl.load(A_batch_ptr + a_idx_f + 0, mask=mask_a, other=0.0)
        A_i = tl.load(A_batch_ptr + a_idx_f + 1, mask=mask_a, other=0.0)
        # Load B tile (real and imag)
        B_r = tl.load(B_batch_ptr + b_idx_f + 0, mask=mask_b, other=0.0)
        B_i = tl.load(B_batch_ptr + b_idx_f + 1, mask=mask_b, other=0.0)

        # Block GEMM accumulation (complex)
        # acc_r += A_r @ B_r - A_i @ B_i
        # acc_i += A_r @ B_i + A_i @ B_r
        for kk in range(0, BLOCK_K):
            a_r = A_r[:, kk]
            a_i = A_i[:, kk]
            b_r = B_r[kk, :]
            b_i = B_i[kk, :]
            acc_r += a_r[:, None] * b_r[None, :] - a_i[:, None] * b_i[None, :]
            acc_i += a_r[:, None] * b_i[None, :] + a_i[:, None] * b_r[None, :]

    # Apply alpha to accumulated result
    out_r = alpha_r * acc_r - alpha_i * acc_i
    out_i = alpha_r * acc_i + alpha_i * acc_r

    # Load C, apply beta, and write result: C = alpha * A*B + beta * C
    # C index (complex elements): n*ldc + m
    c_idx_c = offs_n[None, :] * ldc + offs_m[:, None]
    c_idx_f = c_idx_c * 2
    C_r = tl.load(C_batch_ptr + c_idx_f + 0, mask=omask, other=0.0)
    C_i = tl.load(C_batch_ptr + c_idx_f + 1, mask=omask, other=0.0)

    out_r = out_r + (beta_r * C_r - beta_i * C_i)
    out_i = out_i + (beta_r * C_i + beta_i * C_r)

    tl.store(C_batch_ptr + c_idx_f + 0, out_r, mask=omask)
    tl.store(C_batch_ptr + c_idx_f + 1, out_i, mask=omask)


def cublasCgemmStridedBatched(transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount):
    # Validate inputs
    assert isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor) and isinstance(C, torch.Tensor)
    assert A.is_cuda and B.is_cuda and C.is_cuda
    assert A.dtype == torch.complex64 and B.dtype == torch.complex64 and C.dtype == torch.complex64

    # Convert to float32 view for interleaved real/imag memory access
    A_f = A.view(torch.float32)
    B_f = B.view(torch.float32)
    C_f = C.view(torch.float32)

    # Scalars
    alpha_r = float(alpha.real)
    alpha_i = float(alpha.imag)
    beta_r = float(beta.real)
    beta_i = float(beta.imag)

    # Tiling parameters
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    # Number of K tiles
    NK = (k + BLOCK_K - 1) // BLOCK_K

    # Convert string trans to int if needed (N->0, T->1)
    transa_int = 0 if transa == 'N' else 1 if isinstance(transa, str) else transa
    transb_int = 0 if transb == 'N' else 1 if isinstance(transb, str) else transb

    grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(n, BLOCK_N), batchCount)

    _cgemm_strided_batched_kernel[grid](
        A_f, B_f, C_f,
        m, n, k,
        lda, ldb, ldc,
        strideA, strideB, strideC,
        alpha_r, alpha_i, beta_r, beta_i,
        transa, transb,
        NK=NK,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2
    )
    return C