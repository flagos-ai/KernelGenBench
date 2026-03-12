import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
import torch
from flagbench.dataset.baseline.cublas_ctypes.cublasDgemmStridedBatched import cublasDgemmStridedBatched as baseline_cublasDgemmStridedBatched


def _trans_to_int(trans_str):
    """Convert transpose string to cuBLAS integer: 'N'->0, 'T'->1"""
    return 0 if trans_str == 'N' else 1

@label("cublasDgemmStridedBatched")
@parametrize("M, N, K", [(16, 16, 16), (128, 256, 64)])
@parametrize("alpha, beta", [(1.0, 0.0), (0.5, 0.5)])
@parametrize("transa, transb", [("N", "N"), ("T", "T")])
@parametrize("batchCount", [1, 4])
@parametrize("dtype", [torch.float64])
def test_accuracy_cublasDgemmStridedBatched(M, N, K, alpha, beta, transa, transb, batchCount, dtype):
    A_shape = (batchCount, K, M) if transa == 'T' else (batchCount, M, K)
    B_shape = (batchCount, N, K) if transb == 'T' else (batchCount, K, N)
    A = torch.randn(A_shape, dtype=dtype, device='cuda')
    B = torch.randn(B_shape, dtype=dtype, device='cuda')
    C = torch.randn(batchCount, M, N, dtype=dtype, device='cuda')
    strideA = A.shape[1] * A.shape[2]
    strideB = B.shape[1] * B.shape[2]
    strideC = M * N
    lda = A.shape[1]  # Leading dimension is the first dimension (rows in column-major)
    ldb = B.shape[1]
    ldc = M  # Leading dimension of C is M (rows)
    C_ref = C.clone()
    C_act = C.clone()
    # Convert trans strings to integers for baseline (cuBLAS C API expects int)
    transa_int = _trans_to_int(transa)
    transb_int = _trans_to_int(transb)
    ref_out = baseline_cublasDgemmStridedBatched(transa_int, transb_int, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C_ref, ldc, strideC, batchCount)
    act_out = flagbench.triton.cublasDgemmStridedBatched(transa, transb, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C_act, ldc, strideC, batchCount)
    assert_close(act_out, ref_out, dtype, reduce_dim=K)

    # ========================================
    # Performance Test
    # ========================================
    import triton
    from sandbox.utils.accuracy_utils import CustomBenchmarkResult

    if M < 128 or N < 128 or K < 64:
        return None

    A_bench = torch.randn(A_shape, dtype=dtype, device='cuda')
    B_bench = torch.randn(B_shape, dtype=dtype, device='cuda')

    for _ in range(10):
        C_warmup = torch.randn(batchCount, M, N, dtype=dtype, device='cuda')
        _ = baseline_cublasDgemmStridedBatched(transa_int, transb_int, M, N, K, alpha, A_bench, lda, strideA, B_bench, ldb, strideB, beta, C_warmup.clone(), ldc, strideC, batchCount)
        _ = flagbench.triton.cublasDgemmStridedBatched(transa, transb, M, N, K, alpha, A_bench, lda, strideA, B_bench, ldb, strideB, beta, C_warmup.clone(), ldc, strideC, batchCount)
    torch.cuda.synchronize()

    C_baseline = torch.randn(batchCount, M, N, dtype=dtype, device='cuda')
    ms_baseline = triton.testing.do_bench(
        lambda: baseline_cublasDgemmStridedBatched(transa_int, transb_int, M, N, K, alpha, A_bench, lda, strideA, B_bench, ldb, strideB, beta, C_baseline.clone(), ldc, strideC, batchCount),
        warmup=25, rep=100
    )

    C_triton = torch.randn(batchCount, M, N, dtype=dtype, device='cuda')
    ms_triton = triton.testing.do_bench(
        lambda: flagbench.triton.cublasDgemmStridedBatched(transa, transb, M, N, K, alpha, A_bench, lda, strideA, B_bench, ldb, strideB, beta, C_triton.clone(), ldc, strideC, batchCount),
        warmup=25, rep=100
    )

    speedup = ms_baseline / ms_triton
    print(f"\n{'='*70}")
    print(f"Performance: M={M}, N={N}, K={K}, batch={batchCount}, trans=({transa},{transb})")
    print(f"Baseline: {ms_baseline:.4f} ms | Triton: {ms_triton:.4f} ms | Speedup: {speedup:.2f}x")
    print(f"{'='*70}\n")

    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
