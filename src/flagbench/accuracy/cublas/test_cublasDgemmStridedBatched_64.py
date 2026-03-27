import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
import torch

@label("cublasDgemmStridedBatched_64")
@parametrize("M, N, K", [
    (1, 1, 1),
    (16, 16, 16),
    (64, 32, 48),
    (128, 256, 64),
    (17, 33, 65),
])
@parametrize("alpha, beta", [
    (1.0, 0.0),
    (0.5, 0.5),
    (-1.0, 1.0),
])
@parametrize("transa, transb", [("N", "N"), ("N", "T"), ("T", "N"), ("T", "T")])
@parametrize("batchCount", [1, 2, 4])
@parametrize("dtype", [torch.float64])
def test_accuracy_cublasDgemmStridedBatched_64(M, N, K, alpha, beta, transa, transb, batchCount, dtype):
    A_shape = (batchCount, K, M) if transa == 'T' else (batchCount, M, K)
    B_shape = (batchCount, N, K) if transb == 'T' else (batchCount, K, N)
    A = torch.randn(A_shape, dtype=dtype, device=device)
    B = torch.randn(B_shape, dtype=dtype, device=device)
    C = torch.randn(batchCount, M, N, dtype=dtype, device=device)

    strideA = A.shape[1] * A.shape[2]
    strideB = B.shape[1] * B.shape[2]
    strideC = M * N
    lda = A.shape[1]
    ldb = B.shape[1]
    ldc = M

    C_ref = C.clone()
    C_act = C.clone()

    ref_out = flagbench.baseline.cublasDgemmStridedBatched_64(
        transa, transb, M, N, K, alpha, A, lda, strideA,
        B, ldb, strideB, beta, C_ref, ldc, strideC, batchCount
    )
    act_out = flagbench.triton.cublasDgemmStridedBatched_64(
        transa, transb, M, N, K, alpha, A, lda, strideA,
        B, ldb, strideB, beta, C_act, ldc, strideC, batchCount
    )
    assert_close(act_out, ref_out, dtype, reduce_dim=K)

    from sandbox.utils.accuracy_utils import CustomBenchmarkResult
    if M < 128 or N < 128 or K < 64:
        return None

    A_bench = torch.randn(A_shape, dtype=dtype, device=device)
    B_bench = torch.randn(B_shape, dtype=dtype, device=device)

    # warmup
    for _ in range(10):
        C_w = torch.randn(batchCount, M, N, dtype=dtype, device=device)
        _ = flagbench.baseline.cublasDgemmStridedBatched_64(
            transa, transb, M, N, K, alpha, A_bench, lda, strideA,
            B_bench, ldb, strideB, beta, C_w.clone(), ldc, strideC, batchCount
        )
        _ = flagbench.triton.cublasDgemmStridedBatched_64(
            transa, transb, M, N, K, alpha, A_bench, lda, strideA,
            B_bench, ldb, strideB, beta, C_w.clone(), ldc, strideC, batchCount
        )
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        C_b = torch.randn(batchCount, M, N, dtype=dtype, device=device)
        _ = flagbench.baseline.cublasDgemmStridedBatched_64(
            transa, transb, M, N, K, alpha, A_bench, lda, strideA,
            B_bench, ldb, strideB, beta, C_b.clone(), ldc, strideC, batchCount
        )
    end_event.record()
    torch.cuda.synchronize()
    ms_baseline = start_event.elapsed_time(end_event) / 100

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        C_t = torch.randn(batchCount, M, N, dtype=dtype, device=device)
        _ = flagbench.triton.cublasDgemmStridedBatched_64(
            transa, transb, M, N, K, alpha, A_bench, lda, strideA,
            B_bench, ldb, strideB, beta, C_t.clone(), ldc, strideC, batchCount
        )
    end_event.record()
    torch.cuda.synchronize()
    ms_triton = start_event.elapsed_time(end_event) / 100

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)