import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
import torch
from flagbench.dataset.baseline.cublas_ctypes.cublasCgemvStridedBatched import cublasCgemvStridedBatched as baseline_cublasCgemvStridedBatched


def _trans_to_int(trans_str):
    """Convert transpose string to cuBLAS integer: 'N'->0, 'T'->1"""
    return 0 if trans_str == 'N' else 1

@label("cublasCgemvStridedBatched")
@parametrize("M, N", [(16, 16), (128, 64)])
@parametrize("alpha, beta", [(1.0+0.5j, 0.0+0.0j), (0.5+0.3j, 0.5-0.2j)])
@parametrize("trans", ["N", "T"])
@parametrize("batchCount", [1, 4])
@parametrize("dtype", [torch.complex64])
def test_accuracy_cublasCgemvStridedBatched(M, N, alpha, beta, trans, batchCount, dtype):
    A = torch.randn(batchCount, M, N, dtype=dtype, device='cuda')
    lenx = N if trans == 'N' else M
    leny = M if trans == 'N' else N
    x = torch.randn(batchCount, lenx, dtype=dtype, device='cuda')
    y = torch.randn(batchCount, leny, dtype=dtype, device='cuda')

    incx = 1
    incy = 1
    lda = A.shape[1]  # Leading dimension (rows in column-major)
    strideA = M * N
    stridex = lenx
    stridey = leny

    y_ref = y.clone()
    y_act = y.clone()

    # Convert trans string to integer for baseline (cuBLAS C API expects int)
    trans_int = _trans_to_int(trans)
    ref_out = baseline_cublasCgemvStridedBatched(trans_int, M, N, alpha, A, lda, strideA, x, incx, stridex, beta, y_ref, incy, stridey, batchCount)
    act_out = flagbench.triton.cublasCgemvStridedBatched(trans, M, N, alpha, A, lda, strideA, x, incx, stridex, beta, y_act, incy, stridey, batchCount)
    assert_close(act_out, ref_out, dtype, reduce_dim=N if trans == 'N' else M)

    # Performance Test
    import triton
    from sandbox.utils.accuracy_utils import CustomBenchmarkResult
    if M < 128 or N < 64:
        return None
    A_bench = torch.randn(batchCount, M, N, dtype=dtype, device='cuda')
    x_bench = torch.randn(batchCount, lenx, dtype=dtype, device='cuda')
    for _ in range(10):
        y_warmup = torch.randn(batchCount, leny, dtype=dtype, device='cuda')
        _ = baseline_cublasCgemvStridedBatched(trans_int, M, N, alpha, A_bench, lda, strideA, x_bench, incx, stridex, beta, y_warmup.clone(), incy, stridey, batchCount)
        _ = flagbench.triton.cublasCgemvStridedBatched(trans, M, N, alpha, A_bench, lda, strideA, x_bench, incx, stridex, beta, y_warmup.clone(), incy, stridey, batchCount)
    torch.cuda.synchronize()
    y_baseline = torch.randn(batchCount, leny, dtype=dtype, device='cuda')
    ms_baseline = triton.testing.do_bench(lambda: baseline_cublasCgemvStridedBatched(trans_int, M, N, alpha, A_bench, lda, strideA, x_bench, incx, stridex, beta, y_baseline.clone(), incy, stridey, batchCount), warmup=25, rep=100)
    y_triton = torch.randn(batchCount, leny, dtype=dtype, device='cuda')
    ms_triton = triton.testing.do_bench(lambda: flagbench.triton.cublasCgemvStridedBatched(trans, M, N, alpha, A_bench, lda, strideA, x_bench, incx, stridex, beta, y_triton.clone(), incy, stridey, batchCount), warmup=25, rep=100)
    speedup = ms_baseline / ms_triton
    print(f"\nPerf: M={M}, N={N} | Baseline: {ms_baseline:.4f} ms | Triton: {ms_triton:.4f} ms | Speedup: {speedup:.2f}x\n")
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
