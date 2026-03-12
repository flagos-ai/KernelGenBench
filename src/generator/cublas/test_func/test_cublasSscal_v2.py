import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
import torch
from flagbench.dataset.baseline.cublas_ctypes.cublasSscal_v2 import cublasSscal_v2 as baseline_cublasSscal_v2

@label("cublasSscal_v2")
@parametrize("n", [32, 1024, 4096])
@parametrize("alpha", [1.0, 0.5, 2.0])
@parametrize("incx", [1, 2])
@parametrize("dtype", [torch.float32])
def test_accuracy_cublasSscal_v2(n, alpha, incx, dtype):
    x = torch.randn(n * incx, dtype=dtype, device='cuda')
    x_ref = x.clone()
    x_act = x.clone()

    ref_out = baseline_cublasSscal_v2(n, alpha, x_ref, incx)
    act_out = flagbench.triton.cublasSscal_v2(n, alpha, x_act, incx)

    assert_close(act_out, ref_out, dtype)

    # Performance Test
    import triton
    from sandbox.utils.accuracy_utils import CustomBenchmarkResult
    if n < 1024:
        return None

    x_bench = torch.randn(n * incx, dtype=dtype, device='cuda')
    for _ in range(10):
        _ = baseline_cublasSscal_v2(n, alpha, x_bench.clone(), incx)
        _ = flagbench.triton.cublasSscal_v2(n, alpha, x_bench.clone(), incx)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = baseline_cublasSscal_v2(n, alpha, x_bench.clone(), incx)
    end_event.record()
    torch.cuda.synchronize()
    ms_baseline = start_event.elapsed_time(end_event) / 100

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = flagbench.triton.cublasSscal_v2(n, alpha, x_bench.clone(), incx)
    end_event.record()
    torch.cuda.synchronize()
    ms_triton = start_event.elapsed_time(end_event) / 100

    speedup = ms_baseline / ms_triton
    print(f"\nPerf: n={n}, alpha={alpha} | Baseline: {ms_baseline:.4f} ms | Triton: {ms_triton:.4f} ms | Speedup: {speedup:.2f}x\n")
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
