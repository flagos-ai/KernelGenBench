import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
import torch
from flagbench.dataset.baseline.cublas_ctypes.cublasSaxpy_v2 import cublasSaxpy_v2 as baseline_cublasSaxpy_v2

@label("cublasSaxpy_v2")
@parametrize("n", [32, 1024, 4096])
@parametrize("alpha", [1.0, 0.5, 2.0])
@parametrize("incx", [1, 2])
@parametrize("incy", [1, 2])
@parametrize("dtype", [torch.float32])
def test_accuracy_cublasSaxpy_v2(n, alpha, incx, incy, dtype):
    # Initialize input tensors
    x = torch.randn(n * incx, dtype=dtype, device='cuda')
    y = torch.randn(n * incy, dtype=dtype, device='cuda')

    # Clone y for both calls
    y_ref = y.clone()
    y_act = y.clone()

    # Call baseline
    ref_out = baseline_cublasSaxpy_v2(n, alpha, x, incx, y_ref, incy)

    # Call Triton implementation
    act_out = flagbench.triton.cublasSaxpy_v2(n, alpha, x, incx, y_act, incy)

    # DEBUG: Print function info to verify we're comparing different implementations
    if n == 1024 and alpha == 1.0 and incx == 1 and incy == 1:
        print(f"\n[DEBUG] Baseline function: {baseline_cublasSaxpy_v2}")
        print(f"[DEBUG] Baseline module: {baseline_cublasSaxpy_v2.__module__}")
        print(f"[DEBUG] Triton function: {flagbench.triton.cublasSaxpy_v2}")
        print(f"[DEBUG] Triton module: {flagbench.triton.cublasSaxpy_v2.__module__}")
        print(f"[DEBUG] Are they the same? {baseline_cublasSaxpy_v2 is flagbench.triton.cublasSaxpy_v2}\n")

    # Compare results
    assert_close(act_out, ref_out, dtype)

    # ========================================
    # Performance Test: Multiple Timing Methods
    # ========================================
    import triton
    import time
    from sandbox.utils.accuracy_utils import CustomBenchmarkResult

    # Only run performance test for larger sizes to get meaningful results
    if n < 1024:
        return None

    # Prepare fresh data for benchmarking
    x_bench = torch.randn(n * incx, dtype=dtype, device='cuda')

    # Warmup
    for _ in range(10):
        y_warmup = torch.randn(n * incy, dtype=dtype, device='cuda')
        _ = baseline_cublasSaxpy_v2(n, alpha, x_bench, incx, y_warmup, incy)
        _ = flagbench.triton.cublasSaxpy_v2(n, alpha, x_bench, incx, y_warmup.clone(), incy)
    torch.cuda.synchronize()

    results = {}

    # ========================================
    # Method 1: triton.testing.do_bench
    # ========================================
    y_baseline = torch.randn(n * incy, dtype=dtype, device='cuda')
    ms_baseline_1 = triton.testing.do_bench(
        lambda: baseline_cublasSaxpy_v2(n, alpha, x_bench, incx, y_baseline.clone(), incy),
        warmup=25,
        rep=100
    )

    y_triton = torch.randn(n * incy, dtype=dtype, device='cuda')
    ms_triton_1 = triton.testing.do_bench(
        lambda: flagbench.triton.cublasSaxpy_v2(n, alpha, x_bench, incx, y_triton.clone(), incy),
        warmup=25,
        rep=100
    )

    speedup_1 = ms_baseline_1 / ms_triton_1
    results['triton.testing.do_bench'] = CustomBenchmarkResult(
        ref_time=ms_baseline_1,
        res_time=ms_triton_1,
        speedup=speedup_1,
    )

    # ========================================
    # Method 2: torch.cuda.Event
    # ========================================
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Baseline timing
    y_baseline = torch.randn(n * incy, dtype=dtype, device='cuda')
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = baseline_cublasSaxpy_v2(n, alpha, x_bench, incx, y_baseline.clone(), incy)
    end_event.record()
    torch.cuda.synchronize()
    ms_baseline_2 = start_event.elapsed_time(end_event) / 100

    # Triton timing
    y_triton = torch.randn(n * incy, dtype=dtype, device='cuda')
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = flagbench.triton.cublasSaxpy_v2(n, alpha, x_bench, incx, y_triton.clone(), incy)
    end_event.record()
    torch.cuda.synchronize()
    ms_triton_2 = start_event.elapsed_time(end_event) / 100

    speedup_2 = ms_baseline_2 / ms_triton_2
    results['torch.cuda.Event'] = CustomBenchmarkResult(
        ref_time=ms_baseline_2,
        res_time=ms_triton_2,
        speedup=speedup_2,
    )

    # ========================================
    # Method 3: time.perf_counter
    # ========================================
    # Baseline timing
    y_baseline = torch.randn(n * incy, dtype=dtype, device='cuda')
    torch.cuda.synchronize()
    start_perf = time.perf_counter()
    for _ in range(100):
        _ = baseline_cublasSaxpy_v2(n, alpha, x_bench, incx, y_baseline.clone(), incy)
    torch.cuda.synchronize()
    end_perf = time.perf_counter()
    ms_baseline_3 = (end_perf - start_perf) * 1000 / 100

    # Triton timing
    y_triton = torch.randn(n * incy, dtype=dtype, device='cuda')
    torch.cuda.synchronize()
    start_perf = time.perf_counter()
    for _ in range(100):
        _ = flagbench.triton.cublasSaxpy_v2(n, alpha, x_bench, incx, y_triton.clone(), incy)
    torch.cuda.synchronize()
    end_perf = time.perf_counter()
    ms_triton_3 = (end_perf - start_perf) * 1000 / 100

    speedup_3 = ms_baseline_3 / ms_triton_3
    results['time.perf_counter'] = CustomBenchmarkResult(
        ref_time=ms_baseline_3,
        res_time=ms_triton_3,
        speedup=speedup_3,
    )

    # ========================================
    # Method 4: torch.utils.benchmark.Timer
    # ========================================
    import torch.utils.benchmark as benchmark

    # Baseline timing
    y_baseline = torch.randn(n * incy, dtype=dtype, device='cuda')
    timer_baseline = benchmark.Timer(
        stmt='baseline_cublasSaxpy_v2(n, alpha, x, incx, y.clone(), incy)',
        setup='torch.cuda.synchronize()',
        globals={
            'baseline_cublasSaxpy_v2': baseline_cublasSaxpy_v2,
            'n': n, 'alpha': alpha, 'x': x_bench, 'incx': incx,
            'y': y_baseline, 'incy': incy
        }
    )
    ms_baseline_4 = timer_baseline.blocked_autorange(min_run_time=1).median * 1000

    # Triton timing
    y_triton = torch.randn(n * incy, dtype=dtype, device='cuda')
    timer_triton = benchmark.Timer(
        stmt='flagbench.triton.cublasSaxpy_v2(n, alpha, x, incx, y.clone(), incy)',
        setup='torch.cuda.synchronize()',
        globals={
            'flagbench': flagbench,
            'n': n, 'alpha': alpha, 'x': x_bench, 'incx': incx,
            'y': y_triton, 'incy': incy
        }
    )
    ms_triton_4 = timer_triton.blocked_autorange(min_run_time=1).median * 1000

    speedup_4 = ms_baseline_4 / ms_triton_4
    results['torch.utils.benchmark.Timer'] = CustomBenchmarkResult(
        ref_time=ms_baseline_4,
        res_time=ms_triton_4,
        speedup=speedup_4,
    )

    # Print all timing results
    print(f"\n{'='*70}")
    print(f"Performance Test Results (n={n}, alpha={alpha}, incx={incx}, incy={incy})")
    print(f"{'='*70}")
    print(f"{'Method':<35} {'Baseline (ms)':<15} {'Triton (ms)':<15} {'Speedup':<10}")
    print(f"{'-'*70}")
    for method_name, result in results.items():
        print(f"{method_name:<35} {result.ref_time:<15.4f} {result.res_time:<15.4f} {result.speedup:<10.2f}x")
    print(f"{'='*70}\n")

    # Return the first method's result (framework expects single result)
    return results['triton.testing.do_bench']
