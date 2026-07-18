# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import kernelgenbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import kernelgenbench_assert_close as assert_close
import torch

@label("cublasDsbmv_v2")
@parametrize("n", [1, 32, 71, 1024, 4096, 5333])
@parametrize("alpha", [1.0, 0.0, -0.999, 0.5, 100.001])
@parametrize("incx", [1, 2])
@parametrize("dtype", [torch.float64])
def test_accuracy_cublasDsbmv_v2(n, alpha, incx, dtype):
    # Fixed parameters for this test
    uplo = 1  # 1 for upper storage in cuBLAS convention
    k = 2
    lda = k + 1
    beta = -0.75
    incy = incx

    # Build symmetric band storage AB (lda, n) for upper storage without forming full matrix
    AB = torch.zeros((lda, n), dtype=dtype, device='cuda')
    for j in range(n):
        i_start = max(0, j - k)
        # Fill A[i, j] for i in [i_start, j]
        num = j - i_start + 1
        vals = torch.randn(num, dtype=dtype, device='cuda')
        for idx, i in enumerate(range(i_start, j + 1)):
            row = k + i - j
            AB[row, j] = vals[idx]

    # Convert to column-major layout expected by cuBLAS using transpose trick
    A_band_for_cublas = AB.t().contiguous()

    # Vectors with increments
    x = torch.randn(n * incx, dtype=dtype, device='cuda')
    y = torch.randn(n * incy, dtype=dtype, device='cuda')

    # Clone for ref and act
    x_ref = x.clone()
    y_ref = y.clone()
    x_act = x.clone()
    y_act = y.clone()

    # Reference (baseline) and actual (triton)
    ref_out = kernelgenbench.baseline.cublasDsbmv_v2(uplo, n, k, alpha, A_band_for_cublas, lda, x_ref, incx, beta, y_ref, incy)
    act_out = kernelgenbench.triton.cublasDsbmv_v2(uplo, n, k, alpha, A_band_for_cublas, lda, x_act, incx, beta, y_act, incy)

    assert_close(act_out, ref_out, dtype, reduce_dim=n)

    # Performance Test
    from sandbox.utils.accuracy_utils import CustomBenchmarkResult
    if n < 1024:
        return None

    # Prepare fresh inputs for benchmarking
    AB_bench = torch.zeros((lda, n), dtype=dtype, device='cuda')
    for j in range(n):
        i_start = max(0, j - k)
        num = j - i_start + 1
        vals = torch.randn(num, dtype=dtype, device='cuda')
        for idx, i in enumerate(range(i_start, j + 1)):
            row = k + i - j
            AB_bench[row, j] = vals[idx]
    A_bench = AB_bench.t().contiguous()

    x_bench = torch.randn(n * incx, dtype=dtype, device='cuda')
    y_bench = torch.randn(n * incy, dtype=dtype, device='cuda')

    # warmup
    for _ in range(10):
        _ = kernelgenbench.baseline.cublasDsbmv_v2(uplo, n, k, alpha, A_bench, lda, x_bench, incx, beta, y_bench.clone(), incy)
        _ = kernelgenbench.triton.cublasDsbmv_v2(uplo, n, k, alpha, A_bench, lda, x_bench, incx, beta, y_bench.clone(), incy)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = kernelgenbench.baseline.cublasDsbmv_v2(uplo, n, k, alpha, A_bench, lda, x_bench, incx, beta, y_bench.clone(), incy)
    end_event.record()
    torch.cuda.synchronize()
    ms_baseline = start_event.elapsed_time(end_event) / 100

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = kernelgenbench.triton.cublasDsbmv_v2(uplo, n, k, alpha, A_bench, lda, x_bench, incx, beta, y_bench.clone(), incy)
    end_event.record()
    torch.cuda.synchronize()
    ms_triton = start_event.elapsed_time(end_event) / 100

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
