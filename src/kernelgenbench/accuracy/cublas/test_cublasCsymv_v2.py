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

@label("cublasCsymv_v2")
@parametrize("n", [1, 32, 71, 1024, 4096, 5333])
@parametrize("alpha", [1.0, 0.0, -0.999, 0.5, 100.001])
@parametrize("incx", [1, 2])
@parametrize("dtype", [torch.complex64])
def test_accuracy_cublasCsymv_v2(n, alpha, incx, dtype):
    # Parameters
    CUBLAS_FILL_MODE_UPPER = 0
    uplo = CUBLAS_FILL_MODE_UPPER
    lda = n
    beta = 0.5
    incy = 1

    # Create symmetric complex matrix A (symmetric, not Hermitian)
    M = torch.randn(n, n, dtype=dtype, device='cuda')
    A = 0.5 * (M + M.t())

    # Create vectors with increments
    x = torch.randn(n * incx, dtype=dtype, device='cuda')
    y = torch.randn(n * incy, dtype=dtype, device='cuda')

    # Clone inputs for baseline and triton calls
    y_ref = y.clone()
    y_act = y.clone()

    # Baseline and Triton calls
    ref_out = kernelgenbench.baseline.cublasCsymv_v2(uplo, n, alpha, A, lda, x, incx, beta, y_ref, incy)
    act_out = kernelgenbench.triton.cublasCsymv_v2(uplo, n, alpha, A, lda, x, incx, beta, y_act, incy)

    # Accuracy check
    assert_close(act_out, ref_out, dtype, reduce_dim=n)

    # Performance Test
    from sandbox.utils.accuracy_utils import CustomBenchmarkResult
    if n < 1024:
        return None

    # Benchmark inputs
    M_bench = torch.randn(n, n, dtype=dtype, device='cuda')
    A_bench = 0.5 * (M_bench + M_bench.t())
    x_bench = torch.randn(n * incx, dtype=dtype, device='cuda')
    y_bench = torch.randn(n * incy, dtype=dtype, device='cuda')

    # Warmup
    for _ in range(10):
        _ = kernelgenbench.baseline.cublasCsymv_v2(uplo, n, alpha, A_bench, lda, x_bench, incx, beta, y_bench.clone(), incy)
        _ = kernelgenbench.triton.cublasCsymv_v2(uplo, n, alpha, A_bench, lda, x_bench, incx, beta, y_bench.clone(), incy)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Baseline timing
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = kernelgenbench.baseline.cublasCsymv_v2(uplo, n, alpha, A_bench, lda, x_bench, incx, beta, y_bench.clone(), incy)
    end_event.record()
    torch.cuda.synchronize()
    ms_baseline = start_event.elapsed_time(end_event) / 100

    # Triton timing
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = kernelgenbench.triton.cublasCsymv_v2(uplo, n, alpha, A_bench, lda, x_bench, incx, beta, y_bench.clone(), incy)
    end_event.record()
    torch.cuda.synchronize()
    ms_triton = start_event.elapsed_time(end_event) / 100

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
