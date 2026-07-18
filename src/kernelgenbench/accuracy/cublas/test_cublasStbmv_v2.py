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

@label("cublasStbmv_v2")
@parametrize("n", [1, 32, 71, 1024, 4096, 5333])
@parametrize("incx", [1, 2])
@parametrize("uplo", [0, 1])  # LOWER=0, UPPER=1
@parametrize("trans", [0, 1])  # N=0, T=1
@parametrize("diag", [0, 1])  # NON_UNIT=0, UNIT=1
@parametrize("k", [0, 2])
@parametrize("dtype", [torch.float32])
def test_accuracy_cublasStbmv_v2(n, incx, uplo, trans, diag, k, dtype):
    # Ensure k is valid for given n
    k = min(k, max(0, n - 1))
    lda = k + 1

    # Create a full triangular band matrix A_full on GPU
    A_full = torch.zeros((n, n), dtype=dtype, device='cuda')
    rand_vals = torch.randn((n, n), dtype=dtype, device='cuda')

    if uplo == 1:  # UPPER
        for j in range(n):
            i_start = max(0, j - k)
            for i in range(i_start, j + 1):
                A_full[i, j] = rand_vals[i, j]
    else:  # LOWER
        for j in range(n):
            i_end = min(n - 1, j + k)
            for i in range(j, i_end + 1):
                A_full[i, j] = rand_vals[i, j]

    if diag == 1:  # UNIT
        A_full.fill_diagonal_(1.0)

    # Pack into band storage AB (lda x n) with column-major semantics
    AB = torch.zeros((lda, n), dtype=dtype, device='cuda')
    if uplo == 1:  # UPPER
        for j in range(n):
            i_start = max(0, j - k)
            for i in range(i_start, j + 1):
                AB[k + i - j, j] = A_full[i, j]
    else:  # LOWER
        for j in range(n):
            i_end = min(n - 1, j + k)
            for i in range(j, i_end + 1):
                AB[i - j, j] = A_full[i, j]

    # Transpose to match column-major expectation when passed as row-major
    A_band_t = AB.t().contiguous()

    # Vector x with increment incx
    x = torch.randn(n * incx, dtype=dtype, device='cuda')
    x_ref = x.clone()
    x_act = x.clone()

    # Reference (baseline) and actual (triton)
    ref_out = kernelgenbench.baseline.cublasStbmv_v2(uplo, trans, diag, n, k, A_band_t, lda, x_ref, incx)
    act_out = kernelgenbench.triton.cublasStbmv_v2(uplo, trans, diag, n, k, A_band_t, lda, x_act, incx)

    assert_close(act_out, ref_out, dtype)

    # Performance Test
    from sandbox.utils.accuracy_utils import CustomBenchmarkResult
    if n < 1024:
        return None

    # Prepare benchmark inputs
    x_bench = torch.randn(n * incx, dtype=dtype, device='cuda')

    # warmup
    for _ in range(10):
        _ = kernelgenbench.baseline.cublasStbmv_v2(uplo, trans, diag, n, k, A_band_t, lda, x_bench.clone(), incx)
        _ = kernelgenbench.triton.cublasStbmv_v2(uplo, trans, diag, n, k, A_band_t, lda, x_bench.clone(), incx)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = kernelgenbench.baseline.cublasStbmv_v2(uplo, trans, diag, n, k, A_band_t, lda, x_bench.clone(), incx)
    end_event.record()
    torch.cuda.synchronize()
    ms_baseline = start_event.elapsed_time(end_event) / 100

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = kernelgenbench.triton.cublasStbmv_v2(uplo, trans, diag, n, k, A_band_t, lda, x_bench.clone(), incx)
    end_event.record()
    torch.cuda.synchronize()
    ms_triton = start_event.elapsed_time(end_event) / 100

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
