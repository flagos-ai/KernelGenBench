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

@label("cublasStrsv_v2")
@parametrize("n", [1, 32, 71, 1024, 4096, 5333])
@parametrize("incx", [1, 2])
@parametrize("uplo", [0, 1])
@parametrize("trans", [0, 1])
@parametrize("diag", [0, 1])
@parametrize("dtype", [torch.float32])
def test_accuracy_cublasStrsv_v2(n, incx, uplo, trans, diag, dtype):
    lda = n

    A_full = torch.randn(n, n, dtype=dtype, device='cuda')
    if uplo == 0:
        A = torch.tril(A_full)
    else:
        A = torch.triu(A_full)
    A = A + torch.eye(n, dtype=dtype, device='cuda') * 5.0

    # Create RHS vector x (with stride incx)
    x = torch.randn(n * incx, dtype=dtype, device='cuda')
    x_ref = x.clone()
    x_act = x.clone()

    ref_out = kernelgenbench.baseline.cublasStrsv_v2(uplo, trans, diag, n, A, lda, x_ref, incx)
    act_out = kernelgenbench.triton.cublasStrsv_v2(uplo, trans, diag, n, A, lda, x_act, incx)

    assert_close(act_out, ref_out, dtype)

    # Performance Test
    from sandbox.utils.accuracy_utils import CustomBenchmarkResult
    if n < 1024:
        return None

    # Benchmark inputs
    A_bench_full = torch.randn(n, n, dtype=dtype, device='cuda')
    if uplo == 0:
        A_bench = torch.tril(A_bench_full) + torch.eye(n, dtype=dtype, device='cuda') * 5.0
    else:
        A_bench = torch.triu(A_bench_full) + torch.eye(n, dtype=dtype, device='cuda') * 5.0
    x_bench = torch.randn(n * incx, dtype=dtype, device='cuda')

    # warmup
    for _ in range(10):
        _ = kernelgenbench.baseline.cublasStrsv_v2(uplo, trans, diag, n, A_bench, lda, x_bench.clone(), incx)
        _ = kernelgenbench.triton.cublasStrsv_v2(uplo, trans, diag, n, A_bench, lda, x_bench.clone(), incx)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = kernelgenbench.baseline.cublasStrsv_v2(uplo, trans, diag, n, A_bench, lda, x_bench.clone(), incx)
    end_event.record()
    torch.cuda.synchronize()
    ms_baseline = start_event.elapsed_time(end_event) / 100

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = kernelgenbench.triton.cublasStrsv_v2(uplo, trans, diag, n, A_bench, lda, x_bench.clone(), incx)
    end_event.record()
    torch.cuda.synchronize()
    ms_triton = start_event.elapsed_time(end_event) / 100

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
