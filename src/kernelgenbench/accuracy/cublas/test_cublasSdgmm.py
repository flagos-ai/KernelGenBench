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

@label("cublasSdgmm")
@parametrize("n", [1, 32, 71, 1024, 4096, 5333])
@parametrize("incx", [1, 2, 3])
@parametrize("dtype", [torch.float32])
def test_accuracy_cublasSdgmm(n, incx, dtype):
    # Setup matrix dimensions (square for simplicity)
    m = n

    # Create input matrix A
    A = torch.randn(m, n, dtype=dtype, device=device).contiguous()

    # Prepare vectors with stride incx
    x_right = torch.zeros(n * incx, dtype=dtype, device=device)
    x_right_values = torch.randn(n, dtype=dtype, device=device)
    x_right[::incx] = x_right_values

    x_left = torch.zeros(m * incx, dtype=dtype, device=device)
    x_left_values = torch.randn(m, dtype=dtype, device=device)
    x_left[::incx] = x_left_values

    # Output buffers for reference and actual (right)
    C_right_ref = torch.empty_like(A)
    C_right_act = torch.empty_like(A)

    # Mode constants
    CUBLAS_SIDE_LEFT = 0
    CUBLAS_SIDE_RIGHT = 1

    # Column-major trick: pass (m_param=n, n_param=m, lda=n, ldc=n)
    # Right multiplication: C = A * diag(x_right) => use SIDE_LEFT on column-major view
    ref_right = kernelgenbench.baseline.cublasSdgmm(CUBLAS_SIDE_LEFT, n, m, A.clone(), n, x_right, incx, C_right_ref, n)
    act_right = kernelgenbench.triton.cublasSdgmm(CUBLAS_SIDE_LEFT, n, m, A.clone(), n, x_right, incx, C_right_act, n)

    expected_right = A * x_right_values.view(1, -1)
    assert_close(act_right, ref_right, dtype)
    assert_close(act_right, expected_right, dtype)

    # Output buffers for reference and actual (left)
    C_left_ref = torch.empty_like(A)
    C_left_act = torch.empty_like(A)

    # Left multiplication: C = diag(x_left) * A => use SIDE_RIGHT on column-major view
    ref_left = kernelgenbench.baseline.cublasSdgmm(CUBLAS_SIDE_RIGHT, n, m, A.clone(), n, x_left, incx, C_left_ref, n)
    act_left = kernelgenbench.triton.cublasSdgmm(CUBLAS_SIDE_RIGHT, n, m, A.clone(), n, x_left, incx, C_left_act, n)

    expected_left = x_left_values.view(-1, 1) * A
    assert_close(act_left, ref_left, dtype)
    assert_close(act_left, expected_left, dtype)

    # Performance Test
    from sandbox.utils.accuracy_utils import CustomBenchmarkResult
    if n < 1024:
        return None

    # Benchmark on right multiplication
    A_bench = torch.randn(m, n, dtype=dtype, device=device).contiguous()
    x_bench = torch.zeros(n * incx, dtype=dtype, device=device)
    x_bench_vals = torch.randn(n, dtype=dtype, device=device)
    x_bench[::incx] = x_bench_vals
    C_bench = torch.empty_like(A_bench)

    # warmup
    for _ in range(10):
        _ = kernelgenbench.baseline.cublasSdgmm(CUBLAS_SIDE_LEFT, n, m, A_bench.clone(), n, x_bench.clone(), incx, C_bench, n)
        _ = kernelgenbench.triton.cublasSdgmm(CUBLAS_SIDE_LEFT, n, m, A_bench.clone(), n, x_bench.clone(), incx, C_bench, n)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = kernelgenbench.baseline.cublasSdgmm(CUBLAS_SIDE_LEFT, n, m, A_bench.clone(), n, x_bench.clone(), incx, C_bench, n)
    end_event.record()
    torch.cuda.synchronize()
    ms_baseline = start_event.elapsed_time(end_event) / 100

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = kernelgenbench.triton.cublasSdgmm(CUBLAS_SIDE_LEFT, n, m, A_bench.clone(), n, x_bench.clone(), incx, C_bench, n)
    end_event.record()
    torch.cuda.synchronize()
    ms_triton = start_event.elapsed_time(end_event) / 100

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
