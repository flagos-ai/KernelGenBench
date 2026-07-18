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

@label("cublasSaxpy_v2")
@parametrize("n", [
    1,          # Edge case
    32,         # Edge case
    71,         # Non-symmetric component
    160,        # Non-symmetric component
    497,        # 2D op dimension
    1024,       # Non-symmetric component
    4113,       # Non-aligned size
    4096,       # Power-of-2 aligned
    5333,       # Non-symmetric (large N)
])
@parametrize("alpha", [
    1.0,        # Standard
    0.0,        # Standard
    0.001,      # Fractional/small
    -0.999,     # Fractional/negative
    100.001,    # Large
    -111.999,   # Large negative
    0.5,        # Symmetric
    -0.5,       # Symmetric negative
])
@parametrize("incx", [1, 2, 3])
@parametrize("incy", [1, 2, 3])
@parametrize("dtype", [torch.float32])
def test_accuracy_cublasSaxpy_v2(n, alpha, incx, incy, dtype):
    # Initialize strided input tensors
    x_storage = torch.randn(n * incx, dtype=dtype, device='cuda')
    y_storage0 = torch.randn(n * incy, dtype=dtype, device='cuda')
    y_storage1 = y_storage0.clone()

    # Call baseline
    ref_out = kernelgenbench.baseline.cublasSaxpy_v2(n, alpha, x_storage, incx, y_storage0, incy)

    # Call Triton implementation
    act_out = kernelgenbench.triton.cublasSaxpy_v2(n, alpha, x_storage, incx, y_storage1, incy)

    # Compare results
    assert_close(act_out, ref_out, dtype)

    # ========================================
    # Performance Test
    # ========================================
    from sandbox.utils.accuracy_utils import CustomBenchmarkResult

    # Only run performance test for larger sizes
    if n < 1024:
        return None

    # Prepare fresh data for benchmarking
    x_bench = torch.randn(n * incx, dtype=dtype, device='cuda')
    y_bench = torch.randn(n * incy, dtype=dtype, device='cuda')

    # Warmup
    for _ in range(10):
        _ = kernelgenbench.baseline.cublasSaxpy_v2(n, alpha, x_bench, incx, y_bench.clone(), incy)
        _ = kernelgenbench.triton.cublasSaxpy_v2(n, alpha, x_bench, incx, y_bench.clone(), incy)
    torch.cuda.synchronize()

    # Benchmark with torch.cuda.Event
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Benchmark baseline
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = kernelgenbench.baseline.cublasSaxpy_v2(n, alpha, x_bench, incx, y_bench.clone(), incy)
    end_event.record()
    torch.cuda.synchronize()
    ms_baseline = start_event.elapsed_time(end_event) / 100

    # Benchmark Triton
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = kernelgenbench.triton.cublasSaxpy_v2(n, alpha, x_bench, incx, y_bench.clone(), incy)
    end_event.record()
    torch.cuda.synchronize()
    ms_triton = start_event.elapsed_time(end_event) / 100

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(
        ref_time=ms_baseline,
        res_time=ms_triton,
        speedup=speedup
    )
