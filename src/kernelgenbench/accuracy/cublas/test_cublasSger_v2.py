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

@label("cublasSger_v2")
@parametrize("m", [1, 32, 71, 160, 497, 1024, 4096, 5333])
@parametrize("n", [1, 17, 32, 33, 71, 160, 497, 1024, 4096, 4113, 5333])
@parametrize("alpha", [1.0, 0.0, 0.001, -0.999, 100.001, -111.999, 0.5, -0.5, -1.0])
@parametrize("incx", [1, 2, 3])
@parametrize("incy", [1, 2, 3])
@parametrize("dtype", [torch.float32])
def test_accuracy_cublasSger_v2(m, n, alpha, incx, incy, dtype):
    # Prepare inputs
    x = torch.randn(m * incx, dtype=dtype, device='cuda')
    y = torch.randn(n * incy, dtype=dtype, device='cuda')

    # Create A with column-major layout by using a transposed view of a (n, m) base tensor
    base_ref = torch.randn(n, m, dtype=dtype, device='cuda')
    A_ref = base_ref.t()  # (m, n) with column-major memory layout (view)

    base_act = base_ref.clone()
    A_act = base_act.t()  # separate column-major view for Triton

    lda = m

    # Reference and Actual
    ref_out = kernelgenbench.baseline.cublasSger_v2(m, n, alpha, x, incx, y, incy, A_ref, lda)
    act_out = kernelgenbench.triton.cublasSger_v2(m, n, alpha, x, incx, y, incy, A_act, lda)

    # Accuracy check
    assert_close(act_out, ref_out, dtype)

    # Performance Test
    from sandbox.utils.accuracy_utils import CustomBenchmarkResult
    if n < 1024:
        return None

    # Prepare separate benchmark inputs
    x_bench = torch.randn(m * incx, dtype=dtype, device='cuda')
    y_bench = torch.randn(n * incy, dtype=dtype, device='cuda')
    base_bench_template = torch.randn(n, m, dtype=dtype, device='cuda')

    # warmup
    for _ in range(10):
        base_tmp = base_bench_template.clone()
        _ = kernelgenbench.baseline.cublasSger_v2(m, n, alpha, x_bench, incx, y_bench, incy, base_tmp.t(), lda)
        base_tmp2 = base_bench_template.clone()
        _ = kernelgenbench.triton.cublasSger_v2(m, n, alpha, x_bench, incx, y_bench, incy, base_tmp2.t(), lda)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Baseline timing
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        base_tmp = base_bench_template.clone()
        _ = kernelgenbench.baseline.cublasSger_v2(m, n, alpha, x_bench, incx, y_bench, incy, base_tmp.t(), lda)
    end_event.record()
    torch.cuda.synchronize()
    ms_baseline = start_event.elapsed_time(end_event) / 100

    # Triton timing
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        base_tmp2 = base_bench_template.clone()
        _ = kernelgenbench.triton.cublasSger_v2(m, n, alpha, x_bench, incx, y_bench, incy, base_tmp2.t(), lda)
    end_event.record()
    torch.cuda.synchronize()
    ms_triton = start_event.elapsed_time(end_event) / 100

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
