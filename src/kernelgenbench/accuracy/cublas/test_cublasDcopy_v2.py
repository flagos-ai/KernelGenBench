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

@label("cublasDcopy_v2")
@parametrize("n", [1, 32, 71, 497, 1024, 4096, 5333])
@parametrize("incx", [1, 2, 3])
@parametrize("incy", [1, 2, 3])
@parametrize("dtype", [torch.float64])
def test_accuracy_cublasDcopy_v2(n, incx, incy, dtype):
    size_x = 1 + (n - 1) * incx
    size_y = 1 + (n - 1) * incy

    x = torch.randn(size_x, dtype=dtype, device=device)
    y0 = torch.randn(size_y, dtype=dtype, device=device)
    y1 = y0.clone()

    ref_out = kernelgenbench.baseline.cublasDcopy_v2(n, x, incx, y0, incy)
    act_out = kernelgenbench.triton.cublasDcopy_v2(n, x, incx, y1, incy)

    assert_close(act_out, ref_out, dtype)

    from sandbox.utils.accuracy_utils import CustomBenchmarkResult
    if n < 1024:
        return None

    x_b = torch.randn(size_x, dtype=dtype, device=device)
    y_b = torch.randn(size_y, dtype=dtype, device=device)

    for _ in range(10):
        _ = kernelgenbench.baseline.cublasDcopy_v2(n, x_b, incx, y_b.clone(), incy)
        _ = kernelgenbench.triton.cublasDcopy_v2(n, x_b, incx, y_b.clone(), incy)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = kernelgenbench.baseline.cublasDcopy_v2(n, x_b, incx, y_b.clone(), incy)
    end_event.record()
    torch.cuda.synchronize()
    ms_baseline = start_event.elapsed_time(end_event) / 100

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = kernelgenbench.triton.cublasDcopy_v2(n, x_b, incx, y_b.clone(), incy)
    end_event.record()
    torch.cuda.synchronize()
    ms_triton = start_event.elapsed_time(end_event) / 100

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
