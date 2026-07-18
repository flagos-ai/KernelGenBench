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

@label("cublasSdot_v2")
@parametrize("n", [1, 17, 32, 33, 71, 160, 497, 1024, 4113, 4096, 5333])
@parametrize("incx", [1, 2, 3])
@parametrize("incy", [1, 2, 3])
@parametrize("dtype", [torch.float32])
def test_accuracy_cublasSdot_v2(n, incx, incy, dtype):
    len_x = 1 + (n - 1) * abs(incx)
    len_y = 1 + (n - 1) * abs(incy)

    x = torch.randn(len_x, dtype=dtype, device='cuda')
    y = torch.randn(len_y, dtype=dtype, device='cuda')

    x_ref = x.clone()
    y_ref = y.clone()
    x_act = x.clone()
    y_act = y.clone()

    result_ref = torch.zeros(1, dtype=dtype, device='cuda')
    result_act = torch.zeros(1, dtype=dtype, device='cuda')

    ref_out = kernelgenbench.baseline.cublasSdot_v2(n, x_ref, incx, y_ref, incy, result_ref)
    act_out = kernelgenbench.triton.cublasSdot_v2(n, x_act, incx, y_act, incy, result_act)

    assert_close(act_out, ref_out, dtype)

    # Performance Test
    from sandbox.utils.accuracy_utils import CustomBenchmarkResult
    if n < 1024:
        return None

    x_bench = torch.randn(len_x, dtype=dtype, device='cuda')
    y_bench = torch.randn(len_y, dtype=dtype, device='cuda')
    res_bench_ref = torch.zeros(1, dtype=dtype, device='cuda')
    res_bench_act = torch.zeros(1, dtype=dtype, device='cuda')

    # warmup
    for _ in range(10):
        _ = kernelgenbench.baseline.cublasSdot_v2(n, x_bench, incx, y_bench, incy, res_bench_ref)
        _ = kernelgenbench.triton.cublasSdot_v2(n, x_bench, incx, y_bench, incy, res_bench_act)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = kernelgenbench.baseline.cublasSdot_v2(n, x_bench, incx, y_bench, incy, res_bench_ref)
    end_event.record()
    torch.cuda.synchronize()
    ms_baseline = start_event.elapsed_time(end_event) / 100

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = kernelgenbench.triton.cublasSdot_v2(n, x_bench, incx, y_bench, incy, res_bench_act)
    end_event.record()
    torch.cuda.synchronize()
    ms_triton = start_event.elapsed_time(end_event) / 100

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
