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

@label("cublasDasum_v2")
@parametrize("n", [1, 32, 71, 1024, 4096, 5333])
@parametrize("incx", [1, 2, 3])
@parametrize("dtype", [torch.float64])
def test_accuracy_cublasDasum_v2(n, incx, dtype):
    x = torch.randn(n * incx, dtype=dtype, device='cuda')
    x_ref = x.clone()
    x_act = x.clone()
    result_ref = torch.zeros(1, dtype=dtype, device='cuda')
    result_act = torch.zeros(1, dtype=dtype, device='cuda')

    ref_out = kernelgenbench.baseline.cublasDasum_v2(n, x_ref, incx, result_ref)
    act_out = kernelgenbench.triton.cublasDasum_v2(n, x_act, incx, result_act)

    assert_close(act_out, ref_out, dtype)

    # Performance Test
    from sandbox.utils.accuracy_utils import CustomBenchmarkResult
    if n < 1024:
        return None

    x_bench = torch.randn(n * incx, dtype=dtype, device='cuda')

    # warmup
    for _ in range(10):
        r = torch.zeros(1, dtype=dtype, device='cuda')
        _ = kernelgenbench.baseline.cublasDasum_v2(n, x_bench, incx, r)
        r2 = torch.zeros(1, dtype=dtype, device='cuda')
        _ = kernelgenbench.triton.cublasDasum_v2(n, x_bench, incx, r2)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = kernelgenbench.baseline.cublasDasum_v2(n, x_bench, incx, torch.zeros(1, dtype=dtype, device='cuda'))
    end_event.record()
    torch.cuda.synchronize()
    ms_baseline = start_event.elapsed_time(end_event) / 100

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = kernelgenbench.triton.cublasDasum_v2(n, x_bench, incx, torch.zeros(1, dtype=dtype, device='cuda'))
    end_event.record()
    torch.cuda.synchronize()
    ms_triton = start_event.elapsed_time(end_event) / 100

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
