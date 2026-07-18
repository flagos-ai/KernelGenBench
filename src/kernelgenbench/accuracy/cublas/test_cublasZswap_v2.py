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

@label("cublasZswap_v2")
@parametrize("n", [1, 32, 71, 1024, 4096, 5333])
@parametrize("incx", [1, 2, 3])
@parametrize("incy", [1, 2])
@parametrize("dtype", [torch.complex128])
def test_accuracy_cublasZswap_v2(n, incx, incy, dtype):

    xr = torch.randn(n * incx, dtype=torch.float64, device='cuda')
    xi = torch.randn(n * incx, dtype=torch.float64, device='cuda')
    yr = torch.randn(n * incy, dtype=torch.float64, device='cuda')
    yi = torch.randn(n * incy, dtype=torch.float64, device='cuda')

    x = (xr + 1j * xi).to(dtype)
    y = (yr + 1j * yi).to(dtype)

    x_ref = x.clone()
    y_ref = y.clone()
    x_act = x.clone()
    y_act = y.clone()

    ref_out = kernelgenbench.baseline.cublasZswap_v2(n, x_ref, incx, y_ref, incy)
    act_out = kernelgenbench.triton.cublasZswap_v2(n, x_act, incx, y_act, incy)

    assert_close(act_out, ref_out, dtype)
    assert_close(y_act, y_ref, dtype)

    from sandbox.utils.accuracy_utils import CustomBenchmarkResult
    if n < 1024:
        return None

    xr_b = torch.randn(n * incx, dtype=torch.float64, device='cuda')
    xi_b = torch.randn(n * incx, dtype=torch.float64, device='cuda')
    yr_b = torch.randn(n * incy, dtype=torch.float64, device='cuda')
    yi_b = torch.randn(n * incy, dtype=torch.float64, device='cuda')

    x_bench = (xr_b + 1j * xi_b).to(dtype)
    y_bench = (yr_b + 1j * yi_b).to(dtype)

    for _ in range(10):
        _ = kernelgenbench.baseline.cublasZswap_v2(n, x_bench.clone(), incx, y_bench.clone(), incy)
        _ = kernelgenbench.triton.cublasZswap_v2(n, x_bench.clone(), incx, y_bench.clone(), incy)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = kernelgenbench.baseline.cublasZswap_v2(n, x_bench.clone(), incx, y_bench.clone(), incy)
    end_event.record()
    torch.cuda.synchronize()
    ms_baseline = start_event.elapsed_time(end_event) / 100

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = kernelgenbench.triton.cublasZswap_v2(n, x_bench.clone(), incx, y_bench.clone(), incy)
    end_event.record()
    torch.cuda.synchronize()
    ms_triton = start_event.elapsed_time(end_event) / 100

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
