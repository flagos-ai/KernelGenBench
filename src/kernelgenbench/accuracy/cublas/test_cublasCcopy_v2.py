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

@label("cublasCcopy_v2")
@parametrize("n", [1, 17, 32, 33, 71, 160, 497, 1024, 4113, 4096, 5333])
@parametrize("incx", [1, 2, 3])
@parametrize("incy", [1, 2, 3])
@parametrize("dtype", [torch.complex64])
def test_accuracy_cublasCcopy_v2(n, incx, incy, dtype):
    def rand_complex(size, dtype, device):
        real = torch.randn(size, dtype=torch.float32, device=device)
        imag = torch.randn(size, dtype=torch.float32, device=device)
        return (real + 1j * imag).to(dtype)

    size_x = 1 + (n - 1) * incx
    size_y = 1 + (n - 1) * incy

    x = rand_complex(size_x, dtype, 'cuda')
    y = rand_complex(size_y, dtype, 'cuda')

    x_ref = x.clone()
    y_ref = y.clone()
    x_act = x.clone()
    y_act = y.clone()

    ref_out = kernelgenbench.baseline.cublasCcopy_v2(n, x_ref, incx, y_ref, incy)
    act_out = kernelgenbench.triton.cublasCcopy_v2(n, x_act, incx, y_act, incy)

    assert_close(act_out, ref_out, dtype)

    # Performance Test
    from sandbox.utils.accuracy_utils import CustomBenchmarkResult
    if n < 1024:
        return None

    x_bench = rand_complex(size_x, dtype, 'cuda')
    y_bench = rand_complex(size_y, dtype, 'cuda')

    # warmup
    for _ in range(10):
        _ = kernelgenbench.baseline.cublasCcopy_v2(n, x_bench, incx, y_bench, incy)
        _ = kernelgenbench.triton.cublasCcopy_v2(n, x_bench, incx, y_bench, incy)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = kernelgenbench.baseline.cublasCcopy_v2(n, x_bench, incx, y_bench, incy)
    end_event.record()
    torch.cuda.synchronize()
    ms_baseline = start_event.elapsed_time(end_event) / 100

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = kernelgenbench.triton.cublasCcopy_v2(n, x_bench, incx, y_bench, incy)
    end_event.record()
    torch.cuda.synchronize()
    ms_triton = start_event.elapsed_time(end_event) / 100

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
