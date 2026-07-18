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

@label("cublasCgemv_v2")
@parametrize("m", [1, 16, 64, 128])
@parametrize("n", [1, 16, 64, 128])
@parametrize("trans", ['N', 'T', 'C'])
@parametrize("alpha", [1.0+0j, 0.0+0j, -0.5+0.5j])
@parametrize("beta", [1.0+0j, 0.0+0j])
@parametrize("incx", [1, 2])
@parametrize("incy", [1, 2])
@parametrize("dtype", [torch.complex64])
def test_accuracy_cublasCgemv_v2(m, n, trans, alpha, beta, incx, incy, dtype):
    # Determine dimensions based on trans
    if trans == 'N':
        x_len = n
        y_len = m
    else:  # T or C
        x_len = m
        y_len = n

    lda = m
    A = torch.randn(m, n, dtype=dtype, device=device)
    x = torch.randn(x_len * incx, dtype=dtype, device=device)
    y0 = torch.randn(y_len * incy, dtype=dtype, device=device)
    y1 = y0.clone()

    ref_out = kernelgenbench.baseline.cublasCgemv_v2(trans, m, n, alpha, A, lda, x, incx, beta, y0, incy)
    act_out = kernelgenbench.triton.cublasCgemv_v2(trans, m, n, alpha, A, lda, x, incx, beta, y1, incy)

    assert_close(act_out, ref_out, dtype)

    from sandbox.utils.accuracy_utils import CustomBenchmarkResult
    if m < 64 or n < 64:
        return None

    A_b = torch.randn(m, n, dtype=dtype, device=device)
    x_b = torch.randn(x_len * incx, dtype=dtype, device=device)
    y_b = torch.randn(y_len * incy, dtype=dtype, device=device)

    for _ in range(10):
        _ = kernelgenbench.baseline.cublasCgemv_v2(trans, m, n, alpha, A_b, lda, x_b.clone(), incx, beta, y_b.clone(), incy)
        _ = kernelgenbench.triton.cublasCgemv_v2(trans, m, n, alpha, A_b, lda, x_b.clone(), incx, beta, y_b.clone(), incy)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = kernelgenbench.baseline.cublasCgemv_v2(trans, m, n, alpha, A_b, lda, x_b.clone(), incx, beta, y_b.clone(), incy)
    end_event.record()
    torch.cuda.synchronize()
    ms_baseline = start_event.elapsed_time(end_event) / 100

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = kernelgenbench.triton.cublasCgemv_v2(trans, m, n, alpha, A_b, lda, x_b.clone(), incx, beta, y_b.clone(), incy)
    end_event.record()
    torch.cuda.synchronize()
    ms_triton = start_event.elapsed_time(end_event) / 100

    return CustomBenchmarkResult(
        ref_time=ms_baseline,
        res_time=ms_triton,
        speedup=ms_baseline / ms_triton if ms_triton > 0 else 0,
        params={'dtype': str(dtype), 'trans': trans, 'alpha': alpha, 'beta': beta, 'm': m, 'n': n, 'incx': incx, 'incy': incy}
    )
