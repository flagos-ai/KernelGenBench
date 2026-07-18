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

@label("cublasDgemv_v2")
@parametrize("M_N", [(16, 16), (64, 128), (128, 256), (256, 256), (17, 33)])
@parametrize("alpha_beta", [(1.0, 0.0), (0.5, 0.5), (-1.0, 1.0)])
@parametrize("trans", ["N", "T"])
def test_accuracy_cublasDgemv_v2(M_N, alpha_beta, trans):
    dtype = torch.float64
    m, n = M_N
    alpha, beta = alpha_beta
    lda = m

    if trans == "N":
        len_x = n
        len_y = m
    else:
        len_x = m
        len_y = n

    incx = 1
    incy = 1

    A = torch.randn(m, n, dtype=dtype, device=device)
    x = torch.randn(len_x * incx, dtype=dtype, device=device)
    y0 = torch.randn(len_y * incy, dtype=dtype, device=device)
    y1 = y0.clone()

    ref_out = kernelgenbench.baseline.cublasDgemv_v2(trans, m, n, alpha, A, lda, x, incx, beta, y0, incy)
    act_out = kernelgenbench.triton.cublasDgemv_v2(trans, m, n, alpha, A, lda, x, incx, beta, y1, incy)

    assert_close(act_out, ref_out, dtype)

    from sandbox.utils.accuracy_utils import CustomBenchmarkResult
    if m < 64 or n < 64:
        return None

    for _ in range(10):
        _ = kernelgenbench.baseline.cublasDgemv_v2(trans, m, n, alpha, A, lda, x.clone(), incx, beta, y0.clone(), incy)
        _ = kernelgenbench.triton.cublasDgemv_v2(trans, m, n, alpha, A, lda, x.clone(), incx, beta, y1.clone(), incy)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = kernelgenbench.baseline.cublasDgemv_v2(trans, m, n, alpha, A, lda, x.clone(), incx, beta, y0.clone(), incy)
    end_event.record()
    torch.cuda.synchronize()
    ms_baseline = start_event.elapsed_time(end_event) / 100

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = kernelgenbench.triton.cublasDgemv_v2(trans, m, n, alpha, A, lda, x.clone(), incx, beta, y1.clone(), incy)
    end_event.record()
    torch.cuda.synchronize()
    ms_triton = start_event.elapsed_time(end_event) / 100

    return CustomBenchmarkResult(
        ref_time=ms_baseline,
        res_time=ms_triton,
        speedup=ms_baseline / ms_triton if ms_triton > 0 else 0,
        params={'dtype': str(dtype), 'trans': trans, 'alpha': alpha, 'beta': beta, 'm': m, 'n': n}
    )
