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

@label("cublasZgemvBatched")
@parametrize("M_N", [[16, 16], [64, 128], [128, 256], [256, 256], [17, 33]])
@parametrize("alpha_beta", [[1.0, 0.0], [0.5, 0.5], [-1.0, 1.0]])
@parametrize("trans", ['N', 'T'])
@parametrize("batchCount", [1, 2, 4])
def test_accuracy_cublasZgemvBatched(M_N, alpha_beta, trans, batchCount):
    dtype = torch.complex128
    m, n = M_N
    alpha, beta = alpha_beta

    # Determine vector sizes based on trans
    if trans == 'N':
        x_len = n
        y_len = m
    else:
        x_len = m
        y_len = n

    # Prepare batched inputs
    A_list = []
    x_list = []
    y0_list = []

    for _ in range(batchCount):
        A_real = torch.randn(m, n, device='cuda', dtype=torch.float64)
        A_imag = torch.randn(m, n, device='cuda', dtype=torch.float64)
        A = (A_real + 1j * A_imag).to(dtype).contiguous()

        x_real = torch.randn(x_len, device='cuda', dtype=torch.float64)
        x_imag = torch.randn(x_len, device='cuda', dtype=torch.float64)
        x = (x_real + 1j * x_imag).to(dtype).contiguous()

        y_real = torch.randn(y_len, device='cuda', dtype=torch.float64)
        y_imag = torch.randn(y_len, device='cuda', dtype=torch.float64)
        y = (y_real + 1j * y_imag).to(dtype).contiguous()

        A_list.append(A)
        x_list.append(x)
        y0_list.append(y)

    # Prepare pointer arrays for A and x (shared between baseline and triton)
    A_ptrs = torch.tensor([t.data_ptr() for t in A_list], dtype=torch.int64, device='cuda')
    x_ptrs = torch.tensor([t.data_ptr() for t in x_list], dtype=torch.int64, device='cuda')

    # Prepare y clones for baseline and triton
    y_ref_list = [t.clone() for t in y0_list]
    y_act_list = [t.clone() for t in y0_list]

    y_ref_ptrs = torch.tensor([t.data_ptr() for t in y_ref_list], dtype=torch.int64, device='cuda')
    y_act_ptrs = torch.tensor([t.data_ptr() for t in y_act_list], dtype=torch.int64, device='cuda')

    lda = m
    incx = 1
    incy = 1

    # Reference (baseline) and Actual (triton)
    _ = kernelgenbench.baseline.cublasZgemvBatched(trans, m, n, alpha, A_ptrs, lda, x_ptrs, incx, beta, y_ref_ptrs, incy, batchCount)
    _ = kernelgenbench.triton.cublasZgemvBatched(trans, m, n, alpha, A_ptrs, lda, x_ptrs, incx, beta, y_act_ptrs, incy, batchCount)

    ref_out = torch.stack(y_ref_list, dim=0)
    act_out = torch.stack(y_act_list, dim=0)

    K = x_len
    assert_close(act_out, ref_out, dtype, reduce_dim=K)

    # Performance Test
    from sandbox.utils.accuracy_utils import CustomBenchmarkResult
    if not (m >= 128 and n >= 64):
        return None

    # Prepare benchmark inputs
    A_bench_list = A_list  # reuse A
    x_bench_list = x_list  # reuse x

    A_bench_ptrs = torch.tensor([t.data_ptr() for t in A_bench_list], dtype=torch.int64, device='cuda')
    x_bench_ptrs = torch.tensor([t.data_ptr() for t in x_bench_list], dtype=torch.int64, device='cuda')

    # Fixed initial y to clone from in each iter
    y_bench_init = []
    for _ in range(batchCount):
        yr = torch.randn(y_len, device='cuda', dtype=torch.float64)
        yi = torch.randn(y_len, device='cuda', dtype=torch.float64)
        y0 = (yr + 1j * yi).to(dtype).contiguous()
        y_bench_init.append(y0)

    # warmup
    for _ in range(10):
        y_ref_warm = [t.clone() for t in y_bench_init]
        y_act_warm = [t.clone() for t in y_bench_init]
        y_ref_warm_ptrs = torch.tensor([t.data_ptr() for t in y_ref_warm], dtype=torch.int64, device='cuda')
        y_act_warm_ptrs = torch.tensor([t.data_ptr() for t in y_act_warm], dtype=torch.int64, device='cuda')
        _ = kernelgenbench.baseline.cublasZgemvBatched(trans, m, n, alpha, A_bench_ptrs, lda, x_bench_ptrs, incx, beta, y_ref_warm_ptrs, incy, batchCount)
        _ = kernelgenbench.triton.cublasZgemvBatched(trans, m, n, alpha, A_bench_ptrs, lda, x_bench_ptrs, incx, beta, y_act_warm_ptrs, incy, batchCount)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Baseline timing
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        y_ref_iter = [t.clone() for t in y_bench_init]
        y_ref_iter_ptrs = torch.tensor([t.data_ptr() for t in y_ref_iter], dtype=torch.int64, device='cuda')
        _ = kernelgenbench.baseline.cublasZgemvBatched(trans, m, n, alpha, A_bench_ptrs, lda, x_bench_ptrs, incx, beta, y_ref_iter_ptrs, incy, batchCount)
    end_event.record()
    torch.cuda.synchronize()
    ms_baseline = start_event.elapsed_time(end_event) / 100

    # Triton timing
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        y_act_iter = [t.clone() for t in y_bench_init]
        y_act_iter_ptrs = torch.tensor([t.data_ptr() for t in y_act_iter], dtype=torch.int64, device='cuda')
        _ = kernelgenbench.triton.cublasZgemvBatched(trans, m, n, alpha, A_bench_ptrs, lda, x_bench_ptrs, incx, beta, y_act_iter_ptrs, incy, batchCount)
    end_event.record()
    torch.cuda.synchronize()
    ms_triton = start_event.elapsed_time(end_event) / 100

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
