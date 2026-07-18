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

@label("cublasCgemvBatched_64")
@parametrize("MN", [(16, 16), (64, 128), (128, 256), (256, 256), (17, 33)])
@parametrize("alpha_beta", [(1.0, 0.0), (0.5, 0.5), (-1.0, 1.0)])
@parametrize("trans", ["N", "T"])
@parametrize("batchCount", [1, 2, 4])
@parametrize("dtype", [torch.complex64])
def test_accuracy_cublasCgemvBatched_64(MN, alpha_beta, trans, batchCount, dtype):
    m, n = MN
    alpha, beta = alpha_beta
    lda = m
    incx = 1
    incy = 1

    # Determine vector lengths based on trans
    if trans == 'N':
        x_len = n
        y_len = m
        reduce_dim = n
    else:  # 'T'
        x_len = m
        y_len = n
        reduce_dim = m

    # Prepare per-batch tensors
    A_list = []
    x_list = []
    y0_list = []
    y_ref_list = []
    y_act_list = []

    for _ in range(batchCount):
        A = (torch.randn(m, n, device='cuda', dtype=dtype) +
             0j)  # real-imag combined via dtype
        # For complex randomness, add imaginary part explicitly
        A = (torch.randn(m, n, device='cuda') + 1j * torch.randn(m, n, device='cuda')).to(dtype)

        x = (torch.randn(x_len, device='cuda') + 1j * torch.randn(x_len, device='cuda')).to(dtype)
        y0 = (torch.randn(y_len, device='cuda') + 1j * torch.randn(y_len, device='cuda')).to(dtype)

        A_list.append(A)
        x_list.append(x)
        y0_list.append(y0)
        y_ref_list.append(y0.clone())
        y_act_list.append(y0.clone())

    # Convert matrices to column-major representation for cuBLAS by using A^T row-major
    A_cm_list = [A.t().contiguous() for A in A_list]

    # Build device arrays of pointers (int64 tensors on GPU)
    A_ptrs = torch.tensor([A_cm.data_ptr() for A_cm in A_cm_list], dtype=torch.int64, device='cuda')
    x_ptrs = torch.tensor([x.data_ptr() for x in x_list], dtype=torch.int64, device='cuda')
    y_ref_ptrs = torch.tensor([y.data_ptr() for y in y_ref_list], dtype=torch.int64, device='cuda')
    y_act_ptrs = torch.tensor([y.data_ptr() for y in y_act_list], dtype=torch.int64, device='cuda')

    # Reference (baseline) and Actual (triton) calls
    _ = kernelgenbench.baseline.cublasCgemvBatched_64(trans, m, n, alpha, A_ptrs, lda, x_ptrs, incx, beta, y_ref_ptrs, incy, batchCount)
    _ = kernelgenbench.triton.cublasCgemvBatched_64(trans, m, n, alpha, A_ptrs, lda, x_ptrs, incx, beta, y_act_ptrs, incy, batchCount)

    # Stack results for comparison
    ref_out = torch.stack(y_ref_list, dim=0)
    act_out = torch.stack(y_act_list, dim=0)

    assert_close(act_out, ref_out, dtype, reduce_dim=reduce_dim)

    # Performance Test
    from sandbox.utils.accuracy_utils import CustomBenchmarkResult
    if not (m >= 128 and n >= 64):
        return None

    # Benchmark data
    A_bench_list = []
    x_bench_list = []
    y_bench_templates = []

    for _ in range(batchCount):
        A_b = (torch.randn(m, n, device='cuda') + 1j * torch.randn(m, n, device='cuda')).to(dtype)
        x_b = (torch.randn(x_len, device='cuda') + 1j * torch.randn(x_len, device='cuda')).to(dtype)
        y_b = (torch.randn(y_len, device='cuda') + 1j * torch.randn(y_len, device='cuda')).to(dtype)

        A_bench_list.append(A_b)
        x_bench_list.append(x_b)
        y_bench_templates.append(y_b)

    A_bench_cm_list = [A.t().contiguous() for A in A_bench_list]
    A_bench_ptrs = torch.tensor([A_cm.data_ptr() for A_cm in A_bench_cm_list], dtype=torch.int64, device='cuda')
    x_bench_ptrs = torch.tensor([x.data_ptr() for x in x_bench_list], dtype=torch.int64, device='cuda')

    # warmup
    for _ in range(10):
        # Baseline
        y_b_ref = [y.clone() for y in y_bench_templates]
        y_b_ref_ptrs = torch.tensor([y.data_ptr() for y in y_b_ref], dtype=torch.int64, device='cuda')
        _ = kernelgenbench.baseline.cublasCgemvBatched_64(trans, m, n, alpha, A_bench_ptrs, lda, x_bench_ptrs, incx, beta, y_b_ref_ptrs, incy, batchCount)
        # Triton
        y_b_act = [y.clone() for y in y_bench_templates]
        y_b_act_ptrs = torch.tensor([y.data_ptr() for y in y_b_act], dtype=torch.int64, device='cuda')
        _ = kernelgenbench.triton.cublasCgemvBatched_64(trans, m, n, alpha, A_bench_ptrs, lda, x_bench_ptrs, incx, beta, y_b_act_ptrs, incy, batchCount)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Baseline timing
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        y_b_ref = [y.clone() for y in y_bench_templates]
        y_b_ref_ptrs = torch.tensor([y.data_ptr() for y in y_b_ref], dtype=torch.int64, device='cuda')
        _ = kernelgenbench.baseline.cublasCgemvBatched_64(trans, m, n, alpha, A_bench_ptrs, lda, x_bench_ptrs, incx, beta, y_b_ref_ptrs, incy, batchCount)
    end_event.record()
    torch.cuda.synchronize()
    ms_baseline = start_event.elapsed_time(end_event) / 100

    # Triton timing
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        y_b_act = [y.clone() for y in y_bench_templates]
        y_b_act_ptrs = torch.tensor([y.data_ptr() for y in y_b_act], dtype=torch.int64, device='cuda')
        _ = kernelgenbench.triton.cublasCgemvBatched_64(trans, m, n, alpha, A_bench_ptrs, lda, x_bench_ptrs, incx, beta, y_b_act_ptrs, incy, batchCount)
    end_event.record()
    torch.cuda.synchronize()
    ms_triton = start_event.elapsed_time(end_event) / 100

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
