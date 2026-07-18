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

@label("cublasDgemvBatched")
@parametrize("M_N", [(16, 16), (64, 128), (128, 256), (256, 256), (17, 33)])
@parametrize("alpha_beta", [(1.0, 0.0), (0.5, 0.5), (-1.0, 1.0)])
@parametrize("trans", ["N", "T"])
@parametrize("batchCount", [1, 2, 4])
def test_accuracy_cublasDgemvBatched(M_N, alpha_beta, trans, batchCount):
    dtype = torch.float64
    m, n = M_N
    alpha, beta = alpha_beta
    incx, incy = 1, 1

    # Map row-major tensors to cuBLAS column-major expectations via transpose trick:
    # A_rm (m,n) row-major is treated as A_cm (n,m) column-major, so:
    # - For trans=='N' (row-major), call cuBLAS with trans_cu='T', m_cu=n, n_cu=m, lda=n
    # - For trans=='T' (row-major), call cuBLAS with trans_cu='N', m_cu=n, n_cu=m, lda=n
    if trans == "N":
        trans_cu = "T"
        m_cu, n_cu = n, m
        x_len = n
        y_len = m
    else:  # trans == "T"
        trans_cu = "N"
        m_cu, n_cu = n, m
        x_len = m
        y_len = n
    lda = n  # leading dimension for A_cm (n rows)

    # Create batched inputs
    A_list = [torch.randn(m, n, dtype=dtype, device='cuda').contiguous() for _ in range(batchCount)]
    x_list = [torch.randn(x_len, dtype=dtype, device='cuda').contiguous() for _ in range(batchCount)]
    y_init_list = [torch.randn(y_len, dtype=dtype, device='cuda').contiguous() for _ in range(batchCount)]

    # Reference and actual y buffers
    y_ref_list = [y.clone() for y in y_init_list]
    y_act_list = [y.clone() for y in y_init_list]

    # Build device pointer arrays
    Aarray = torch.tensor([A.data_ptr() for A in A_list], dtype=torch.int64, device='cuda')
    xarray = torch.tensor([x.data_ptr() for x in x_list], dtype=torch.int64, device='cuda')
    yarray_ref = torch.tensor([y.data_ptr() for y in y_ref_list], dtype=torch.int64, device='cuda')
    yarray_act = torch.tensor([y.data_ptr() for y in y_act_list], dtype=torch.int64, device='cuda')

    # Call baseline (reference)
    _ = kernelgenbench.baseline.cublasDgemvBatched(
        trans_cu, m_cu, n_cu, alpha, Aarray, lda, xarray, incx, beta, yarray_ref, incy, batchCount
    )

    # Call triton (actual)
    _ = kernelgenbench.triton.cublasDgemvBatched(
        trans_cu, m_cu, n_cu, alpha, Aarray, lda, xarray, incx, beta, yarray_act, incy, batchCount
    )

    ref_out = torch.stack(y_ref_list, dim=0)
    act_out = torch.stack(y_act_list, dim=0)

    # Reduce dimension K is the inner dimension of the matvec
    K = n if trans == "N" else m
    assert_close(act_out, ref_out, dtype, reduce_dim=K)

    # Performance Test
    from sandbox.utils.accuracy_utils import CustomBenchmarkResult
    if not (m >= 128 and n >= 64):
        return None

    # Prepare benchmark inputs
    A_bench_list = [torch.randn(m, n, dtype=dtype, device='cuda').contiguous() for _ in range(batchCount)]
    x_bench_list = [torch.randn(x_len, dtype=dtype, device='cuda').contiguous() for _ in range(batchCount)]
    y_seed_list = [torch.randn(y_len, dtype=dtype, device='cuda').contiguous() for _ in range(batchCount)]

    Aarray_bench = torch.tensor([A.data_ptr() for A in A_bench_list], dtype=torch.int64, device='cuda')
    xarray_bench = torch.tensor([x.data_ptr() for x in x_bench_list], dtype=torch.int64, device='cuda')

    # warmup
    for _ in range(10):
        # Baseline warmup
        y_ref_tmp = [y.clone() for y in y_seed_list]
        yarray_ref_tmp = torch.tensor([y.data_ptr() for y in y_ref_tmp], dtype=torch.int64, device='cuda')
        _ = kernelgenbench.baseline.cublasDgemvBatched(
            trans_cu, m_cu, n_cu, alpha, Aarray_bench, lda, xarray_bench, incx, beta, yarray_ref_tmp, incy, batchCount
        )
        # Triton warmup
        y_act_tmp = [y.clone() for y in y_seed_list]
        yarray_act_tmp = torch.tensor([y.data_ptr() for y in y_act_tmp], dtype=torch.int64, device='cuda')
        _ = kernelgenbench.triton.cublasDgemvBatched(
            trans_cu, m_cu, n_cu, alpha, Aarray_bench, lda, xarray_bench, incx, beta, yarray_act_tmp, incy, batchCount
        )
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Benchmark baseline
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        y_ref_tmp = [y.clone() for y in y_seed_list]
        yarray_ref_tmp = torch.tensor([y.data_ptr() for y in y_ref_tmp], dtype=torch.int64, device='cuda')
        _ = kernelgenbench.baseline.cublasDgemvBatched(
            trans_cu, m_cu, n_cu, alpha, Aarray_bench, lda, xarray_bench, incx, beta, yarray_ref_tmp, incy, batchCount
        )
    end_event.record()
    torch.cuda.synchronize()
    ms_baseline = start_event.elapsed_time(end_event) / 100

    # Benchmark triton
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        y_act_tmp = [y.clone() for y in y_seed_list]
        yarray_act_tmp = torch.tensor([y.data_ptr() for y in y_act_tmp], dtype=torch.int64, device='cuda')
        _ = kernelgenbench.triton.cublasDgemvBatched(
            trans_cu, m_cu, n_cu, alpha, Aarray_bench, lda, xarray_bench, incx, beta, yarray_act_tmp, incy, batchCount
        )
    end_event.record()
    torch.cuda.synchronize()
    ms_triton = start_event.elapsed_time(end_event) / 100

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
