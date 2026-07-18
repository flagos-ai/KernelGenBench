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

@label("cublasSgemvBatched")
@parametrize("M_N", [
    (1, 1),
    (1, 32),
    (32, 1),
    (16, 16),
    (32, 64),
    (64, 128),
    (128, 64),
    (128, 128),
    (128, 256),
    (256, 128),
    (256, 256),
    (17, 33),
    (33, 17),
    (71, 497),
    (497, 71),
    (1024, 4096),
    (4096, 1024),
    (5333, 5333),
])
@parametrize("alpha_beta", [
    (1.0, 0.0),
    (0.0, 1.0),
    (0.5, 0.5),
    (-1.0, 1.0),
    (1.0, 0.5),
    (0.001, 0.999),
    (-0.999, -1.0),
    (100.001, 0.0),
    (0.5, -0.5),
])
@parametrize("trans", ['N', 'T'])
@parametrize("batchCount", [1, 2, 4, 8])
@parametrize("incx", [1, 2, 3])
@parametrize("incy", [1, 2, 3])
@parametrize("dtype", [torch.float32])
def test_accuracy_cublasSgemvBatched(M_N, alpha_beta, trans, batchCount, incx, incy, dtype):
    m, n = M_N
    alpha, beta = alpha_beta
    lda = max(1, m)

    # Determine logical vector lengths based on trans
    if trans == 'N':
        Lx = n
        Ly = m
        K = n
    else:
        Lx = m
        Ly = n
        K = m

    # Allocate per-batch matrices and vectors
    A_buf_list = []
    x_buf_list = []
    y_ref_buf_list = []
    y_act_buf_list = []

    for _ in range(batchCount):
        # Create A in desired row-major (m, n), then store as column-major buffer via transpose
        A_desired = torch.randn((m, n), dtype=dtype, device=device)
        A_buf = A_desired.t().contiguous()  # represents column-major A
        A_buf_list.append(A_buf)

        # Strided x and y buffers
        x_len = (Lx - 1) * incx + 1 if Lx > 0 else 0
        y_len = (Ly - 1) * incy + 1 if Ly > 0 else 0
        x_buf = torch.randn(x_len if x_len > 0 else 1, dtype=dtype, device=device)
        y_init = torch.randn(y_len if y_len > 0 else 1, dtype=dtype, device=device)
        y_ref = y_init.clone()
        y_act = y_init.clone()

        x_buf_list.append(x_buf)
        y_ref_buf_list.append(y_ref)
        y_act_buf_list.append(y_act)

    # Build device arrays of pointers (int64 on device)
    A_ptrs = torch.tensor([t.data_ptr() for t in A_buf_list], dtype=torch.int64, device=device)
    X_ptrs = torch.tensor([t.data_ptr() for t in x_buf_list], dtype=torch.int64, device=device)
    Y_ref_ptrs = torch.tensor([t.data_ptr() for t in y_ref_buf_list], dtype=torch.int64, device=device)
    Y_act_ptrs = torch.tensor([t.data_ptr() for t in y_act_buf_list], dtype=torch.int64, device=device)

    # Reference (cuBLAS ctypes baseline)
    _ = kernelgenbench.baseline.cublasSgemvBatched(trans, m, n, alpha, A_ptrs, lda, X_ptrs, incx, beta, Y_ref_ptrs, incy, batchCount)
    # Actual (Triton)
    _ = kernelgenbench.triton.cublasSgemvBatched(trans, m, n, alpha, A_ptrs, lda, X_ptrs, incx, beta, Y_act_ptrs, incy, batchCount)

    # Extract logical y from strided buffers and stack
    def extract_y(buf, length, stride):
        if length == 0:
            return torch.empty(0, dtype=dtype, device=device)
        return buf[0: (length - 1) * stride + 1: stride].contiguous()

    ref_out = torch.stack([extract_y(y_ref_buf_list[i], Ly, incy) for i in range(batchCount)], dim=0)
    act_out = torch.stack([extract_y(y_act_buf_list[i], Ly, incy) for i in range(batchCount)], dim=0)

    assert_close(act_out, ref_out, dtype, reduce_dim=K)

    # Performance Test
    from sandbox.utils.accuracy_utils import CustomBenchmarkResult
    if not (m >= 128 and n >= 64):
        return None

    # Prepare separate buffers for benchmarking
    bench_A_buf_list = []
    bench_x_buf_list = []
    bench_y_ref_buf_list = []
    bench_y_act_buf_list = []

    for _ in range(batchCount):
        A_desired = torch.randn((m, n), dtype=dtype, device=device)
        A_buf = A_desired.t().contiguous()
        bench_A_buf_list.append(A_buf)

        x_len = (Lx - 1) * incx + 1 if Lx > 0 else 0
        y_len = (Ly - 1) * incy + 1 if Ly > 0 else 0
        x_buf = torch.randn(x_len if x_len > 0 else 1, dtype=dtype, device=device)
        y_init = torch.randn(y_len if y_len > 0 else 1, dtype=dtype, device=device)
        y_ref = y_init.clone()
        y_act = y_init.clone()

        bench_x_buf_list.append(x_buf)
        bench_y_ref_buf_list.append(y_ref)
        bench_y_act_buf_list.append(y_act)

    bench_A_ptrs = torch.tensor([t.data_ptr() for t in bench_A_buf_list], dtype=torch.int64, device=device)
    bench_X_ptrs = torch.tensor([t.data_ptr() for t in bench_x_buf_list], dtype=torch.int64, device=device)
    bench_Y_ref_ptrs = torch.tensor([t.data_ptr() for t in bench_y_ref_buf_list], dtype=torch.int64, device=device)
    bench_Y_act_ptrs = torch.tensor([t.data_ptr() for t in bench_y_act_buf_list], dtype=torch.int64, device=device)

    # warmup
    for _ in range(10):
        _ = kernelgenbench.baseline.cublasSgemvBatched(trans, m, n, alpha, bench_A_ptrs, lda, bench_X_ptrs, incx, beta, bench_Y_ref_ptrs, incy, batchCount)
        _ = kernelgenbench.triton.cublasSgemvBatched(trans, m, n, alpha, bench_A_ptrs, lda, bench_X_ptrs, incx, beta, bench_Y_act_ptrs, incy, batchCount)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = kernelgenbench.baseline.cublasSgemvBatched(trans, m, n, alpha, bench_A_ptrs, lda, bench_X_ptrs, incx, beta, bench_Y_ref_ptrs, incy, batchCount)
    end_event.record()
    torch.cuda.synchronize()
    ms_baseline = start_event.elapsed_time(end_event) / 100

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = kernelgenbench.triton.cublasSgemvBatched(trans, m, n, alpha, bench_A_ptrs, lda, bench_X_ptrs, incx, beta, bench_Y_act_ptrs, incy, batchCount)
    end_event.record()
    torch.cuda.synchronize()
    ms_triton = start_event.elapsed_time(end_event) / 100

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
