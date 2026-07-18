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


def _trans_to_int(trans_str):
    """Convert transpose string to cuBLAS integer: 'N'->0, 'T'->1"""
    return 0 if trans_str == 'N' else 1

@label("cublasDgemvStridedBatched")
@parametrize("M, N", [
    (1, 1),
    (16, 16),
    (32, 64),
    (64, 32),
    (128, 64),
    (256, 128),
    (17, 33),
    (128, 128),
    (256, 256),
])
@parametrize("alpha, beta", [
    (1.0, 0.0),
    (0.0, 1.0),
    (0.5, 0.5),
    (-1.0, 1.0),
    (2.0, -0.5),
    (0.001, 0.999),
])
@parametrize("trans", ["N", "T"])
@parametrize("batchCount", [1, 2, 4, 8])
@parametrize("dtype", [torch.float64])
def test_accuracy_cublasDgemvStridedBatched(M, N, alpha, beta, trans, batchCount, dtype):
    A = torch.randn(batchCount, M, N, dtype=dtype, device='cuda')
    lenx = N if trans == 'N' else M
    leny = M if trans == 'N' else N
    x = torch.randn(batchCount, lenx, dtype=dtype, device='cuda')
    y = torch.randn(batchCount, leny, dtype=dtype, device='cuda')

    incx = 1
    incy = 1
    lda = A.shape[1]  # Leading dimension (rows in column-major)
    strideA = M * N
    stridex = lenx
    stridey = leny

    y_ref = y.clone()
    y_act = y.clone()

    # Convert trans string to integer for baseline (cuBLAS C API expects int)
    trans_int = _trans_to_int(trans)
    ref_out = kernelgenbench.baseline.cublasDgemvStridedBatched(trans_int, M, N, alpha, A, lda, strideA, x, incx, stridex, beta, y_ref, incy, stridey, batchCount)
    act_out = kernelgenbench.triton.cublasDgemvStridedBatched(trans, M, N, alpha, A, lda, strideA, x, incx, stridex, beta, y_act, incy, stridey, batchCount)
    assert_close(act_out, ref_out, dtype, reduce_dim=N if trans == 'N' else M)

    # Performance Test
    import triton
    from sandbox.utils.accuracy_utils import CustomBenchmarkResult
    if M < 128 or N < 64:
        return None
    A_bench = torch.randn(batchCount, M, N, dtype=dtype, device='cuda')
    x_bench = torch.randn(batchCount, lenx, dtype=dtype, device='cuda')
    for _ in range(10):
        y_warmup = torch.randn(batchCount, leny, dtype=dtype, device='cuda')
        _ = kernelgenbench.baseline.cublasDgemvStridedBatched(trans_int, M, N, alpha, A_bench, lda, strideA, x_bench, incx, stridex, beta, y_warmup.clone(), incy, stridey, batchCount)
        _ = kernelgenbench.triton.cublasDgemvStridedBatched(trans, M, N, alpha, A_bench, lda, strideA, x_bench, incx, stridex, beta, y_warmup.clone(), incy, stridey, batchCount)
    torch.cuda.synchronize()

    # Benchmark with torch.cuda.Event
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    y_baseline = torch.randn(batchCount, leny, dtype=dtype, device='cuda')
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = kernelgenbench.baseline.cublasDgemvStridedBatched(trans_int, M, N, alpha, A_bench, lda, strideA, x_bench, incx, stridex, beta, y_baseline.clone(), incy, stridey, batchCount)
    end_event.record()
    torch.cuda.synchronize()
    ms_baseline = start_event.elapsed_time(end_event) / 100

    y_triton = torch.randn(batchCount, leny, dtype=dtype, device='cuda')
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = kernelgenbench.triton.cublasDgemvStridedBatched(trans, M, N, alpha, A_bench, lda, strideA, x_bench, incx, stridex, beta, y_triton.clone(), incy, stridey, batchCount)
    end_event.record()
    torch.cuda.synchronize()
    ms_triton = start_event.elapsed_time(end_event) / 100
    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
