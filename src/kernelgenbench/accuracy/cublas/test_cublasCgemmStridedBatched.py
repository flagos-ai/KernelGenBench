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

@label("cublasCgemmStridedBatched")
@parametrize("M, N, K", [
    (1, 1, 1),
    (16, 16, 16),
    (64, 32, 48),
    (128, 256, 64),
    (17, 33, 65),
])
@parametrize("alpha, beta", [
    (1.0, 0.0),
    (0.5, 0.5),
    (-1.0, 1.0),
])
@parametrize("transa, transb", [("N", "N"), ("N", "T"), ("T", "N"), ("T", "T")])
@parametrize("batchCount", [2, 4])
@parametrize("dtype", [torch.complex64])
def test_accuracy_cublasCgemmStridedBatched(M, N, K, alpha, beta, transa, transb, batchCount, dtype):
    def rand_complex(shape, dtype, device):
        real = torch.randn(shape, dtype=torch.float32, device=device)
        imag = torch.randn(shape, dtype=torch.float32, device=device)
        return torch.complex(real, imag).to(dtype)

    # Row-major shapes based on trans
    A_rows = K if transa == 'T' else M
    A_cols = M if transa == 'T' else K
    B_rows = N if transb == 'T' else K
    B_cols = K if transb == 'T' else N

    A_rm = rand_complex((batchCount, A_rows, A_cols), dtype=dtype, device=device)
    B_rm = rand_complex((batchCount, B_rows, B_cols), dtype=dtype, device=device)
    C_rm = rand_complex((batchCount, M, N), dtype=dtype, device=device)

    # Column-major via transpose+contiguous
    At = A_rm.transpose(-1, -2).contiguous()
    Bt = B_rm.transpose(-1, -2).contiguous()
    Ct_ref = C_rm.transpose(-1, -2).contiguous()
    Ct_act = C_rm.transpose(-1, -2).contiguous()

    lda = A_rows
    ldb = B_rows
    ldc = M
    strideA = A_rows * A_cols
    strideB = B_rows * B_cols
    strideC = M * N

    ref_out = kernelgenbench.baseline.cublasCgemmStridedBatched(
        transa, transb, M, N, K, alpha,
        At, lda, strideA, Bt, ldb, strideB,
        beta, Ct_ref, ldc, strideC, batchCount
    )
    act_out = kernelgenbench.triton.cublasCgemmStridedBatched(
        transa, transb, M, N, K, alpha,
        At.clone(), lda, strideA, Bt.clone(), ldb, strideB,
        beta, Ct_act, ldc, strideC, batchCount
    )

    assert_close(Ct_act, Ct_ref, dtype, reduce_dim=K)

    from sandbox.utils.accuracy_utils import CustomBenchmarkResult
    if M < 128 or N < 128 or K < 64:
        return None

    for _ in range(10):
        Cw = C_rm.transpose(-1, -2).contiguous()
        _ = kernelgenbench.baseline.cublasCgemmStridedBatched(
            transa, transb, M, N, K, alpha,
            At, lda, strideA, Bt, ldb, strideB,
            beta, Cw, ldc, strideC, batchCount)
        Cw2 = C_rm.transpose(-1, -2).contiguous()
        _ = kernelgenbench.triton.cublasCgemmStridedBatched(
            transa, transb, M, N, K, alpha,
            At.clone(), lda, strideA, Bt.clone(), ldb, strideB,
            beta, Cw2, ldc, strideC, batchCount)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        Cb = C_rm.transpose(-1, -2).contiguous()
        _ = kernelgenbench.baseline.cublasCgemmStridedBatched(
            transa, transb, M, N, K, alpha,
            At, lda, strideA, Bt, ldb, strideB,
            beta, Cb, ldc, strideC, batchCount)
    end_event.record()
    torch.cuda.synchronize()
    ms_baseline = start_event.elapsed_time(end_event) / 100

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        Ct2 = C_rm.transpose(-1, -2).contiguous()
        _ = kernelgenbench.triton.cublasCgemmStridedBatched(
            transa, transb, M, N, K, alpha,
            At.clone(), lda, strideA, Bt.clone(), ldb, strideB,
            beta, Ct2, ldc, strideC, batchCount)
    end_event.record()
    torch.cuda.synchronize()
    ms_triton = start_event.elapsed_time(end_event) / 100

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
