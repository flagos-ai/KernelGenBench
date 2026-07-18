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

@label("cublasSgemm_v2")
@parametrize("M, N, K", [
    (1, 1, 1),
    (16, 16, 16),
    (32, 64, 16),
    (64, 32, 48),
    (128, 256, 64),
    (256, 128, 128),
    (17, 33, 65),
    (128, 128, 128),
    (256, 256, 256),
    (71, 71, 71),
    (33, 71, 497),
])
@parametrize("alpha, beta", [
    (1.0, 0.0),
    (0.0, 1.0),
    (0.5, 0.5),
    (-1.0, 1.0),
    (2.0, -0.5),
    (0.001, 0.999),
    (-0.999, 0.5),
    (100.001, -1.0),
])
@parametrize("transa, transb", [("N", "N"), ("N", "T"), ("T", "N"), ("T", "T")])
@parametrize("dtype", [torch.float32])
def test_accuracy_cublasSgemm_v2(M, N, K, alpha, beta, transa, transb, dtype):
    A_shape = (K, M) if transa == 'T' else (M, K)
    B_shape = (N, K) if transb == 'T' else (K, N)
    A = torch.randn(A_shape, dtype=dtype, device='cuda')
    B = torch.randn(B_shape, dtype=dtype, device='cuda')
    C = torch.randn(M, N, dtype=dtype, device='cuda')

    # Leading dimension: for row-major PyTorch tensors viewed as column-major by cuBLAS
    # lda is the first dimension (rows) of the matrix
    lda = A.shape[0]
    ldb = B.shape[0]
    ldc = M

    C_ref = C.clone()
    C_act = C.clone()

    ref_out = kernelgenbench.baseline.cublasSgemm_v2(
        transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C_ref, ldc
    )
    act_out = kernelgenbench.triton.cublasSgemm_v2(
        transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C_act, ldc
    )
    assert_close(act_out, ref_out, dtype, reduce_dim=K)

    from sandbox.utils.accuracy_utils import CustomBenchmarkResult
    if M < 128 or N < 128 or K < 64:
        return None

    A_bench = torch.randn(A_shape, dtype=dtype, device='cuda')
    B_bench = torch.randn(B_shape, dtype=dtype, device='cuda')

    # warmup
    for _ in range(10):
        C_w = torch.randn(M, N, dtype=dtype, device='cuda')
        _ = kernelgenbench.baseline.cublasSgemm_v2(
            transa, transb, M, N, K, alpha, A_bench, lda, B_bench, ldb, beta, C_w.clone(), ldc
        )
        _ = kernelgenbench.triton.cublasSgemm_v2(
            transa, transb, M, N, K, alpha, A_bench, lda, B_bench, ldb, beta, C_w.clone(), ldc
        )
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    C_b = torch.randn(M, N, dtype=dtype, device='cuda')
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = kernelgenbench.baseline.cublasSgemm_v2(
            transa, transb, M, N, K, alpha, A_bench, lda, B_bench, ldb, beta, C_b.clone(), ldc
        )
    end_event.record()
    torch.cuda.synchronize()
    ms_baseline = start_event.elapsed_time(end_event) / 100

    C_t = torch.randn(M, N, dtype=dtype, device='cuda')
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = kernelgenbench.triton.cublasSgemm_v2(
            transa, transb, M, N, K, alpha, A_bench, lda, B_bench, ldb, beta, C_t.clone(), ldc
        )
    end_event.record()
    torch.cuda.synchronize()
    ms_triton = start_event.elapsed_time(end_event) / 100

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
