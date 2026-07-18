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

@label("cublasSgemmEx")
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
])
@parametrize("alpha, beta", [
    (1.0, 0.0),
    (0.5, 0.5),
    (-1.0, 1.0),
])
@parametrize("transa, transb", [("N", "N"), ("N", "T"), ("T", "N"), ("T", "T")])
@parametrize("dtype", [torch.float32])
def test_accuracy_cublasSgemmEx(M, N, K, alpha, beta, transa, transb, dtype):
    CUDA_R_32F = 0
    
    A_shape = (K, M) if transa == 'T' else (M, K)
    B_shape = (N, K) if transb == 'T' else (K, N)
    A = torch.randn(A_shape, dtype=dtype, device='cuda')
    B = torch.randn(B_shape, dtype=dtype, device='cuda')
    C_ref = torch.randn((M, N), dtype=dtype, device='cuda')
    C_act = C_ref.clone()
    
    lda = A.shape[0]
    ldb = B.shape[0]
    ldc = M
    
    ref_out = kernelgenbench.baseline.cublasSgemmEx(
        transa, transb, M, N, K, alpha,
        A, CUDA_R_32F, lda,
        B, CUDA_R_32F, ldb,
        beta, C_ref, CUDA_R_32F, ldc
    )
    
    act_out = kernelgenbench.triton.cublasSgemmEx(
        transa, transb, M, N, K, alpha,
        A, CUDA_R_32F, lda,
        B, CUDA_R_32F, ldb,
        beta, C_act, CUDA_R_32F, ldc
    )
    
    assert_close(act_out, ref_out, dtype, reduce_dim=K)

    from sandbox.utils.accuracy_utils import CustomBenchmarkResult
    if M < 128 or N < 128 or K < 64:
        return None

    for _ in range(10):
        C_w = torch.randn((M, N), dtype=dtype, device='cuda')
        _ = kernelgenbench.baseline.cublasSgemmEx(transa, transb, M, N, K, alpha, A, CUDA_R_32F, lda, B, CUDA_R_32F, ldb, beta, C_w.clone(), CUDA_R_32F, ldc)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        _ = kernelgenbench.baseline.cublasSgemmEx(transa, transb, M, N, K, alpha, A, CUDA_R_32F, lda, B, CUDA_R_32F, ldb, beta, torch.randn((M,N), dtype=dtype, device='cuda'), CUDA_R_32F, ldc)
    end.record()
    torch.cuda.synchronize()
    ms_baseline = start.elapsed_time(end) / 100

    start.record()
    for _ in range(100):
        _ = kernelgenbench.triton.cublasSgemmEx(transa, transb, M, N, K, alpha, A, CUDA_R_32F, lda, B, CUDA_R_32F, ldb, beta, torch.randn((M,N), dtype=dtype, device='cuda'), CUDA_R_32F, ldc)
    end.record()
    torch.cuda.synchronize()
    ms_triton = start.elapsed_time(end) / 100

    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=ms_baseline/ms_triton)
