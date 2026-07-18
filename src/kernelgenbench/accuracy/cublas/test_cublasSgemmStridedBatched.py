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

@label("cublasSgemmStridedBatched")
@parametrize("M, N, K", [
    (1, 1, 1),           # Edge case
    (16, 16, 16),        # Small square
    (32, 64, 16),        # Non-square
    (64, 32, 48),        # Non-square reversed
    (128, 256, 64),      # Medium rectangular
    (256, 128, 128),     # Medium rectangular reversed
    (17, 33, 65),        # Non-aligned sizes
    (128, 128, 128),     # Medium square
    (256, 256, 256),     # Larger square
])
@parametrize("alpha, beta", [
    (1.0, 0.0),          # Standard: C = alpha*A*B
    (0.0, 1.0),          # Zero alpha: C = beta*C
    (0.5, 0.5),          # Fractional
    (-1.0, 1.0),         # Negative alpha
    (2.0, -0.5),         # Large alpha, negative beta
    (0.001, 0.999),      # Small alpha
])
@parametrize("transa, transb", [("N", "N"), ("N", "T"), ("T", "N"), ("T", "T")])
@parametrize("batchCount", [1, 2, 4, 8])
@parametrize("dtype", [torch.float32])
def test_accuracy_cublasSgemmStridedBatched(M, N, K, alpha, beta, transa, transb, batchCount, dtype):
    A_shape = (batchCount, K, M) if transa == 'T' else (batchCount, M, K)
    B_shape = (batchCount, N, K) if transb == 'T' else (batchCount, K, N)
    A = torch.randn(A_shape, dtype=dtype, device='cuda')
    B = torch.randn(B_shape, dtype=dtype, device='cuda')
    C = torch.randn(batchCount, M, N, dtype=dtype, device='cuda')
    strideA = A.shape[1] * A.shape[2]
    strideB = B.shape[1] * B.shape[2]
    strideC = M * N
    lda = A.shape[1]  # Leading dimension is the first dimension (rows in column-major)
    ldb = B.shape[1]
    ldc = M  # Leading dimension of C is M (rows)
    C_ref = C.clone()
    C_act = C.clone()
    # Convert trans strings to integers for baseline (cuBLAS C API expects int)
    transa_int = _trans_to_int(transa)
    transb_int = _trans_to_int(transb)
    ref_out = kernelgenbench.baseline.cublasSgemmStridedBatched(transa_int, transb_int, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C_ref, ldc, strideC, batchCount)
    act_out = kernelgenbench.triton.cublasSgemmStridedBatched(transa, transb, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C_act, ldc, strideC, batchCount)
    assert_close(act_out, ref_out, dtype, reduce_dim=K)

    # ========================================
    # Performance Test: triton.testing.do_bench
    # ========================================
    import triton
    from sandbox.utils.accuracy_utils import CustomBenchmarkResult

    # Only run performance test for larger sizes
    if M < 128 or N < 128 or K < 64:
        return None

    # Prepare fresh data for benchmarking
    A_bench = torch.randn(A_shape, dtype=dtype, device='cuda')
    B_bench = torch.randn(B_shape, dtype=dtype, device='cuda')

    # Warmup
    for _ in range(10):
        C_warmup = torch.randn(batchCount, M, N, dtype=dtype, device='cuda')
        _ = kernelgenbench.baseline.cublasSgemmStridedBatched(transa_int, transb_int, M, N, K, alpha, A_bench, lda, strideA, B_bench, ldb, strideB, beta, C_warmup.clone(), ldc, strideC, batchCount)
        _ = kernelgenbench.triton.cublasSgemmStridedBatched(transa, transb, M, N, K, alpha, A_bench, lda, strideA, B_bench, ldb, strideB, beta, C_warmup.clone(), ldc, strideC, batchCount)
    torch.cuda.synchronize()

    # Benchmark with torch.cuda.Event
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Benchmark baseline
    C_baseline = torch.randn(batchCount, M, N, dtype=dtype, device='cuda')
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = kernelgenbench.baseline.cublasSgemmStridedBatched(transa_int, transb_int, M, N, K, alpha, A_bench, lda, strideA, B_bench, ldb, strideB, beta, C_baseline.clone(), ldc, strideC, batchCount)
    end_event.record()
    torch.cuda.synchronize()
    ms_baseline = start_event.elapsed_time(end_event) / 100

    # Benchmark Triton
    C_triton = torch.randn(batchCount, M, N, dtype=dtype, device='cuda')
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = kernelgenbench.triton.cublasSgemmStridedBatched(transa, transb, M, N, K, alpha, A_bench, lda, strideA, B_bench, ldb, strideB, beta, C_triton.clone(), ldc, strideC, batchCount)
    end_event.record()
    torch.cuda.synchronize()
    ms_triton = start_event.elapsed_time(end_event) / 100

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(
        ref_time=ms_baseline,
        res_time=ms_triton,
        speedup=speedup,
    )
