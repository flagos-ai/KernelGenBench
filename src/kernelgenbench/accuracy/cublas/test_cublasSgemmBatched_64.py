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

@label("cublasSgemmBatched_64")
@parametrize("M, N, K", [(1, 1, 1), (16, 16, 16), (64, 32, 48), (128, 256, 64), (17, 33, 65)])
@parametrize("alpha, beta", [(1.0, 0.0), (0.5, 0.5), (-1.0, 1.0)])
@parametrize("transa, transb", [("N", "N"), ("N", "T"), ("T", "N"), ("T", "T")])
@parametrize("batchCount", [1, 2, 4])
@parametrize("dtype", [torch.float32])
def test_accuracy_cublasSgemmBatched_64(M, N, K, alpha, beta, transa, transb, batchCount, dtype):
    # Create per-batch matrices
    A_list = []
    B_list = []
    C_list = []
    for _ in range(batchCount):
        A_shape = (K, M) if transa == 'T' else (M, K)
        B_shape = (N, K) if transb == 'T' else (K, N)
        A_list.append(torch.randn(A_shape, dtype=dtype, device='cuda'))
        B_list.append(torch.randn(B_shape, dtype=dtype, device='cuda'))
        C_list.append(torch.randn(M, N, dtype=dtype, device='cuda'))

    # Build pointer arrays (int64 tensors holding data_ptr)
    Aarray = torch.tensor([a.data_ptr() for a in A_list], dtype=torch.int64, device='cuda')
    Barray = torch.tensor([b.data_ptr() for b in B_list], dtype=torch.int64, device='cuda')

    C_ref_list = [c.clone() for c in C_list]
    C_act_list = [c.clone() for c in C_list]
    Carray_ref = torch.tensor([c.data_ptr() for c in C_ref_list], dtype=torch.int64, device='cuda')
    Carray_act = torch.tensor([c.data_ptr() for c in C_act_list], dtype=torch.int64, device='cuda')

    lda = A_list[0].shape[0]
    ldb = B_list[0].shape[0]
    ldc = M

    ref_out = kernelgenbench.baseline.cublasSgemmBatched_64(
        transa, transb, M, N, K, alpha, Aarray, lda, Barray, ldb, beta, Carray_ref, ldc, batchCount)
    act_out = kernelgenbench.triton.cublasSgemmBatched_64(
        transa, transb, M, N, K, alpha, Aarray, lda, Barray, ldb, beta, Carray_act, ldc, batchCount)

    for i in range(batchCount):
        assert_close(C_act_list[i], C_ref_list[i], dtype, reduce_dim=K)

    # Performance Test
    from sandbox.utils.accuracy_utils import CustomBenchmarkResult
    if M < 128 or N < 128 or K < 64:
        return None

    # warmup
    for _ in range(10):
        C_w = [torch.randn(M, N, dtype=dtype, device='cuda') for _ in range(batchCount)]
        Cw_ptrs = torch.tensor([c.data_ptr() for c in C_w], dtype=torch.int64, device='cuda')
        _ = kernelgenbench.baseline.cublasSgemmBatched_64(
            transa, transb, M, N, K, alpha, Aarray, lda, Barray, ldb, beta, Cw_ptrs, ldc, batchCount)
        C_w2 = [torch.randn(M, N, dtype=dtype, device='cuda') for _ in range(batchCount)]
        Cw2_ptrs = torch.tensor([c.data_ptr() for c in C_w2], dtype=torch.int64, device='cuda')
        _ = kernelgenbench.triton.cublasSgemmBatched_64(
            transa, transb, M, N, K, alpha, Aarray, lda, Barray, ldb, beta, Cw2_ptrs, ldc, batchCount)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        C_b = [torch.randn(M, N, dtype=dtype, device='cuda') for _ in range(batchCount)]
        Cb_ptrs = torch.tensor([c.data_ptr() for c in C_b], dtype=torch.int64, device='cuda')
        _ = kernelgenbench.baseline.cublasSgemmBatched_64(
            transa, transb, M, N, K, alpha, Aarray, lda, Barray, ldb, beta, Cb_ptrs, ldc, batchCount)
    end_event.record()
    torch.cuda.synchronize()
    ms_baseline = start_event.elapsed_time(end_event) / 100

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        C_t = [torch.randn(M, N, dtype=dtype, device='cuda') for _ in range(batchCount)]
        Ct_ptrs = torch.tensor([c.data_ptr() for c in C_t], dtype=torch.int64, device='cuda')
        _ = kernelgenbench.triton.cublasSgemmBatched_64(
            transa, transb, M, N, K, alpha, Aarray, lda, Barray, ldb, beta, Ct_ptrs, ldc, batchCount)
    end_event.record()
    torch.cuda.synchronize()
    ms_triton = start_event.elapsed_time(end_event) / 100

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
