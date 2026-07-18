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

@label("cublasDtrsmBatched")
@parametrize("m", [4, 16, 32, 64])
@parametrize("n", [4, 16, 32])
@parametrize("alpha", [1.0, 0.5, -1.0])
@parametrize("batchCount", [1, 2, 4])
@parametrize("dtype", [torch.float64])
def test_accuracy_cublasDtrsmBatched(m, n, alpha, batchCount, dtype):
    side = 0   # CUBLAS_SIDE_LEFT
    uplo = 1   # CUBLAS_FILL_MODE_UPPER
    trans = 0  # CUBLAS_OP_N
    diag = 0   # CUBLAS_DIAG_NON_UNIT
    lda = m
    ldb = m

    # Create per-batch A (upper triangular, invertible) and B (column-major)
    As_ref = []
    Bs_ref = []
    As_act = []
    Bs_act = []
    for _ in range(batchCount):
        diag_vals = 1.0 + torch.rand(m, dtype=dtype, device=device)
        A = torch.diag(diag_vals).to(dtype=dtype, device=device)
        A = A + torch.triu(torch.randn(m, m, dtype=dtype, device=device), diagonal=1) * 0.1
        B_rm = torch.randn(m, n, dtype=dtype, device=device)
        B_cm = B_rm.t().contiguous()  # column-major (n, m)

        As_ref.append(A.clone())
        Bs_ref.append(B_cm.clone())
        As_act.append(A.clone())
        Bs_act.append(B_cm.clone())

    # Build pointer arrays (int64 tensors holding data_ptr)
    Aarray_ref = torch.tensor([a.data_ptr() for a in As_ref], dtype=torch.int64, device=device)
    Barray_ref = torch.tensor([b.data_ptr() for b in Bs_ref], dtype=torch.int64, device=device)
    Aarray_act = torch.tensor([a.data_ptr() for a in As_act], dtype=torch.int64, device=device)
    Barray_act = torch.tensor([b.data_ptr() for b in Bs_act], dtype=torch.int64, device=device)

    kernelgenbench.baseline.cublasDtrsmBatched(side, uplo, trans, diag, m, n, alpha, Aarray_ref, lda, Barray_ref, ldb, batchCount)
    kernelgenbench.triton.cublasDtrsmBatched(side, uplo, trans, diag, m, n, alpha, Aarray_act, lda, Barray_act, ldb, batchCount)

    # Compare results: B is modified in-place, compare each batch
    for i in range(batchCount):
        assert_close(Bs_act[i], Bs_ref[i], dtype, reduce_dim=m)

    from sandbox.utils.accuracy_utils import CustomBenchmarkResult
    if m < 64 or n < 32:
        return None

    # Benchmark setup
    As_b = []
    Bs_b_base = []
    for _ in range(batchCount):
        diag_vals = 1.0 + torch.rand(m, dtype=dtype, device=device)
        A = torch.diag(diag_vals).to(dtype=dtype, device=device)
        B_cm = torch.randn(m, n, dtype=dtype, device=device).t().contiguous()
        As_b.append(A)
        Bs_b_base.append(B_cm)
    Aarray_b = torch.tensor([a.data_ptr() for a in As_b], dtype=torch.int64, device=device)

    # warmup
    for _ in range(10):
        Bs_w = [b.clone() for b in Bs_b_base]
        Barray_w = torch.tensor([b.data_ptr() for b in Bs_w], dtype=torch.int64, device=device)
        kernelgenbench.baseline.cublasDtrsmBatched(side, uplo, trans, diag, m, n, alpha, Aarray_b, lda, Barray_w, ldb, batchCount)
        Bs_w2 = [b.clone() for b in Bs_b_base]
        Barray_w2 = torch.tensor([b.data_ptr() for b in Bs_w2], dtype=torch.int64, device=device)
        kernelgenbench.triton.cublasDtrsmBatched(side, uplo, trans, diag, m, n, alpha, Aarray_b, lda, Barray_w2, ldb, batchCount)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        Bs_bm = [b.clone() for b in Bs_b_base]
        Barray_bm = torch.tensor([b.data_ptr() for b in Bs_bm], dtype=torch.int64, device=device)
        kernelgenbench.baseline.cublasDtrsmBatched(side, uplo, trans, diag, m, n, alpha, Aarray_b, lda, Barray_bm, ldb, batchCount)
    end_event.record()
    torch.cuda.synchronize()
    ms_baseline = start_event.elapsed_time(end_event) / 100

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        Bs_tm = [b.clone() for b in Bs_b_base]
        Barray_tm = torch.tensor([b.data_ptr() for b in Bs_tm], dtype=torch.int64, device=device)
        kernelgenbench.triton.cublasDtrsmBatched(side, uplo, trans, diag, m, n, alpha, Aarray_b, lda, Barray_tm, ldb, batchCount)
    end_event.record()
    torch.cuda.synchronize()
    ms_triton = start_event.elapsed_time(end_event) / 100

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
