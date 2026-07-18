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

@label("cublasStrsm_v2")
@parametrize("m", [1, 16, 48, 64])
@parametrize("n", [1, 8, 16])
@parametrize("alpha", [1.0])
@parametrize("trans", [0, 1])
@parametrize("uplo", [0, 1])
@parametrize("diag", [0, 1])
@parametrize("dtype", [torch.float32])
def test_accuracy_cublasStrsm_v2(m, n, alpha, trans, uplo, diag, dtype):
    side = 0  # CUBLAS_SIDE_LEFT
    lda = m
    ldb = m

    A_full = torch.randn(m, m, dtype=dtype, device=device)
    if uplo == 0:
        A = torch.tril(A_full)
    else:
        A = torch.triu(A_full)
    if m > 0:
        A.diagonal().add_(5.0)

    B = torch.randn(m, n, dtype=dtype, device=device)
    B_ref = B.clone()
    B_act = B.clone()

    ref_out = kernelgenbench.baseline.cublasStrsm_v2(
        side, uplo, trans, diag, m, n, alpha, A, lda, B_ref, ldb
    )
    act_out = kernelgenbench.triton.cublasStrsm_v2(
        side, uplo, trans, diag, m, n, alpha, A, lda, B_act, ldb
    )

    assert_close(act_out, ref_out, dtype, reduce_dim=m)

    from sandbox.utils.accuracy_utils import CustomBenchmarkResult
    if m < 48 or n < 8:
        return None

    A_b = torch.randn(m, m, dtype=dtype, device=device)
    if uplo == 0:
        A_b = torch.tril(A_b)
    else:
        A_b = torch.triu(A_b)
    A_b.diagonal().add_(5.0)
    B_b = torch.randn(m, n, dtype=dtype, device=device)

    for _ in range(10):
        _ = kernelgenbench.baseline.cublasStrsm_v2(
            side, uplo, trans, diag, m, n, alpha, A_b, lda, B_b.clone(), ldb
        )
        _ = kernelgenbench.triton.cublasStrsm_v2(
            side, uplo, trans, diag, m, n, alpha, A_b, lda, B_b.clone(), ldb
        )
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = kernelgenbench.baseline.cublasStrsm_v2(
            side, uplo, trans, diag, m, n, alpha, A_b, lda, B_b.clone(), ldb
        )
    end_event.record()
    torch.cuda.synchronize()
    ms_baseline = start_event.elapsed_time(end_event) / 100

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = kernelgenbench.triton.cublasStrsm_v2(
            side, uplo, trans, diag, m, n, alpha, A_b, lda, B_b.clone(), ldb
        )
    end_event.record()
    torch.cuda.synchronize()
    ms_triton = start_event.elapsed_time(end_event) / 100

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
