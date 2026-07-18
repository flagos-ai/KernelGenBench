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

@label("cublasCsymm_v2")
@parametrize("n", [1, 32, 64, 128, 160])
@parametrize("alpha", [1.0, 0.0, -1.0])
@parametrize("beta", [0.0, 0.5])
@parametrize("side", [0, 1])
@parametrize("dtype", [torch.complex64])
def test_accuracy_cublasCsymm_v2(n, alpha, beta, side, dtype):
    m = n
    # side=0: LEFT, A is m×m; side=1: RIGHT, A is n×n
    k = m if side == 0 else n

    A_rand = torch.randn(k, k, dtype=dtype, device=device)
    A_sym = A_rand + A_rand.t()

    B_rm = torch.randn(m, n, dtype=dtype, device=device)
    C_rm = torch.randn(m, n, dtype=dtype, device=device)

    A_cm = A_sym.t().contiguous()
    B_cm = B_rm.t().contiguous()

    uplo = 1  # UPPER
    lda = k
    ldb = m
    ldc = m

    C_cm_ref = C_rm.t().contiguous().clone()
    C_cm_act = C_rm.t().contiguous().clone()

    ref_out_cm = kernelgenbench.baseline.cublasCsymm_v2(
        side, uplo, m, n, alpha, A_cm.clone(), lda, B_cm.clone(), ldb, beta, C_cm_ref, ldc
    )
    act_out_cm = kernelgenbench.triton.cublasCsymm_v2(
        side, uplo, m, n, alpha, A_cm.clone(), lda, B_cm.clone(), ldb, beta, C_cm_act, ldc
    )

    assert_close(act_out_cm, ref_out_cm, dtype, reduce_dim=k)

    from sandbox.utils.accuracy_utils import CustomBenchmarkResult
    if n < 128:
        return None

    A_b = (torch.randn(k, k, dtype=dtype, device=device))
    A_b = (A_b + A_b.t()).t().contiguous()
    B_b = torch.randn(m, n, dtype=dtype, device=device).t().contiguous()
    C_b = torch.randn(m, n, dtype=dtype, device=device).t().contiguous()

    for _ in range(10):
        _ = kernelgenbench.baseline.cublasCsymm_v2(side, uplo, m, n, alpha, A_b.clone(), lda, B_b.clone(), ldb, beta, C_b.clone(), ldc)
        _ = kernelgenbench.triton.cublasCsymm_v2(side, uplo, m, n, alpha, A_b.clone(), lda, B_b.clone(), ldb, beta, C_b.clone(), ldc)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = kernelgenbench.baseline.cublasCsymm_v2(side, uplo, m, n, alpha, A_b.clone(), lda, B_b.clone(), ldb, beta, C_b.clone(), ldc)
    end_event.record()
    torch.cuda.synchronize()
    ms_baseline = start_event.elapsed_time(end_event) / 100

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = kernelgenbench.triton.cublasCsymm_v2(side, uplo, m, n, alpha, A_b.clone(), lda, B_b.clone(), ldb, beta, C_b.clone(), ldc)
    end_event.record()
    torch.cuda.synchronize()
    ms_triton = start_event.elapsed_time(end_event) / 100

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
