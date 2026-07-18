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

@label("cublasSgeam")
@parametrize("m", [1, 32, 1024])
@parametrize("n", [1, 71, 1024])
@parametrize("alpha, beta", [
    (1.0, 0.0),
    (0.5, 0.5),
    (-1.0, 1.0),
])
@parametrize("transa, transb", [(0, 0), (0, 1), (1, 0), (1, 1)])
@parametrize("dtype", [torch.float32])
def test_accuracy_cublasSgeam(m, n, alpha, beta, transa, transb, dtype):
    A_shape = (n, m) if transa == 1 else (m, n)
    B_shape = (n, m) if transb == 1 else (m, n)

    A = torch.randn(A_shape, dtype=dtype, device=device)
    B = torch.randn(B_shape, dtype=dtype, device=device)

    lda = A_shape[0]
    ldb = B_shape[0]
    ldc = m

    C_ref = torch.empty(m, n, dtype=dtype, device=device)
    C_act = torch.empty(m, n, dtype=dtype, device=device)

    ref_out = kernelgenbench.baseline.cublasSgeam(
        transa, transb, m, n, alpha, A, lda, beta, B, ldb, C_ref, ldc
    )
    act_out = kernelgenbench.triton.cublasSgeam(
        transa, transb, m, n, alpha, A, lda, beta, B, ldb, C_act, ldc
    )
    assert_close(act_out, ref_out, dtype)

    from sandbox.utils.accuracy_utils import CustomBenchmarkResult
    if m < 128 or n < 128:
        return None

    A_b = torch.randn(A_shape, dtype=dtype, device=device)
    B_b = torch.randn(B_shape, dtype=dtype, device=device)

    for _ in range(10):
        C_w = torch.empty(m, n, dtype=dtype, device=device)
        _ = kernelgenbench.baseline.cublasSgeam(transa, transb, m, n, alpha, A_b, lda, beta, B_b, ldb, C_w, ldc)
        _ = kernelgenbench.triton.cublasSgeam(transa, transb, m, n, alpha, A_b, lda, beta, B_b, ldb, C_w, ldc)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        C_w = torch.empty(m, n, dtype=dtype, device=device)
        _ = kernelgenbench.baseline.cublasSgeam(transa, transb, m, n, alpha, A_b, lda, beta, B_b, ldb, C_w, ldc)
    end_event.record()
    torch.cuda.synchronize()
    ms_baseline = start_event.elapsed_time(end_event) / 100

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        C_w = torch.empty(m, n, dtype=dtype, device=device)
        _ = kernelgenbench.triton.cublasSgeam(transa, transb, m, n, alpha, A_b, lda, beta, B_b, ldb, C_w, ldc)
    end_event.record()
    torch.cuda.synchronize()
    ms_triton = start_event.elapsed_time(end_event) / 100

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
