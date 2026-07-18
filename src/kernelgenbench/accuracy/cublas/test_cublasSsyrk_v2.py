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

@label("cublasSsyrk_v2")
@parametrize("n", [1, 32, 71, 1024, 4096, 5333])
@parametrize("alpha", [1.0, 0.0, -0.999, 0.5, 100.001])
@parametrize("trans", [0, 1])  # CUBLAS_OP_N=0, CUBLAS_OP_T=1
@parametrize("dtype", [torch.float32])
def test_accuracy_cublasSsyrk_v2(n, alpha, trans, dtype):
    beta = 0.25
    k = max(1, n // 2)

    if trans == 0:  # N
        A = torch.randn(n, k, dtype=dtype, device='cuda').contiguous()
        lda = n
    else:  # T
        A = torch.randn(k, n, dtype=dtype, device='cuda').contiguous()
        lda = k

    C = torch.randn(n, n, dtype=dtype, device='cuda').contiguous()
    C_ref = C.clone()
    C_act = C.clone()
    ldc = n
    uplo = 0  # CUBLAS_FILL_MODE_LOWER

    ref_out = kernelgenbench.baseline.cublasSsyrk_v2(uplo, trans, n, k, alpha, A, lda, beta, C_ref, ldc)
    act_out = kernelgenbench.triton.cublasSsyrk_v2(uplo, trans, n, k, alpha, A, lda, beta, C_act, ldc)

    assert_close(act_out, ref_out, dtype, reduce_dim=k)

    from sandbox.utils.accuracy_utils import CustomBenchmarkResult
    if n < 1024:
        return None

    if trans == 0:  # N
        A_bench = torch.randn(n, k, dtype=dtype, device='cuda').contiguous()
        lda_bench = n
    else:  # T
        A_bench = torch.randn(k, n, dtype=dtype, device='cuda').contiguous()
        lda_bench = k

    C_bench = torch.randn(n, n, dtype=dtype, device='cuda').contiguous()

    for _ in range(10):
        _ = kernelgenbench.baseline.cublasSsyrk_v2(uplo, trans, n, k, alpha, A_bench, lda_bench, beta, C_bench.clone(), ldc)
        _ = kernelgenbench.triton.cublasSsyrk_v2(uplo, trans, n, k, alpha, A_bench, lda_bench, beta, C_bench.clone(), ldc)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = kernelgenbench.baseline.cublasSsyrk_v2(uplo, trans, n, k, alpha, A_bench, lda_bench, beta, C_bench.clone(), ldc)
    end_event.record()
    torch.cuda.synchronize()
    ms_baseline = start_event.elapsed_time(end_event) / 100

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = kernelgenbench.triton.cublasSsyrk_v2(uplo, trans, n, k, alpha, A_bench, lda_bench, beta, C_bench.clone(), ldc)
    end_event.record()
    torch.cuda.synchronize()
    ms_triton = start_event.elapsed_time(end_event) / 100

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
