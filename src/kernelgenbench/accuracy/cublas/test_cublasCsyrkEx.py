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

@label("cublasCsyrkEx")
@parametrize("n", [1, 17, 32, 33, 71, 160, 497, 1024, 4096, 4113, 5333])
@parametrize("k", [1, 17, 33, 71, 160, 497])
@parametrize("uplo", [0, 1])  # 0: LOWER/UPPER (implementation-defined), 1: opposite
@parametrize("trans", [0, 1, 2])  # 0: N, 1: T, 2: C
@parametrize("alpha", [1+0j, 0+0j, 0.5+0.3j, 0.001+0.001j, -0.999+0j, 100.001+0j, -1+0j])
@parametrize("beta", [0+0j, 1+0j, 0.5-0.25j, -1+0j, 0.001+0.001j])
@parametrize("dtype", [torch.complex64])
def test_accuracy_cublasCsyrkEx(n, k, uplo, trans, alpha, beta, dtype):
    CUDA_C_32F = 8
    Atype = CUDA_C_32F
    Ctype = CUDA_C_32F

    # Allocate inputs
    A = torch.randn(n, k, dtype=dtype, device=device)
    C = torch.randn(n, n, dtype=dtype, device=device)

    # Clone for baseline and triton
    A_ref = A.clone()
    C_ref = C.clone()
    A_act = A.clone()
    C_act = C.clone()

    # Leading dimensions (use column-major conventions; safe with row-major buffers as both paths share)
    lda = n if trans == 0 else k
    ldc = n

    # Reference (baseline) and Actual (triton)
    ref_out = kernelgenbench.baseline.cublasCsyrkEx(uplo, trans, n, k, alpha, A_ref, Atype, lda, beta, C_ref, Ctype, ldc)
    act_out = kernelgenbench.triton.cublasCsyrkEx(uplo, trans, n, k, alpha, A_act, Atype, lda, beta, C_act, Ctype, ldc)

    assert_close(act_out, ref_out, dtype, reduce_dim=k)

    # Performance Test
    from sandbox.utils.accuracy_utils import CustomBenchmarkResult
    if n < 1024:
        return None

    # Benchmark tensors
    A_bench = torch.randn(n, k, dtype=dtype, device=device)
    C_bench = torch.randn(n, n, dtype=dtype, device=device)

    # Warmup
    for _ in range(10):
        _ = kernelgenbench.baseline.cublasCsyrkEx(uplo, trans, n, k, alpha, A_bench, Atype, lda, beta, C_bench.clone(), Ctype, ldc)
        _ = kernelgenbench.triton.cublasCsyrkEx(uplo, trans, n, k, alpha, A_bench, Atype, lda, beta, C_bench.clone(), Ctype, ldc)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = kernelgenbench.baseline.cublasCsyrkEx(uplo, trans, n, k, alpha, A_bench, Atype, lda, beta, C_bench.clone(), Ctype, ldc)
    end_event.record()
    torch.cuda.synchronize()
    ms_baseline = start_event.elapsed_time(end_event) / 100

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = kernelgenbench.triton.cublasCsyrkEx(uplo, trans, n, k, alpha, A_bench, Atype, lda, beta, C_bench.clone(), Ctype, ldc)
    end_event.record()
    torch.cuda.synchronize()
    ms_triton = start_event.elapsed_time(end_event) / 100

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
