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
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton

@label("rms_norm")
@parametrize("shape", [(1, 32), (71, 497), (128, 512), (1024, 4096), (5333, 8192)])
@parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@parametrize("epsilon", [1e-6, 1e-5])
@parametrize("weight_mode", ["ones", "randn"])
def test_accuracy_rms_norm(shape, dtype, epsilon, weight_mode):
    # ===== Accuracy Test =====
    M, N = shape
    x = torch.randn(M, N, device='cuda', dtype=dtype)

    if weight_mode == "ones":
        w = torch.ones(N, device='cuda', dtype=dtype)
    else:
        w = torch.randn(N, device='cuda', dtype=dtype)

    # Prepare output tensors
    ref_out = torch.empty_like(x, device='cuda', dtype=dtype)
    act_out = torch.empty_like(x, device='cuda', dtype=dtype)

    # Call baseline: kernelgenbench.baseline.rms_norm(out, input, weight, epsilon)
    kernelgenbench.baseline.rms_norm(ref_out, x, w, float(epsilon))

    # Call triton:   kernelgenbench.triton.rms_norm(out, input, weight, epsilon)
    kernelgenbench.triton.rms_norm(act_out, x, w, float(epsilon))

    # Compare: assert_close(act_out, ref_out, dtype)
    assert_close(act_out, ref_out, dtype)

    # ===== Performance Test =====
    # Skip small sizes and reduce runs by gating on num_experts and dtype
    if (M < 1024) or (N < 4096) or (dtype == torch.float32):
        return None

    # Prepare fresh data for benchmarking
    x_bench = torch.randn(M, N, device='cuda', dtype=dtype)
    if weight_mode == "ones":
        w_bench = torch.ones(N, device='cuda', dtype=dtype)
    else:
        w_bench = torch.randn(N, device='cuda', dtype=dtype)

    out_baseline = torch.empty_like(x_bench)
    out_triton = torch.empty_like(x_bench)

    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.rms_norm(out_baseline, x_bench, w_bench, float(epsilon)),
        warmup=25, rep=100
    )

    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.rms_norm(out_triton, x_bench, w_bench, float(epsilon)),
        warmup=25, rep=100
    )

    speedup = ms_baseline / ms_triton if ms_triton > 0 else float('inf')
    return CustomBenchmarkResult(
        ref_time=ms_baseline, res_time=ms_triton, speedup=speedup
    )
