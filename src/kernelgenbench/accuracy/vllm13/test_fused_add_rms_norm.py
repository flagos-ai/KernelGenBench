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

@label("fused_add_rms_norm")
@parametrize("num_tokens", [1, 71, 128, 1024, 5333])
@parametrize("hidden_size", [32, 497, 512, 4096, 8192])
@parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@parametrize("epsilon", [1e-6, 1e-5, 1e-4])
def test_accuracy_fused_add_rms_norm(num_tokens, hidden_size, dtype, epsilon):
    # ===== Accuracy Test =====
    m, n = num_tokens, hidden_size

    # Inputs
    input_ref = torch.randn(m, n, device='cuda', dtype=dtype)
    residual_ref = torch.randn(m, n, device='cuda', dtype=dtype)
    weight_ref = torch.randn(n, device='cuda', dtype=dtype)

    # Clone for triton path
    input_act = input_ref.clone()
    residual_act = residual_ref.clone()
    weight_act = weight_ref.clone()

    # Call baseline: kernelgenbench.baseline.fused_add_rms_norm(...)
    # This is an in-place op: it mutates the input tensor
    kernelgenbench.baseline.fused_add_rms_norm(input_ref, residual_ref, weight_ref, float(epsilon))

    # Call triton:   kernelgenbench.triton.fused_add_rms_norm(...)
    kernelgenbench.triton.fused_add_rms_norm(input_act, residual_act, weight_act, float(epsilon))

    # Compare mutated tensors (the input tensor is the in-place output)
    assert_close(input_act, input_ref, dtype)

    # ===== Performance Test =====
    # Skip small sizes for performance test
    if m * n < (1 << 20):  # ~1M elements
        return None

    # Prepare fresh data for benchmarking
    input_baseline = torch.randn(m, n, device='cuda', dtype=dtype)
    residual_baseline = torch.randn(m, n, device='cuda', dtype=dtype)
    input_triton = input_baseline.clone()
    residual_triton = residual_baseline.clone()
    base_weight = torch.randn(n, device='cuda', dtype=dtype)

    # For in-place ops, ensure each call gets fresh tensors
    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.fused_add_rms_norm(input_baseline, residual_baseline, base_weight, float(epsilon)),
        warmup=25, rep=100
    )

    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.fused_add_rms_norm(input_triton, residual_triton, base_weight, float(epsilon)),
        warmup=25, rep=100
    )

    speedup = ms_baseline / ms_triton if ms_triton > 0 else float("inf")
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
