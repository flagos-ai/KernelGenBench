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

@label("cutlass_scaled_mm_azp")
@parametrize("config", [
    # (M, K, N, has_azp, has_bias)
    (32, 128, 128, False, False),
    (32, 128, 128, True, False),
    (128, 256, 256, False, True),
    (128, 256, 256, True, True),
    (128, 256, 256, False, False),
    (256, 512, 512, False, False),
    (256, 512, 512, True, True),
    (256, 512, 512, True, False),
])
def test_accuracy_cutlass_scaled_mm_azp(config):
    # ===== Accuracy Test =====
    M, K, N = config[0], config[1], config[2]
    has_azp, has_bias = config[3], config[4]

    A = torch.randint(-128, 127, (M, K), device=device, dtype=torch.int8)
    # B needs column-major stride: stride(0)==1
    B = torch.randint(-128, 127, (K, N), device=device, dtype=torch.int8).t().contiguous().t()

    scale_a = torch.rand(M, 1, device=device, dtype=torch.float32) + 0.5
    scale_b = torch.rand(1, N, device=device, dtype=torch.float32) + 0.5
    azp_adj = torch.randint(-8, 9, (N,), device=device, dtype=torch.int32)

    azp_t = torch.randint(-8, 9, (M,), device=device, dtype=torch.int32) if has_azp else None
    bias_t = torch.randn(N, device=device, dtype=torch.float16) if has_bias else None

    ref_out = kernelgenbench.baseline.cutlass_scaled_mm_azp(
        A, B, scale_a, scale_b, torch.float16, azp_adj, azp=azp_t, bias=bias_t)
    act_out = kernelgenbench.triton.cutlass_scaled_mm_azp(
        A, B, scale_a, scale_b, torch.float16, azp_adj, azp=azp_t, bias=bias_t)

    assert_close(act_out, ref_out, torch.float16)

    # ===== Performance Test =====
    if M < 128:
        return None

    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.cutlass_scaled_mm_azp(
            A, B, scale_a, scale_b, torch.float16, azp_adj, azp=azp_t, bias=bias_t),
        warmup=25, rep=100)
    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.cutlass_scaled_mm_azp(
            A, B, scale_a, scale_b, torch.float16, azp_adj, azp=azp_t, bias=bias_t),
        warmup=25, rep=100)

    speedup = ms_baseline / ms_triton if ms_triton > 0 else float("inf")
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
