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

@label("awq_gemm")
@parametrize("config", [
    # (M, K, N, group_size, split_k_iters)
    # --- Small: accuracy-only ---
    (1, 256, 128, 128, 1),
    (16, 256, 128, 128, 1),
    (32, 512, 256, 128, 1),
    # --- Medium: accuracy + perf ---
    (64, 1024, 512, 128, 1),
    (128, 4096, 4096, 128, 1),
    # --- Large / real LLM shapes: accuracy + perf ---
    (1, 4096, 11008, 128, 1),       # decode, Llama-7B FFN
    (32, 8192, 8192, 128, 1),       # Llama-70B scale
    (512, 4096, 11008, 128, 1),     # large prefill, Llama-7B FFN
])
def test_accuracy_awq_gemm(config):
    # ===== Accuracy Test =====
    M, K, N, group_size, split_k_iters = config
    dtype = torch.float16
    bit = 4
    pack_factor = 32 // bit  # 8 values per int32

    # Input activation: (M, K)
    input_tensor = torch.randn(M, K, device=device, dtype=dtype)

    # AWQ quantized weight: qweight is (K, N // pack_factor) int32
    # Each int32 packs 8 int4 values along N dimension
    qweight = torch.randint(0, 2**31, (K, N // pack_factor), device=device, dtype=torch.int32)

    # Scales: (K // group_size, N) float16
    group_count = K // group_size
    scales = (torch.rand(group_count, N, device=device, dtype=dtype) + 0.01)

    # Quantized zeros: (K // group_size, N // pack_factor) int32
    qzeros = torch.randint(0, 2**31, (group_count, N // pack_factor), device=device, dtype=torch.int32)

    ref_out = kernelgenbench.baseline.awq_gemm(input_tensor, qweight, scales, qzeros, split_k_iters)
    act_out = kernelgenbench.triton.awq_gemm(input_tensor, qweight, scales, qzeros, split_k_iters)

    assert_close(act_out, ref_out, dtype)

    # ===== Performance Test =====
    if M * K < 65536:
        return None

    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.awq_gemm(input_tensor, qweight, scales, qzeros, split_k_iters),
        warmup=25, rep=100)
    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.awq_gemm(input_tensor, qweight, scales, qzeros, split_k_iters),
        warmup=25, rep=100)

    speedup = ms_baseline / ms_triton if ms_triton > 0 else float("inf")
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
