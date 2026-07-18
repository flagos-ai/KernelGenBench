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

@label("cutlass_scaled_mm")
@parametrize("shape", [(1, 128, 128), (71, 512, 256), (128, 512, 512), (1024, 4096, 4096)])
@parametrize("out_dtype", [torch.float16, torch.bfloat16])
@parametrize("scale_case", ["all_scalar", "row_a_col_b"])
@parametrize("bias_mode", [None, "vector"])
def test_accuracy_cutlass_scaled_mm(shape, out_dtype, scale_case, bias_mode):
    # ===== Accuracy Test =====
    M, K, N = shape

    # cutlass_scaled_mm on A100 requires int8 inputs
    # b must be column-major (stride(0)==1)
    a = torch.randint(-128, 127, (M, K), device="cuda", dtype=torch.int8)
    b = torch.randint(-128, 127, (N, K), device="cuda", dtype=torch.int8).t()

    # Scales must be float32
    if scale_case == "all_scalar":
        scale_a = torch.randn((1, 1), device="cuda", dtype=torch.float32)
        scale_b = torch.randn((1, 1), device="cuda", dtype=torch.float32)
    elif scale_case == "row_a_col_b":
        scale_a = torch.randn((M, 1), device="cuda", dtype=torch.float32)
        scale_b = torch.randn((1, N), device="cuda", dtype=torch.float32)
    else:
        raise ValueError(f"Unknown scale_case: {scale_case}")

    # Create optional bias
    if bias_mode is None:
        bias = None
    elif bias_mode == "vector":
        # Typical linear bias with shape (N,)
        bias = torch.randn((N,), device="cuda", dtype=out_dtype)
    else:
        raise ValueError(f"Unknown bias_mode: {bias_mode}")

    # Call baseline
    ref_out = kernelgenbench.baseline.cutlass_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)

    # Call triton
    act_out = kernelgenbench.triton.cutlass_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)

    # Compare
    assert_close(act_out, ref_out, out_dtype)

    # ===== Performance Test =====
    # Skip small sizes for performance test
    if max(M, N, K) < 1024:
        return None

    # Prepare fresh data for benchmarking (same shapes/dtypes)
    a_bench = torch.randint(-128, 127, (M, K), device="cuda", dtype=torch.int8)
    b_bench = torch.randint(-128, 127, (N, K), device="cuda", dtype=torch.int8).t()
    if bias_mode is None:
        bias_bench = None
    else:
        bias_bench = torch.randn((N,), device="cuda", dtype=out_dtype)

    # Benchmark baseline
    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.cutlass_scaled_mm(a_bench, b_bench, scale_a, scale_b, out_dtype, bias_bench),
        warmup=25, rep=100
    )

    # Benchmark triton
    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.cutlass_scaled_mm(a_bench, b_bench, scale_a, scale_b, out_dtype, bias_bench),
        warmup=25, rep=100
    )

    speedup = ms_baseline / ms_triton if ms_triton > 0 else float("inf")
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
