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

@label("permute_cols")
@parametrize("shape", [(1, 32), (71, 504), (128, 512), (1024, 4096), (5333, 8192)])
@parametrize("perm_pattern", ["identity", "reverse", "random"])
@parametrize("dtype", [torch.float16, torch.bfloat16])
def test_accuracy_permute_cols(shape, perm_pattern, dtype):
    # ===== Accuracy Test =====
    # Create inputs
    rows, cols = shape
    a = torch.randn(rows, cols, device='cuda', dtype=dtype)

    # Build permutation (gather semantics: output[:, j] = input[:, perm[j]])
    if perm_pattern == "identity":
        perm = torch.arange(cols, device='cuda', dtype=torch.int32)
    elif perm_pattern == "reverse":
        perm = torch.arange(cols - 1, -1, -1, device='cuda', dtype=torch.int32)
    elif perm_pattern == "random":
        gen = torch.Generator(device='cuda')
        gen.manual_seed(42)
        perm = torch.randperm(cols, device='cuda', generator=gen).to(torch.int32)
    else:
        perm = torch.arange(cols, device='cuda', dtype=torch.int32)

    # Call baseline: kernelgenbench.baseline.permute_cols(...)
    ref_out = kernelgenbench.baseline.permute_cols(a, perm)

    # Call triton:   kernelgenbench.triton.permute_cols(...)
    act_out = kernelgenbench.triton.permute_cols(a, perm)

    # Compare: assert_close(act_out, ref_out, dtype)
    assert_close(act_out, ref_out, dtype)

    # ===== Performance Test =====
    # Skip small sizes for performance test
    if cols < 2048 or rows < 512:
        return None

    # Prepare fresh data for benchmarking
    x_baseline = torch.randn(rows, cols, device='cuda', dtype=dtype)
    x_triton = x_baseline.clone()

    # Benchmark baseline
    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.permute_cols(x_baseline, perm),
        warmup=25, rep=100)

    # Benchmark triton
    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.permute_cols(x_triton, perm),
        warmup=25, rep=100)

    speedup = ms_baseline / ms_triton if ms_triton > 0 else float('inf')
    return CustomBenchmarkResult(
        ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
