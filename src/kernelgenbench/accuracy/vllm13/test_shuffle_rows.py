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

@label("shuffle_rows")
@parametrize("rows", [1, 128, 1024, 5333])
@parametrize("cols", [32, 512, 4096])
@parametrize("dst_factor", [1, 2])
@parametrize("dtype", [torch.float16, torch.bfloat16])
def test_accuracy_shuffle_rows(rows, cols, dst_factor, dtype):
    # ===== Accuracy Test =====
    # Create inputs
    src_rows = rows
    dst_rows = rows * dst_factor
    input_tensor = torch.randn(src_rows, cols, device='cuda', dtype=dtype)

    if dst_factor == 1:
        # Pure shuffle (permutation) when destination size equals source size
        # Use permutation for better coverage when rows > 1, otherwise just [0]
        if src_rows > 1:
            map64 = torch.randperm(src_rows, device='cuda')
        else:
            map64 = torch.zeros(dst_rows, device='cuda', dtype=torch.long)
    else:
        # Expansion: allow repeats by sampling with replacement
        map64 = torch.randint(0, src_rows, (dst_rows,), device='cuda')

    dst2src_map = map64.to(torch.int32).contiguous()

    # Call baseline
    ref_out = kernelgenbench.baseline.shuffle_rows(input_tensor, dst2src_map)

    # Call triton
    act_out = kernelgenbench.triton.shuffle_rows(input_tensor, dst2src_map)

    # Compare
    assert_close(act_out, ref_out, dtype)

    # ===== Performance Test =====
    # Skip small sizes for performance test
    if dst_rows < 1024 or cols < 512:
        return None

    # Prepare fresh data for benchmarking
    x_bench = torch.randn(src_rows, cols, device='cuda', dtype=dtype)
    m_bench = dst2src_map

    # Benchmark baseline
    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.shuffle_rows(x_bench, m_bench), warmup=25, rep=100)

    # Benchmark triton
    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.shuffle_rows(x_bench, m_bench), warmup=25, rep=100)

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(
        ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
