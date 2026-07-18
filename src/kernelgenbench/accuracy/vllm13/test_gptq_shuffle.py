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

@label("gptq_shuffle")
@parametrize("q_weight_bit", [
    # bit=8: verified stable combos
    ((128, 512), 8),
    ((256, 1024), 8),
    ((1024, 4096), 8),
    # bit=4: verified stable combos
    ((32, 256), 4),
    ((512, 4096), 4),
    ((1024, 4096), 4),
])
def test_accuracy_gptq_shuffle(q_weight_bit):
    # ===== Accuracy Test =====
    q_weight_shape, bit = q_weight_bit[0], q_weight_bit[1]
    rows, cols = q_weight_shape[0], q_weight_shape[1]
    assert cols % 32 == 0, "Number of columns must be a multiple of 32 for GPTQ packing."
    # Packed columns: for each block of 32 columns, we store `bit` int32 words.
    packed_cols = (cols // 32) * bit
    dtype = torch.int32

    # Create inputs
    # q_weight: packed int32 tensor on CUDA
    q_weight_init = torch.randint(
        low=0, high=2**31 - 1,
        size=(rows, packed_cols),
        device=device, dtype=dtype,
    )
    # q_perm: permutation of the original (unpacked) column indices
    q_perm_tensor = torch.randperm(cols, device=device).to(torch.int32)

    # Clone inputs for baseline and triton runs
    q_weight_ref = q_weight_init.clone()
    q_weight_act = q_weight_init.clone()

    # Call baseline: kernelgenbench.baseline.gptq_shuffle(...)
    kernelgenbench.baseline.gptq_shuffle(q_weight_ref, q_perm_tensor, bit)

    # Call triton:   kernelgenbench.triton.gptq_shuffle(...)
    kernelgenbench.triton.gptq_shuffle(q_weight_act, q_perm_tensor, bit)

    # Compare: int32 output, use exact comparison
    assert torch.equal(q_weight_act, q_weight_ref), f"Mismatch: max diff={(q_weight_act - q_weight_ref).abs().max()}"

    # ===== Performance Test =====
    # Skip small sizes for performance test
    if cols < 1024:
        return None

    # Prepare fresh data for benchmarking
    q_weight_baseline = torch.randint(
        low=0, high=2**31 - 1,
        size=(rows, packed_cols),
        device=device, dtype=dtype,
    )
    q_weight_triton = q_weight_baseline.clone()
    q_perm_bench = torch.randperm(cols, device=device).to(torch.int32)

    # Benchmark baseline
    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.gptq_shuffle(q_weight_baseline, q_perm_bench, bit),
        warmup=25, rep=100
    )

    # Benchmark triton
    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.gptq_shuffle(q_weight_triton, q_perm_bench, bit),
        warmup=25, rep=100
    )

    speedup = ms_baseline / ms_triton if ms_triton > 0 else float("inf")
    return CustomBenchmarkResult(
        ref_time=ms_baseline, res_time=ms_triton, speedup=speedup
    )
