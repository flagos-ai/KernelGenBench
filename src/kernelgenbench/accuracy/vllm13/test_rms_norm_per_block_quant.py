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

@label("rms_norm_per_block_quant")
@parametrize("input", [(1, 128), (128, 256), (1024, 4096), (4096, 8192)])
@parametrize("epsilon", [1e-5, 1e-6])
@parametrize("quant_dtype", [torch.int8, torch.float8_e4m3fn])
@parametrize("group_size", [[128, 128], [64, 64]])
@parametrize("is_scale_transposed", [False, True])
def test_accuracy_rms_norm_per_block_quant(input, epsilon, quant_dtype, group_size, is_scale_transposed):
    # ===== Accuracy Test =====
    M, N = input
    # N must be divisible by group_size
    if N % group_size[0] != 0:
        return None

    x = torch.randn(M, N, device=device, dtype=torch.float16)
    w = torch.randn(N, device=device, dtype=torch.float16)

    ref_quant, ref_scale = kernelgenbench.baseline.rms_norm_per_block_quant(
        x, w, float(epsilon), quant_dtype, group_size, None, None, bool(is_scale_transposed)
    )

    act_quant, act_scale = kernelgenbench.triton.rms_norm_per_block_quant(
        x, w, float(epsilon), quant_dtype, group_size, None, None, bool(is_scale_transposed)
    )

    # Compare quantized output and scales
    assert torch.equal(act_quant, ref_quant), f"quant mismatch: max diff={(act_quant.float() - ref_quant.float()).abs().max()}"
    assert_close(act_scale, ref_scale, torch.float32)

    # ===== Performance Test =====
    if M * N < 131072:
        return None

    x_bench = torch.randn(M, N, device=device, dtype=torch.float16)
    w_bench = torch.randn(N, device=device, dtype=torch.float16)

    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.rms_norm_per_block_quant(
            x_bench, w_bench, float(epsilon), quant_dtype, group_size, None, None, bool(is_scale_transposed)
        ),
        warmup=25, rep=100,
    )

    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.rms_norm_per_block_quant(
            x_bench, w_bench, float(epsilon), quant_dtype, group_size, None, None, bool(is_scale_transposed)
        ),
        warmup=25, rep=100,
    )

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
