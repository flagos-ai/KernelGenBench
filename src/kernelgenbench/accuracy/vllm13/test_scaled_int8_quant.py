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

@label("scaled_int8_quant")
@parametrize("shape", [(1, 32), (71, 497), (128, 512), (1024, 4096), (5333, 8192)])
@parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@parametrize("symmetric", [True, False])
@parametrize("provide_scale", [False, True])
@parametrize("provide_azp", [False, True])
def test_accuracy_scaled_int8_quant(shape, dtype, symmetric, provide_scale, provide_azp):
    # ===== Accuracy Test =====
    M, N = shape

    # Skip invalid combo: asymmetric quant with static scale requires azp
    if not symmetric and provide_scale and not provide_azp:
        return None

    # Create input activations
    x = torch.randn(M, N, device='cuda', dtype=dtype)

    # Prepare optional scale and azp
    scale = None
    azp = None
    if provide_scale:
        # vllm expects scale.numel() == 1 (scalar tensor)
        scale = torch.tensor(0.1, device='cuda', dtype=torch.float32)
        if not symmetric:
            azp = torch.tensor(0, device='cuda', dtype=torch.int32)
    if not provide_scale:
        azp = None

    # Call baseline (returns tuple: (quantized_tensor, scale_tensor) or (quantized_tensor, scale_tensor, azp_tensor))
    ref_result = kernelgenbench.baseline.scaled_int8_quant(x, scale, azp, symmetric)

    # Call triton
    act_result = kernelgenbench.triton.scaled_int8_quant(x, scale, azp, symmetric)

    # Compare: unpack tuple and compare each element
    if isinstance(ref_result, tuple):
        for ref_elem, act_elem in zip(ref_result, act_result):
            if isinstance(ref_elem, torch.Tensor):
                assert_close(act_elem, ref_elem, ref_elem.dtype)
    else:
        assert_close(act_result, ref_result, dtype)

    # ===== Performance Test =====
    # Skip small sizes for performance test
    total_elems = M * N
    threshold = 1024 * 4096
    if total_elems < threshold:
        return None

    # Prepare fresh data for benchmarking
    x_bench = torch.randn(M, N, device='cuda', dtype=dtype)
    if provide_scale:
        scale_bench = torch.tensor(0.1, device='cuda', dtype=torch.float32)
        if not symmetric:
            azp_bench = torch.tensor(0, device='cuda', dtype=torch.int32)
        else:
            azp_bench = None
    else:
        scale_bench = None
        azp_bench = None

    # Benchmark baseline
    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.scaled_int8_quant(x_bench, scale_bench, azp_bench, symmetric),
        warmup=25, rep=100)

    # Benchmark triton
    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.scaled_int8_quant(x_bench, scale_bench, azp_bench, symmetric),
        warmup=25, rep=100)

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(
        ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
