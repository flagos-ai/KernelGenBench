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

@label("rms_norm_dynamic_per_token_quant")
@parametrize("tokens", [1, 128, 1024, 5333])
@parametrize("hidden_size", [32, 512, 4096])
@parametrize("dtype", [torch.float16, torch.bfloat16])
@parametrize("quant_dtype", [torch.int8])
@parametrize("epsilon", [1e-6])
@parametrize("use_residual", [False, True])
def test_accuracy_rms_norm_dynamic_per_token_quant(tokens, hidden_size, dtype, quant_dtype, epsilon, use_residual):
    # ===== Accuracy Test =====
    x = torch.randn(tokens, hidden_size, device='cuda', dtype=dtype)
    w = torch.randn(hidden_size, device='cuda', dtype=dtype)
    residual = None
    if use_residual:
        residual = torch.randn(tokens, hidden_size, device='cuda', dtype=dtype)

    # Clone inputs for each call since residual may be modified in-place
    ref_result = kernelgenbench.baseline.rms_norm_dynamic_per_token_quant(
        x.clone(), w, float(epsilon), quant_dtype, None,
        residual.clone() if residual is not None else None
    )

    act_result = kernelgenbench.triton.rms_norm_dynamic_per_token_quant(
        x.clone(), w, float(epsilon), quant_dtype, None,
        residual.clone() if residual is not None else None
    )

    # Compare each element in the returned tuple
    if isinstance(ref_result, tuple):
        for ref_elem, act_elem in zip(ref_result, act_result):
            if isinstance(ref_elem, torch.Tensor):
                if ref_elem.is_floating_point() and ref_elem.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
                    assert_close(act_elem, ref_elem, ref_elem.dtype)
                else:
                    diff = (act_elem.float() - ref_elem.float()).abs()
                    assert diff.max() <= 1, f"Mismatch for {ref_elem.dtype}, max diff={diff.max()}"
    else:
        assert_close(act_result, ref_result, dtype)

    # ===== Performance Test =====
    total_elems = tokens * hidden_size
    if total_elems < (1 << 20):
        return None

    x_bench = torch.randn(tokens, hidden_size, device='cuda', dtype=dtype)
    w_bench = torch.randn(hidden_size, device='cuda', dtype=dtype)
    residual_bench = None
    if use_residual:
        residual_bench = torch.randn(tokens, hidden_size, device='cuda', dtype=dtype)

    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.rms_norm_dynamic_per_token_quant(
            x_bench.clone(), w_bench, float(epsilon), quant_dtype, None,
            residual_bench.clone() if residual_bench is not None else None
        ),
        warmup=25, rep=100
    )

    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.rms_norm_dynamic_per_token_quant(
            x_bench.clone(), w_bench, float(epsilon), quant_dtype, None,
            residual_bench.clone() if residual_bench is not None else None
        ),
        warmup=25, rep=100
    )

    speedup = ms_baseline / ms_triton if ms_triton > 0 else float("inf")
    return CustomBenchmarkResult(
        ref_time=ms_baseline, res_time=ms_triton, speedup=speedup
    )
