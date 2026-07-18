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

@label("scaled_fp8_quant")
@parametrize("shape", [(1, 32), (71, 497), (128, 512), (1024, 4096), (5333, 8192)])
@parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@parametrize("num_token_padding", [None, 0, 64])
@parametrize("scale_kind", ["dynamic", "per_tensor", "group_2d_1x1"])
@parametrize("use_per_token_if_dynamic", [False, True])
@parametrize("scale_ub_kind", [None, "rowwise"])
def test_accuracy_scaled_fp8_quant(shape, dtype, num_token_padding, scale_kind, use_per_token_if_dynamic, scale_ub_kind):
    # ===== Accuracy Test =====
    # Create inputs
    M, N = shape
    x = torch.randn(M, N, device='cuda', dtype=dtype)

    # Prepare scale, scale_ub, and group_shape based on scale_kind and dynamic settings
    scale = None
    scale_ub = None
    group_shape = None

    if scale_kind == "dynamic":
        scale = None
        group_shape = None
        if use_per_token_if_dynamic and (scale_ub_kind == "rowwise"):
            # Row-wise upper bound for dynamic per-token scaling: shape [M]
            scale_ub = torch.rand(M, device='cuda', dtype=torch.float32) + 0.5  # positive bounds
        else:
            scale_ub = None
    elif scale_kind == "per_tensor":
        # Static per-tensor scaling: 0D scalar and explicit group_shape
        scale = torch.tensor(1.0, device='cuda', dtype=torch.float32)
        group_shape = (-1, -1)
        scale_ub = None
    elif scale_kind == "per_channel_1d":
        # Static per-channel scaling: 1D over N, requires group_shape (-1, 1)
        scale = (torch.rand(N, device='cuda', dtype=torch.float32) + 0.5)
        group_shape = (-1, 1)
        scale_ub = None
    elif scale_kind == "per_token_1d":
        # Static per-token scaling: 1D over M, requires group_shape (1, -1)
        scale = (torch.rand(M, device='cuda', dtype=torch.float32) + 0.5)
        group_shape = (1, -1)
        scale_ub = None
    elif scale_kind == "group_2d_1x1":
        # Static 2D scaling with shape [1, 1] (per-tensor expressed as 2D)
        scale = (torch.rand(1, 1, device='cuda', dtype=torch.float32) + 0.5)
        group_shape = None
        scale_ub = None
    else:
        raise ValueError(f"Unsupported scale_kind: {scale_kind}")

    # Call baseline (returns tuple: (quantized_tensor, scale_tensor))
    ref_result = kernelgenbench.baseline.scaled_fp8_quant(
        x,
        scale=scale,
        num_token_padding=num_token_padding,
        scale_ub=scale_ub,
        use_per_token_if_dynamic=use_per_token_if_dynamic,
        output=None,
    )

    # Call triton (returns tuple: (quantized_tensor, scale_tensor))
    act_result = kernelgenbench.triton.scaled_fp8_quant(
        x,
        scale=scale,
        num_token_padding=num_token_padding,
        scale_ub=scale_ub,
        use_per_token_if_dynamic=use_per_token_if_dynamic,
        output=None,
    )

    # Compare: unpack tuple and compare each element
    # FP8 tensors need to be cast to float16 for comparison since assert_close doesn't support FP8
    # When num_token_padding > M, output is padded; only compare the first M rows (valid data)
    def _slice_valid(t):
        if isinstance(t, torch.Tensor) and t.dim() >= 2 and t.shape[0] > M:
            return t[:M]
        return t

    if isinstance(ref_result, tuple):
        for ref_elem, act_elem in zip(ref_result, act_result):
            if isinstance(ref_elem, torch.Tensor):
                ref_v = _slice_valid(ref_elem)
                act_v = _slice_valid(act_elem)
                if ref_v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                    assert_close(act_v.to(torch.float16), ref_v.to(torch.float16), torch.float16)
                else:
                    assert_close(act_v, ref_v, ref_v.dtype)
    else:
        assert_close(_slice_valid(act_result), _slice_valid(ref_result), dtype)

    # ===== Performance Test =====
    # Skip small sizes for performance test (use M as main size param)
    if M < 1024:
        return None

    # Prepare fresh data for benchmarking
    x_perf = torch.randn(M, N, device='cuda', dtype=dtype)
    # Reuse the same scale/params prepared above for consistent benchmarking

    # Benchmark baseline
    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.scaled_fp8_quant(
            x_perf,
            scale=scale,
            num_token_padding=num_token_padding,
            scale_ub=scale_ub,
            use_per_token_if_dynamic=use_per_token_if_dynamic,
            output=None,
        ),
        warmup=25, rep=100
    )

    # Benchmark triton
    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.scaled_fp8_quant(
            x_perf,
            scale=scale,
            num_token_padding=num_token_padding,
            scale_ub=scale_ub,
            use_per_token_if_dynamic=use_per_token_if_dynamic,
            output=None,
        ),
        warmup=25, rep=100
    )

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(
        ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
