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

@label("hadacore_transform")
@parametrize("shape", [(1, 32), (16, 128), (128, 512), (1024, 4096), (4096, 1024), (5333, 8192), (2, 8192), (777, 32)])
@parametrize("inplace", [True])
@parametrize("dtype", [torch.float16, torch.bfloat16])
def test_accuracy_hadacore_transform(shape, inplace, dtype):
    # ===== Accuracy Test =====
    x_base = torch.randn(shape, device=device, dtype=dtype)

    ref_in = x_base.clone()
    ref_out = kernelgenbench.baseline.hadacore_transform(ref_in, inplace=inplace)

    act_in = x_base.clone()
    act_out = kernelgenbench.triton.hadacore_transform(act_in, inplace=inplace)

    assert_close(act_out, ref_out, dtype)

    # ===== Performance Test =====
    numel = x_base.numel()
    if numel < (1 << 18):
        return None

    x_baseline = torch.randn(shape, device=device, dtype=dtype)
    x_triton = x_baseline.clone()

    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.hadacore_transform(x_baseline, inplace=inplace),
        warmup=25, rep=100
    )

    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.hadacore_transform(x_triton, inplace=inplace),
        warmup=25, rep=100
    )

    speedup = ms_baseline / ms_triton if ms_triton > 0 else float("inf")
    return CustomBenchmarkResult(
        ref_time=ms_baseline, res_time=ms_triton, speedup=speedup
    )
