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

@label("reshape_and_cache_flash")
@parametrize("shape", [
    (4, 8, 64, 16, 16),
    (32, 8, 64, 32, 16),
    (128, 8, 64, 64, 16),
    (128, 16, 128, 64, 16),
    (256, 8, 64, 64, 16),
    (256, 16, 128, 64, 16),
    (512, 16, 64, 128, 16),
    (512, 32, 128, 128, 16),
    (1024, 16, 128, 256, 16),
])
@parametrize("dtype", [torch.float16, torch.bfloat16])
def test_accuracy_reshape_and_cache_flash(shape, dtype):
    # ===== Accuracy Test =====
    T, H, D, num_blocks, block_size = shape[0], shape[1], shape[2], shape[3], shape[4]

    key = torch.randn(T, H, D, device=device, dtype=dtype)
    value = torch.randn(T, H, D, device=device, dtype=dtype)
    slot_mapping = torch.randperm(num_blocks * block_size, device=device, dtype=torch.int64)[:T]
    k_scale = torch.tensor(1.0, device=device, dtype=torch.float32)
    v_scale = torch.tensor(1.0, device=device, dtype=torch.float32)

    # flash version: both caches are 4D (num_blocks, block_size, num_heads, head_dim)
    ref_kc = torch.zeros(num_blocks, block_size, H, D, device=device, dtype=dtype)
    ref_vc = torch.zeros(num_blocks, block_size, H, D, device=device, dtype=dtype)
    act_kc = ref_kc.clone()
    act_vc = ref_vc.clone()

    kernelgenbench.baseline.reshape_and_cache_flash(key, value, ref_kc, ref_vc, slot_mapping, "auto", k_scale, v_scale)
    kernelgenbench.triton.reshape_and_cache_flash(key, value, act_kc, act_vc, slot_mapping, "auto", k_scale, v_scale)

    assert_close(act_kc, ref_kc, dtype)
    assert_close(act_vc, ref_vc, dtype)

    # ===== Performance Test =====
    if T < 128:
        return None

    kc_baseline = torch.zeros(num_blocks, block_size, H, D, device=device, dtype=dtype)
    vc_baseline = torch.zeros(num_blocks, block_size, H, D, device=device, dtype=dtype)
    kc_triton = torch.zeros(num_blocks, block_size, H, D, device=device, dtype=dtype)
    vc_triton = torch.zeros(num_blocks, block_size, H, D, device=device, dtype=dtype)

    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.reshape_and_cache_flash(
            key, value, kc_baseline, vc_baseline, slot_mapping, "auto", k_scale, v_scale),
        warmup=25, rep=100)
    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.reshape_and_cache_flash(
            key, value, kc_triton, vc_triton, slot_mapping, "auto", k_scale, v_scale),
        warmup=25, rep=100)

    speedup = ms_baseline / ms_triton if ms_triton > 0 else float("inf")
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
