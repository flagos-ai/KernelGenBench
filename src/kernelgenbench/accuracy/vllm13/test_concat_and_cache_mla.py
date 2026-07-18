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

@label("concat_and_cache_mla")
@parametrize("shape", [
    # (num_tokens, kv_lora_rank, pe_dim, num_blocks, block_size)
    (4, 512, 64, 16, 16),
    (32, 512, 64, 32, 16),
    (128, 256, 64, 64, 16),
    (128, 256, 128, 64, 16),
    (256, 256, 64, 64, 16),
    (256, 512, 64, 64, 16),
    (512, 256, 128, 128, 16),
    (512, 512, 64, 128, 16),
    (1024, 512, 64, 256, 16),
])
@parametrize("dtype", [torch.float16, torch.bfloat16])
def test_accuracy_concat_and_cache_mla(shape, dtype):
    # ===== Accuracy Test =====
    T, kv_lora_rank, pe_dim = shape[0], shape[1], shape[2]
    num_blocks, block_size = shape[3], shape[4]

    kv_c = torch.randn(T, kv_lora_rank, device=device, dtype=dtype)
    k_pe = torch.randn(T, pe_dim, device=device, dtype=dtype)
    slot_mapping = torch.randperm(num_blocks * block_size, device=device, dtype=torch.int64)[:T]
    scale = torch.tensor(1.0, device=device, dtype=torch.float32)

    # kv_cache: 3D (num_blocks, block_size, kv_lora_rank + pe_dim)
    ref_cache = torch.zeros(num_blocks, block_size, kv_lora_rank + pe_dim, device=device, dtype=dtype)
    act_cache = ref_cache.clone()

    kernelgenbench.baseline.concat_and_cache_mla(kv_c, k_pe, ref_cache, slot_mapping, "auto", scale)
    kernelgenbench.triton.concat_and_cache_mla(kv_c, k_pe, act_cache, slot_mapping, "auto", scale)

    assert_close(act_cache, ref_cache, dtype)

    # ===== Performance Test =====
    if T < 128:
        return None

    total_dim = kv_lora_rank + pe_dim
    cache_baseline = torch.zeros(num_blocks, block_size, total_dim, device=device, dtype=dtype)
    cache_triton = torch.zeros(num_blocks, block_size, total_dim, device=device, dtype=dtype)

    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.concat_and_cache_mla(
            kv_c, k_pe, cache_baseline, slot_mapping, "auto", scale),
        warmup=25, rep=100)
    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.concat_and_cache_mla(
            kv_c, k_pe, cache_triton, slot_mapping, "auto", scale),
        warmup=25, rep=100)

    speedup = ms_baseline / ms_triton if ms_triton > 0 else float("inf")
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
