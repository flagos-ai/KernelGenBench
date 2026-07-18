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

@label("copy_blocks")
@parametrize("config", [
    (8, 4, 64, 16, 2),
    (32, 8, 128, 16, 4),
    (64, 4, 64, 32, 8),
    (64, 8, 128, 32, 8),
    (64, 16, 128, 32, 16),
    (128, 8, 64, 32, 8),
    (128, 16, 128, 32, 16),
    (128, 8, 128, 32, 16),
    (16, 8, 64, 16, 4),
    (256, 16, 128, 64, 16),
    (256, 32, 256, 64, 32),
    (512, 16, 128, 32, 16),
])
def test_accuracy_copy_blocks(config):
    num_blocks = config[0]
    num_heads = config[1]
    head_size = config[2]
    block_size = config[3]
    num_mappings = config[4]

    kc_ref = [torch.randn(num_blocks, num_heads, head_size, block_size, device=device, dtype=torch.float16)]
    vc_ref = [torch.randn(num_blocks, num_heads, head_size, block_size, device=device, dtype=torch.float16)]

    # Use non-overlapping src/dst to avoid CUDA race conditions
    perm = torch.randperm(num_blocks)
    src_idx = perm[:num_mappings].to(torch.int64)
    dst_idx = perm[num_mappings:2*num_mappings].to(torch.int64)
    block_mapping = torch.stack([src_idx, dst_idx], dim=1).to(device)

    kc_act = [k.clone() for k in kc_ref]
    vc_act = [v.clone() for v in vc_ref]
    kc_ref2 = [k.clone() for k in kc_ref]
    vc_ref2 = [v.clone() for v in vc_ref]

    kernelgenbench.baseline.copy_blocks(kc_ref2, vc_ref2, block_mapping)
    kernelgenbench.triton.copy_blocks(kc_act, vc_act, block_mapping)

    assert torch.equal(kc_act[0], kc_ref2[0]), f"key mismatch: max diff={(kc_act[0]-kc_ref2[0]).abs().max()}"
    assert torch.equal(vc_act[0], vc_ref2[0]), f"val mismatch: max diff={(vc_act[0]-vc_ref2[0]).abs().max()}"

    # ===== Performance Test =====
    if num_blocks < 64 or num_mappings < 8:
        return None

    kc_baseline = [torch.randn(num_blocks, num_heads, head_size, block_size, device=device, dtype=torch.float16)]
    vc_baseline = [torch.randn(num_blocks, num_heads, head_size, block_size, device=device, dtype=torch.float16)]
    kc_triton = [k.clone() for k in kc_baseline]
    vc_triton = [v.clone() for v in vc_baseline]

    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.copy_blocks(kc_baseline, vc_baseline, block_mapping),
        warmup=25, rep=100
    )

    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.copy_blocks(kc_triton, vc_triton, block_mapping),
        warmup=25, rep=100
    )

    speedup = ms_baseline / ms_triton if ms_triton > 0 else float('inf')
    return CustomBenchmarkResult(
        ref_time=ms_baseline, res_time=ms_triton, speedup=speedup
    )
