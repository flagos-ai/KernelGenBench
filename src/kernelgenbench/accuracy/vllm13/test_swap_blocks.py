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

@label("swap_blocks")
@parametrize("num_blocks", [1, 32, 512])
@parametrize("block_shape", [(16, 64), (32, 128), (64, 256)])
@parametrize("dtype", [torch.float16, torch.bfloat16])
@parametrize("mapping_kind", ["identity", "reverse", "random_subset"])
def test_accuracy_swap_blocks(num_blocks, block_shape, dtype, mapping_kind):
    # ===== Accuracy Test =====
    # swap_blocks signature: (src, dst, block_mapping) -> None
    # src/dst are 3D KV cache tensors: (num_blocks, block_size, head_dim)
    block_h, block_w = block_shape
    src = torch.randn(num_blocks, block_h, block_w, dtype=dtype, device='cuda')
    base_dst = torch.randn(num_blocks, block_h, block_w, dtype=dtype, device='cuda')

    # Prepare block mapping on CPU (int64), shape (N, 2) with [src_idx, dst_idx]
    if mapping_kind == "identity":
        idx = torch.arange(num_blocks, dtype=torch.int64)
        block_mapping = torch.stack([idx, idx], dim=1)
    elif mapping_kind == "reverse":
        src_idx = torch.arange(num_blocks, dtype=torch.int64)
        dst_idx = torch.arange(num_blocks - 1, -1, -1, dtype=torch.int64)
        block_mapping = torch.stack([src_idx, dst_idx], dim=1)
    else:
        k = max(1, num_blocks // 2)
        src_idx = torch.randperm(num_blocks, dtype=torch.int64)[:k]
        dst_idx = torch.randperm(num_blocks, dtype=torch.int64)[:k]
        block_mapping = torch.stack([src_idx, dst_idx], dim=1)
    block_mapping = block_mapping.to('cpu', dtype=torch.int64)

    ref_dst = base_dst.clone()
    act_dst = base_dst.clone()

    kernelgenbench.baseline.swap_blocks(src, ref_dst, block_mapping)
    kernelgenbench.triton.swap_blocks(src, act_dst, block_mapping)

    assert_close(act_dst, ref_dst, dtype)

    # ===== Performance Test =====
    numel = num_blocks * block_h * block_w
    if numel < 131072:
        return None

    src_perf = torch.randn(num_blocks, block_h, block_w, dtype=dtype, device='cuda')
    dst_baseline = torch.randn(num_blocks, block_h, block_w, dtype=dtype, device='cuda')
    dst_triton = dst_baseline.clone()

    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.swap_blocks(src_perf, dst_baseline, block_mapping),
        warmup=25, rep=100
    )
    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.swap_blocks(src_perf, dst_triton, block_mapping),
        warmup=25, rep=100
    )

    speedup = ms_baseline / ms_triton if ms_triton > 0 else float('inf')
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
