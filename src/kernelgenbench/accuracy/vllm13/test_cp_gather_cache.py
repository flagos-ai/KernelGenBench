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

@label("cp_gather_cache")
@parametrize("config", [
    (8, 16, 4, 64, 2),
    (16, 16, 8, 128, 2),
    (32, 32, 4, 64, 4),
    (16, 32, 8, 64, 2),
    (32, 16, 4, 128, 4),
    (64, 16, 4, 64, 4),
    (64, 32, 8, 128, 4),
    (128, 32, 8, 256, 8),
])
def test_accuracy_cp_gather_cache(config):
    num_blocks = config[0]
    block_size = config[1]
    num_heads = config[2]
    head_size = config[3]
    batch_size = config[4]

    src_cache = torch.randn(num_blocks, block_size, num_heads, head_size,
                            device=device, dtype=torch.float16)

    blocks_per_seq = 2
    total_tokens = batch_size * blocks_per_seq * block_size
    block_table = torch.randint(0, num_blocks, (batch_size, blocks_per_seq),
                                device=device, dtype=torch.int32)
    cu_seq_lens = torch.arange(0, total_tokens + 1,
                               blocks_per_seq * block_size,
                               device=device, dtype=torch.int32)

    ref_dst = torch.zeros(total_tokens, num_heads, head_size,
                          device=device, dtype=torch.float16)
    act_dst = torch.zeros(total_tokens, num_heads, head_size,
                          device=device, dtype=torch.float16)

    kernelgenbench.baseline.cp_gather_cache(src_cache, ref_dst, block_table,
                                       cu_seq_lens, batch_size)
    kernelgenbench.triton.cp_gather_cache(src_cache, act_dst, block_table,
                                     cu_seq_lens, batch_size)

    assert torch.equal(act_dst, ref_dst), \
        f"Mismatch: max diff={(act_dst - ref_dst).abs().max()}"

    # ===== Performance Test =====
    if num_blocks < 32 or batch_size < 4:
        return None

    src_cache_bench = torch.randn(num_blocks, block_size, num_heads, head_size,
                                  device=device, dtype=torch.float16)
    dst_baseline = torch.zeros(total_tokens, num_heads, head_size,
                           device=device, dtype=torch.float16)
    dst_triton = torch.zeros(total_tokens, num_heads, head_size,
                           device=device, dtype=torch.float16)

    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.cp_gather_cache(src_cache_bench, dst_baseline, block_table,
                                                   cu_seq_lens, batch_size),
        warmup=25, rep=100
    )

    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.cp_gather_cache(src_cache_bench, dst_triton, block_table,
                                                 cu_seq_lens, batch_size),
        warmup=25, rep=100
    )

    speedup = ms_baseline / ms_triton if ms_triton > 0 else float('inf')
    return CustomBenchmarkResult(
        ref_time=ms_baseline, res_time=ms_triton, speedup=speedup
    )
