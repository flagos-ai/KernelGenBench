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

@label("cp_gather_indexer_k_quant_cache")
@parametrize("config", [
    (4, 16, 4, 64, 2),
    (8, 16, 8, 128, 2),
    (8, 32, 4, 64, 4),
    (16, 16, 8, 64, 2),
    (16, 16, 4, 64, 4),
    (16, 32, 4, 128, 4),
    (32, 16, 4, 64, 4),
    (32, 32, 8, 128, 4),
    (32, 64, 4, 128, 4),
    (64, 32, 8, 128, 4),
    (64, 64, 8, 256, 4),
    (64, 64, 8, 256, 8),
    (128, 64, 8, 256, 8),
])
def test_accuracy_cp_gather_indexer_k_quant_cache(config):
    num_blocks = config[0]
    block_size = config[1]
    num_heads = config[2]
    head_size = config[3]
    batch_size = config[4]

    kv_cache = torch.randn(num_blocks, block_size, num_heads, head_size,
                           device=device, dtype=torch.float16)
    blocks_per_seq = 2
    total_tokens = batch_size * blocks_per_seq * block_size

    block_table = torch.randint(0, num_blocks, (batch_size, blocks_per_seq),
                                device=device, dtype=torch.int32)
    cu_seq = torch.arange(0, total_tokens + 1,
                          blocks_per_seq * block_size,
                          device=device, dtype=torch.int32)

    ref_k = torch.zeros(total_tokens, num_heads, head_size,
                        device=device, dtype=torch.float16)
    ref_s = torch.zeros(total_tokens, num_heads,
                        device=device, dtype=torch.float32)
    act_k = torch.zeros(total_tokens, num_heads, head_size,
                        device=device, dtype=torch.float16)
    act_s = torch.zeros(total_tokens, num_heads,
                        device=device, dtype=torch.float32)

    kernelgenbench.baseline.cp_gather_indexer_k_quant_cache(
        kv_cache, ref_k, ref_s, block_table, cu_seq)
    kernelgenbench.triton.cp_gather_indexer_k_quant_cache(
        kv_cache, act_k, act_s, block_table, cu_seq)

    assert torch.equal(act_k, ref_k), \
        f"k mismatch: max diff={(act_k - ref_k).abs().max()}"
    assert torch.equal(act_s, ref_s), \
        f"scale mismatch: max diff={(act_s - ref_s).abs().max()}"

    # ===== Performance Test =====
    if num_blocks < 16 or batch_size < 4:
        return None

    kv_cache_bench = torch.randn(num_blocks, block_size, num_heads, head_size,
                                 device=device, dtype=torch.float16)
    k_baseline = torch.zeros(total_tokens, num_heads, head_size,
                            device=device, dtype=torch.float16)
    s_baseline = torch.zeros(total_tokens, num_heads,
                            device=device, dtype=torch.float32)
    k_triton = torch.zeros(total_tokens, num_heads, head_size,
                          device=device, dtype=torch.float16)
    s_triton = torch.zeros(total_tokens, num_heads,
                          device=device, dtype=torch.float32)

    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.cp_gather_indexer_k_quant_cache(
            kv_cache_bench, k_baseline, s_baseline, block_table, cu_seq),
        warmup=25, rep=100
    )

    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.cp_gather_indexer_k_quant_cache(
            kv_cache_bench, k_triton, s_triton, block_table, cu_seq),
        warmup=25, rep=100
    )

    speedup = ms_baseline / ms_triton if ms_triton > 0 else float('inf')
    return CustomBenchmarkResult(
        ref_time=ms_baseline, res_time=ms_triton, speedup=speedup
    )
