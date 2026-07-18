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

@label("gather_and_maybe_dequant_cache")
@parametrize("config", [
    (4, 16, 1, 576, 2),
    (8, 16, 1, 576, 2),
    (8, 32, 1, 576, 4),
    (16, 16, 1, 576, 4),
    (32, 32, 1, 576, 4),
    (32, 64, 1, 576, 4),
    (64, 32, 1, 576, 4),
    (64, 32, 1, 576, 8),
    (64, 64, 1, 576, 4),
    (64, 64, 1, 576, 8),
    (128, 32, 1, 576, 4),
    (128, 64, 1, 576, 4),
    (128, 64, 1, 576, 8),
])
def test_accuracy_gather_and_maybe_dequant_cache(config):
    num_blocks = config[0]
    block_size = config[1]
    num_heads = config[2]
    head_dim = config[3]
    batch_size = config[4]

    src = torch.randn(num_blocks, block_size, num_heads, head_dim,
                      device=device, dtype=torch.float16)
    blocks_per_seq = 2
    seq_len = blocks_per_seq * block_size
    total_tokens = batch_size * seq_len

    block_table = torch.randint(0, num_blocks, (batch_size, blocks_per_seq),
                                device=device, dtype=torch.int32)
    cu_seq = torch.arange(0, total_tokens + 1, seq_len,
                          device=device, dtype=torch.int32)
    token_to_seq = torch.cat([
        torch.full((seq_len,), i, dtype=torch.int32)
        for i in range(batch_size)
    ]).to(device)
    scale = torch.tensor([1.0], device=device, dtype=torch.float32)

    ref_dst = torch.zeros(total_tokens, num_heads, head_dim,
                          device=device, dtype=torch.float16)
    act_dst = torch.zeros(total_tokens, num_heads, head_dim,
                          device=device, dtype=torch.float16)

    kernelgenbench.baseline.gather_and_maybe_dequant_cache(
        src, ref_dst, block_table, cu_seq, token_to_seq,
        total_tokens, "auto", scale)
    kernelgenbench.triton.gather_and_maybe_dequant_cache(
        src, act_dst, block_table, cu_seq, token_to_seq,
        total_tokens, "auto", scale)

    assert_close(act_dst, ref_dst, torch.float16)

    # ===== Performance Test =====
    if num_blocks < 32 or batch_size < 4:
        return None

    src_bench = torch.randn(num_blocks, block_size, num_heads, head_dim,
                           device=device, dtype=torch.float16)
    dst_bench = torch.zeros(total_tokens, num_heads, head_dim,
                           device=device, dtype=torch.float16)

    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.gather_and_maybe_dequant_cache(
            src_bench, dst_bench.clone(), block_table, cu_seq, token_to_seq,
            total_tokens, "auto", scale),
        warmup=25, rep=100
    )

    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.gather_and_maybe_dequant_cache(
            src_bench, dst_bench.clone(), block_table, cu_seq, token_to_seq,
            total_tokens, "auto", scale),
        warmup=25, rep=100
    )

    speedup = ms_baseline / ms_triton if ms_triton > 0 else float('inf')
    return CustomBenchmarkResult(
        ref_time=ms_baseline, res_time=ms_triton, speedup=speedup
    )
