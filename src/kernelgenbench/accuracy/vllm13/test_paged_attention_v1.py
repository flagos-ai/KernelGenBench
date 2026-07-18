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
import math

@label("paged_attention_v1")
@parametrize("num_seqs", [1, 4, 16, 71])
@parametrize("context_len", [1, 33, 128, 512, 2048])
@parametrize("num_heads", [8, 16])
@parametrize("head_size", [64, 80, 128])
@parametrize("kv_divisor", [1, 4])
@parametrize("block_size", [16, 32])
@parametrize("dtype", [torch.float16, torch.bfloat16])
def test_accuracy_paged_attention_v1(num_seqs, context_len, num_heads, head_size, kv_divisor, block_size, dtype):
    # ===== Accuracy Test =====
    num_kv_heads = max(1, num_heads // kv_divisor)

    # paged_attention_v1 is a decode-phase op: each seq has exactly 1 query token
    # query shape: (num_seqs, num_heads, head_size)
    query = torch.randn(num_seqs, num_heads, head_size, dtype=dtype, device='cuda')
    ref_out = torch.empty_like(query)
    act_out = torch.empty_like(query)

    # Vary context lengths across sequences
    seq_lens_list = []
    for i in range(num_seqs):
        # Vary around context_len: ensure at least 1
        sl = max(1, context_len - i % 3)
        seq_lens_list.append(sl)
    max_seq_len = max(seq_lens_list)

    # Build block tables
    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    blocks_per_seq = [(sl + block_size - 1) // block_size for sl in seq_lens_list]
    total_blocks = sum(blocks_per_seq)

    block_tables = torch.zeros(num_seqs, max_num_blocks_per_seq, dtype=torch.int32, device='cuda')
    current_block = 0
    for i in range(num_seqs):
        b = blocks_per_seq[i]
        if b > 0:
            block_tables[i, :b] = torch.arange(current_block, current_block + b, dtype=torch.int32, device='cuda')
            current_block += b

    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device='cuda')

    # KV caches: (total_blocks, block_size, num_kv_heads, head_size)
    key_cache = torch.randn(total_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device='cuda')
    value_cache = torch.randn(total_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device='cuda')

    scale = 1.0 / math.sqrt(float(head_size))

    # k_scale, v_scale: scalar tensors for non-quantized cache
    k_scale = torch.ones(1, dtype=torch.float32, device='cuda')
    v_scale = torch.ones(1, dtype=torch.float32, device='cuda')

    common = dict(
        num_kv_heads=num_kv_heads, scale=scale,
        block_tables=block_tables, seq_lens=seq_lens,
        block_size=block_size, max_seq_len=max_seq_len,
        alibi_slopes=None, kv_cache_dtype="auto",
        k_scale=k_scale, v_scale=v_scale,
        tp_rank=0, blocksparse_local_blocks=0,
        blocksparse_vert_stride=0, blocksparse_block_size=64,
        blocksparse_head_sliding_step=0,
    )

    kernelgenbench.baseline.paged_attention_v1(
        ref_out, query, key_cache, value_cache, **common)

    kernelgenbench.triton.paged_attention_v1(
        act_out, query, key_cache, value_cache, **common)

    assert_close(act_out, ref_out, dtype)

    # ===== Performance Test =====
    if num_seqs * max_seq_len < 4096:
        return None

    out_baseline = torch.empty_like(query)
    out_triton = torch.empty_like(query)

    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.paged_attention_v1(out_baseline, query, key_cache, value_cache, **common),
        warmup=25, rep=100)
    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.paged_attention_v1(out_triton, query, key_cache, value_cache, **common),
        warmup=25, rep=100)

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
