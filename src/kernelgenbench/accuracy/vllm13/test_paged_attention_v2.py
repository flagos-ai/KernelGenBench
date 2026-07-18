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

@label("paged_attention_v2")
@parametrize("shape", [
    # (num_seqs, num_heads, num_kv_heads, head_dim, block_size, num_blocks, max_seq_len)
    (2, 8, 8, 64, 16, 32, 128),
    (4, 8, 8, 64, 16, 64, 256),
    (4, 32, 8, 128, 16, 128, 512),
    (8, 16, 8, 128, 16, 256, 1024),
])
@parametrize("dtype", [torch.float16, torch.bfloat16])
def test_accuracy_paged_attention_v2(shape, dtype):
    # ===== Accuracy Test =====
    num_seqs = shape[0]
    num_heads, num_kv_heads, head_dim = shape[1], shape[2], shape[3]
    block_size, num_blocks, max_seq_len = shape[4], shape[5], shape[6]
    x = 8

    seq_lens = torch.randint(1, max_seq_len + 1, (num_seqs,), device=device, dtype=torch.int32)
    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    block_tables = torch.randint(0, num_blocks, (num_seqs, max_num_blocks_per_seq), device=device, dtype=torch.int32)

    query = torch.randn(num_seqs, num_heads, head_dim, device=device, dtype=dtype)
    key_cache = torch.randn(num_blocks, num_kv_heads, head_dim // x, block_size, x, device=device, dtype=dtype)
    value_cache = torch.randn(num_blocks, num_kv_heads, head_dim, block_size, device=device, dtype=dtype)

    scale = 1.0 / (head_dim ** 0.5)
    k_scale = torch.tensor(1.0, device=device, dtype=torch.float32)
    v_scale = torch.tensor(1.0, device=device, dtype=torch.float32)
    num_partitions = (max_seq_len + 511) // 512

    ref_out = torch.empty(num_seqs, num_heads, head_dim, device=device, dtype=dtype)
    ref_exp = torch.empty(num_seqs, num_heads, num_partitions, device=device, dtype=torch.float32)
    ref_ml = torch.empty(num_seqs, num_heads, num_partitions, device=device, dtype=torch.float32)
    ref_tmp = torch.empty(num_seqs, num_heads, num_partitions, head_dim, device=device, dtype=dtype)

    act_out = torch.empty(num_seqs, num_heads, head_dim, device=device, dtype=dtype)
    act_exp = torch.empty(num_seqs, num_heads, num_partitions, device=device, dtype=torch.float32)
    act_ml = torch.empty(num_seqs, num_heads, num_partitions, device=device, dtype=torch.float32)
    act_tmp = torch.empty(num_seqs, num_heads, num_partitions, head_dim, device=device, dtype=dtype)

    kernelgenbench.baseline.paged_attention_v2(
        ref_out, ref_exp, ref_ml, ref_tmp, query, key_cache, value_cache,
        num_kv_heads, scale, block_tables, seq_lens, block_size, max_seq_len,
        None, "auto", k_scale, v_scale)
    kernelgenbench.triton.paged_attention_v2(
        act_out, act_exp, act_ml, act_tmp, query, key_cache, value_cache,
        num_kv_heads, scale, block_tables, seq_lens, block_size, max_seq_len,
        None, "auto", k_scale, v_scale)

    assert_close(act_out, ref_out, dtype)

    # ===== Performance Test =====
    if max_seq_len < 256:
        return None

    out_baseline = torch.empty(num_seqs, num_heads, head_dim, device=device, dtype=dtype)
    exp_baseline = torch.empty(num_seqs, num_heads, num_partitions, device=device, dtype=torch.float32)
    ml_baseline = torch.empty(num_seqs, num_heads, num_partitions, device=device, dtype=torch.float32)
    tmp_baseline = torch.empty(num_seqs, num_heads, num_partitions, head_dim, device=device, dtype=dtype)
    out_triton = torch.empty(num_seqs, num_heads, head_dim, device=device, dtype=dtype)
    exp_triton = torch.empty(num_seqs, num_heads, num_partitions, device=device, dtype=torch.float32)
    ml_triton = torch.empty(num_seqs, num_heads, num_partitions, device=device, dtype=torch.float32)
    tmp_triton = torch.empty(num_seqs, num_heads, num_partitions, head_dim, device=device, dtype=dtype)

    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.paged_attention_v2(
            out_baseline, exp_baseline, ml_baseline, tmp_baseline, query, key_cache, value_cache,
            num_kv_heads, scale, block_tables, seq_lens, block_size, max_seq_len,
            None, "auto", k_scale, v_scale),
        warmup=25, rep=100)
    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.paged_attention_v2(
            out_triton, exp_triton, ml_triton, tmp_triton, query, key_cache, value_cache,
            num_kv_heads, scale, block_tables, seq_lens, block_size, max_seq_len,
            None, "auto", k_scale, v_scale),
        warmup=25, rep=100)

    speedup = ms_baseline / ms_triton if ms_triton > 0 else float("inf")
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
