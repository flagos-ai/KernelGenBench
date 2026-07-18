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

@label("fused_qk_norm_rope")
@parametrize("seq_len", [1, 71, 128, 1024, 5333])
@parametrize("heads", [(4, 4, 4), (16, 16, 16), (32, 16, 8)])
@parametrize("head_dim", [64, 128])
@parametrize("dtype", [torch.float16, torch.bfloat16])
@parametrize("is_neox", [False, True])
def test_accuracy_fused_qk_norm_rope(seq_len, heads, head_dim, dtype, is_neox):
    # ===== Accuracy Test =====
    num_heads_q, num_heads_k, num_heads_v = heads
    eps = 1e-5

    # qkv shape: (seq_len, q_size + k_size + v_size)
    q_size = num_heads_q * head_dim
    k_size = num_heads_k * head_dim
    v_size = num_heads_v * head_dim
    total_hidden = q_size + k_size + v_size

    # Create inputs with correct shapes
    qkv = torch.randn(seq_len, total_hidden, device='cuda', dtype=dtype)
    # q_weight/k_weight: (head_dim,) — 1D RMSNorm weight
    q_weight = torch.randn(head_dim, device='cuda', dtype=dtype)
    k_weight = torch.randn(head_dim, device='cuda', dtype=dtype)
    # cos_sin_cache: (max_position, head_dim)
    max_position = max(seq_len, 4096)
    cos_sin_cache = torch.randn(max_position, head_dim, device='cuda', dtype=dtype)

    position_ids = torch.arange(seq_len, device='cuda', dtype=torch.long)

    # Clone qkv for baseline and triton (in-place op modifies qkv)
    ref_qkv = qkv.clone()
    act_qkv = qkv.clone()

    # Call baseline (in-place modifies ref_qkv)
    kernelgenbench.baseline.fused_qk_norm_rope(
        ref_qkv, num_heads_q, num_heads_k, num_heads_v, head_dim, eps,
        q_weight, k_weight, cos_sin_cache, is_neox, position_ids
    )

    # Call triton (in-place modifies act_qkv)
    kernelgenbench.triton.fused_qk_norm_rope(
        act_qkv, num_heads_q, num_heads_k, num_heads_v, head_dim, eps,
        q_weight, k_weight, cos_sin_cache, is_neox, position_ids
    )

    # Compare mutated qkv tensors
    assert_close(act_qkv, ref_qkv, dtype)

    # ===== Performance Test =====
    if seq_len < 1024 or total_hidden < 4096:
        return None

    # Prepare fresh data for benchmarking
    qkv_bench = torch.randn(seq_len, total_hidden, device='cuda', dtype=dtype)

    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.fused_qk_norm_rope(
            qkv_bench.clone(), num_heads_q, num_heads_k, num_heads_v, head_dim, eps,
            q_weight, k_weight, cos_sin_cache, is_neox, position_ids
        ),
        warmup=25, rep=100
    )

    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.fused_qk_norm_rope(
            qkv_bench.clone(), num_heads_q, num_heads_k, num_heads_v, head_dim, eps,
            q_weight, k_weight, cos_sin_cache, is_neox, position_ids
        ),
        warmup=25, rep=100
    )

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
