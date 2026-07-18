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

@label("rotary_embedding")
@parametrize("shape_pair", [(1, 32), (71, 497), (128, 512), (1024, 4096), (5333, 8192)])
@parametrize("num_heads", [1, 4, 16])
@parametrize("head_size", [32, 64])
@parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@parametrize("key_is_none", [True, False])
@parametrize("is_neox", [True, False])
def test_accuracy_rotary_embedding(shape_pair, num_heads, head_size, dtype, key_is_none, is_neox):
    # ===== Accuracy Test =====
    seq_len, cache_len = shape_pair

    # Create positions tensor (indices into cos_sin_cache)
    positions = torch.randint(low=0, high=cache_len, size=(seq_len,), dtype=torch.long, device='cuda')

    # Build cos/sin cache: shape [cache_len, 2, head_size]
    # Standard RoPE cache with base=10000, repeated along last dim to match head_size
    half = head_size // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_size, 2, device='cuda', dtype=torch.float32) / head_size))
    t = torch.arange(cache_len, device='cuda', dtype=torch.float32).unsqueeze(1)  # [cache_len, 1]
    freqs = t * inv_freq  # [cache_len, half]
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1)  # [cache_len, head_size]
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1)  # [cache_len, head_size]
    cos_sin_cache = torch.stack([cos, sin], dim=1).to(dtype=dtype)  # [cache_len, 2, head_size]

    # Create inputs
    q0 = torch.randn(seq_len, num_heads, head_size, device='cuda', dtype=dtype)
    k0 = None if key_is_none else torch.randn(seq_len, num_heads, head_size, device='cuda', dtype=dtype)

    # Clone for baseline and triton (in-place mutation expected)
    ref_q = q0.clone()
    ref_k = None if key_is_none else k0.clone()
    act_q = q0.clone()
    act_k = None if key_is_none else k0.clone()

    # Call baseline
    kernelgenbench.baseline.rotary_embedding(positions, ref_q, ref_k, head_size, cos_sin_cache, is_neox)
    # Call triton
    kernelgenbench.triton.rotary_embedding(positions, act_q, act_k, head_size, cos_sin_cache, is_neox)

    # Compare mutated tensors
    assert_close(act_q, ref_q, dtype)
    if not key_is_none:
        assert_close(act_k, ref_k, dtype)

    # ===== Performance Test =====
    # Skip small sizes for performance test
    if seq_len < 1024:
        return None

    # Prepare fresh data for benchmarking
    positions_bench = torch.randint(low=0, high=cache_len, size=(seq_len,), dtype=torch.long, device='cuda')
    # Reuse the same cos_sin_cache structure but allocate fresh to avoid potential caching artifacts
    freqs_b = torch.arange(cache_len, device='cuda', dtype=torch.float32).unsqueeze(1) * inv_freq
    cos_b = torch.cos(freqs_b).repeat_interleave(2, dim=-1)
    sin_b = torch.sin(freqs_b).repeat_interleave(2, dim=-1)
    cos_sin_cache_bench = torch.stack([cos_b, sin_b], dim=1).to(dtype=dtype)

    q0_b = torch.randn(seq_len, num_heads, head_size, device='cuda', dtype=dtype)
    k0_b = None if key_is_none else torch.randn(seq_len, num_heads, head_size, device='cuda', dtype=dtype)

    def bench_baseline():
        q = q0_b.clone()
        k = None if key_is_none else k0_b.clone()
        kernelgenbench.baseline.rotary_embedding(positions_bench, q, k, head_size, cos_sin_cache_bench, is_neox)

    def bench_triton():
        q = q0_b.clone()
        k = None if key_is_none else k0_b.clone()
        kernelgenbench.triton.rotary_embedding(positions_bench, q, k, head_size, cos_sin_cache_bench, is_neox)

    ms_baseline = triton.testing.do_bench(bench_baseline, warmup=25, rep=100)
    ms_triton = triton.testing.do_bench(bench_triton, warmup=25, rep=100)

    speedup = ms_baseline / ms_triton if ms_triton > 0 else float('inf')
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
