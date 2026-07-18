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

@label("merge_attn_states")
@parametrize("shape", [(1, 8, 64), (4, 8, 64), (4, 32, 128), (16, 8, 128), (16, 32, 64), (64, 32, 128)])
@parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_accuracy_merge_attn_states(shape, dtype):
    # ===== Accuracy Test =====
    num_seqs, num_heads, head_dim = shape[0], shape[1], shape[2]

    # Create inputs: output/prefix/suffix are 3D, lse are 2D
    prefix_output = torch.randn(num_seqs, num_heads, head_dim, device=device, dtype=dtype)
    suffix_output = torch.randn(num_seqs, num_heads, head_dim, device=device, dtype=dtype)
    prefix_lse = (torch.rand(num_seqs, num_heads, device=device, dtype=torch.float32) - 0.5) * 4.0
    suffix_lse = (torch.rand(num_seqs, num_heads, device=device, dtype=torch.float32) - 0.5) * 4.0

    ref_output = torch.empty(num_seqs, num_heads, head_dim, device=device, dtype=dtype)
    act_output = torch.empty(num_seqs, num_heads, head_dim, device=device, dtype=dtype)

    kernelgenbench.baseline.merge_attn_states(ref_output, prefix_output, prefix_lse, suffix_output, suffix_lse)
    kernelgenbench.triton.merge_attn_states(act_output, prefix_output, prefix_lse, suffix_output, suffix_lse)

    assert_close(act_output, ref_output, dtype)

    # ===== Performance Test =====
    if num_seqs < 16:
        return None

    bm_prefix = torch.randn(num_seqs, num_heads, head_dim, device=device, dtype=dtype)
    bm_suffix = torch.randn(num_seqs, num_heads, head_dim, device=device, dtype=dtype)
    bm_plse = (torch.rand(num_seqs, num_heads, device=device, dtype=torch.float32) - 0.5) * 4.0
    bm_slse = (torch.rand(num_seqs, num_heads, device=device, dtype=torch.float32) - 0.5) * 4.0

    out_baseline = torch.empty(num_seqs, num_heads, head_dim, device=device, dtype=dtype)
    out_triton = torch.empty(num_seqs, num_heads, head_dim, device=device, dtype=dtype)

    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.merge_attn_states(
            out_baseline, bm_prefix, bm_plse, bm_suffix, bm_slse,
        ),
        warmup=25, rep=100,
    )

    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.merge_attn_states(
            out_triton, bm_prefix, bm_plse, bm_suffix, bm_slse,
        ),
        warmup=25, rep=100,
    )

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
