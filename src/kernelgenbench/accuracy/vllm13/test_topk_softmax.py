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

@label("topk_softmax")
@parametrize("num_tokens", [1, 7, 32, 128, 1024, 4096])
@parametrize("num_experts", [8, 16, 64])
@parametrize("top_k", [1, 2, 4])
@parametrize("renormalize", [False, True])
def test_accuracy_topk_softmax(num_tokens, num_experts, top_k, renormalize):
    # ===== Accuracy Test =====
    # top_k must not exceed num_experts
    if top_k > num_experts:
        return None

    gating_output = torch.randn(num_tokens, num_experts, device='cuda', dtype=torch.float32)

    # Output tensors for baseline
    ref_weights = torch.empty(num_tokens, top_k, device='cuda', dtype=torch.float32)
    ref_ids = torch.empty(num_tokens, top_k, device='cuda', dtype=torch.int32)
    ref_indices = torch.empty(num_tokens, top_k, device='cuda', dtype=torch.int32)

    # Output tensors for triton
    act_weights = torch.empty(num_tokens, top_k, device='cuda', dtype=torch.float32)
    act_ids = torch.empty(num_tokens, top_k, device='cuda', dtype=torch.int32)
    act_indices = torch.empty(num_tokens, top_k, device='cuda', dtype=torch.int32)

    kernelgenbench.baseline.topk_softmax(
        ref_weights, ref_ids, ref_indices, gating_output, renormalize)

    kernelgenbench.triton.topk_softmax(
        act_weights, act_ids, act_indices, gating_output, renormalize)

    # Compare weights and ids
    assert_close(act_weights, ref_weights, torch.float32)
    assert (act_ids == ref_ids).all(), "topk_ids mismatch"

    # ===== Performance Test =====
    if num_tokens < 1024:
        return None

    w_baseline = torch.empty(num_tokens, top_k, device='cuda', dtype=torch.float32)
    i_baseline = torch.empty(num_tokens, top_k, device='cuda', dtype=torch.int32)
    t_baseline = torch.empty(num_tokens, top_k, device='cuda', dtype=torch.int32)
    w_triton = torch.empty(num_tokens, top_k, device='cuda', dtype=torch.float32)
    i_triton = torch.empty(num_tokens, top_k, device='cuda', dtype=torch.int32)
    t_triton = torch.empty(num_tokens, top_k, device='cuda', dtype=torch.int32)

    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.topk_softmax(w_baseline, i_baseline, t_baseline, gating_output, renormalize),
        warmup=25, rep=100)
    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.topk_softmax(w_triton, i_triton, t_triton, gating_output, renormalize),
        warmup=25, rep=100)

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
