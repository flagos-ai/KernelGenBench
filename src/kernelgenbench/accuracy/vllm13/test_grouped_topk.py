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

@label("grouped_topk")
@parametrize("scores", [(1, 32), (128, 256), (1024, 512)])
@parametrize("num_expert_group", [1, 2, 4])
@parametrize("topk_group", [1, 2])
@parametrize("topk", [1, 2])
@parametrize("renormalize", [False, True])
@parametrize("scoring_func", [0, 1])
@parametrize("dtype", [torch.float16, torch.bfloat16])
def test_accuracy_grouped_topk(scores, num_expert_group, topk_group, topk, renormalize, scoring_func, dtype):
    # ===== Accuracy Test =====
    num_tokens, total_experts = scores

    # Skip invalid combos
    if total_experts % num_expert_group != 0:
        return None
    topk_group_eff = min(topk_group, num_expert_group)
    routed_scaling_factor = 1.0

    scores_t = torch.randn(num_tokens, total_experts, device=device, dtype=dtype)
    bias = torch.randn(total_experts, device=device, dtype=dtype)

    ref_weights, ref_ids = kernelgenbench.baseline.grouped_topk(
        scores_t, num_expert_group, topk_group_eff, topk, renormalize, routed_scaling_factor, bias, scoring_func
    )

    act_weights, act_ids = kernelgenbench.triton.grouped_topk(
        scores_t, num_expert_group, topk_group_eff, topk, renormalize, routed_scaling_factor, bias, scoring_func
    )

    # Compare weights (float) and ids (int)
    assert_close(act_weights, ref_weights, torch.float32)
    assert torch.equal(act_ids, ref_ids), f"topk_ids mismatch: max diff={(act_ids - ref_ids).abs().max()}"

    # ===== Performance Test =====
    if num_tokens < 1024:
        return None

    scores_bench = torch.randn(num_tokens, total_experts, device=device, dtype=dtype)
    bias_bench = torch.randn(total_experts, device=device, dtype=dtype)

    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.grouped_topk(
            scores_bench, num_expert_group, topk_group_eff, topk, renormalize, routed_scaling_factor, bias_bench, scoring_func
        ),
        warmup=25, rep=100
    )

    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.grouped_topk(
            scores_bench, num_expert_group, topk_group_eff, topk, renormalize, routed_scaling_factor, bias_bench, scoring_func
        ),
        warmup=25, rep=100
    )

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(
        ref_time=ms_baseline, res_time=ms_triton, speedup=speedup
    )
