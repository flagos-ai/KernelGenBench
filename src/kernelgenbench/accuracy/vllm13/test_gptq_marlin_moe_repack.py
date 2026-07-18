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

@label("gptq_marlin_moe_repack")
@parametrize("config", [
    (4, 256, 256, 4),
    (8, 256, 512, 4),
    (4, 512, 256, 4),
    (16, 256, 256, 4),
    (4, 256, 256, 8),
    (8, 512, 512, 4),
    (8, 512, 512, 8),
    (16, 512, 512, 4),
    (16, 512, 512, 8),
    (8, 1024, 512, 4),
])
def test_accuracy_gptq_marlin_moe_repack(config):
    num_experts = config[0]
    K = config[1]
    N = config[2]
    num_bits = config[3]
    pack_factor = 32 // num_bits

    b_q_weight = torch.randint(0, 2**31, (num_experts, K // pack_factor, N),
                               device=device, dtype=torch.int32)
    perm = torch.arange(K, device=device, dtype=torch.int32).unsqueeze(0).expand(num_experts, -1).contiguous()

    ref_out = kernelgenbench.baseline.gptq_marlin_moe_repack(b_q_weight, perm, K, N, num_bits)
    act_out = kernelgenbench.triton.gptq_marlin_moe_repack(b_q_weight, perm, K, N, num_bits)

    assert torch.equal(act_out, ref_out), f"Mismatch: max diff={(act_out - ref_out).abs().max()}"

    # ===== Performance Test =====
    if num_experts < 8 or K < 512 or N < 512:
        return None

    b_q_weight_bench = torch.randint(0, 2**31, (num_experts, K // pack_factor, N),
                                     device=device, dtype=torch.int32)
    perm_bench = torch.arange(K, device=device, dtype=torch.int32).unsqueeze(0).expand(num_experts, -1).contiguous()

    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.gptq_marlin_moe_repack(
            b_q_weight_bench, perm_bench, K, N, num_bits),
        warmup=25, rep=100
    )

    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.gptq_marlin_moe_repack(
            b_q_weight_bench, perm_bench, K, N, num_bits),
        warmup=25, rep=100
    )

    speedup = ms_baseline / ms_triton if ms_triton > 0 else float('inf')
    return CustomBenchmarkResult(
        ref_time=ms_baseline, res_time=ms_triton, speedup=speedup
    )
