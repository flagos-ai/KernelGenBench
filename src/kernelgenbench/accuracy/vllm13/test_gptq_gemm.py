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

@label("gptq_gemm")
@parametrize("config", [
    # (M, K, N, bit, use_exllama, use_v2_format)
    (1, 256, 256, 4, True, False),
    (16, 256, 256, 4, True, False),
    (32, 256, 256, 4, True, False),
    (64, 512, 512, 4, True, False),
])
def test_accuracy_gptq_gemm(config):
    # ===== Accuracy Test =====
    M = config[0]
    K = config[1]
    N = config[2]
    bit = config[3]
    use_exllama = config[4]
    use_v2_format = config[5]
    dtype = torch.float16
    group_size = 128
    group_count = K // group_size
    pack_factor = 32 // bit

    a_fp = torch.randn(M, K, device=device, dtype=dtype)

    b_g_idx = (torch.arange(K, device=device, dtype=torch.int32) // group_size).contiguous()

    shifts = (torch.arange(pack_factor, device=device, dtype=torch.int32) * bit).view(1, 1, -1)

    Kp = K // pack_factor
    qvals_w = torch.randint(0, (1 << bit), (Kp, N, pack_factor), device=device, dtype=torch.int32)
    b_q_weight = torch.sum(qvals_w << shifts, dim=-1).to(torch.int32).contiguous()

    gp = (group_count + pack_factor - 1) // pack_factor
    qvals_z = torch.randint(0, (1 << bit), (gp, N, pack_factor), device=device, dtype=torch.int32)
    qzeros_packed = torch.sum(qvals_z << shifts, dim=-1).to(torch.int32).contiguous()

    if use_v2_format:
        b_gptq_qzeros = qzeros_packed.transpose(0, 1).contiguous()
        b_gptq_scales = (torch.rand(N, group_count, device=device, dtype=dtype) + 0.01)
    else:
        b_gptq_qzeros = qzeros_packed
        b_gptq_scales = (torch.rand(group_count, N, device=device, dtype=dtype) + 0.01)

    ref_out = kernelgenbench.baseline.gptq_gemm(
        a_fp, b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx,
        use_exllama, use_v2_format, bit)
    act_out = kernelgenbench.triton.gptq_gemm(
        a_fp, b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx,
        use_exllama, use_v2_format, bit)

    assert_close(act_out, ref_out, torch.float16)

    # ===== Performance Test =====
    if K < 256:
        return None

    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.gptq_gemm(
            a_fp, b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx,
            use_exllama, use_v2_format, bit),
        warmup=25, rep=100)
    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.gptq_gemm(
            a_fp, b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx,
            use_exllama, use_v2_format, bit),
        warmup=25, rep=100)

    speedup = ms_baseline / ms_triton if ms_triton > 0 else float("inf")
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
