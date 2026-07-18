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

"""vLLM gptq_marlin_gemm baseline wrapper."""
import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def gptq_marlin_gemm(
    a: torch.Tensor,
    c: "torch.Tensor | None",
    b_q_weight: torch.Tensor,
    b_bias: "torch.Tensor | None",
    b_scales: torch.Tensor,
    a_scales: "torch.Tensor | None",
    global_scale: "torch.Tensor | None",
    b_zeros: torch.Tensor,
    g_idx: torch.Tensor,
    perm: torch.Tensor,
    workspace: torch.Tensor,
    b_q_type: "object",
    size_m: int,
    size_n: int,
    size_k: int,
    is_k_full: bool = True,
    use_atomic_add: bool = False,
    use_fp32_reduce: bool = False,
    is_zp_float: bool = False
) -> torch.Tensor:
    return _custom_ops.gptq_marlin_gemm(
        a, c, b_q_weight, b_bias, b_scales, a_scales, global_scale,
        b_zeros, g_idx, perm, workspace, b_q_type, size_m, size_n, size_k,
        is_k_full, use_atomic_add, use_fp32_reduce, is_zp_float
    )
