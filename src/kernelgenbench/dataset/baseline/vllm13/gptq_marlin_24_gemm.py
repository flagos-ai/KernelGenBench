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

"""
vLLM gptq_marlin_24_gemm baseline wrapper.
"""
import torch
try:
    import vllm
except ModuleNotFoundError:
    vllm = None
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def gptq_marlin_24_gemm(
    a: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_meta: torch.Tensor,
    b_scales: torch.Tensor,
    workspace: torch.Tensor,
    b_q_type: vllm.scalar_type.ScalarType,
    size_m: int,
    size_n: int,
    size_k: int
) -> torch.Tensor:
    """Wrapper for vLLM gptq_marlin_24_gemm implementation."""
    return _custom_ops.gptq_marlin_24_gemm(
        a,
        b_q_weight,
        b_meta,
        b_scales,
        workspace,
        b_q_type,
        size_m,
        size_n,
        size_k
    )
