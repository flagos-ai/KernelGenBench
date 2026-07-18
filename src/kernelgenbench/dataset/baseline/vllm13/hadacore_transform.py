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
vLLM hadacore_transform baseline wrapper.
"""
import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def hadacore_transform(
    x: torch.Tensor,
    inplace: bool = True
) -> torch.Tensor:
    """Perform Hadamard transforms using [Hadacore](https://arxiv.org/abs/2412.08832)"""
    return _custom_ops.hadacore_transform(
        x,
        inplace
    )
