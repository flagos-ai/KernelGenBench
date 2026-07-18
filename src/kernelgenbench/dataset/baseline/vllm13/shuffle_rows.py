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
vLLM shuffle_rows baseline wrapper.
"""
import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def shuffle_rows(
    input_tensor: torch.Tensor,
    dst2src_map: torch.Tensor
) -> None:
    """Shuffle and expand the input tensor according to the dst2src_map and store the result in output_tensor."""
    return _custom_ops.shuffle_rows(
        input_tensor,
        dst2src_map
    )
