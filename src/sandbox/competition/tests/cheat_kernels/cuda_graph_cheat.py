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
Cheat kernel #2: CUDA Graph attack.
Attempts to capture and replay CUDA Graph to hide kernel launch overhead.

Should be caught by Layer 4 (CUDA Protection).
"""
import torch

_g = None

def forward(x, weight):
    global _g

    if _g is not None:
        _g.replay()
        return x

    # Capture CUDA Graph
    _g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(_g):
        y = x @ weight.T
    _g.replay()
    return x
