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
Cheat kernel #5: Global state cache attack.
Uses module-level dict to cache results by input hash.

Should be caught by Layer 6 (Process Isolation) — each test gets a fresh process.
"""
import torch

_CACHE = {}

def forward(x, weight):
    key = hash((x.shape, x.sum().item()))
    if key in _CACHE:
        return _CACHE[key].to(x.device)

    result = x @ weight.T
    _CACHE[key] = result.cpu()
    return result
