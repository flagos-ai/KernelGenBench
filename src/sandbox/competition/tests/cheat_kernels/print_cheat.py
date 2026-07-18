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
Cheat kernel #4: Print sniffing attack.
Attempts to print() tensor data to stdout for data extraction.

Should be caught by Layer 3 (Import Hook) — print is no-op.
"""
import torch

def forward(x, weight):
    # Try to extract data via print
    print("DATA_START")
    print(x.shape)
    print(x.mean().item())
    print("DATA_END")
    return x @ weight.T
