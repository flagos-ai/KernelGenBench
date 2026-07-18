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
Cheat kernel #1: Disk cache attack.
Attempts to cache results to disk to avoid recomputation.

Should be caught by Layer 1 (Cache Isolation) + Layer 3 (Import Hook).
"""
import torch
import os
import pickle

_CACHE_FILE = "/tmp/cheat_cache.pkl"

def forward(x, weight):
    # Try to load from disk cache
    if os.path.exists(_CACHE_FILE):
        with open(_CACHE_FILE, "rb") as f:
            return pickle.load(f).to(x.device)

    # Compute
    result = x @ weight.T

    # Try to save to disk cache
    try:
        with open(_CACHE_FILE, "wb") as f:
            pickle.dump(result.cpu(), f)
    except Exception:
        pass

    return result
