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
FlagBench Runtime — unified multi-chip backend management.

Usage:
    from kernelgenbench.runtime import detect_vendor, get_backend

    vendor = detect_vendor()       # -> Vendor.HYGON
    backend = get_backend()        # -> HygonBackend instance
    backend.map_blas_func_name("cublasSgemm_v2")  # -> "hipblasSgemm"
    backend.map_op(0)              # -> 111
"""
from .vendor import Vendor, detect_vendor
from .backends import VendorBackend, get_backend, register_backend

__all__ = [
    "Vendor",
    "detect_vendor",
    "VendorBackend",
    "get_backend",
    "register_backend",
]
