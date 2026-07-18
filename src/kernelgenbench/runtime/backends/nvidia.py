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

"""NVIDIA backend — identity mapping (cuBLAS native)."""
import os

from . import VendorBackend, register_backend
from ..vendor import Vendor


class NvidiaBackend(VendorBackend):
    vendor = Vendor.NVIDIA
    device_name = "cuda"
    visible_devices_env = "CUDA_VISIBLE_DEVICES"

    @property
    def blas_lib_path(self):
        cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
        return os.path.join(cuda_home, "lib64", "libcublas.so.12")

    blas_create_handle_fn = "cublasCreate_v2"
    blas_set_pointer_mode_fn = "cublasSetPointerMode_v2"

    # All mappings are identity — cuBLAS names and enums pass through unchanged.


register_backend(NvidiaBackend())
