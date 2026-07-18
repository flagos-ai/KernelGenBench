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

"""Ascend NPU backend — placeholder for future implementation."""
from . import VendorBackend, register_backend
from ..vendor import Vendor


class AscendBackend(VendorBackend):
    vendor = Vendor.ASCEND
    device_name = "npu"
    visible_devices_env = "ASCEND_RT_VISIBLE_DEVICES"

    device_constraints = """\
## Device-Specific Requirements
It should be noted that the operator runs on Ascend NPU devices.
1. If `import torch` is used, it must be immediately followed by `import torch_npu`.
2. The device type is `npu`. Use `npu` for all device-related APIs.
3. All GPU-related commands must use `ASCEND_RT_VISIBLE_DEVICES` instead of `CUDA_VISIBLE_DEVICES`.
"""


register_backend(AscendBackend())
