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
