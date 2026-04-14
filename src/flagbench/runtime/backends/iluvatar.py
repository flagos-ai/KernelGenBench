"""Iluvatar backend — placeholder for future implementation."""
from . import VendorBackend, register_backend
from ..vendor import Vendor


class IluvatarBackend(VendorBackend):
    vendor = Vendor.ILUVATAR
    device_name = "cuda"
    visible_devices_env = "CUDA_VISIBLE_DEVICES"

    device_constraints = """\
## Device-Specific Requirements
It should be noted that the operator runs on Iluvatar BI-V150 GPUs with CoreX software stack.
1. The device type is `cuda`. No special import is needed beyond `import torch`.
2. Avoid NVIDIA-specific hardware features (e.g., Tensor Core specific instructions).
3. Use `allow_tf32=False` for `tl.dot` to ensure precision.
4. Prefer smaller BLOCK_SIZE values (e.g., 512 or 1024).
"""


register_backend(IluvatarBackend())
