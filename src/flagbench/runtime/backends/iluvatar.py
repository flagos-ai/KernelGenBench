"""Iluvatar backend — CoreX cuBLAS (API-compatible with NVIDIA cuBLAS)."""
import os

from . import VendorBackend, register_backend
from ..vendor import Vendor


class IluvatarBackend(VendorBackend):
    vendor = Vendor.ILUVATAR
    device_name = "cuda"
    visible_devices_env = "CUDA_VISIBLE_DEVICES"

    @property
    def blas_lib_path(self):
        corex_home = os.environ.get(
            "COREX_HOME",
            os.environ.get("CUDA_HOME", "/usr/local/corex-4.4.0"),
        )
        return os.path.join(corex_home, "lib64", "libcublas.so")

    blas_create_handle_fn = "cublasCreate_v2"
    blas_set_pointer_mode_fn = "cublasSetPointerMode_v2"

    device_constraints = """\
## Device-Specific Requirements
It should be noted that the operator runs on Iluvatar BI-V150 GPUs with CoreX software stack.
1. The device type is `cuda`. No special import is needed beyond `import torch`.
2. Avoid NVIDIA-specific hardware features (e.g., Tensor Core specific instructions).
3. Use `allow_tf32=False` for `tl.dot` to ensure precision.
4. Prefer smaller BLOCK_SIZE values (e.g., 512 or 1024).
"""

    def map_blas_func_name(self, cublas_name: str) -> str:
        """CoreX cuBLAS is API-compatible; only strip _64 suffix (not available)."""
        if cublas_name.endswith("_64"):
            return cublas_name[:-3]
        return cublas_name


register_backend(IluvatarBackend())
