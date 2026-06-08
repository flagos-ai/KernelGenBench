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
