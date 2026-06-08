"""Moore Threads MUSA backend — MUBLAS mapping."""
import os

from . import VendorBackend, register_backend
from ..vendor import Vendor


class MusaBackend(VendorBackend):
    vendor = Vendor.MTHREADS
    device_name = "musa"
    visible_devices_env = "MUSA_VISIBLE_DEVICES"

    @property
    def blas_lib_path(self):
        musa_home = os.environ.get("MUSA_HOME", "/usr/local/musa")
        return os.path.join(musa_home, "lib", "libmublas.so")

    blas_create_handle_fn = "mublasCreate"
    blas_set_pointer_mode_fn = "mublasSetPointerMode"

    device_constraints = """\
## Device-Specific Requirements
It should be noted that the operator runs on Moore Threads MUSA GPUs.
1. The device type is `musa`. Use `import torch_musa` before any GPU operations.
2. Moore Threads GPUs use the MUSA SDK. Avoid NVIDIA-specific hardware features (e.g., Tensor Core instructions).
3. Some advanced Triton features may not be supported on MUSA backend. Prefer basic Triton operations.
4. Use `allow_tf32=False` for `tl.dot` to ensure precision.
5. Prefer moderate BLOCK_SIZE values (e.g., 256 or 512).
"""

    def map_blas_func_name(self, cublas_name: str) -> str:
        # Special case: cublasSgemmEx → mublasGemmEx (drops type prefix S)
        if cublas_name == "cublasSgemmEx":
            return "mublasGemmEx"
        mu_name = cublas_name.replace("cublas", "mublas", 1)
        if mu_name.endswith("_64"):
            mu_name = mu_name[:-3]
        if mu_name.endswith("_v2"):
            mu_name = mu_name[:-3]
        return mu_name

    def map_op(self, op) -> int:
        # MUBLAS uses same enum values as cuBLAS
        if isinstance(op, str):
            return {"N": 0, "T": 1, "C": 2}[op]
        return op

    def map_fill_mode(self, fill: int) -> int:
        return fill

    def map_side(self, side: int) -> int:
        return side

    def map_diag(self, diag: int) -> int:
        return diag

    def map_data_type(self, dtype: int) -> int:
        return dtype


register_backend(MusaBackend())
