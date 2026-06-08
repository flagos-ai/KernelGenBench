"""Hygon DCU backend — hipBLAS mapping."""
import os

from . import VendorBackend, register_backend
from ..vendor import Vendor

# hipBLAS naming: cublas→hipblas, drop _v2 (except complex types), drop _64
_HIPBLAS_KEEP_V2 = {
    "cublasCcopy_v2", "cublasCdotu_v2", "cublasCgemm_v2", "cublasCgemv_v2",
    "cublasCgeru_v2", "cublasCsymm_v2", "cublasCsymv_v2",
    "cublasZdotc_v2", "cublasZgerc_v2", "cublasZswap_v2",
}

# Enum mappings: cuBLAS → hipBLAS
_OP_MAP = {0: 111, 1: 112, 2: 113}
_FILL_MAP = {0: 122, 1: 121}
_SIDE_MAP = {0: 141, 1: 142}
_DIAG_MAP = {0: 131, 1: 132}

# cudaDataType → hipblasDatatype_t
_DATA_TYPE_MAP = {
    0: 151,   # CUDA_R_32F → HIPBLAS_R_32F
    1: 152,   # CUDA_R_64F → HIPBLAS_R_64F
    2: 150,   # CUDA_R_16F → HIPBLAS_R_16F
    4: 154,   # CUDA_C_32F → HIPBLAS_C_32F
    5: 155,   # CUDA_C_64F → HIPBLAS_C_64F
    6: 160,   # CUDA_R_8I  → HIPBLAS_R_8I
    7: 161,   # CUDA_R_8U  → HIPBLAS_R_8U
    8: 154,   # CUDA_C_32F (alt) → HIPBLAS_C_32F
}


class HygonBackend(VendorBackend):
    vendor = Vendor.HYGON
    device_name = "cuda"
    visible_devices_env = "HIP_VISIBLE_DEVICES"

    @property
    def blas_lib_path(self):
        dtk_home = os.environ.get("DTK_HOME", "/opt/dtk-25.04")
        return os.path.join(dtk_home, "lib", "libhipblas.so")

    blas_create_handle_fn = "hipblasCreate"
    blas_set_pointer_mode_fn = "hipblasSetPointerMode"

    device_constraints = """\
## Device-Specific Requirements
It should be noted that the operator runs on Hygon DCU (Deep Computing Unit).
1. The device type is `cuda` (standard PyTorch CUDA API via ROCm/HIP). No special import is needed beyond `import torch`.
2. Hygon DCU is based on ROCm/HIP ecosystem. Avoid NVIDIA-specific hardware features (e.g., Tensor Core instructions).
3. Some advanced Triton features may not be supported on HIP backend. Prefer basic Triton operations.
4. Use `allow_tf32=False` for `tl.dot` to ensure precision.
5. Prefer moderate BLOCK_SIZE values (e.g., 256 or 512).
"""

    def map_blas_func_name(self, cublas_name: str) -> str:
        hip_name = cublas_name.replace("cublas", "hipblas", 1)
        if hip_name.endswith("_64"):
            hip_name = hip_name[:-3]
        if cublas_name not in _HIPBLAS_KEEP_V2 and hip_name.endswith("_v2"):
            hip_name = hip_name[:-3]
        return hip_name

    def map_op(self, op) -> int:
        if isinstance(op, str):
            op = {"N": 0, "T": 1, "C": 2}[op]
        return _OP_MAP.get(op, op)

    def map_fill_mode(self, fill: int) -> int:
        return _FILL_MAP.get(fill, fill)

    def map_side(self, side: int) -> int:
        return _SIDE_MAP.get(side, side)

    def map_diag(self, diag: int) -> int:
        return _DIAG_MAP.get(diag, diag)

    def map_data_type(self, dtype: int) -> int:
        return _DATA_TYPE_MAP.get(dtype, dtype)


register_backend(HygonBackend())
