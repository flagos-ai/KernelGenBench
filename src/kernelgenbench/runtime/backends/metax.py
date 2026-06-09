"""MetaX (沐曦) backend — mcblas mapping (API-compatible with NVIDIA cuBLAS)."""
import os

from . import VendorBackend, register_backend
from ..vendor import Vendor

# mcbLAS naming: cublas→mcblas, drop _v2 and _64 suffixes

# Enum values are IDENTICAL to cuBLAS — no mapping needed
# MCBLAS_OP_N=0, MCBLAS_OP_T=1, MCBLAS_OP_C=2
# MCBLAS_FILL_MODE_LOWER=0, MCBLAS_FILL_MODE_UPPER=1
# MCBLAS_SIDE_LEFT=0, MCBLAS_SIDE_RIGHT=1
# MCBLAS_DIAG_NON_UNIT=0, MCBLAS_DIAG_UNIT=1
# mcblasDataType_t values are API-compatible with cudaDataType


class MetaXBackend(VendorBackend):
    vendor = Vendor.METAX
    device_name = "cuda"
    visible_devices_env = "CUDA_VISIBLE_DEVICES"

    @property
    def blas_lib_path(self):
        maca_home = os.environ.get("MACA_PATH", "/opt/maca")
        return os.path.join(maca_home, "lib", "libmcblas.so")

    blas_create_handle_fn = "mcblasCreate"
    blas_set_pointer_mode_fn = "mcblasSetPointerMode"

    device_constraints = """\
## Device-Specific Requirements
It should be noted that the operator runs on MetaX C500/C550 GPUs with MACA software stack.
1. The device type is `cuda` (MetaX implements the CUDA/HIP API via MACA). No special import is needed beyond `import torch`.
2. MACA uses `mcblas` (API-compatible with cuBLAS) for BLAS operations.
3. The MACA Triton backend may have limitations:
   - Avoid large memory offsets that exceed 32-bit pointer range.
   - Some MLIR passes may fail on complex Triton patterns; prefer simpler kernel designs.
   - Use `allow_tf32=False` for `tl.dot` to ensure precision.
4. Prefer moderate BLOCK_SIZE values (e.g., 256 or 512).
5. The LLVM toolchain version is LLVM 12; avoid emitting bitcode that requires newer LLVM features.
"""

    def map_blas_func_name(self, cublas_name: str) -> str:
        """Map cuBLAS function name to mcblas function name."""
        mcblas_name = cublas_name.replace("cublas", "mcblas", 1)
        if mcblas_name.endswith("_64"):
            mcblas_name = mcblas_name[:-3]
        if mcblas_name.endswith("_v2"):
            mcblas_name = mcblas_name[:-3]
        return mcblas_name

    # All enum mappings are identity — mcblas uses identical values to cuBLAS.


register_backend(MetaXBackend())
