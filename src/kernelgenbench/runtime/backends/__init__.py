"""Vendor backend base class and registry."""
from __future__ import annotations

from typing import Callable, Optional

from ..vendor import Vendor, detect_vendor


class VendorBackend:
    """Base class for vendor-specific backend configuration."""

    vendor: Vendor = Vendor.NVIDIA

    # -- BLAS library --
    blas_lib_path: str = ""
    blas_create_handle_fn: str = ""
    blas_set_pointer_mode_fn: str = ""

    # -- Device --
    device_name: str = "cuda"           # torch device name
    visible_devices_env: str = "CUDA_VISIBLE_DEVICES"
    device_constraints: str = ""        # LLM prompt constraints

    # -- BLAS function name mapping --
    def map_blas_func_name(self, cublas_name: str) -> str:
        """Map cuBLAS function name to this backend's function name."""
        return cublas_name

    # -- Enum mappings --
    def map_op(self, op) -> int:
        """Map cuBLAS operation enum (0/1/2 or 'N'/'T'/'C') to backend enum."""
        if isinstance(op, str):
            op = {"N": 0, "T": 1, "C": 2}[op]
        return op

    def map_fill_mode(self, fill: int) -> int:
        return fill

    def map_side(self, side: int) -> int:
        return side

    def map_diag(self, diag: int) -> int:
        return diag

    def map_data_type(self, dtype: int) -> int:
        """Map cudaDataType enum to backend enum. Identity on NVIDIA."""
        return dtype


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_registry: dict[Vendor, VendorBackend] = {}


def register_backend(backend: VendorBackend):
    """Register a vendor backend instance."""
    _registry[backend.vendor] = backend


def get_backend(vendor: Optional[Vendor] = None) -> VendorBackend:
    """Get backend for given vendor (auto-detected if None)."""
    if vendor is None:
        vendor = detect_vendor()
    if vendor not in _registry:
        # Fallback to NVIDIA (identity mapping)
        if Vendor.NVIDIA in _registry:
            return _registry[Vendor.NVIDIA]
        raise RuntimeError(f"No backend registered for vendor: {vendor}")
    return _registry[vendor]


# Auto-import all backend modules to trigger registration
from . import nvidia, hygon, ascend, iluvatar, metax, musa  # noqa: E402, F401
