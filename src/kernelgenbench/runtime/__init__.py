"""
FlagBench Runtime — unified multi-chip backend management.

Usage:
    from kernelgenbench.runtime import detect_vendor, get_backend

    vendor = detect_vendor()       # -> Vendor.HYGON
    backend = get_backend()        # -> HygonBackend instance
    backend.map_blas_func_name("cublasSgemm_v2")  # -> "hipblasSgemm"
    backend.map_op(0)              # -> 111
"""
from .vendor import Vendor, detect_vendor
from .backends import VendorBackend, get_backend, register_backend

__all__ = [
    "Vendor",
    "detect_vendor",
    "VendorBackend",
    "get_backend",
    "register_backend",
]
