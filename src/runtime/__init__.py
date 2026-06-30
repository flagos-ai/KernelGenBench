import triton.testing as _triton_testing


def get_triton_testing():
    """Return triton.testing module for benchmark utilities."""
    return _triton_testing


"""
Runtime device detection and constraints.

Standalone implementation (no flag_gems dependency).
Provides unified device detection, visibility env var mapping,
and device-specific prompt constraints for kernel generation.
"""
import os
import glob
import shutil
import subprocess


# ─── Device Detection ───────────────────────────────────────────────────────

def _detect_device_name() -> str:
    """Detect device name: 'cuda', 'npu', 'musa'."""
    vendor = os.environ.get("GEMS_VENDOR", "")
    if vendor == "ascend" or os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        return "npu"
    if vendor == "mthreads" or os.environ.get("MUSA_VISIBLE_DEVICES"):
        return "musa"
    # iluvatar, hygon, muxi all present as 'cuda'
    return "cuda"


class _Device:
    """Minimal device descriptor (replaces flag_gems.runtime.device)."""

    @property
    def name(self) -> str:
        return _detect_device_name()


device = _Device()


# ─── Visibility Env Var ─────────────────────────────────────────────────────

VISIBLE_DEVICES_ENV = {
    "cuda": "CUDA_VISIBLE_DEVICES",
    "npu": "ASCEND_RT_VISIBLE_DEVICES",
    "musa": "MUSA_VISIBLE_DEVICES",
    "hygon": "HIP_VISIBLE_DEVICES",
    "muxi": "MACA_VISIBLE_DEVICES",
}

ENABLE_DEVICE_CONSTRAINTS = os.environ.get("KERNELGENBENCH_ENABLE_DEVICE_CONSTRAINTS", "1") == "1"


# ─── Device Constraints (injected into LLM prompts) ─────────────────────────

DEVICE_CONSTRAINTS = {
    "npu": """
## Device-Specific Requirements
The operator runs on Ascend NPU devices.
1. `import torch` must be immediately followed by `import torch_npu`, otherwise the npu device is unavailable.
2. Device type is `npu`. All device APIs use `npu`: `torch.device("npu:0")`, `torch.npu.synchronize()`, `tensor.to('npu')`.
3. Use `ASCEND_RT_VISIBLE_DEVICES` instead of `CUDA_VISIBLE_DEVICES`.
4. Triton notes: `tl.dot` does NOT support `allow_tf32` — remove it. Prefer basic Triton operations. Compile error "ub overflow" means split into smaller tiles.
""",
    "musa": """
## Device-Specific Requirements
The operator runs on MUSA devices (Moore Threads).
1. `import torch` must be immediately followed by `import torch_musa`, otherwise the musa device is unavailable.
2. Device type is `musa`. All device APIs use `musa`: `torch.device("musa:0")`, `torch.musa.synchronize()`, `tensor.to('musa')`.
3. Use `MUSA_VISIBLE_DEVICES` instead of `CUDA_VISIBLE_DEVICES`.
4. Triton notes: prefer basic Triton operations. Avoid CUDA-specific hardware features.
""",
    "iluvatar": """
## Device-Specific Requirements
The operator runs on Iluvatar GPUs.
1. Device type is `cuda` (standard PyTorch CUDA API). No special import needed beyond `import torch`.
2. Iluvatar GPUs provide a CUDA-compatible interface but hardware differs from NVIDIA. Avoid NVIDIA-specific features (e.g. Tensor Core instructions).
3. Use `CUDA_VISIBLE_DEVICES` for device visibility (same as NVIDIA).
4. Triton notes: use `allow_tf32=False` for `tl.dot`. Prefer basic Triton operations.
""",
    "hygon": """
## Device-Specific Requirements
The operator runs on Hygon DCU (Deep Computing Unit).
1. Device type is `cuda` (PyTorch CUDA API via ROCm/HIP). No special import needed beyond `import torch`.
2. Hygon DCU uses the ROCm/HIP ecosystem with CUDA-compatible interface. Avoid NVIDIA-specific features (Tensor Cores, CUDA intrinsics).
3. Use `HIP_VISIBLE_DEVICES` instead of `CUDA_VISIBLE_DEVICES`.
4. Triton notes: use `allow_tf32=False` for `tl.dot` (TF32 is NVIDIA-specific). Some advanced Triton features may behave differently on HIP backend.
""",
    "muxi": """
## Device-Specific Requirements
The operator runs on MetaX (MUXI) GPUs.
1. Device type is `cuda` (standard PyTorch CUDA API). No special import needed beyond `import torch`.
2. MetaX GPUs use MACA SDK with CUDA-compatible interface. Avoid NVIDIA-specific hardware features.
3. Use `MACA_VISIBLE_DEVICES` instead of `CUDA_VISIBLE_DEVICES`.
4. Triton notes: use `allow_tf32=False` for `tl.dot`. Limited bfloat16 support — prefer float32 accumulation when encountering precision issues.
""",
}


# ─── Internal Detection Helpers ──────────────────────────────────────────────

def _is_iluvatar() -> bool:
    if os.environ.get("GEMS_VENDOR") == "iluvatar":
        return True
    try:
        result = subprocess.run(
            ["ixsmi", "-L"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and "Iluvatar" in result.stdout:
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    if glob.glob("/usr/local/corex*") or shutil.which("ixsmi"):
        return True
    return False


def _is_hygon() -> bool:
    if os.environ.get("GEMS_VENDOR") == "hygon":
        return True
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and (
            "Hygon" in result.stdout or "DCU" in result.stdout or "BW" in result.stdout
        ):
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return False


def _is_muxi() -> bool:
    if os.environ.get("GEMS_VENDOR") == "muxi":
        return True
    try:
        result = subprocess.run(
            ["mx-smi", "-L"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and ("MetaX" in result.stdout or "MXC" in result.stdout):
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return False


# ─── Public API ──────────────────────────────────────────────────────────────

def get_visible_devices_env() -> str:
    """Get the environment variable name for device visibility."""
    return VISIBLE_DEVICES_ENV.get(device.name, "CUDA_VISIBLE_DEVICES")


def get_device_constraints() -> str:
    """Get device-specific prompt constraints (if enabled)."""
    if not ENABLE_DEVICE_CONSTRAINTS:
        return ""
    if device.name == "cuda":
        if _is_iluvatar():
            return DEVICE_CONSTRAINTS.get("iluvatar", "")
        if _is_hygon():
            return DEVICE_CONSTRAINTS.get("hygon", "")
        if _is_muxi():
            return DEVICE_CONSTRAINTS.get("muxi", "")
        return ""
    return DEVICE_CONSTRAINTS.get(device.name, "")


def get_device_type() -> str:
    """Get device type string for configuration.

    Returns:
        "nvidia", "iluvatar", "hygon", "muxi", "ascend", or "mthreads"
    """
    if device.name == "npu":
        return "ascend"
    if device.name == "musa":
        return "mthreads"
    if device.name == "cuda":
        if _is_iluvatar():
            return "iluvatar"
        if _is_hygon():
            return "hygon"
        if _is_muxi():
            return "muxi"
        return "nvidia"
    return "nvidia"


__all__ = [
    "device",
    "get_visible_devices_env",
    "get_device_constraints",
    "get_device_type",
    "VISIBLE_DEVICES_ENV",
    "DEVICE_CONSTRAINTS",
    "ENABLE_DEVICE_CONSTRAINTS",
]
