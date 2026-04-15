"""
FlagGems runtime 扩展

补充 FlagGems runtime 中缺失的设备信息，提供统一的 runtime 接口。
"""
import os

# 防御性 import：确保设备扩展包在 FlagGems runtime 初始化前被加载
# 这样 torch.npu / torch.musa API 才能正常使用
_vendor = os.environ.get('GEMS_VENDOR', '')
if _vendor == 'ascend' or os.environ.get('ASCEND_RT_VISIBLE_DEVICES'):
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        pass

if _vendor == 'mthreads' or os.environ.get('MUSA_VISIBLE_DEVICES'):
    try:
        import torch_musa  # noqa: F401
    except ImportError:
        pass

if _vendor == 'hygon' or os.environ.get('HIP_VISIBLE_DEVICES'):
    try:
        # import torch_hygon  # noqa: F401  # 如果海光有专门的 torch 扩展包
        pass
    except ImportError:
        pass

from flag_gems.runtime import device, torch_device_fn

# 设备可见性环境变量映射
VISIBLE_DEVICES_ENV = {
    'cuda': 'CUDA_VISIBLE_DEVICES',
    'npu': 'ASCEND_RT_VISIBLE_DEVICES',
    'musa': 'MUSA_VISIBLE_DEVICES',
    'hygon': 'HIP_VISIBLE_DEVICES',
    'muxi': 'MACA_VISIBLE_DEVICES',
}

# 设备约束开关（通过环境变量控制）
ENABLE_DEVICE_CONSTRAINTS = os.environ.get('FLAGBENCH_ENABLE_DEVICE_CONSTRAINTS', '1') == '1'

# 设备特定的 Prompt 约束
DEVICE_CONSTRAINTS = {
    'npu': """
## Device-Specific Requirements
It should be noted that the operator runs on Ascend NPU devices.
1. In the generated operator implementation, if `import torch` is used, it must be immediately followed by `import torch_npu`.
2. The device type is `npu`, and all device-related APIs should use `npu`, for example `device = torch.device("npu:0")`, `torch.npu.synchronize()`, etc. Always ensure consistent use of the `npu` device.
3. All GPU-related commands must use `ASCEND_RT_VISIBLE_DEVICES` instead of `CUDA_VISIBLE_DEVICES`.
""",
    'musa': """
## Device-Specific Requirements
It should be noted that the operator runs on MUSA devices.
1. In the generated operator implementation, if `import torch` is used, it must be immediately followed by `import torch_musa`.
2. The device type is `musa`, and all device-related APIs should use `musa`, for example `device = torch.device("musa:0")`, `torch.musa.synchronize()`, etc. Always ensure consistent use of the `musa` device.
""",
    'iluvatar': """
## Device-Specific Requirements
It should be noted that the operator runs on Iluvatar GPUs.
1. The device type is `cuda` (standard PyTorch CUDA API). No special import is needed beyond `import torch`.
2. Iluvatar GPUs provide a CUDA-compatible interface, but the underlying hardware architecture differs from NVIDIA. Avoid relying on NVIDIA-specific hardware features (e.g., Tensor Core specific instructions).
3. Some advanced Triton features may not be supported or may behave differently. Prefer basic Triton operations.
4. Use `allow_tf32=False` for `tl.dot` to ensure precision.
""",
    'hygon': """
## Device-Specific Requirements
It should be noted that the operator runs on Hygon DCU (Deep Computing Unit).
1. The device type is `cuda` (standard PyTorch CUDA API via ROCm/HIP). No special import is needed beyond `import torch`.
2. Hygon DCU is based on ROCm/HIP ecosystem, providing CUDA-compatible interface. Avoid relying on NVIDIA-specific hardware features (e.g., Tensor Core instructions, CUDA-specific intrinsics).
3. Some advanced Triton features may not be supported or may behave differently on HIP backend. Prefer basic Triton operations.
4. Use `allow_tf32=False` for `tl.dot` to ensure precision (TF32 is an NVIDIA-specific feature).
""",
    'muxi': """
## Device-Specific Requirements
It should be noted that the operator runs on MetaX (MUXI) GPUs.
1. The device type is `cuda` (standard PyTorch CUDA API). No special import is needed beyond `import torch`.
2. MetaX GPUs are based on MACA SDK, providing CUDA-compatible interface, but the underlying hardware architecture differs from NVIDIA. Avoid relying on NVIDIA-specific hardware features (e.g., Tensor Core specific instructions).
3. Some advanced Triton features may not be supported or may behave differently. Prefer basic Triton operations.
4. Use `allow_tf32=False` for `tl.dot` to ensure precision.
5. Some operators have limited bfloat16 support. When encountering precision issues, prefer float32 accumulation.
""",
}


def _is_iluvatar() -> bool:
    """Detect if running on Iluvatar GPU."""
    if os.environ.get('GEMS_VENDOR') == 'iluvatar':
        return True
    try:
        import torch
        if torch.cuda.is_available():
            return 'Iluvatar' in torch.cuda.get_device_name(0)
    except Exception:
        pass
    return False


def _is_hygon() -> bool:
    """Detect if running on Hygon DCU."""
    if os.environ.get('GEMS_VENDOR') == 'hygon':
        return True
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            # Hygon DCU reports as 'BW' (series name)
            return 'Hygon' in name or 'DCU' in name or name.strip() == 'BW'
    except Exception:
        pass
    return False


def _is_muxi() -> bool:
    """Detect if running on MetaX (MUXI) GPU."""
    if os.environ.get('GEMS_VENDOR') == 'muxi':
        return True
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return 'MetaX' in name
    except Exception:
        pass
    return False


def get_visible_devices_env() -> str:
    """获取当前设备的可见性环境变量名"""
    return VISIBLE_DEVICES_ENV.get(device.name, 'CUDA_VISIBLE_DEVICES')


def get_device_constraints() -> str:
    """获取当前设备的 Prompt 约束（如果启用）"""
    if not ENABLE_DEVICE_CONSTRAINTS:
        return ""
    # Iluvatar/Hygon/MUXI report as 'cuda' but need their own constraints
    if device.name == 'cuda' and _is_iluvatar():
        return DEVICE_CONSTRAINTS.get('iluvatar', "")
    if device.name == 'cuda' and _is_hygon():
        return DEVICE_CONSTRAINTS.get('hygon', "")
    if device.name == 'cuda' and _is_muxi():
        return DEVICE_CONSTRAINTS.get('muxi', "")
    return DEVICE_CONSTRAINTS.get(device.name, "")


def get_device_type() -> str:
    """获取当前设备类型，用于 anti-hack 等场景。

    Returns:
        "nvidia", "iluvatar", "hygon", "muxi", "ascend", or "mthreads"
    """
    if device.name == 'npu':
        return 'ascend'
    if device.name == 'musa':
        return 'mthreads'
    if device.name == 'cuda':
        if _is_iluvatar():
            return 'iluvatar'
        if _is_hygon():
            return 'hygon'
        if _is_muxi():
            return 'muxi'
        return 'nvidia'
    return 'nvidia'


def get_triton_testing():
    """
    获取当前设备的 triton testing 模块

    不同设备使用不同的 benchmark API：
    - CUDA: triton.testing
    - MUSA: triton.musa_testing

    Returns:
        triton testing 模块，包含 do_bench 等函数
    """
    import triton
    if device.name == 'musa':
        return triton.musa_testing
    return triton.testing


__all__ = [
    'device',
    'torch_device_fn',
    'get_visible_devices_env',
    'get_device_constraints',
    'get_device_type',
    'get_triton_testing',
    'VISIBLE_DEVICES_ENV',
    'DEVICE_CONSTRAINTS',
    'ENABLE_DEVICE_CONSTRAINTS',
]
