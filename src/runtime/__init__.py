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

from flag_gems.runtime import device, torch_device_fn

# 设备可见性环境变量映射
VISIBLE_DEVICES_ENV = {
    'cuda': 'CUDA_VISIBLE_DEVICES',
    'npu': 'ASCEND_RT_VISIBLE_DEVICES',
    'musa': 'MUSA_VISIBLE_DEVICES',
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
}


def get_visible_devices_env() -> str:
    """获取当前设备的可见性环境变量名"""
    return VISIBLE_DEVICES_ENV.get(device.name, 'CUDA_VISIBLE_DEVICES')


def get_device_constraints() -> str:
    """获取当前设备的 Prompt 约束（如果启用）"""
    if not ENABLE_DEVICE_CONSTRAINTS:
        return ""
    return DEVICE_CONSTRAINTS.get(device.name, "")


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
    'get_triton_testing',
    'VISIBLE_DEVICES_ENV',
    'DEVICE_CONSTRAINTS',
    'ENABLE_DEVICE_CONSTRAINTS',
]
