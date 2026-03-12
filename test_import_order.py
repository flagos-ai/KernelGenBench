"""
尝试解决循环导入问题的测试
"""
import os
os.environ["DISPATCH_TORCH_LIB"] = "0"
os.environ["FLAGBENCH_UPCAST"] = "0"

import sys

# 添加源代码路径
sys.path.insert(0, '/share/project/zpy/flagbench/src')

print("=" * 80)
print("步骤 1: 先导入 flagbench 核心模块")
print("=" * 80)

# 先导入 flagbench，让循环依赖完成初始化
import flagbench
from sandbox.register import REGISTERED_OPS, register

print("✓ 成功导入 flagbench 和 register")
print(f"当前注册的操作数: {len(REGISTERED_OPS)}")

print("\n" + "=" * 80)
print("步骤 2: 导入 baseline 实现")
print("=" * 80)

# 添加 baseline 目录
baseline_dir = '/share/project/zpy/flagbench/output_baseline_cublas/baseline_cublas_deepseek-v3-0324_temp_0.0_20260119-165244/baseline_0'
sys.path.insert(0, baseline_dir)

# 导入所有 baseline 实现
import saxpy as saxpy_baseline
import daxpy as daxpy_baseline  
import caxpy as caxpy_baseline
import zaxpy as zaxpy_baseline

print("✓ 成功导入所有 axpy baseline 实现")

# 检查注册情况
print(f"\n当前注册的 namespace: {list(REGISTERED_OPS.keys())}")
if 'baseline' in REGISTERED_OPS:
    print(f"baseline namespace 中注册的操作: {list(REGISTERED_OPS['baseline'].keys())}")

print("\n" + "=" * 80)
print("步骤 3: 导入 triton 实现")
print("=" * 80)

# 移除 baseline 目录，添加 triton 目录
sys.path.remove(baseline_dir)
triton_dir = '/share/project/zpy/flagbench/output_triton_cublas/triton_cublas_deepseek-v3-0324_num_samples_1_temp_0.0_max_tokens_16384_20260119-165610/triton_0'
sys.path.insert(0, triton_dir)

# 需要删除已经导入的模块，重新导入 triton 版本
for mod in ['saxpy', 'daxpy', 'caxpy', 'zaxpy']:
    if mod in sys.modules:
        del sys.modules[mod]

# 重新导入 (这次是 triton 版本)
import saxpy as saxpy_triton
import daxpy as daxpy_triton
import caxpy as caxpy_triton
import zaxpy as zaxpy_triton

print("✓ 成功导入所有 axpy triton 实现")

print(f"\n当前注册的 namespace: {list(REGISTERED_OPS.keys())}")
if 'triton' in REGISTERED_OPS:
    print(f"triton namespace 中注册的操作: {list(REGISTERED_OPS['triton'].keys())}")

print("\n" + "=" * 80)
print("步骤 4: 检查 flagbench 命名空间")
print("=" * 80)

print(f"flagbench.baseline 存在: {hasattr(flagbench, 'baseline')}")
if hasattr(flagbench, 'baseline'):
    print(f"  - flagbench.baseline.saxpy 存在: {hasattr(flagbench.baseline, 'saxpy')}")
    print(f"  - flagbench.baseline.daxpy 存在: {hasattr(flagbench.baseline, 'daxpy')}")
    print(f"  - flagbench.baseline.caxpy 存在: {hasattr(flagbench.baseline, 'caxpy')}")
    print(f"  - flagbench.baseline.zaxpy 存在: {hasattr(flagbench.baseline, 'zaxpy')}")

print(f"\nflagbench.triton 存在: {hasattr(flagbench, 'triton')}")
if hasattr(flagbench, 'triton'):
    print(f"  - flagbench.triton.saxpy 存在: {hasattr(flagbench.triton, 'saxpy')}")
    print(f"  - flagbench.triton.daxpy 存在: {hasattr(flagbench.triton, 'daxpy')}")
    print(f"  - flagbench.triton.caxpy 存在: {hasattr(flagbench.triton, 'caxpy')}")
    print(f"  - flagbench.triton.zaxpy 存在: {hasattr(flagbench.triton, 'zaxpy')}")

print("\n" + "=" * 80)
print("步骤 5: 简单功能测试")
print("=" * 80)

import torch

n = 16
alpha = 2.0
incx = 1
incy = 1

x = torch.randn(n, dtype=torch.float32, device='cuda')
y_baseline_tensor = torch.randn(n, dtype=torch.float32, device='cuda')
y_triton_tensor = y_baseline_tensor.clone()

print(f"\n测试数据: n={n}, alpha={alpha}, dtype=float32")
print(f"x shape: {x.shape}, y shape: {y_baseline_tensor.shape}")

if hasattr(flagbench, 'baseline') and hasattr(flagbench.baseline, 'saxpy'):
    print("\n调用 flagbench.baseline.saxpy...")
    result_baseline = flagbench.baseline.saxpy(n, alpha, x, incx, y_baseline_tensor, incy)
    print(f"✓ baseline.saxpy 调用成功, result shape: {result_baseline.shape}")
else:
    print("✗ 无法找到 flagbench.baseline.saxpy")
    result_baseline = None

if hasattr(flagbench, 'triton') and hasattr(flagbench.triton, 'saxpy'):
    print("\n调用 flagbench.triton.saxpy...")
    result_triton = flagbench.triton.saxpy(n, alpha, x, incx, y_triton_tensor, incy)
    print(f"✓ triton.saxpy 调用成功, result shape: {result_triton.shape}")
else:
    print("✗ 无法找到 flagbench.triton.saxpy")
    result_triton = None

if result_baseline is not None and result_triton is not None:
    diff = torch.abs(result_baseline - result_triton).max().item()
    print(f"\n最大差异: {diff:.2e}")
    if diff < 1e-5:
        print("✓ 结果匹配！")
    else:
        print(f"⚠ 结果不匹配，差异: {diff:.2e}")
        print(f"Baseline 结果: {result_baseline[:5]}")
        print(f"Triton 结果:   {result_triton[:5]}")

print("\n" + "=" * 80)
print("测试完成！")
print("=" * 80)
