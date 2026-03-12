"""
测试 cuBLAS baseline + Triton 实现的 UT
"""
import os
os.environ["DISPATCH_TORCH_LIB"] = "0"
os.environ["FLAGBENCH_UPCAST"] = "0"

import sys

# 添加源代码路径
sys.path.insert(0, '/share/project/zpy/flagbench/src')

# 先导入 baseline 实现 (这会触发注册)
baseline_dir = '/share/project/zpy/flagbench/output_baseline_cublas/baseline_cublas_deepseek-v3-0324_temp_0.0_20260119-165244/baseline_0'
sys.path.insert(0, baseline_dir)

print("=" * 80)
print("步骤 1: 导入 baseline 实现")
print("=" * 80)

# 导入所有 baseline 实现
import saxpy
import daxpy  
import caxpy
import zaxpy

print("✓ 成功导入所有 axpy baseline 实现")

# 检查注册情况
import flagbench
from sandbox.register import REGISTERED_OPS

print(f"\n当前注册的 namespace: {list(REGISTERED_OPS.keys())}")
if 'baseline' in REGISTERED_OPS:
    print(f"baseline namespace 中注册的操作: {list(REGISTERED_OPS['baseline'].keys())}")

print("\n" + "=" * 80)
print("步骤 2: 导入 triton 实现")
print("=" * 80)

# 移除 baseline 目录，添加 triton 目录
sys.path.remove(baseline_dir)
triton_dir = '/share/project/zpy/flagbench/output_triton_cublas/triton_cublas_deepseek-v3-0324_num_samples_1_temp_0.0_max_tokens_16384_20260119-165610/triton_0'
sys.path.insert(0, triton_dir)

# 重新导入 (这次是 triton 版本)
import importlib
import sys

# 需要用不同的名字导入，避免冲突
triton_saxpy = importlib.import_module('saxpy')
triton_daxpy = importlib.import_module('daxpy')
triton_caxpy = importlib.import_module('caxpy')
triton_zaxpy = importlib.import_module('zaxpy')

print("✓ 成功导入所有 axpy triton 实现")

print(f"\n当前注册的 namespace: {list(REGISTERED_OPS.keys())}")
if 'triton' in REGISTERED_OPS:
    print(f"triton namespace 中注册的操作: {list(REGISTERED_OPS['triton'].keys())}")

print("\n" + "=" * 80)
print("步骤 3: 检查 flagbench 命名空间")
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
print("步骤 4: 简单功能测试")
print("=" * 80)

import torch

n = 16
alpha = 2.0
incx = 1
incy = 1

x = torch.randn(n, dtype=torch.float32, device='cuda')
y_baseline = torch.randn(n, dtype=torch.float32, device='cuda')
y_triton = y_baseline.clone()

print(f"\n测试数据: n={n}, alpha={alpha}, dtype=float32")

if hasattr(flagbench.baseline, 'saxpy') and hasattr(flagbench.triton, 'saxpy'):
    result_baseline = flagbench.baseline.saxpy(n, alpha, x, incx, y_baseline, incy)
    print(f"✓ baseline.saxpy 调用成功")
    
    result_triton = flagbench.triton.saxpy(n, alpha, x, incx, y_triton, incy)
    print(f"✓ triton.saxpy 调用成功")
    
    diff = torch.abs(result_baseline - result_triton).max().item()
    print(f"\n最大差异: {diff}")
    if diff < 1e-5:
        print("✓ 结果匹配！")
    else:
        print(f"⚠ 结果不匹配，差异: {diff}")
else:
    print("✗ 无法找到 saxpy 函数")

print("\n" + "=" * 80)
print("步骤 5: 导入并运行 UT 测试")
print("=" * 80)

# 添加 UT 目录到路径
ut_dir = '/share/project/zpy/flagbench/output_ut_cublas/ut_cublas_deepseek-v3-0324_num_samples_1_temp_0.0_max_tokens_16384_20260119-165531/ut_0'
sys.path.insert(0, ut_dir)

# 导入测试文件
import test_saxpy_cublas_baseline

print("✓ 成功导入 test_saxpy_cublas_baseline")

# 获取测试函数
test_func = test_saxpy_cublas_baseline.test_saxpy_cublas_baseline

print(f"✓ 找到测试函数: {test_func.__name__}")

# 使用 verifier 运行测试
print("\n使用 Verifier 运行测试...")
from sandbox.verifier import Verifier, VerifyConfig, VerifyRequest, Source

config = VerifyConfig(
    run_name="test_cublas_saxpy",
    test_type="accuracy",
    run_dir="/share/project/zpy/flagbench/runs",
    store_type="local",
    strict_check=False,
    seed=42,
    sample_id=0,
    save_log=True,
    acc_timeout=300,
)

verifier = Verifier(config)

# 设置要使用的测试模块
verifier.set_modules(modules=[ut_dir + '/test_saxpy_cublas_baseline.py'], mode="accuracy")

# 创建验证请求
requests = [
    VerifyRequest(
        source=[Source(
            source="mock code",  # 实际上不需要，因为我们已经有了实现
            function_name="saxpy"
        )]
    )
]

try:
    result = verifier.only_verify(
        name_source_map=requests,
        test_type="accuracy",
        device_count=1
    )
    print(f"\n✓ 测试完成！")
    print(f"结果: {result}")
except Exception as e:
    print(f"\n✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("测试流程完成")
print("=" * 80)
