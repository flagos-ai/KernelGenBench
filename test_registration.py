"""
测试注册系统是否正常工作
验证 baseline 和 triton 实现是否能正确注册和调用
"""

import sys
import os

# 添加必要的路径
sys.path.insert(0, '/share/project/zpy/flagbench/src')

# 导入 baseline 实现
baseline_dir = '/share/project/zpy/flagbench/output_baseline_cublas/baseline_cublas_deepseek-v3-0324_temp_0.0_20260119-165244/baseline_0'
sys.path.insert(0, baseline_dir)

# 导入 triton 实现
triton_dir = '/share/project/zpy/flagbench/output_triton_cublas/triton_cublas_deepseek-v3-0324_num_samples_1_temp_0.0_max_tokens_16384_20260119-165610/triton_0'
sys.path.insert(0, triton_dir)

print("=" * 80)
print("测试 1: 导入 baseline 和 triton 实现")
print("=" * 80)

try:
    # 导入 saxpy baseline
    import saxpy as saxpy_baseline
    print("✓ Successfully imported saxpy baseline")
    
    # 导入 saxpy triton (重命名以避免冲突)
    sys.path.remove(baseline_dir)  # 临时移除 baseline 目录
    import saxpy as saxpy_triton
    sys.path.insert(0, baseline_dir)  # 重新添加
    print("✓ Successfully imported saxpy triton")
    
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("测试 2: 检查注册信息")
print("=" * 80)

try:
    from sandbox.register import REGISTERED_OPS
    
    print(f"\nTotal registered operations: {len(REGISTERED_OPS)}")
    
    # 查找 saxpy 相关的注册
    saxpy_ops = [op for op in REGISTERED_OPS if 'axpy' in op.lower()]
    print(f"\nFound {len(saxpy_ops)} axpy-related operations:")
    for op in saxpy_ops:
        print(f"  - {op}")
    
    # 检查是否有 baseline 和 triton namespace
    baseline_ops = [op for op in REGISTERED_OPS if 'baseline' in op]
    triton_ops = [op for op in REGISTERED_OPS if 'triton' in op]
    
    print(f"\nBaseline namespace operations: {len(baseline_ops)}")
    print(f"Triton namespace operations: {len(triton_ops)}")
    
except Exception as e:
    print(f"✗ Registration check failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("测试 3: 通过 flagbench 访问函数")
print("=" * 80)

try:
    import flagbench
    
    # 检查是否可以访问 baseline.saxpy
    if hasattr(flagbench, 'baseline'):
        print("✓ flagbench.baseline exists")
        if hasattr(flagbench.baseline, 'saxpy'):
            print("✓ flagbench.baseline.saxpy exists")
        else:
            print("✗ flagbench.baseline.saxpy NOT found")
            print(f"Available baseline operations: {dir(flagbench.baseline)}")
    else:
        print("✗ flagbench.baseline NOT found")
        print(f"Available flagbench attributes: {dir(flagbench)}")
    
    # 检查是否可以访问 triton.saxpy
    if hasattr(flagbench, 'triton'):
        print("✓ flagbench.triton exists")
        if hasattr(flagbench.triton, 'saxpy'):
            print("✓ flagbench.triton.saxpy exists")
        else:
            print("✗ flagbench.triton.saxpy NOT found")
            print(f"Available triton operations: {dir(flagbench.triton)}")
    else:
        print("✗ flagbench.triton NOT found")
        print(f"Available flagbench attributes: {dir(flagbench)}")
    
except Exception as e:
    print(f"✗ Access check failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("测试 4: 简单调用测试 (如果可以访问)")
print("=" * 80)

try:
    import torch
    
    # 创建测试数据
    n = 16
    alpha = 2.0
    incx = 1
    incy = 1
    
    x = torch.randn(n, dtype=torch.float32, device='cuda')
    y_baseline = torch.randn(n, dtype=torch.float32, device='cuda')
    y_triton = y_baseline.clone()
    
    print(f"\nTest data created: n={n}, alpha={alpha}")
    print(f"x shape: {x.shape}, y shape: {y_baseline.shape}")
    
    # 尝试调用 baseline
    if hasattr(flagbench, 'baseline') and hasattr(flagbench.baseline, 'saxpy'):
        result_baseline = flagbench.baseline.saxpy(n, alpha, x, incx, y_baseline, incy)
        print(f"✓ Baseline call successful, result shape: {result_baseline.shape}")
    else:
        print("⊘ Skipping baseline call (not accessible)")
    
    # 尝试调用 triton
    if hasattr(flagbench, 'triton') and hasattr(flagbench.triton, 'saxpy'):
        result_triton = flagbench.triton.saxpy(n, alpha, x, incx, y_triton, incy)
        print(f"✓ Triton call successful, result shape: {result_triton.shape}")
    else:
        print("⊘ Skipping triton call (not accessible)")
    
    # 比较结果
    if hasattr(flagbench, 'baseline') and hasattr(flagbench, 'triton') and \
       hasattr(flagbench.baseline, 'saxpy') and hasattr(flagbench.triton, 'saxpy'):
        diff = torch.abs(result_baseline - result_triton).max().item()
        print(f"\nMax difference between baseline and triton: {diff}")
        if diff < 1e-5:
            print("✓ Results match!")
        else:
            print(f"⚠ Results differ by {diff}")
    
except Exception as e:
    print(f"✗ Call test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("测试完成")
print("=" * 80)
