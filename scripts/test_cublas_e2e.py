#!/usr/bin/env python3
"""
cuBLAS Triton Baseline - 端到端测试

完整流程：
1. 导入 Triton 内核（自动注册到 flagbench.cublas 命名空间）
2. 导入生成的测试文件
3. 运行测试
"""

import sys
import os

# Step 0: 配置路径
REPO_ROOT = '/share/project/zpy/flagbench'
TRITON_DIR = 'output_triton_cublas/triton_cublas_deepseek-v3-0324_num_samples_1_temp_0.0_max_tokens_16384_20260119-144713/triton_0'
UT_DIR = 'output_ut_cublas/ut_cublas_deepseek-v3-0324_num_samples_1_temp_0.0_max_tokens_16384_20260119-143016/ut_0'

sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))

# Step 1: 导入 flagbench（避免循环导入）
print("=" * 80)
print("步骤 1: 导入 flagbench 框架")
print("=" * 80)

import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.register import REGISTERED_OPS
import torch
import cupy as cp
from torch.utils.dlpack import to_dlpack, from_dlpack

print("✓ flagbench 框架已加载\n")

# Step 2: 导入 Triton 内核（触发注册）
print("=" * 80)
print("步骤 2: 导入 Triton 内核（自动注册）")
print("=" * 80)

sys.path.insert(0, os.path.join(REPO_ROOT, TRITON_DIR))

# 导入所有内核
from sgemm import sgemm
from dgemm import dgemm
from cgemm import cgemm
from zgemm import zgemm
from hgemm import hgemm

print("✓ 所有 Triton 内核已导入并注册\n")

# 验证注册
print("注册验证:")
print(f"  - flagbench.cublas 存在: {hasattr(flagbench, 'cublas')}")
kernels = ['sgemm', 'dgemm', 'cgemm', 'zgemm', 'hgemm']
for kernel in kernels:
    status = "✓" if hasattr(flagbench.cublas, kernel) else "✗"
    print(f"  {status} flagbench.cublas.{kernel}")

# Step 3: 加载并运行测试
print("\n" + "=" * 80)
print("步骤 3: 运行单元测试")
print("=" * 80)

# 定义 CuPy baseline (从生成的 UT 文件复制)
def cublas_baseline(A, B, alpha=1.0, beta=0.0):
    A_cp = cp.from_dlpack(to_dlpack(A))
    B_cp = cp.from_dlpack(to_dlpack(B))
    C_cp = alpha * cp.dot(A_cp, B_cp)
    return from_dlpack(C_cp.toDlpack())

# 测试用例
test_cases = [
    {
        'name': 'sgemm_small',
        'kernel': 'sgemm',
        'M': 2, 'N': 3, 'K': 4,
        'dtype': torch.float32,
        'alpha': 1.0, 'beta': 0.0,
    },
    {
        'name': 'sgemm_medium',
        'kernel': 'sgemm',
        'M': 128, 'N': 256, 'K': 64,
        'dtype': torch.float32,
        'alpha': 1.0, 'beta': 0.0,
    },
    {
        'name': 'dgemm_medium',
        'kernel': 'dgemm',
        'M': 64, 'N': 128, 'K': 32,
        'dtype': torch.float64,
        'alpha': 1.0, 'beta': 0.0,
    },
]

passed = 0
failed = 0

for test in test_cases:
    print(f"\n测试: {test['name']}")
    print(f"  配置: M={test['M']}, N={test['N']}, K={test['K']}, dtype={test['dtype']}")
    
    try:
        # 准备输入
        A = torch.randn(test['M'], test['K'], dtype=test['dtype'], device='cuda')
        B = torch.randn(test['K'], test['N'], dtype=test['dtype'], device='cuda')
        C = torch.zeros(test['M'], test['N'], dtype=test['dtype'], device='cuda')
        
        # Baseline
        ref_A = A.clone()
        ref_B = B.clone()
        ref_out = cublas_baseline(ref_A, ref_B, alpha=test['alpha'], beta=test['beta'])
        if test['beta'] != 0.0:
            ref_out += test['beta'] * C.clone()
        
        # Triton
        kernel_func = getattr(flagbench.cublas, test['kernel'])
        act_out = kernel_func(A, B, C, alpha=test['alpha'], beta=test['beta'])
        
        # 比较
        diff = torch.norm(act_out - ref_out) / (torch.norm(ref_out) + 1e-10)
        
        # 结果
        threshold = 1e-3  # float32 允许更大的误差
        if diff < threshold:
            print(f"  ✓ PASS - relative_error={diff.item():.6e} (threshold={threshold})")
            passed += 1
        else:
            print(f"  ✗ FAIL - relative_error={diff.item():.6e} (threshold={threshold})")
            failed += 1
            
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        failed += 1
        import traceback
        traceback.print_exc()

# 总结
print("\n" + "=" * 80)
print(f"测试总结")
print("=" * 80)
print(f"  通过: {passed}")
print(f"  失败: {failed}")
print(f"  总计: {passed + failed}")
print(f"  成功率: {100 * passed / (passed + failed) if (passed + failed) > 0 else 0:.1f}%")
print("=" * 80)

if failed == 0:
    print("\n🎉 所有测试通过！cuBLAS Triton baseline 流程验证成功！")
    sys.exit(0)
else:
    print(f"\n⚠️  有 {failed} 个测试失败")
    sys.exit(1)
