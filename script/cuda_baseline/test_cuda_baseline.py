"""
Test CUDA Baseline - 完整测试 3 个 Kaldi kernel

这个脚本演示完整的工作流程：
1. 从 Kaldi 提取 CUDA 源码
2. 生成 adapter 代码
3. 使用 CudaBaselineBuilder 编译
4. 测试正确性和性能

作者: FlagBench Team
日期: 2026-01-16
"""

import sys
import os
import time
from pathlib import Path

# 添加脚本目录到路径
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

import torch
from cuda_baseline_builder import CudaBaselineBuilder
from extract_cuda_source import KaldiCudaExtractor

print("=" * 80)
print("CUDA Baseline Test - Kaldi K1 Kernels")
print("=" * 80)

# ========================================================================
# 步骤 1: 提取 CUDA 源码
# ========================================================================
print("\n[Step 1] Extracting CUDA source code from Kaldi...")

kaldi_repo = "/share/project/zpy/k1_repo"
extractor = KaldiCudaExtractor(kaldi_repo)

kernel_names = ["copy_low_upp", "copy_upp_low", "add_mat"]
cuda_sources = {}

for name in kernel_names:
    try:
        cuda_sources[name] = extractor.extract_full_cuda_source(name)
        print(f"  ✓ Extracted: {name}")
    except Exception as e:
        print(f"  ✗ Failed: {name} - {e}")
        sys.exit(1)

# ========================================================================
# 步骤 2: 准备 adapter 代码
# ========================================================================
print("\n[Step 2] Loading adapter code...")

adapter_dir = Path("cache/generated_adapters")
adapters = {}

for name in kernel_names:
    adapter_file = adapter_dir / f"{name}_adapter.cpp"
    if not adapter_file.exists():
        print(f"  ✗ Adapter not found: {adapter_file}")
        print("  Please run: python script/cuda_baseline/generate_adapter.py")
        sys.exit(1)
    
    with open(adapter_file, 'r') as f:
        adapters[name] = f.read()
    print(f"  ✓ Loaded adapter: {name}")

# ========================================================================
# 步骤 3: 编译 CUDA kernels
# ========================================================================
print("\n[Step 3] Compiling CUDA kernels with load_inline...")

builder = CudaBaselineBuilder(build_dir="./cache/cuda_jit", verbose=False)

compiled_funcs = {}

for name in kernel_names:
    print(f"\n  Compiling: {name}...")
    try:
        func = builder.load_kernel(
            kernel_name=f"kaldi_{name}",
            cuda_source=cuda_sources[name],
            adapter_source=adapters[name],
            func_name=name
        )
        compiled_funcs[name] = func
        print(f"  ✓ Successfully compiled: {name}")
    except Exception as e:
        print(f"  ✗ Compilation failed: {name}")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

print(f"\n✓ All {len(compiled_funcs)} kernels compiled successfully!")

# ========================================================================
# 步骤 4: 正确性测试
# ========================================================================
print("\n" + "=" * 80)
print("[Step 4] Correctness Tests")
print("=" * 80)

def test_copy_low_upp():
    """测试 copy_low_upp kernel"""
    print("\n[Test 1] copy_low_upp")
    
    N = 32
    A = torch.randn(N, N, device='cuda', dtype=torch.float32)
    A_copy = A.clone()
    
    # 提取下三角
    lower = torch.tril(A_copy, diagonal=-1)
    
    # 运行 kernel
    compiled_funcs['copy_low_upp'](A)
    
    # 验证：上三角应该等于下三角的转置
    upper = torch.triu(A, diagonal=1)
    lower_t = lower.t()
    
    # 只比较上三角部分
    upper_mask = torch.triu(torch.ones(N, N, device='cuda'), diagonal=1).bool()
    if torch.allclose(A[upper_mask], A_copy.t()[upper_mask]):
        print("  ✓ PASSED - Upper triangle correctly copied from lower triangle")
        return True
    else:
        print("  ✗ FAILED - Results don't match")
        return False

def test_copy_upp_low():
    """测试 copy_upp_low kernel"""
    print("\n[Test 2] copy_upp_low")
    
    N = 32
    A = torch.randn(N, N, device='cuda', dtype=torch.float32)
    A_copy = A.clone()
    
    # 提取上三角
    upper = torch.triu(A_copy, diagonal=1)
    
    # 运行 kernel
    compiled_funcs['copy_upp_low'](A)
    
    # 验证：下三角应该等于上三角的转置
    lower_mask = torch.tril(torch.ones(N, N, device='cuda'), diagonal=-1).bool()
    if torch.allclose(A[lower_mask], A_copy.t()[lower_mask]):
        print("  ✓ PASSED - Lower triangle correctly copied from upper triangle")
        return True
    else:
        print("  ✗ FAILED - Results don't match")
        return False

def test_add_mat():
    """测试 add_mat kernel"""
    print("\n[Test 3] add_mat")
    
    M, N = 64, 128
    dst = torch.randn(M, N, device='cuda', dtype=torch.float32)
    src = torch.randn(M, N, device='cuda', dtype=torch.float32)
    alpha = 2.5
    
    # 计算期望结果
    dst_copy = dst.clone()
    expected = alpha * src + dst_copy
    
    # 运行 kernel
    compiled_funcs['add_mat'](dst, src, alpha)
    
    # 验证
    if torch.allclose(dst, expected, rtol=1e-5, atol=1e-6):
        print(f"  ✓ PASSED - dst = {alpha} * src + dst computed correctly")
        return True
    else:
        max_diff = (dst - expected).abs().max().item()
        print(f"  ✗ FAILED - Max difference: {max_diff}")
        return False

# 运行测试
test_results = []
test_results.append(("copy_low_upp", test_copy_low_upp()))
test_results.append(("copy_upp_low", test_copy_upp_low()))
test_results.append(("add_mat", test_add_mat()))

print("\n" + "=" * 80)
print("Test Summary")
print("=" * 80)
passed = sum(1 for _, result in test_results if result)
total = len(test_results)
print(f"Passed: {passed}/{total}")

for name, result in test_results:
    status = "✓ PASS" if result else "✗ FAIL"
    print(f"  {status}: {name}")

if passed < total:
    print("\n⚠ Some tests failed!")
    sys.exit(1)

# ========================================================================
# 步骤 5: 性能测试
# ========================================================================
print("\n" + "=" * 80)
print("[Step 5] Performance Tests")
print("=" * 80)

def benchmark_kernel(func, *args, warmup=10, iterations=100):
    """测量 kernel 性能"""
    # Warmup
    for _ in range(warmup):
        func(*args)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(iterations):
        func(*args)
    torch.cuda.synchronize()
    end = time.time()
    
    avg_time_ms = (end - start) / iterations * 1000
    return avg_time_ms

print("\n[Perf 1] copy_low_upp (N=512)")
N = 512
A = torch.randn(N, N, device='cuda', dtype=torch.float32)
time_ms = benchmark_kernel(compiled_funcs['copy_low_upp'], A, iterations=1000)
print(f"  Average time: {time_ms:.4f} ms")

print("\n[Perf 2] copy_upp_low (N=512)")
A = torch.randn(N, N, device='cuda', dtype=torch.float32)
time_ms = benchmark_kernel(compiled_funcs['copy_upp_low'], A, iterations=1000)
print(f"  Average time: {time_ms:.4f} ms")

print("\n[Perf 3] add_mat (512x512)")
dst = torch.randn(512, 512, device='cuda', dtype=torch.float32)
src = torch.randn(512, 512, device='cuda', dtype=torch.float32)
time_ms = benchmark_kernel(compiled_funcs['add_mat'], dst, src, 2.0, iterations=1000)
print(f"  Average time: {time_ms:.4f} ms")

print("\n" + "=" * 80)
print("✓ All tests completed successfully!")
print("=" * 80)
print("\nNext steps:")
print("1. Compare with CuPy baseline performance")
print("2. Compare with Triton implementation")
print("3. Scale to all 169 Kaldi kernels")
print("=" * 80)
