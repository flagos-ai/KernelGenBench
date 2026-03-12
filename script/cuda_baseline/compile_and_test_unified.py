"""
编译并测试unified Kaldi K1 kernels (3个kernels)

使用torch.utils.cpp_extension.load_inline编译
"""

import sys
import torch
from pathlib import Path

# 添加路径
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from cuda_baseline_builder import CudaBaselineBuilder


def compile_unified_kernels():
    """
    编译3个Kaldi kernels的统一接口
    """
    
    print("="*60)
    print("Compiling Unified Kaldi K1 Kernels (3 kernels)")
    print("="*60)
    
    # 读取CUDA源码和adapter
    cuda_file = Path("/share/project/zpy/flagbench/cache/extracted_cuda/unified_3kernels.cu")
    adapter_file = Path("/share/project/zpy/flagbench/cache/generated_adapters/unified_kaldi_adapter.cpp")
    
    if not cuda_file.exists():
        print(f"ERROR: CUDA file not found: {cuda_file}")
        return None
    
    if not adapter_file.exists():
        print(f"ERROR: Adapter file not found: {adapter_file}")
        return None
    
    cuda_source = cuda_file.read_text()
    adapter_source = adapter_file.read_text()
    
    print(f"\n✓ Loaded sources:")
    print(f"  CUDA: {len(cuda_source)} bytes")
    print(f"  Adapter: {len(adapter_source)} bytes")
    
    # 使用CudaBaselineBuilder编译
    print("\n" + "="*60)
    print("Compiling with load_inline...")
    print("="*60)
    print("This may take 1-2 minutes on first compile...")
    
    builder = CudaBaselineBuilder(
        build_dir="/share/project/zpy/flagbench/cache/cuda_jit_unified",
        verbose=True
    )
    
    try:
        # 注意：我们传一个假的func_name来绕过检查，然后捕获异常获取模块
        try:
            _ = builder.load_kernel(
                kernel_name="kaldi_k1_unified",
                cuda_source=cuda_source,
                adapter_source=adapter_source,
                func_name="copy_low_upp",  # 传一个实际存在的函数名
            )
        except Exception:
            pass
        
        # 直接从缓存获取模块
        cache_key = f"kaldi_k1_unified_{hash(cuda_source + adapter_source)}"
        if cache_key in builder._compiled_modules:
            module = builder._compiled_modules[cache_key]
        else:
            print("ERROR: Module not in cache")
            return None
        
        print("\n" + "="*60)
        print("✓ Compilation successful!")
        print("="*60)
        
        # 检查可用函数
        available_funcs = [name for name in dir(module) if not name.startswith('_')]
        print(f"\nAvailable functions in module:")
        for func in available_funcs:
            print(f"  - {func}")
        
        return module
        
    except Exception as e:
        print(f"\n✗ Compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_kernels(module):
    """
    测试编译好的kernels
    """
    
    if module is None:
        print("No module to test")
        return False
    
    print("\n" + "="*60)
    print("Testing Compiled Kernels")
    print("="*60)
    
    all_passed = True
    
    # Test 1: copy_low_upp
    print("\nTest 1: copy_low_upp")
    print("-" * 60)
    try:
        A = torch.randn(32, 32, device='cuda', dtype=torch.float32)
        A_before = A.clone()
        
        print(f"Before: A[0,1]={A[0,1].item():.4f}, A[1,0]={A[1,0].item():.4f}")
        
        module.copy_low_upp(A)
        
        print(f"After:  A[0,1]={A[0,1].item():.4f}, A[1,0]={A[1,0].item():.4f}")
        
        # 验证：上三角 = 原下三角
        correct = True
        for i in range(32):
            for j in range(i+1, 32):
                if not torch.isclose(A[i, j], A_before[j, i]):
                    correct = False
                    break
            if not correct:
                break
        
        if correct:
            print("✓ PASSED")
        else:
            print("✗ FAILED")
            all_passed = False
            
    except Exception as e:
        print(f"✗ FAILED: {e}")
        all_passed = False
    
    # Test 2: copy_upp_low  
    print("\nTest 2: copy_upp_low")
    print("-" * 60)
    try:
        A = torch.randn(32, 32, device='cuda', dtype=torch.float32)
        A_before = A.clone()
        
        print(f"Before: A[0,1]={A[0,1].item():.4f}, A[1,0]={A[1,0].item():.4f}")
        
        module.copy_upp_low(A)
        
        print(f"After:  A[0,1]={A[0,1].item():.4f}, A[1,0]={A[1,0].item():.4f}")
        
        # 验证：下三角 = 原上三角
        correct = True
        for i in range(32):
            for j in range(i+1, 32):
                if not torch.isclose(A[j, i], A_before[i, j]):
                    correct = False
                    break
            if not correct:
                break
        
        if correct:
            print("✓ PASSED")
        else:
            print("✗ FAILED")
            all_passed = False
            
    except Exception as e:
        print(f"✗ FAILED: {e}")
        all_passed = False
    
    # Test 3: add_mat
    print("\nTest 3: add_mat")
    print("-" * 60)
    try:
        dst = torch.ones(32, 32, device='cuda', dtype=torch.float32)
        src = torch.ones(32, 32, device='cuda', dtype=torch.float32) * 2.0
        dst_before = dst.clone()
        alpha = 3.0
        
        print(f"Before: dst[0,0]={dst[0,0].item():.4f}")
        print(f"Formula: dst = alpha * src + dst")
        print(f"  alpha={alpha}, src[0,0]={src[0,0].item():.4f}, dst[0,0]={dst_before[0,0].item():.4f}")
        
        module.add_mat(dst, src, alpha)
        
        expected = alpha * src[0,0].item() + dst_before[0,0].item()
        print(f"After:  dst[0,0]={dst[0,0].item():.4f} (expected: {expected:.4f})")
        
        if torch.allclose(dst, alpha * src + dst_before):
            print("✓ PASSED")
        else:
            print("✗ FAILED")
            all_passed = False
            
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("✓ All tests PASSED!")
    else:
        print("✗ Some tests FAILED")
    print("="*60)
    
    return all_passed


def main():
    # 编译
    module = compile_unified_kernels()
    
    if module is None:
        print("\nCompilation failed, cannot run tests")
        return False
    
    # 测试
    success = test_kernels(module)
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
