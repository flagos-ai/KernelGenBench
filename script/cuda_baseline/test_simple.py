"""简单测试已编译的kernels"""
import sys
from pathlib import Path
import torch

# 添加路径
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from cuda_baseline_builder import CudaBaselineBuilder

def test_copy_low_upp():
    """测试 copy_low_upp"""
    print("="*60)
    print("Test: copy_low_upp")
    print("="*60)
    
    # 直接加载已编译的 .so 文件
    builder = CudaBaselineBuilder(
        build_dir="/share/project/zpy/flagbench/cache/cuda_jit"
    )
    
    # 读取 CUDA 和 adapter 代码
    cuda_file = Path("/share/project/zpy/flagbench/cache/extracted_cuda/copy_low_upp.cu")
    adapter_file = Path("/share/project/zpy/flagbench/cache/generated_adapters/copy_low_upp_adapter.cpp")
    
    if not cuda_file.exists():
        print(f"ERROR: CUDA file not found: {cuda_file}")
        return False
    if not adapter_file.exists():
        print(f"ERROR: Adapter file not found: {adapter_file}")
        return False
    
    cuda_code = cuda_file.read_text()
    adapter_code = adapter_file.read_text()
    
    print(f"CUDA source: {len(cuda_code)} bytes")
    print(f"Adapter code: {len(adapter_code)} bytes")
    
    # 编译（或从缓存加载）
    print("\nCompiling kernel...")
    func = builder.load_kernel(
        kernel_name="kaldi_copy_low_upp",
        cuda_source=cuda_code,
        adapter_source=adapter_code,
        func_name="copy_low_upp"
    )
    
    print("✓ Kernel loaded successfully")
    
    # 测试
    print("\nTesting kernel...")
    A = torch.randn(32, 32, device='cuda')
    A_before = A.clone()
    
    print(f"Before: A[0,1] = {A[0,1].item():.4f}, A[1,0] = {A[1,0].item():.4f}")
    
    func(A)
    
    print(f"After:  A[0,1] = {A[0,1].item():.4f}, A[1,0] = {A[1,0].item():.4f}")
    
    # 验证：上三角应该等于原始下三角
    passed = True
    for i in range(32):
        for j in range(i+1, 32):
            if not torch.isclose(A[i, j], A_before[j, i]):
                print(f"FAILED at ({i},{j}): {A[i,j].item():.4f} != {A_before[j,i].item():.4f}")
                passed = False
                break
        if not passed:
            break
    
    if passed:
        print("✓ PASSED: Upper triangle correctly copied from lower triangle")
    
    return passed

if __name__ == "__main__":
    success = test_copy_low_upp()
    sys.exit(0 if success else 1)
