"""直接测试已编译的.so文件"""
import sys
import torch
import ctypes

# 直接加载 .so 文件
so_path = "/share/project/zpy/flagbench/cache/cuda_jit/kaldi_copy_low_upp.so"

print("="*60)
print("Direct .so test: copy_low_upp")
print("="*60)

try:
    # 方法1：使用 ctypes
    print(f"\nLoading {so_path}...")
    lib = ctypes.CDLL(so_path)
    print("✓ Library loaded successfully")
    print(f"Available symbols (first 10): {dir(lib)[:10]}")
    
except Exception as e:
    print(f"✗ Failed to load library: {e}")
    sys.exit(1)

# 方法2：使用torch.utils.cpp_extension
print("\n" + "="*60)
print("Try importing as PyTorch extension")
print("="*60)

try:
    # 这个应该会快很多，因为.so已经存在
    from torch.utils.cpp_extension import load
    
    module = load(
        name="kaldi_copy_low_upp",
        sources=[],  # 空列表，因为已经编译了
        build_directory="/share/project/zpy/flagbench/cache/cuda_jit",
        is_python_module=True,
        verbose=True
    )
    
    print("✓ Module loaded")
    print(f"Module attributes: {dir(module)}")
    
    if hasattr(module, 'copy_low_upp'):
        print("\n Testing kernel...")
        A = torch.randn(32, 32, device='cuda')
        print(f"Before: A[0,1]={A[0,1].item():.4f}, A[1,0]={A[1,0].item():.4f}")
        
        module.copy_low_upp(A)
        
        print(f"After:  A[0,1]={A[0,1].item():.4f}, A[1,0]={A[1,0].item():.4f}")
        print("✓ Kernel executed successfully!")
    else:
        print("✗ copy_low_upp function not found in module")
        
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
