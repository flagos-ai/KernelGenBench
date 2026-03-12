#!/usr/bin/env python3
"""
测试cuFFT调用的示例脚本

展示三种方法：
1. 使用ctypes直接调用libcufft.so
2. 使用CuPy (如果已安装)
3. 使用PyCUDA (如果已安装)
"""

import ctypes
import os
import sys

def test_ctypes_cufft():
    """方法1: 使用ctypes直接调用libcufft.so"""
    print("=" * 70)
    print("方法1: 使用ctypes直接调用libcufft.so")
    print("=" * 70)
    
    # 查找libcufft.so
    cuda_paths = ['/usr/local/cuda', '/opt/cuda']
    libcufft_path = None
    
    for path in cuda_paths:
        lib_path = os.path.join(path, 'lib64', 'libcufft.so')
        if os.path.exists(lib_path):
            libcufft_path = lib_path
            break
    
    if not libcufft_path:
        print("✗ 未找到libcufft.so")
        return False
    
    try:
        # 加载库
        libcufft = ctypes.CDLL(libcufft_path)
        print(f"✓ 成功加载: {libcufft_path}")
        
        # 定义常量
        CUFFT_SUCCESS = 0
        CUFFT_C2C = 0x29
        CUFFT_FORWARD = -1
        CUFFT_INVERSE = 1
        
        # 测试cufftGetVersion
        cufftGetVersion = libcufft.cufftGetVersion
        cufftGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        cufftGetVersion.restype = ctypes.c_int
        
        version = ctypes.c_int()
        result = cufftGetVersion(ctypes.byref(version))
        if result == CUFFT_SUCCESS:
            major = version.value // 10000
            minor = (version.value // 100) % 100
            patch = version.value % 100
            print(f"✓ cuFFT版本: {major}.{minor}.{patch} (version code: {version.value})")
            return True
        else:
            print(f"✗ 获取版本失败: {result}")
            return False
            
    except Exception as e:
        print(f"✗ 加载库失败: {e}")
        return False


def test_cupy_cufft():
    """方法2: 使用CuPy (推荐)"""
    print("=" * 70)
    print("方法2: 使用CuPy (推荐)")
    print("=" * 70)
    
    try:
        import cupy as cp
        import numpy as np
        
        print(f"✓ CuPy已安装: {cp.__version__}")
        print(f"  CUDA版本: {cp.cuda.runtime.runtimeGetVersion()}")
        print()
        
        # 测试使用CuPy的FFT (底层使用cuFFT)
        print("测试CuPy FFT (底层使用cuFFT):")
        
        # 创建测试数据
        nx, ny = 128, 128
        data = cp.random.randn(nx, ny, dtype=cp.complex64)
        print(f"  输入数据shape: {data.shape}, dtype: {data.dtype}")
        
        # 执行FFT
        result = cp.fft.fft2(data)
        print(f"  输出数据shape: {result.shape}, dtype: {result.dtype}")
        print("  ✓ FFT执行成功")
        
        # 验证结果
        inverse = cp.fft.ifft2(result)
        diff = cp.abs(data - inverse)
        max_diff = cp.max(diff)
        print(f"  逆FFT验证: 最大误差 = {max_diff:.2e}")
        if max_diff < 1e-5:
            print("  ✓ 结果正确")
            return True
        else:
            print("  ⚠️ 误差较大")
            return False
            
    except ImportError:
        print("✗ CuPy未安装")
        print("  安装方法:")
        print("    pip install cupy-cuda11x  # CUDA 11.x")
        print("    pip install cupy-cuda12x  # CUDA 12.x")
        return False
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pycuda_cufft():
    """方法3: 使用PyCUDA"""
    print("=" * 70)
    print("方法3: 使用PyCUDA")
    print("=" * 70)
    
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        import pycuda.gpuarray as gpuarray
        import numpy as np
        
        print("✓ PyCUDA已安装")
        print()
        print("注意: PyCUDA本身不直接封装cuFFT")
        print("需要结合ctypes使用，或者使用CuPy")
        return True
        
    except ImportError:
        print("✗ PyCUDA未安装")
        print("  安装方法: pip install pycuda")
        return False
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("cuFFT调用测试")
    print("=" * 70)
    print()
    
    results = []
    
    # 测试方法1
    results.append(("ctypes", test_ctypes_cufft()))
    print()
    
    # 测试方法2
    results.append(("CuPy", test_cupy_cufft()))
    print()
    
    # 测试方法3
    results.append(("PyCUDA", test_pycuda_cufft()))
    print()
    
    # 总结
    print("=" * 70)
    print("测试总结")
    print("=" * 70)
    for method, success in results:
        status = "✓ 可用" if success else "✗ 不可用"
        print(f"{method:10s}: {status}")
    print()
    
    print("推荐:")
    print("  - 如果CuPy可用: 使用CuPy (最简单)")
    print("  - 如果只有ctypes: 使用ctypes (需要更多代码)")
    print("  - 如果需要精确控制: 写C++扩展")

