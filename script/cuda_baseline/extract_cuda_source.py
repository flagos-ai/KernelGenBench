"""
CUDA Source Extractor - 从 Kaldi 等仓库提取 CUDA 源码

从开源仓库中提取 CUDA kernel 和 wrapper 函数的源码，
准备用于 load_inline 编译。

作者: FlagBench Team
日期: 2026-01-16
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class KernelInfo:
    """Kernel 信息"""
    name: str                    # kernel 名称，如 "copy_low_upp"
    kernel_code: str             # __global__ kernel 代码
    wrapper_code: str            # host wrapper 函数代码
    dependencies: List[str]      # 依赖的头文件或其他代码


class KaldiCudaExtractor:
    """
    从 Kaldi 仓库提取 CUDA kernel 源码
    
    使用方式：
        extractor = KaldiCudaExtractor("/path/to/k1_repo")
        cuda_source = extractor.extract_kernel("copy_low_upp")
        print(cuda_source)
    """
    
    def __init__(self, kaldi_repo_path: str):
        """
        初始化提取器
        
        Args:
            kaldi_repo_path: Kaldi 仓库根目录
        """
        self.repo_path = Path(kaldi_repo_path)
        self.cu_kernels_path = self.repo_path / "src" / "cudamatrix" / "cu-kernels.cu"
        self.cu_matrixdim_path = self.repo_path / "src" / "cudamatrix" / "cu-matrixdim.h"
        
        if not self.cu_kernels_path.exists():
            raise FileNotFoundError(f"cu-kernels.cu not found at {self.cu_kernels_path}")
        
        # 读取文件内容
        with open(self.cu_kernels_path, 'r') as f:
            self.cu_kernels_content = f.read()
        
        # 读取 MatrixDim 定义
        with open(self.cu_matrixdim_path, 'r') as f:
            self.matrixdim_header = f.read()
    
    def extract_matrixdim_struct(self) -> str:
        """提取 MatrixDim 结构定义"""
        # 提取 typedef 部分（包括 int32_cuda）
        pattern = r'typedef\s+(?:uint|int)32_t\s+(?:uint32|int32)_cuda;'
        typedefs = re.findall(pattern, self.matrixdim_header)
        
        # 简化版本：直接用 int 替代 int32_cuda
        return """
// Type definitions
typedef int int32_cuda;
typedef unsigned int uint32_cuda;

// MatrixDim structure
typedef struct MatrixDim_ {
    int32_cuda rows;
    int32_cuda cols;
    int32_cuda stride;
} MatrixDim;
"""
    
    def extract_kernel_template(self, kernel_name: str) -> str:
        """
        提取 __global__ kernel 模板代码
        
        Args:
            kernel_name: kernel 名称（不带下划线前缀），如 "copy_low_upp"
        
        Returns:
            kernel 代码字符串
        """
        # 查找 template<typename Real> __global__ static void _kernel_name(...)
        # 或者 __global__ static void _kernel_name(...)
        pattern = rf'(template<typename\s+Real>\s+)?__global__\s+static\s+void\s+_{kernel_name}\s*\([^)]*\)\s*{{[^{{}}]*(?:{{[^{{}}]*}}[^{{}}]*)*}}'
        
        match = re.search(pattern, self.cu_kernels_content, re.DOTALL)
        if not match:
            raise ValueError(f"Kernel '_{kernel_name}' not found in cu-kernels.cu")
        
        kernel_code = match.group(0)
        
        # 将 template<typename Real> 替换为具体类型 float
        kernel_code = re.sub(r'template<typename\s+Real>\s+', '', kernel_code)
        kernel_code = kernel_code.replace('Real*', 'float*')
        kernel_code = kernel_code.replace('Real', 'float')
        
        return kernel_code
    
    def extract_wrapper_function(self, kernel_name: str) -> str:
        """
        提取 wrapper 函数代码
        
        Args:
            kernel_name: kernel 名称，如 "copy_low_upp"
        
        Returns:
            wrapper 函数代码（extern "C"）
        """
        # 查找 cudaF_{kernel_name} 函数
        wrapper_pattern = rf'void\s+cudaF_{kernel_name}\s*\([^)]*\)\s*{{[^{{}}]*(?:{{[^{{}}]*}}[^{{}}]*)*}}'
        
        match = re.search(wrapper_pattern, self.cu_kernels_content, re.DOTALL)
        if not match:
            raise ValueError(f"Wrapper function 'cudaF_{kernel_name}' not found")
        
        wrapper_code = match.group(0)
        
        # 提取函数签名和内容
        func_match = re.match(r'void\s+cudaF_(\w+)\s*\(([^)]*)\)\s*{(.*)}', wrapper_code, re.DOTALL)
        if not func_match:
            return wrapper_code
        
        func_name, params, body = func_match.groups()
        
        # 生成 extern "C" 版本
        extern_c_wrapper = f'''
extern "C" void launch_{kernel_name}({params}) {{
{body}
}}
'''
        return extern_c_wrapper
    
    def extract_full_cuda_source(self, kernel_name: str) -> str:
        """
        提取完整的 CUDA 源码（包括 kernel + wrapper + 依赖）
        
        Args:
            kernel_name: kernel 名称，如 "copy_low_upp"
        
        Returns:
            完整的可编译 CUDA 源码
        """
        # 提取各部分
        matrixdim = self.extract_matrixdim_struct()
        kernel = self.extract_kernel_template(kernel_name)
        wrapper = self.extract_wrapper_function(kernel_name)
        
        # 组装完整源码
        full_source = f"""
// Auto-extracted from Kaldi cu-kernels.cu
// Kernel: {kernel_name}

#include <cuda_runtime.h>

// MatrixDim structure
{matrixdim}

// Kernel definition
{kernel}

// Host wrapper function
{wrapper}
"""
        return full_source
    
    def extract_multiple_kernels(self, kernel_names: List[str]) -> Dict[str, str]:
        """
        批量提取多个 kernel
        
        Args:
            kernel_names: kernel 名称列表
        
        Returns:
            Dict[kernel_name, cuda_source]
        """
        results = {}
        for name in kernel_names:
            try:
                results[name] = self.extract_full_cuda_source(name)
                print(f"✓ Extracted: {name}")
            except Exception as e:
                print(f"✗ Failed to extract {name}: {e}")
        return results


def demo_extract_kaldi_kernels():
    """演示：提取 Kaldi K1 的 3 个测试 kernel"""
    print("=" * 80)
    print("Kaldi CUDA Source Extractor - Demo")
    print("=" * 80)
    
    # 初始化提取器
    kaldi_repo = "/share/project/zpy/k1_repo"
    extractor = KaldiCudaExtractor(kaldi_repo)
    
    # 提取 3 个 kernel
    kernel_names = ["copy_low_upp", "copy_upp_low", "add_mat"]
    
    print(f"\nExtracting {len(kernel_names)} kernels from {kaldi_repo}...\n")
    
    results = extractor.extract_multiple_kernels(kernel_names)
    
    # 保存到文件
    output_dir = Path("./cache/extracted_cuda")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for kernel_name, source in results.items():
        output_file = output_dir / f"{kernel_name}.cu"
        with open(output_file, 'w') as f:
            f.write(source)
        print(f"  Saved to: {output_file}")
    
    # 显示第一个 kernel 的内容作为示例
    if results:
        first_kernel = list(results.keys())[0]
        print(f"\n{'=' * 80}")
        print(f"Example: {first_kernel}.cu")
        print("=" * 80)
        print(results[first_kernel])
    
    print("=" * 80)
    print(f"✓ Extraction complete! {len(results)} kernels saved to {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    demo_extract_kaldi_kernels()
