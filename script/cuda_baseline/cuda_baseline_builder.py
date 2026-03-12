"""
CudaBaselineBuilder - 轻量级 CUDA Baseline 构建工具

使用 torch.utils.cpp_extension.load_inline 动态编译原始 CUDA 代码，
无需修改源码，完美复用开源仓库（如 Kaldi）的 CUDA kernel。

核心优势：
1. 零侵入：直接复用原始 CUDA 代码，不需要修改
2. 零开销：C++ 直接调用，无 Python/CuPy 中间层
3. 自动缓存：首次编译后缓存，后续加载极快
4. 易复用：适配器代码可自动生成

作者: FlagBench Team
日期: 2026-01-16
"""

import os
import logging
from typing import List, Optional, Callable
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

logger = logging.getLogger(__name__)


class CudaBaselineBuilder:
    """
    动态编译 CUDA 源码并生成 Python 可调用函数的构建器。
    
    使用场景：
    1. 将开源项目的 CUDA kernel 包装成 PyTorch 函数
    2. 快速原型开发和测试
    3. 性能基准测试（baseline）
    
    示例：
        builder = CudaBaselineBuilder(build_dir="./cache/cuda_jit")
        
        # 原始 CUDA 代码（从 Kaldi 等项目拷贝）
        cuda_source = '''
        __global__ void _add_kernel(float* a, float* b, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) a[idx] += b[idx];
        }
        
        extern "C" void launch_add(float* a, float* b, int n) {
            int threads = 256;
            int blocks = (n + threads - 1) / threads;
            _add_kernel<<<blocks, threads>>>(a, b, n);
        }
        '''
        
        # 适配器代码（可自动生成）
        adapter_source = '''
        #include <torch/extension.h>
        
        extern "C" void launch_add(float*, float*, int);
        
        void add_wrapper(torch::Tensor a, torch::Tensor b) {
            launch_add(a.data_ptr<float>(), b.data_ptr<float>(), a.numel());
        }
        
        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            m.def("add", &add_wrapper);
        }
        '''
        
        # 编译并获取函数
        add_fn = builder.load_kernel(
            kernel_name="add_kernel",
            cuda_source=cuda_source,
            adapter_source=adapter_source,
            func_name="add"
        )
        
        # 使用
        a = torch.randn(1000, device='cuda')
        b = torch.randn(1000, device='cuda')
        add_fn(a, b)
    """
    
    def __init__(self, build_dir: str = "./cache/cuda_jit", verbose: bool = False):
        """
        初始化构建器。
        
        Args:
            build_dir: 编译缓存目录，默认 "./cache/cuda_jit"
            verbose: 是否显示详细编译日志
        """
        self.build_dir = Path(build_dir).resolve()
        self.build_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self._compiled_modules = {}  # 缓存已编译的模块
        
        logger.info(f"CudaBaselineBuilder initialized with build_dir: {self.build_dir}")
    
    def load_kernel(
        self,
        kernel_name: str,
        cuda_source: str,
        adapter_source: str,
        include_dirs: Optional[List[str]] = None,
        func_name: Optional[str] = None,
        extra_cuda_cflags: Optional[List[str]] = None,
        extra_cflags: Optional[List[str]] = None,
    ) -> Callable:
        """
        编译 CUDA 源码并返回 Python 可调用函数。
        
        Args:
            kernel_name: 内核唯一标识符（用于缓存），建议用描述性名称如 "kaldi_copy_low_upp"
            cuda_source: 原始 CUDA 代码字符串（包含 __global__ kernel 和 launcher）
            adapter_source: C++ 适配器代码（包含 torch/extension.h 和 pybind11 绑定）
            include_dirs: 头文件搜索路径列表，用于处理内部依赖
            func_name: 要暴露的 Python 函数名（默认与 kernel_name 相同）
            extra_cuda_cflags: 额外的 CUDA 编译选项（默认 ['-O3', '--use_fast_math']）
            extra_cflags: 额外的 C++ 编译选项（默认 ['-O3']）
        
        Returns:
            Python 可调用函数
        
        Raises:
            RuntimeError: 编译失败时抛出，包含详细错误信息
        """
        if func_name is None:
            func_name = kernel_name
        
        # 检查是否已缓存
        cache_key = f"{kernel_name}_{hash(cuda_source + adapter_source)}"
        if cache_key in self._compiled_modules:
            logger.info(f"Using cached module for kernel: {kernel_name}")
            return getattr(self._compiled_modules[cache_key], func_name)
        
        # 设置编译选项
        if extra_cuda_cflags is None:
            extra_cuda_cflags = ['-O3', '--use_fast_math']
        if extra_cflags is None:
            extra_cflags = ['-O3']
        
        # 准备头文件路径
        if include_dirs is None:
            include_dirs = []
        include_paths = [f"-I{path}" for path in include_dirs]
        extra_cuda_cflags.extend(include_paths)
        extra_cflags.extend(include_paths)
        
        logger.info(f"Compiling kernel: {kernel_name}")
        if self.verbose:
            logger.info(f"  Build directory: {self.build_dir}")
            logger.info(f"  Include dirs: {include_dirs}")
            logger.info(f"  CUDA flags: {extra_cuda_cflags}")
        
        try:
            # 使用 load_inline 编译
            module = load_inline(
                name=kernel_name,
                cpp_sources=[adapter_source],
                cuda_sources=[cuda_source],
                extra_cflags=extra_cflags,
                extra_cuda_cflags=extra_cuda_cflags,
                build_directory=str(self.build_dir),
                verbose=self.verbose,
                with_cuda=True,
            )
            
            # 缓存模块
            self._compiled_modules[cache_key] = module
            
            # 验证函数存在
            if not hasattr(module, func_name):
                raise RuntimeError(
                    f"Compilation succeeded but function '{func_name}' not found in module. "
                    f"Available functions: {dir(module)}"
                )
            
            logger.info(f"✓ Kernel '{kernel_name}' compiled successfully")
            return getattr(module, func_name)
        
        except Exception as e:
            logger.error(f"✗ Failed to compile kernel '{kernel_name}'")
            logger.error(f"Error: {str(e)}")
            if self.verbose:
                logger.error("=== CUDA Source ===")
                logger.error(cuda_source)
                logger.error("=== Adapter Source ===")
                logger.error(adapter_source)
            raise RuntimeError(f"Kernel compilation failed: {kernel_name}") from e
    
    def clear_cache(self):
        """清空内存中的模块缓存（不删除磁盘文件）"""
        self._compiled_modules.clear()
        logger.info("Module cache cleared")
    
    def get_cache_size(self) -> int:
        """返回已缓存的模块数量"""
        return len(self._compiled_modules)


def simple_test():
    """简单的功能测试"""
    print("=" * 80)
    print("CudaBaselineBuilder - Simple Test")
    print("=" * 80)
    
    # 创建构建器
    builder = CudaBaselineBuilder(build_dir="./cache/cuda_jit", verbose=True)
    
    # 定义一个简单的 add kernel
    cuda_source = """
    __global__ void _simple_add_kernel(float* a, float* b, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            a[idx] += b[idx];
        }
    }
    
    extern "C" void launch_simple_add(float* a, float* b, int n) {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        _simple_add_kernel<<<blocks, threads>>>(a, b, n);
        cudaDeviceSynchronize();
    }
    """
    
    adapter_source = """
    #include <torch/extension.h>
    
    extern "C" void launch_simple_add(float*, float*, int);
    
    void simple_add(torch::Tensor a, torch::Tensor b) {
        TORCH_CHECK(a.device().is_cuda(), "Tensor a must be on CUDA");
        TORCH_CHECK(b.device().is_cuda(), "Tensor b must be on CUDA");
        TORCH_CHECK(a.is_contiguous(), "Tensor a must be contiguous");
        TORCH_CHECK(b.is_contiguous(), "Tensor b must be contiguous");
        TORCH_CHECK(a.numel() == b.numel(), "Tensors must have same size");
        
        launch_simple_add(
            a.data_ptr<float>(),
            b.data_ptr<float>(),
            a.numel()
        );
    }
    
    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("simple_add", &simple_add, "Simple add kernel");
    }
    """
    
    # 编译
    print("\n[1] Compiling kernel...")
    add_fn = builder.load_kernel(
        kernel_name="simple_add_test",
        cuda_source=cuda_source,
        adapter_source=adapter_source,
        func_name="simple_add"
    )
    
    # 测试
    print("\n[2] Testing kernel...")
    a = torch.ones(1000, device='cuda', dtype=torch.float32)
    b = torch.ones(1000, device='cuda', dtype=torch.float32) * 2
    
    print(f"Before: a[0:5] = {a[:5].cpu().tolist()}")
    add_fn(a, b)
    print(f"After:  a[0:5] = {a[:5].cpu().tolist()}")
    
    expected = torch.ones(1000, device='cuda') * 3
    if torch.allclose(a, expected):
        print("\n✓ Test PASSED")
    else:
        print("\n✗ Test FAILED")
    
    print(f"\n[3] Cache info: {builder.get_cache_size()} modules cached")
    print("=" * 80)


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s"
    )
    
    # 运行测试
    simple_test()
