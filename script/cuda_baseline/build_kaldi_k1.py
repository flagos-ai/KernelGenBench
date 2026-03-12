"""
Kaldi K1 Baseline Builder - 批量构建所有 Kaldi kernels

一键构建并注册所有 Kaldi K1 CUDA kernels 作为 PyTorch 可调用函数。

用法:
    python build_kaldi_k1.py --build-all          # 构建所有 kernel
    python build_kaldi_k1.py --kernel copy_low_upp # 构建单个 kernel
    python build_kaldi_k1.py --list                # 列出所有可用 kernel

作者: FlagBench Team
日期: 2026-01-16
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional
import logging

# 添加脚本目录到路径
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from cuda_baseline_builder import CudaBaselineBuilder
from extract_cuda_source import KaldiCudaExtractor
from generate_adapter import AdapterGenerator, TensorParam, ScalarParam

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class KaldiK1Builder:
    """
    Kaldi K1 批量构建器
    
    负责提取、生成 adapter、编译并注册所有 Kaldi kernels
    """
    
    def __init__(
        self,
        kaldi_repo_path: str = "/share/project/zpy/k1_repo",
        cache_dir: str = "./cache",
        verbose: bool = False
    ):
        self.kaldi_repo = Path(kaldi_repo_path)
        self.cache_dir = Path(cache_dir)
        self.verbose = verbose
        
        # 创建缓存目录
        self.cuda_cache = self.cache_dir / "extracted_cuda"
        self.adapter_cache = self.cache_dir / "generated_adapters"
        self.jit_cache = self.cache_dir / "cuda_jit"
        
        for dir_path in [self.cuda_cache, self.adapter_cache, self.jit_cache]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化工具
        self.extractor = KaldiCudaExtractor(str(self.kaldi_repo))
        self.adapter_gen = AdapterGenerator()
        self.builder = CudaBaselineBuilder(
            build_dir=str(self.jit_cache),
            verbose=verbose
        )
        
        # 编译后的函数
        self.compiled_kernels: Dict[str, any] = {}
        
        logger.info(f"Kaldi K1 Builder initialized")
        logger.info(f"  Kaldi repo: {self.kaldi_repo}")
        logger.info(f"  Cache dir: {self.cache_dir}")
    
    def get_kernel_config(self, kernel_name: str) -> Optional[Dict]:
        """
        获取 kernel 的配置信息
        
        目前支持的 kernel 配置（手动定义）
        """
        configs = {
            "copy_low_upp": {
                "tensor_params": [TensorParam("A", is_output=True)],
                "scalar_params": [],
                "grid_config": "simple_2d",
                "need_matrixdim": True,
                "description": "Copy lower triangle to upper triangle",
                "has_multiple_kernels": False,
            },
            "copy_upp_low": {
                "tensor_params": [TensorParam("A", is_output=True)],
                "scalar_params": [],
                "grid_config": "simple_2d",
                "need_matrixdim": True,
                "description": "Copy upper triangle to lower triangle",
                "has_multiple_kernels": False,
            },
            "add_mat": {
                "custom": True,  # 使用自定义 adapter
                "description": "Matrix addition: dst = alpha * src + dst",
                "has_multiple_kernels": True,  # 需要 _add_mat 和 _add_mat_trans
            },
        }
        return configs.get(kernel_name)
    
    def extract_kernel(self, kernel_name: str) -> str:
        """提取 kernel 的 CUDA 源码"""
        cuda_file = self.cuda_cache / f"{kernel_name}.cu"
        
        # 检查缓存
        if cuda_file.exists():
            logger.info(f"  Using cached CUDA source: {kernel_name}")
            with open(cuda_file, 'r') as f:
                return f.read()
        
        # 提取源码
        logger.info(f"  Extracting CUDA source: {kernel_name}")
        
        config = self.get_kernel_config(kernel_name)
        
        # 特殊处理：add_mat 需要两个 kernel
        if kernel_name == "add_mat":
            cuda_source = self._extract_add_mat_with_trans()
        else:
            cuda_source = self.extractor.extract_full_cuda_source(kernel_name)
        
        # 保存到缓存
        with open(cuda_file, 'w') as f:
            f.write(cuda_source)
        
        return cuda_source
    
    def _extract_add_mat_with_trans(self) -> str:
        """特殊处理：提取 add_mat 及其依赖的 _add_mat_trans"""
        # 手动构建包含两个 kernel 的完整源码
        source = """
// Auto-extracted from Kaldi cu-kernels.cu
// Kernel: add_mat (with _add_mat and _add_mat_trans)

#include <cuda_runtime.h>

// Type definitions
typedef int int32_cuda;
typedef unsigned int uint32_cuda;

// MatrixDim structure
typedef struct MatrixDim_ {
    int32_cuda rows;
    int32_cuda cols;
    int32_cuda stride;
} MatrixDim;

// Kernel: _add_mat
__global__
static void _add_mat(float alpha, const float* src, float* dst, MatrixDim d,
                     int src_stride) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j * d.stride;
  int32_cuda index_src = i + j * src_stride;
  if (i < d.cols && j < d.rows)
    dst[index] = alpha * src[index_src] + dst[index];
}

// Kernel: _add_mat_trans
__global__
static void _add_mat_trans(float alpha, const float* src, float* dst, MatrixDim d,
                           int src_stride) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j * d.stride;
  int32_cuda index_src = j + i * src_stride;
  if (i < d.cols && j < d.rows)
    dst[index] = alpha * src[index_src] + dst[index];
}

// Host wrapper
extern "C" void launch_add_mat(dim3 Gr, dim3 Bl, float alpha, const float* src, 
                               float* dst, MatrixDim d, int src_stride, int A_trans) {
  if (A_trans) {
    _add_mat_trans<<<Gr,Bl>>>(alpha, src, dst, d, src_stride);
  } else {
    _add_mat<<<Gr,Bl>>>(alpha, src, dst, d, src_stride);
  }
}
"""
        return source
    
    def generate_adapter(self, kernel_name: str) -> str:
        """生成 kernel 的 adapter 代码"""
        adapter_file = self.adapter_cache / f"{kernel_name}_adapter.cpp"
        
        # 检查缓存
        if adapter_file.exists():
            logger.info(f"  Using cached adapter: {kernel_name}")
            with open(adapter_file, 'r') as f:
                return f.read()
        
        logger.info(f"  Generating adapter: {kernel_name}")
        
        config = self.get_kernel_config(kernel_name)
        if not config:
            raise ValueError(f"No configuration found for kernel: {kernel_name}")
        
        # 特殊处理：add_mat 使用自定义 adapter
        if kernel_name == "add_mat":
            adapter = self._generate_add_mat_adapter()
        else:
            adapter = self.adapter_gen.generate_adapter(
                kernel_name=kernel_name,
                tensor_params=config["tensor_params"],
                scalar_params=config["scalar_params"],
                grid_config=config["grid_config"],
                need_matrixdim=config["need_matrixdim"],
                description=config["description"],
            )
        
        # 保存到缓存
        with open(adapter_file, 'w') as f:
            f.write(adapter)
        
        return adapter
    
    def _generate_add_mat_adapter(self) -> str:
        """生成 add_mat 的自定义 adapter"""
        return '''
#include <torch/extension.h>
#include <cuda_runtime.h>

// MatrixDim structure
typedef struct MatrixDim_ {
    int rows;
    int cols;
    int stride;
} MatrixDim;

// Forward declaration
extern "C" void launch_add_mat(dim3 Gr, dim3 Bl, float alpha, const float* src, 
                               float* dst, MatrixDim d, int src_stride, int A_trans);

// dst = alpha * src + dst
void add_mat(torch::Tensor dst, torch::Tensor src, float alpha) {
    TORCH_CHECK(dst.device().is_cuda(), "dst must be on CUDA");
    TORCH_CHECK(src.device().is_cuda(), "src must be on CUDA");
    TORCH_CHECK(dst.is_contiguous(), "dst must be contiguous");
    TORCH_CHECK(src.is_contiguous(), "src must be contiguous");
    TORCH_CHECK(dst.sizes() == src.sizes(), "dst and src must have same shape");

    int rows = dst.size(0);
    int cols = dst.size(1);

    // 2D grid configuration
    const int BLOCK_SIZE = 16;
    dim3 Bl(BLOCK_SIZE, BLOCK_SIZE);
    dim3 Gr((cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    float* dst_ptr = dst.data_ptr<float>();
    const float* src_ptr = src.data_ptr<float>();

    MatrixDim d;
    d.rows = rows;
    d.cols = cols;
    d.stride = dst.stride(0);

    int src_stride = src.stride(0);
    int A_trans = 0;

    launch_add_mat(Gr, Bl, alpha, src_ptr, dst_ptr, d, src_stride, A_trans);
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_mat", &add_mat, "Matrix addition: dst = alpha * src + dst");
}
'''
    
    def compile_kernel(self, kernel_name: str) -> any:
        """编译 kernel 并返回 Python 可调用函数"""
        if kernel_name in self.compiled_kernels:
            logger.info(f"  Using cached compiled kernel: {kernel_name}")
            return self.compiled_kernels[kernel_name]
        
        logger.info(f"  Compiling kernel: {kernel_name}")
        
        # 提取和生成
        cuda_source = self.extract_kernel(kernel_name)
        adapter_source = self.generate_adapter(kernel_name)
        
        # 编译
        func = self.builder.load_kernel(
            kernel_name=f"kaldi_{kernel_name}",
            cuda_source=cuda_source,
            adapter_source=adapter_source,
            func_name=kernel_name
        )
        
        self.compiled_kernels[kernel_name] = func
        logger.info(f"  ✓ Successfully compiled: {kernel_name}")
        
        return func
    
    def build_kernel(self, kernel_name: str) -> any:
        """构建单个 kernel（提取 + 生成 + 编译）"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Building kernel: {kernel_name}")
        logger.info(f"{'='*60}")
        
        try:
            func = self.compile_kernel(kernel_name)
            return func
        except Exception as e:
            logger.error(f"✗ Failed to build {kernel_name}: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return None
    
    def build_all(self, kernel_names: Optional[List[str]] = None) -> Dict[str, any]:
        """批量构建多个 kernel"""
        if kernel_names is None:
            # 默认构建有配置的 kernel
            kernel_names = list(self.get_all_configured_kernels())
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Building {len(kernel_names)} Kaldi K1 kernels")
        logger.info(f"{'='*60}")
        
        results = {}
        succeeded = 0
        failed = 0
        
        for kernel_name in kernel_names:
            func = self.build_kernel(kernel_name)
            if func:
                results[kernel_name] = func
                succeeded += 1
            else:
                failed += 1
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Build Summary")
        logger.info(f"{'='*60}")
        logger.info(f"  Succeeded: {succeeded}/{len(kernel_names)}")
        logger.info(f"  Failed: {failed}/{len(kernel_names)}")
        logger.info(f"{'='*60}")
        
        return results
    
    def get_all_configured_kernels(self) -> List[str]:
        """获取所有已配置的 kernel 名称"""
        # 目前手动配置了 3 个
        return ["copy_low_upp", "copy_upp_low", "add_mat"]
    
    def save_manifest(self, output_file: str = "cache/kaldi_k1_manifest.json"):
        """保存构建清单"""
        manifest = {
            "total_kernels": len(self.compiled_kernels),
            "kernels": list(self.compiled_kernels.keys()),
            "cache_dir": str(self.cache_dir),
            "kaldi_repo": str(self.kaldi_repo),
        }
        
        with open(output_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"\n✓ Manifest saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Build Kaldi K1 CUDA kernels")
    parser.add_argument("--build-all", action="store_true", help="Build all configured kernels")
    parser.add_argument("--kernel", type=str, help="Build a specific kernel")
    parser.add_argument("--list", action="store_true", help="List all configured kernels")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--kaldi-repo", type=str, default="/share/project/zpy/k1_repo", 
                       help="Path to Kaldi repository")
    parser.add_argument("--cache-dir", type=str, default="./cache", help="Cache directory")
    
    args = parser.parse_args()
    
    builder = KaldiK1Builder(
        kaldi_repo_path=args.kaldi_repo,
        cache_dir=args.cache_dir,
        verbose=args.verbose
    )
    
    if args.list:
        kernels = builder.get_all_configured_kernels()
        print(f"\nConfigured Kaldi K1 kernels ({len(kernels)}):")
        for name in kernels:
            print(f"  - {name}")
        return
    
    if args.build_all:
        results = builder.build_all()
        builder.save_manifest()
        print(f"\n✓ Built {len(results)} kernels successfully!")
        print(f"\nYou can now use them in Python:")
        print(f"  from kaldi_k1_kernels import KaldiK1Kernels")
        print(f"  kaldi = KaldiK1Kernels()")
        print(f"  kaldi.copy_low_upp(A)  # Call any kernel")
    
    elif args.kernel:
        func = builder.build_kernel(args.kernel)
        if func:
            print(f"\n✓ Kernel '{args.kernel}' built successfully!")
        else:
            print(f"\n✗ Failed to build kernel '{args.kernel}'")
            sys.exit(1)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
