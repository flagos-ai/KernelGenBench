"""
Kaldi K1 Kernels - 统一的 Python 接口

一键加载并使用所有 Kaldi K1 CUDA kernels。

用法:
    from kaldi_k1_kernels import KaldiK1Kernels
    
    # 加载所有 kernel
    kaldi = KaldiK1Kernels()
    
    # 使用 kernel（就像普通函数一样）
    import torch
    A = torch.randn(64, 64, device='cuda')
    kaldi.copy_low_upp(A)  # 调用 kernel
    
    # 查看所有可用 kernel
    print(kaldi.available_kernels())

作者: FlagBench Team
日期: 2026-01-16
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Callable
import logging

# 添加路径
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from build_kaldi_k1 import KaldiK1Builder

logger = logging.getLogger(__name__)


class KaldiK1Kernels:
    """
    Kaldi K1 Kernels 统一接口
    
    提供所有 Kaldi K1 CUDA kernels 的 Python 调用接口。
    
    示例:
        >>> kaldi = KaldiK1Kernels()
        >>> import torch
        >>> A = torch.randn(64, 64, device='cuda')
        >>> kaldi.copy_low_upp(A)  # 原地操作
        
        >>> dst = torch.randn(64, 64, device='cuda')
        >>> src = torch.randn(64, 64, device='cuda')
        >>> kaldi.add_mat(dst, src, 2.0)  # dst = 2.0 * src + dst
    """
    
    def __init__(
        self,
        kaldi_repo_path: str = "/share/project/zpy/k1_repo",
        cache_dir: str = None,
        auto_build: bool = True,
        verbose: bool = False
    ):
        """
        初始化 Kaldi K1 Kernels
        
        Args:
            kaldi_repo_path: Kaldi 仓库路径
            cache_dir: 缓存目录（默认：./cache）
            auto_build: 是否自动构建缺失的 kernel
            verbose: 是否显示详细日志
        """
        if cache_dir is None:
            # 使用项目根目录的 cache
            cache_dir = str(Path(__file__).parent.parent.parent / "cache")
        
        self.builder = KaldiK1Builder(
            kaldi_repo_path=kaldi_repo_path,
            cache_dir=cache_dir,
            verbose=verbose
        )
        
        self.auto_build = auto_build
        self._kernels: Dict[str, Callable] = {}
        self._manifest_file = Path(cache_dir) / "kaldi_k1_manifest.json"
        
        # 尝试加载已构建的 kernel
        if self._manifest_file.exists():
            self._load_from_manifest()
        
        # 如果自动构建，构建所有配置的 kernel
        if auto_build and len(self._kernels) == 0:
            logger.info("No kernels found, auto-building...")
            self.build_all()
    
    def _load_from_manifest(self):
        """从 manifest 加载已构建的 kernel"""
        try:
            with open(self._manifest_file, 'r') as f:
                manifest = json.load(f)
            
            kernel_names = manifest.get("kernels", [])
            logger.info(f"Loading {len(kernel_names)} kernels from manifest...")
            
            for name in kernel_names:
                try:
                    func = self.builder.compile_kernel(name)
                    self._kernels[name] = func
                except Exception as e:
                    logger.warning(f"Failed to load kernel {name}: {e}")
            
            logger.info(f"Loaded {len(self._kernels)} kernels successfully")
        
        except Exception as e:
            logger.warning(f"Failed to load manifest: {e}")
    
    def build_all(self):
        """构建所有配置的 kernel"""
        results = self.builder.build_all()
        self._kernels.update(results)
        self.builder.save_manifest(str(self._manifest_file))
        return results
    
    def build_kernel(self, kernel_name: str) -> Callable:
        """构建单个 kernel"""
        func = self.builder.build_kernel(kernel_name)
        if func:
            self._kernels[kernel_name] = func
        return func
    
    def available_kernels(self) -> List[str]:
        """返回所有可用的 kernel 名称"""
        return list(self._kernels.keys())
    
    def __getattr__(self, name: str) -> Callable:
        """
        允许通过属性访问 kernel
        
        例如: kaldi.copy_low_upp(A)
        """
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        # 检查是否已加载
        if name in self._kernels:
            return self._kernels[name]
        
        # 自动构建
        if self.auto_build:
            logger.info(f"Kernel '{name}' not loaded, building...")
            func = self.build_kernel(name)
            if func:
                return func
        
        raise AttributeError(
            f"Kernel '{name}' not available. "
            f"Available kernels: {', '.join(self.available_kernels())}"
        )
    
    def __repr__(self) -> str:
        return (
            f"KaldiK1Kernels({len(self._kernels)} kernels loaded)\n"
            f"Available: {', '.join(self.available_kernels())}"
        )
    
    # 显式定义常用 kernel 方法（提供自动补全和文档）
    
    def copy_low_upp(self, A):
        """
        Copy lower triangle to upper triangle of a square matrix (in-place).
        
        Args:
            A (torch.Tensor): Square matrix (N x N), must be on CUDA device
        
        Returns:
            None (operation is in-place)
        
        Example:
            >>> A = torch.randn(64, 64, device='cuda')
            >>> kaldi.copy_low_upp(A)
            >>> # Now A[i,j] == A[j,i] for i > j
        """
        return self._kernels['copy_low_upp'](A)
    
    def copy_upp_low(self, A):
        """
        Copy upper triangle to lower triangle of a square matrix (in-place).
        
        Args:
            A (torch.Tensor): Square matrix (N x N), must be on CUDA device
        
        Returns:
            None (operation is in-place)
        
        Example:
            >>> A = torch.randn(64, 64, device='cuda')
            >>> kaldi.copy_upp_low(A)
            >>> # Now A[i,j] == A[j,i] for j > i
        """
        return self._kernels['copy_upp_low'](A)
    
    def add_mat(self, dst, src, alpha):
        """
        Matrix addition with scalar multiplier (in-place): dst = alpha * src + dst
        
        Args:
            dst (torch.Tensor): Destination matrix, must be on CUDA device (modified in-place)
            src (torch.Tensor): Source matrix (same shape as dst), must be on CUDA device
            alpha (float): Scalar multiplier for source matrix
        
        Returns:
            None (operation is in-place)
        
        Example:
            >>> dst = torch.randn(64, 64, device='cuda')
            >>> src = torch.randn(64, 64, device='cuda')
            >>> kaldi.add_mat(dst, src, 2.0)
            >>> # Now dst = 2.0 * src + dst (original)
        """
        return self._kernels['add_mat'](dst, src, alpha)


# 便捷函数：快速加载
def load_kaldi_kernels(auto_build: bool = True, verbose: bool = False) -> KaldiK1Kernels:
    """
    快速加载 Kaldi K1 kernels
    
    Args:
        auto_build: 是否自动构建缺失的 kernel
        verbose: 是否显示详细日志
    
    Returns:
        KaldiK1Kernels 实例
    
    Example:
        >>> kaldi = load_kaldi_kernels()
        >>> kaldi.copy_low_upp(A)
    """
    return KaldiK1Kernels(auto_build=auto_build, verbose=verbose)


if __name__ == "__main__":
    # 测试
    print("="*60)
    print("Kaldi K1 Kernels - Test")
    print("="*60)
    
    # 加载 kernels
    kaldi = KaldiK1Kernels(auto_build=True, verbose=True)
    
    print(f"\n{kaldi}")
    
    # 测试
    import torch
    
    print("\n" + "="*60)
    print("Test 1: copy_low_upp")
    print("="*60)
    A = torch.randn(32, 32, device='cuda')
    print(f"Before: A[0,1] = {A[0,1].item():.4f}, A[1,0] = {A[1,0].item():.4f}")
    kaldi.copy_low_upp(A)
    print(f"After:  A[0,1] = {A[0,1].item():.4f}, A[1,0] = {A[1,0].item():.4f}")
    print("✓ PASSED" if torch.allclose(A, A.t()) else "✗ FAILED")
    
    print("\n" + "="*60)
    print("Test 2: add_mat")
    print("="*60)
    dst = torch.ones(32, 32, device='cuda')
    src = torch.ones(32, 32, device='cuda') * 2
    alpha = 3.0
    kaldi.add_mat(dst, src, alpha)
    expected = 1 + 3.0 * 2  # 7.0
    print(f"Result: dst[0,0] = {dst[0,0].item():.4f} (expected: {expected:.4f})")
    print("✓ PASSED" if torch.allclose(dst, torch.ones(32, 32, device='cuda') * expected) else "✗ FAILED")
    
    print("\n" + "="*60)
    print("✓ All tests completed!")
    print("="*60)
