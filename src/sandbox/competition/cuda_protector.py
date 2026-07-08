"""
Layer 3: CUDA Layer Protection.
From: triton_competition_anti_cheat_guide.md - Section 5
"""
import torch
import os


class CUDALayerProtector:
    """CUDA层保护器"""

    def __init__(self):
        self.original_graph = None
        self.original_make_graphed_callables = None

    def disable_cuda_graph(self):
        """禁用CUDA Graph"""
        if hasattr(torch, 'cuda'):
            # 保存原始函数
            if hasattr(torch.cuda, 'graph'):
                self.original_graph = torch.cuda.graph
            if hasattr(torch.cuda, 'make_graphed_callables'):
                self.original_make_graphed_callables = torch.cuda.make_graphed_callables

            # 替换为禁用版本
            torch.cuda.graph = DisabledCUDAGraphContext
            torch.cuda.make_graphed_callables = lambda *a, **k: a[0] if a else None

    def reset_cuda_state(self):
        """重置CUDA状态"""
        if torch.cuda.is_available():
            # 清空缓存
            torch.cuda.empty_cache()

            # 重置cuBLAS
            if hasattr(torch.cuda, 'blas_handle'):
                del torch.cuda.blas_handle

            # 同步
            torch.cuda.synchronize()

            # 创建新的stream
            torch.cuda.set_stream(torch.cuda.Stream())

    def disable_cuda_profiler(self):
        """禁用CUDA Profiler（防止利用profiler缓存）"""
        os.environ['CUDA_PROFILER_DISABLE'] = '1'

    def setup(self):
        """完整设置"""
        self.disable_cuda_graph()
        self.reset_cuda_state()
        self.disable_cuda_profiler()

        # 禁用TF32（防止精度相关缓存）
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def restore(self):
        """恢复原始状态"""
        if self.original_graph and hasattr(torch.cuda, 'graph'):
            torch.cuda.graph = self.original_graph
        if self.original_make_graphed_callables and hasattr(torch.cuda, 'make_graphed_callables'):
            torch.cuda.make_graphed_callables = self.original_make_graphed_callables


class DisabledCUDAGraphContext:
    """禁用的CUDA Graph上下文管理器"""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "CUDA Graph is disabled in competition mode. "
            "This prevents kernel capture and replay caching."
        )

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass
