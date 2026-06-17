"""
CUDA Layer Protection for KernelGenBench sandbox.

Disables/protects:
- CUDA Graph (capture + replay)
- cuBLAS/cuDNN state (reset between tests)
- TF32 (prevents precision-related cache variance)
- CUDA Profiler (prevents profiler-based caching)
"""
import torch
import logging

logger = logging.getLogger(__name__)


class _DisabledGraphCtx:
    """Context manager that raises when CUDA Graph is used."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "CUDA Graph is disabled in competition mode. "
            "This prevents kernel capture and replay caching."
        )

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


class CUDALayerProtector:
    """Protects CUDA layer from cache-based cheating."""

    def __init__(self):
        self._original_graph = None
        self._original_make_graphed = None
        self._original_tf32_matmul = None
        self._original_tf32_cudnn = None

    def disable_cuda_graph(self):
        if not hasattr(torch, "cuda"):
            return
        if hasattr(torch.cuda, "graph"):
            self._original_graph = torch.cuda.graph
            torch.cuda.graph = _DisabledGraphCtx
            logger.debug("CUDALayerProtector: CUDA Graph disabled")
        if hasattr(torch.cuda, "make_graphed_callables"):
            self._original_make_graphed = torch.cuda.make_graphed_callables
            torch.cuda.make_graphed_callables = lambda *a, **k: a[0] if a else None

    def disable_tf32(self):
        """Disable TF32 to prevent precision-dependent caching."""
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            self._original_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            self._original_tf32_cudnn = torch.backends.cudnn.allow_tf32
            torch.backends.cudnn.allow_tf32 = False
        logger.debug("CUDALayerProtector: TF32 disabled")

    def reset_cuda_state(self):
        """Reset CUDA state between tests."""
        if not torch.cuda.is_available():
            return
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.set_stream(torch.cuda.Stream())
        logger.debug("CUDALayerProtector: CUDA state reset")

    def disable_profiler(self):
        import os
        os.environ["CUDA_PROFILER_DISABLE"] = "1"

    def setup(self):
        self.disable_cuda_graph()
        self.disable_tf32()
        self.reset_cuda_state()
        self.disable_profiler()

    def restore(self):
        if self._original_graph is not None and hasattr(torch.cuda, "graph"):
            torch.cuda.graph = self._original_graph
        if self._original_make_graphed is not None and hasattr(torch.cuda, "make_graphed_callables"):
            torch.cuda.make_graphed_callables = self._original_make_graphed
        if self._original_tf32_matmul is not None:
            torch.backends.cuda.matmul.allow_tf32 = self._original_tf32_matmul
        if self._original_tf32_cudnn is not None:
            torch.backends.cudnn.allow_tf32 = self._original_tf32_cudnn
        logger.debug("CUDALayerProtector: restored")

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, *args):
        self.restore()
