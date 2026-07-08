"""
Tests for Layer 4: CUDA Protection (cuda_protector.py).

Verifies that:
1. CUDA Graph is disabled
2. TF32 is disabled
3. CUDA state is reset
4. Protection can be restored
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from sandbox.competition.cuda_protector import (
    CUDALayerProtector,
    DisabledCUDAGraphContext,
)


class TestCUDALayerProtector:
    """Test the CUDA layer protection."""

    def test_create_protector(self):
        """CUDALayerProtector can be created."""
        protector = CUDALayerProtector()
        assert protector is not None

    def test_setup_runs(self):
        """setup() runs without error."""
        protector = CUDALayerProtector()
        protector.setup()
        protector.restore()

    def test_disable_cuda_graph(self):
        """CUDA Graph is disabled after setup."""
        import torch
        protector = CUDALayerProtector()
        protector.setup()
        try:
            # torch.cuda.graph should be DisabledCUDAGraphContext
            assert torch.cuda.graph is DisabledCUDAGraphContext
        finally:
            protector.restore()

    def test_disable_tf32(self):
        """TF32 is disabled after setup."""
        import torch
        protector = CUDALayerProtector()
        protector.setup()
        try:
            assert torch.backends.cuda.matmul.allow_tf32 is False
            assert torch.backends.cudnn.allow_tf32 is False
        finally:
            protector.restore()

    def test_restore_cuda_graph(self):
        """restore() brings back original CUDA Graph."""
        import torch
        protector = CUDALayerProtector()
        protector.setup()
        protector.restore()
        # After restore, torch.cuda.graph should be back to original
        assert torch.cuda.graph is not DisabledCUDAGraphContext


class TestDisabledCUDAGraph:
    """Test the disabled CUDA Graph placeholder."""

    def test_cuda_graph_raises(self):
        """Creating a CUDA Graph raises RuntimeError."""
        with pytest.raises(RuntimeError, match="disabled"):
            DisabledCUDAGraphContext()

    def test_context_manager_raises(self):
        """Using CUDA Graph as context manager raises."""
        with pytest.raises(RuntimeError):
            with DisabledCUDAGraphContext():
                pass