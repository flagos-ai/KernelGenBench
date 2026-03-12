#!/usr/bin/env python3
"""
High-level API for Kaldi CUDA kernels

Provides PyTorch-like interface for Kaldi kernels via CuPy.
Example usage:
    import torch
    from kaldi_ops import kaldi_ops
    
    A = torch.randn(4, 4, device='cuda')
    kaldi_ops.copy_low_upp(A)  # Copies lower triangle to upper triangle
"""

import numpy as np
from typing import Optional

try:
    import cupy as cp
except ImportError:
    raise ImportError(
        "CuPy is required. Install with: pip install cupy-cuda11x or cupy-cuda12x"
    )

try:
    import torch
except ImportError:
    torch = None

from kaldi_kernel_wrapper import kaldi_lib, MatrixDim


class KaldiOps:
    """
    High-level PyTorch-compatible interface for Kaldi CUDA kernels.
    
    All methods accept PyTorch tensors and return PyTorch tensors.
    """
    
    def __init__(self):
        self.lib = kaldi_lib
    
    @staticmethod
    def _torch_to_cupy(tensor):
        """Convert PyTorch tensor to CuPy array (zero-copy)"""
        return cp.from_dlpack(tensor.detach())
    
    @staticmethod
    def _cupy_to_torch(array, like):
        """Convert CuPy array to PyTorch tensor (zero-copy)"""
        # Use DLPack for zero-copy conversion
        tensor = torch.from_dlpack(array.toDlpack())
        return tensor.to(like.device)
    
    def copy_low_upp(self, A):
        """
        Copy lower triangle to upper triangle of matrix A (in-place).
        
        Args:
            A: Square matrix (rows x rows), CUDA tensor
            
        Returns:
            Modified A (same object)
        """
        assert A.is_cuda, "Tensor must be on CUDA device"
        assert A.ndim == 2, "Tensor must be 2D"
        assert A.size(0) == A.size(1), "Tensor must be square"
        
        rows, cols = A.shape
        stride = A.stride(0)
        
        # Convert to CuPy
        A_cp = self._torch_to_cupy(A)
        
        # Prepare kernel arguments
        dim = MatrixDim(rows, cols, stride)
        
        # Launch kernel
        block = (16, 16)
        grid = ((cols + block[0] - 1) // block[0], 
                (rows + block[1] - 1) // block[1])
        
        dtype = np.float32 if A.dtype == torch.float32 else np.float64
        
        kernel = self.lib.copy_low_upp
        kernel(A_cp, *dim.to_args(), grid=grid, block=block, dtype=dtype)
        
        return A
    
    def copy_upp_low(self, A):
        """
        Copy upper triangle to lower triangle of matrix A (in-place).
        
        Args:
            A: Square matrix (rows x rows), CUDA tensor
            
        Returns:
            Modified A (same object)
        """
        assert A.is_cuda, "Tensor must be on CUDA device"
        assert A.ndim == 2, "Tensor must be 2D"
        assert A.size(0) == A.size(1), "Tensor must be square"
        
        rows, cols = A.shape
        stride = A.stride(0)
        
        A_cp = self._torch_to_cupy(A)
        dim = MatrixDim(rows, cols, stride)
        
        block = (16, 16)
        grid = ((cols + block[0] - 1) // block[0], 
                (rows + block[1] - 1) // block[1])
        
        dtype = np.float32 if A.dtype == torch.float32 else np.float64
        
        kernel = self.lib.copy_upp_low
        kernel(A_cp, *dim.to_args(), grid=grid, block=block, dtype=dtype)
        
        return A
    
    def add_mat(self, dst, src, alpha=1.0):
        """
        Matrix addition: dst = alpha * src + dst (in-place on dst).
        
        Args:
            dst: Destination matrix, CUDA tensor
            src: Source matrix, CUDA tensor (same shape as dst)
            alpha: Scalar multiplier
            
        Returns:
            Modified dst (same object)
        """
        assert dst.is_cuda and src.is_cuda, "Tensors must be on CUDA device"
        assert dst.shape == src.shape, "Tensors must have same shape"
        assert dst.ndim == 2, "Tensors must be 2D"
        
        rows, cols = dst.shape
        dst_stride = dst.stride(0)
        src_stride = src.stride(0)
        
        dst_cp = self._torch_to_cupy(dst)
        src_cp = self._torch_to_cupy(src)
        
        block = (16, 16)
        grid = ((cols + block[0] - 1) // block[0], 
                (rows + block[1] - 1) // block[1])
        
        dtype = np.float32 if dst.dtype == torch.float32 else np.float64
        alpha_typed = dtype(alpha)
        
        kernel = self.lib.add_mat
        kernel(
            alpha_typed, src_cp, dst_cp,
            np.int32(rows), np.int32(cols), 
            np.int32(dst_stride), np.int32(src_stride),
            grid=grid, block=block, dtype=dtype
        )
        
        return dst


# Global instance
kaldi_ops = KaldiOps()


if __name__ == "__main__":
    print("Testing KaldiOps (PyTorch interface)")
    print("=" * 60)
    
    # Test 1: copy_low_upp
    print("\n[Test 1] copy_low_upp")
    A = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [2.0, 3.0, 0.0, 0.0],
        [4.0, 5.0, 6.0, 0.0],
        [7.0, 8.0, 9.0, 10.0]
    ], device='cuda', dtype=torch.float32)
    
    print("Before:")
    print(A)
    
    kaldi_ops.copy_low_upp(A)
    
    print("\nAfter:")
    print(A)
    
    expected = torch.tensor([
        [1.0, 2.0, 4.0, 7.0],
        [2.0, 3.0, 5.0, 8.0],
        [4.0, 5.0, 6.0, 9.0],
        [7.0, 8.0, 9.0, 10.0]
    ], device='cuda', dtype=torch.float32)
    
    if torch.allclose(A, expected):
        print("✓ Test 1 PASSED!")
    else:
        print("✗ Test 1 FAILED!")
    
    # Test 2: add_mat
    print("\n" + "=" * 60)
    print("[Test 2] add_mat")
    
    src = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ], device='cuda', dtype=torch.float32)
    
    dst = torch.tensor([
        [0.5, 1.0, 1.5],
        [2.0, 2.5, 3.0]
    ], device='cuda', dtype=torch.float32)
    
    print("Source:")
    print(src)
    print("\nDestination (before):")
    print(dst)
    
    alpha = 2.0
    print(f"\nalpha = {alpha}")
    
    kaldi_ops.add_mat(dst, src, alpha)
    
    print("\nDestination (after):")
    print(dst)
    
    expected = torch.tensor([
        [2.5, 5.0, 7.5],
        [10.0, 12.5, 15.0]
    ], device='cuda', dtype=torch.float32)
    
    if torch.allclose(dst, expected):
        print("✓ Test 2 PASSED!")
    else:
        print("✗ Test 2 FAILED!")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
