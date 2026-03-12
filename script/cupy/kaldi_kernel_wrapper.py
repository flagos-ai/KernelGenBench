#!/usr/bin/env python3
"""
Kaldi CUDA Kernel Wrapper using CuPy

This module provides a wrapper to load and execute Kaldi CUDA kernels
using CuPy, without needing to compile the entire Kaldi library.
"""

import re
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

try:
    import cupy as cp
except ImportError:
    raise ImportError(
        "CuPy is required for Kaldi kernel wrapper. "
        "Install with: pip install cupy-cuda11x or cupy-cuda12x"
    )

# Project paths
REPO_ROOT = Path(__file__).parent.parent.parent
KALDI_KERNELS_CU = REPO_ROOT / "kaldi/src/cudamatrix/cu-kernels.cu"
KERNEL_LIST_JSON = REPO_ROOT / "csrc/kaldi_kernels.json"


class MatrixDim:
    """Helper class to represent MatrixDim structure"""
    def __init__(self, rows: int, cols: int, stride: Optional[int] = None):
        self.rows = rows
        self.cols = cols
        self.stride = stride if stride is not None else cols
    
    def to_args(self) -> Tuple[np.int32, np.int32, np.int32]:
        """Convert to tuple of int32 arguments for kernel calls"""
        return (np.int32(self.rows), np.int32(self.cols), np.int32(self.stride))


class KaldiKernel:
    """Wrapper for a single Kaldi CUDA kernel"""
    
    def __init__(self, name: str, kernel_code: str):
        """
        Initialize a Kaldi kernel wrapper.
        
        Args:
            name: Kernel name (e.g., "copy_low_upp")
            kernel_code: CUDA source code for the kernel
        """
        self.name = name
        self.kernel_code = kernel_code
        self._module = None
        self._func_f32 = None
        self._func_f64 = None
        self._compiled = False
    
    def compile(self):
        """Compile the kernel if not already compiled"""
        if self._compiled:
            return
        
        try:
            self._module = cp.RawModule(code=self.kernel_code)
            # Try to get both float and double versions
            try:
                self._func_f32 = self._module.get_function(f'{self.name}_float')
            except:
                pass
            
            try:
                self._func_f64 = self._module.get_function(f'{self.name}_double')
            except:
                pass
            
            if self._func_f32 is None and self._func_f64 is None:
                raise RuntimeError(f"Could not find {self.name}_float or {self.name}_double")
            
            self._compiled = True
            
        except Exception as e:
            raise RuntimeError(f"Failed to compile kernel {self.name}: {e}")
    
    def __call__(self, *args, grid: Tuple, block: Tuple, dtype=np.float32):
        """
        Execute the kernel.
        
        Args:
            *args: Kernel arguments
            grid: Grid dimensions (tuple)
            block: Block dimensions (tuple)
            dtype: Data type (np.float32 or np.float64)
        """
        if not self._compiled:
            self.compile()
        
        # Select appropriate kernel based on dtype
        if dtype == np.float32:
            if self._func_f32 is None:
                raise RuntimeError(f"float32 version of {self.name} not available")
            func = self._func_f32
        elif dtype == np.float64:
            if self._func_f64 is None:
                raise RuntimeError(f"float64 version of {self.name} not available")
            func = self._func_f64
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
        
        func(grid=grid, block=block, args=args)
        cp.cuda.Device().synchronize()


class KaldiKernelLibrary:
    """
    Library of Kaldi CUDA kernels.
    
    Provides easy access to all Kaldi kernels via CuPy.
    """
    
    def __init__(self):
        self._kernels: Dict[str, KaldiKernel] = {}
        self._kernel_info: Dict[str, dict] = {}
        self._load_kernel_info()
    
    def _load_kernel_info(self):
        """Load kernel metadata from JSON"""
        if not KERNEL_LIST_JSON.exists():
            raise FileNotFoundError(f"Kernel list not found: {KERNEL_LIST_JSON}")
        
        with open(KERNEL_LIST_JSON, 'r') as f:
            self._kernel_info = json.load(f)
    
    def _extract_kernel_from_cu(self, kernel_name: str) -> str:
        """
        Extract kernel source code from cu-kernels.cu file.
        
        Args:
            kernel_name: Name of the kernel to extract
            
        Returns:
            CUDA source code as string
        """
        if not KALDI_KERNELS_CU.exists():
            raise FileNotFoundError(f"Kaldi kernels file not found: {KALDI_KERNELS_CU}")
        
        # For now, return a placeholder
        # In the full implementation, we'll parse the .cu file
        raise NotImplementedError(
            f"Automatic kernel extraction from .cu file not yet implemented. "
            f"Please use register_kernel() to manually add kernel source code."
        )
    
    def register_kernel(self, name: str, kernel_code: str):
        """
        Manually register a kernel.
        
        Args:
            name: Kernel name
            kernel_code: CUDA source code (including both template and extern "C" functions)
        """
        kernel = KaldiKernel(name, kernel_code)
        self._kernels[name] = kernel
    
    def get_kernel(self, name: str) -> KaldiKernel:
        """
        Get a kernel by name.
        
        Args:
            name: Kernel name
            
        Returns:
            KaldiKernel instance
        """
        if name not in self._kernels:
            # Try to auto-extract from .cu file
            try:
                kernel_code = self._extract_kernel_from_cu(name)
                self.register_kernel(name, kernel_code)
            except NotImplementedError:
                raise KeyError(
                    f"Kernel '{name}' not registered. "
                    f"Use register_kernel() to add it manually."
                )
        
        return self._kernels[name]
    
    def __getattr__(self, name: str):
        """Allow accessing kernels as attributes"""
        try:
            return self.get_kernel(name)
        except KeyError:
            raise AttributeError(f"Kernel '{name}' not found")
    
    def list_kernels(self) -> List[str]:
        """List all available kernel names from JSON"""
        return list(self._kernel_info.keys())
    
    def list_registered_kernels(self) -> List[str]:
        """List currently registered kernels"""
        return list(self._kernels.keys())


# Global kernel library instance
kaldi_lib = KaldiKernelLibrary()


# ============================================================================
# Pre-registered kernels (manually added)
# ============================================================================

# copy_low_upp
_COPY_LOW_UPP_CODE = """
template<typename Real>
__global__
void _copy_low_upp(Real* A, int rows, int cols, int stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i <= j || i >= rows)
    return;
  int index_1 = i * stride + j;
  int index_2 = j * stride + i;
  A[index_2] = A[index_1];
}

extern "C" __global__
void copy_low_upp_float(float* A, int rows, int cols, int stride) {
  _copy_low_upp<float>(A, rows, cols, stride);
}

extern "C" __global__
void copy_low_upp_double(double* A, int rows, int cols, int stride) {
  _copy_low_upp<double>(A, rows, cols, stride);
}
"""

kaldi_lib.register_kernel("copy_low_upp", _COPY_LOW_UPP_CODE)


# copy_upp_low
_COPY_UPP_LOW_CODE = """
template<typename Real>
__global__
static void _copy_upp_low(Real* A, int rows, int cols, int stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (j <= i || j >= rows)
    return;
  int index_1 = i * stride + j;
  int index_2 = j * stride + i;
  A[index_2] = A[index_1];
}

extern "C" __global__
void copy_upp_low_float(float* A, int rows, int cols, int stride) {
  _copy_upp_low<float>(A, rows, cols, stride);
}

extern "C" __global__
void copy_upp_low_double(double* A, int rows, int cols, int stride) {
  _copy_upp_low<double>(A, rows, cols, stride);
}
"""

kaldi_lib.register_kernel("copy_upp_low", _COPY_UPP_LOW_CODE)


# add_mat
_ADD_MAT_CODE = """
template<typename Real>
__global__
static void _add_mat(Real alpha, const Real* src, Real* dst, 
                     int rows, int cols, int stride, int src_stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // column index
  int j = blockIdx.y * blockDim.y + threadIdx.y;  // row index
  int index = i + j * stride;
  int index_src = i + j * src_stride;
  if (i < cols && j < rows)
    dst[index] = alpha * src[index_src] + dst[index];
}

extern "C" __global__
void add_mat_float(float alpha, const float* src, float* dst, 
                   int rows, int cols, int stride, int src_stride) {
  _add_mat<float>(alpha, src, dst, rows, cols, stride, src_stride);
}

extern "C" __global__
void add_mat_double(double alpha, const double* src, double* dst, 
                    int rows, int cols, int stride, int src_stride) {
  _add_mat<double>(alpha, src, dst, rows, cols, stride, src_stride);
}
"""

kaldi_lib.register_kernel("add_mat", _ADD_MAT_CODE)


if __name__ == "__main__":
    # Quick test
    print("Kaldi Kernel Library")
    print("=" * 60)
    print(f"Total kernels in JSON: {len(kaldi_lib.list_kernels())}")
    print(f"Registered kernels: {len(kaldi_lib.list_registered_kernels())}")
    print(f"\nRegistered: {kaldi_lib.list_registered_kernels()}")
    
    # Test copy_low_upp
    print("\n" + "=" * 60)
    print("Testing copy_low_upp kernel...")
    
    rows, cols = 4, 4
    A = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [2.0, 3.0, 0.0, 0.0],
        [4.0, 5.0, 6.0, 0.0],
        [7.0, 8.0, 9.0, 10.0]
    ], dtype=np.float32)
    
    A_gpu = cp.asarray(A)
    dim = MatrixDim(rows, cols)
    
    block = (16, 16)
    grid = ((cols + block[0] - 1) // block[0], 
            (rows + block[1] - 1) // block[1])
    
    kernel = kaldi_lib.copy_low_upp
    kernel(A_gpu, *dim.to_args(), grid=grid, block=block, dtype=np.float32)
    
    result = cp.asnumpy(A_gpu)
    print("Result:")
    print(result)
    
    expected = np.array([
        [1.0, 2.0, 4.0, 7.0],
        [2.0, 3.0, 5.0, 8.0],
        [4.0, 5.0, 6.0, 9.0],
        [7.0, 8.0, 9.0, 10.0]
    ], dtype=np.float32)
    
    if np.allclose(result, expected):
        print("✓ Test PASSED!")
    else:
        print("✗ Test FAILED!")
    
    print("=" * 60)
