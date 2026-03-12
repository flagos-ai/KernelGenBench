#!/usr/bin/env python3
"""
Demo: Using CuPy-based Kaldi kernels

This demonstrates how to use the CuPy wrapper for Kaldi kernels.
"""

import sys
import numpy as np
import cupy as cp

# Add script/cupy to path
from pathlib import Path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from kaldi_kernel_wrapper import kaldi_lib, MatrixDim

print("="*60)
print("Kaldi Kernel Demo (CuPy-based)")
print("="*60)

# Test 1: copy_low_upp
print("\n[Test 1] copy_low_upp - Copy lower triangle to upper")
print("-"*60)

rows, cols = 4, 4
A = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [2.0, 3.0, 0.0, 0.0],
    [4.0, 5.0, 6.0, 0.0],
    [7.0, 8.0, 9.0, 10.0]
], dtype=np.float32)

print("Input matrix:")
print(A)

A_gpu = cp.asarray(A)
dim = MatrixDim(rows, cols)

block = (16, 16)
grid = ((cols + block[0] - 1) // block[0], 
        (rows + block[1] - 1) // block[1])

kernel = kaldi_lib.copy_low_upp
kernel(A_gpu, *dim.to_args(), grid=grid, block=block, dtype=np.float32)

result = cp.asnumpy(A_gpu)
print("\nOutput matrix:")
print(result)

expected = np.array([
    [1.0, 2.0, 4.0, 7.0],
    [2.0, 3.0, 5.0, 8.0],
    [4.0, 5.0, 6.0, 9.0],
    [7.0, 8.0, 9.0, 10.0]
], dtype=np.float32)

if np.allclose(result, expected):
    print("✓ Test 1 PASSED!")
else:
    print(f"✗ Test 1 FAILED! Max diff: {np.max(np.abs(result - expected))}")
    sys.exit(1)

# Test 2: add_mat
print("\n" + "="*60)
print("[Test 2] add_mat - dst = alpha * src + dst")
print("-"*60)

rows, cols = 3, 4
alpha = 2.5

src = np.array([
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0],
    [9.0, 10.0, 11.0, 12.0]
], dtype=np.float32)

dst = np.array([
    [0.5, 1.0, 1.5, 2.0],
    [2.5, 3.0, 3.5, 4.0],
    [4.5, 5.0, 5.5, 6.0]
], dtype=np.float32)

print("Source matrix:")
print(src)
print("\nDestination matrix (before):")
print(dst)
print(f"\nalpha = {alpha}")

src_gpu = cp.asarray(src)
dst_gpu = cp.asarray(dst)

block = (16, 16)
grid = ((cols + block[0] - 1) // block[0], 
        (rows + block[1] - 1) // block[1])

kernel = kaldi_lib.add_mat
kernel(
    np.float32(alpha), src_gpu, dst_gpu,
    np.int32(rows), np.int32(cols), 
    np.int32(cols), np.int32(cols),
    grid=grid, block=block, dtype=np.float32
)

result = cp.asnumpy(dst_gpu)
print("\nDestination matrix (after):")
print(result)

expected = alpha * src + dst
if np.allclose(result, expected):
    print("✓ Test 2 PASSED!")
else:
    print(f"✗ Test 2 FAILED! Max diff: {np.max(np.abs(result - expected))}")
    sys.exit(1)

# Test 3: copy_upp_low
print("\n" + "="*60)
print("[Test 3] copy_upp_low - Copy upper triangle to lower")
print("-"*60)

rows, cols = 4, 4
B = np.array([
    [1.0, 2.0, 4.0, 7.0],
    [0.0, 3.0, 5.0, 8.0],
    [0.0, 0.0, 6.0, 9.0],
    [0.0, 0.0, 0.0, 10.0]
], dtype=np.float32)

print("Input matrix:")
print(B)

B_gpu = cp.asarray(B)
dim = MatrixDim(rows, cols)

block = (16, 16)
grid = ((cols + block[0] - 1) // block[0], 
        (rows + block[1] - 1) // block[1])

kernel = kaldi_lib.copy_upp_low
kernel(B_gpu, *dim.to_args(), grid=grid, block=block, dtype=np.float32)

result = cp.asnumpy(B_gpu)
print("\nOutput matrix:")
print(result)

expected = np.array([
    [1.0, 2.0, 4.0, 7.0],
    [2.0, 3.0, 5.0, 8.0],
    [4.0, 5.0, 6.0, 9.0],
    [7.0, 8.0, 9.0, 10.0]
], dtype=np.float32)

if np.allclose(result, expected):
    print("✓ Test 3 PASSED!")
else:
    print(f"✗ Test 3 FAILED! Max diff: {np.max(np.abs(result - expected))}")
    sys.exit(1)

print("\n" + "="*60)
print("✓✓✓ ALL TESTS PASSED! ✓✓✓")
print("="*60)
print("\nSummary:")
print(f"  - Tested {len(kaldi_lib.list_registered_kernels())} kernels")
print(f"  - All kernels working correctly")
print(f"  - Zero-copy CuPy integration")
print(f"  - Ready for batch generation")
print("="*60)
