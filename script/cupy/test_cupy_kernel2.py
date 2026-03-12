#!/usr/bin/env python3
"""
CuPy Validation - Test 2: More complex kernels

Testing:
1. add_mat - Matrix addition with scalar (dst = alpha * src + dst)
2. copy_upp_low - Copy upper triangle to lower triangle
"""

import sys
import numpy as np

try:
    import cupy as cp
except ImportError:
    print("ERROR: CuPy not installed!")
    print("Install: pip install cupy-cuda11x  (or cupy-cuda12x)")
    sys.exit(1)

# ============================================================================
# Test 1: add_mat kernel
# ============================================================================

add_mat_kernel = """
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

print("="*60)
print("CuPy Validation - Test 2: Complex Kernels")
print("="*60)

# Compile add_mat kernel
print("\n[Test 1] add_mat kernel")
print("-" * 60)
print("[1.1] Compiling kernel...")
try:
    module1 = cp.RawModule(code=add_mat_kernel)
    add_mat_f32 = module1.get_function('add_mat_float')
    print("OK - Kernel compiled successfully!")
except Exception as e:
    print(f"ERROR - Compilation failed: {e}")
    sys.exit(1)

# Test add_mat
print("\n[1.2] Testing add_mat...")
rows, cols = 3, 4
alpha = 2.5

# Source matrix
src = np.array([
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0],
    [9.0, 10.0, 11.0, 12.0]
], dtype=np.float32)

# Destination matrix (initial values)
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

# Copy to GPU
src_gpu = cp.asarray(src)
dst_gpu = cp.asarray(dst)

# Launch kernel
block = (16, 16)
grid = ((cols + block[0] - 1) // block[0], 
        (rows + block[1] - 1) // block[1])

add_mat_f32(
    grid=grid,
    block=block,
    args=(np.float32(alpha), src_gpu, dst_gpu, 
          np.int32(rows), np.int32(cols), np.int32(cols), np.int32(cols))
)

cp.cuda.Device().synchronize()

# Verify result
dst_result = cp.asnumpy(dst_gpu)
print("\nDestination matrix (after):")
print(dst_result)

# Expected: dst = alpha * src + dst_original
expected = alpha * src + dst
print("\nExpected matrix:")
print(expected)

if np.allclose(dst_result, expected):
    print("\n✓ Test 1 PASSED!")
else:
    print(f"\n✗ Test 1 FAILED! Max difference: {np.max(np.abs(dst_result - expected))}")
    sys.exit(1)

# ============================================================================
# Test 2: copy_upp_low kernel
# ============================================================================

copy_upp_low_kernel = """
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

print("\n" + "="*60)
print("[Test 2] copy_upp_low kernel")
print("-" * 60)
print("[2.1] Compiling kernel...")
try:
    module2 = cp.RawModule(code=copy_upp_low_kernel)
    copy_upp_low_f32 = module2.get_function('copy_upp_low_float')
    print("OK - Kernel compiled successfully!")
except Exception as e:
    print(f"ERROR - Compilation failed: {e}")
    sys.exit(1)

# Test copy_upp_low
print("\n[2.2] Testing copy_upp_low...")
rows, cols = 4, 4

# Create matrix with upper triangle filled, lower triangle zeros
A = np.array([
    [1.0, 2.0, 4.0, 7.0],
    [0.0, 3.0, 5.0, 8.0],
    [0.0, 0.0, 6.0, 9.0],
    [0.0, 0.0, 0.0, 10.0]
], dtype=np.float32)

print("Input matrix (upper triangle filled):")
print(A)

# Copy to GPU
A_gpu = cp.asarray(A)

# Launch kernel
block = (16, 16)
grid = ((cols + block[0] - 1) // block[0], 
        (rows + block[1] - 1) // block[1])

copy_upp_low_f32(
    grid=grid,
    block=block,
    args=(A_gpu, np.int32(rows), np.int32(cols), np.int32(cols))
)

cp.cuda.Device().synchronize()

# Verify result
A_result = cp.asnumpy(A_gpu)
print("\nOutput matrix (after copying upper to lower):")
print(A_result)

# Expected: lower triangle should equal upper triangle's transpose
expected = np.array([
    [1.0, 2.0, 4.0, 7.0],
    [2.0, 3.0, 5.0, 8.0],
    [4.0, 5.0, 6.0, 9.0],
    [7.0, 8.0, 9.0, 10.0]
], dtype=np.float32)

print("\nExpected matrix:")
print(expected)

if np.allclose(A_result, expected):
    print("\n✓ Test 2 PASSED!")
else:
    print(f"\n✗ Test 2 FAILED! Max difference: {np.max(np.abs(A_result - expected))}")
    sys.exit(1)

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*60)
print("✓✓✓ ALL TESTS PASSED! ✓✓✓")
print("="*60)
print("\nKey findings:")
print("  1. CuPy can compile kernels with multiple parameters")
print("  2. Scalar parameters (alpha) work correctly")
print("  3. Read-write patterns (dst = f(src, dst)) work")
print("  4. Complex indexing patterns work")
print("\nNext steps:")
print("  - Create generic wrapper class for all Kaldi kernels")
print("  - Auto-generate wrappers for 129 kernels")
print("  - Integrate into FlagBench test generation")
print("="*60)
