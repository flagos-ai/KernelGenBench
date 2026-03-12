#!/usr/bin/env python3
"""
Comprehensive test: Compare CuPy baseline vs Triton implementation
for K1 CUDA kernels (copy_low_upp, copy_upp_low, add_mat)
"""

import sys
import os
from pathlib import Path

# Add paths
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "triton_kernels_k1"))

import torch
import numpy as np

# Import CuPy baseline (PyTorch custom ops)
from flagbench.ops import kaldi_ops

# Import Triton implementations
from copy_low_upp_kernel import copy_low_upp as triton_copy_low_upp
from copy_upp_low_kernel import copy_upp_low as triton_copy_upp_low
from add_mat_kernel import add_mat as triton_add_mat


def test_copy_low_upp():
    """Test copy_low_upp: CuPy baseline vs Triton"""
    print("\n" + "="*70)
    print("TEST 1: copy_low_upp - Copy lower triangle to upper triangle")
    print("="*70)
    
    # Test different matrix sizes
    sizes = [4, 8, 16, 32, 64]
    all_passed = True
    
    for N in sizes:
        # Create test matrix (lower triangle filled, upper triangle zeros)
        A_baseline = torch.tril(torch.randn(N, N, device='cuda', dtype=torch.float32))
        A_triton = A_baseline.clone()
        
        # Run CuPy baseline
        torch.ops.kaldi.copy_low_upp(A_baseline)
        
        # Run Triton
        triton_copy_low_upp(A_triton)
        
        # Compare results
        passed = torch.allclose(A_baseline, A_triton, rtol=1e-5, atol=1e-6)
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  N={N:3d}: {status}")
        
        if not passed:
            all_passed = False
            max_diff = torch.max(torch.abs(A_baseline - A_triton)).item()
            print(f"         Max difference: {max_diff:.2e}")
    
    return all_passed


def test_copy_upp_low():
    """Test copy_upp_low: CuPy baseline vs Triton"""
    print("\n" + "="*70)
    print("TEST 2: copy_upp_low - Copy upper triangle to lower triangle")
    print("="*70)
    
    # Test different matrix sizes
    sizes = [4, 8, 16, 32, 64]
    all_passed = True
    
    for N in sizes:
        # Create test matrix (upper triangle filled, lower triangle zeros)
        A_baseline = torch.triu(torch.randn(N, N, device='cuda', dtype=torch.float32))
        A_triton = A_baseline.clone()
        
        # Run CuPy baseline
        torch.ops.kaldi.copy_upp_low(A_baseline)
        
        # Run Triton
        triton_copy_upp_low(A_triton)
        
        # Compare results
        passed = torch.allclose(A_baseline, A_triton, rtol=1e-5, atol=1e-6)
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  N={N:3d}: {status}")
        
        if not passed:
            all_passed = False
            max_diff = torch.max(torch.abs(A_baseline - A_triton)).item()
            print(f"         Max difference: {max_diff:.2e}")
    
    return all_passed


def test_add_mat():
    """Test add_mat: CuPy baseline vs Triton"""
    print("\n" + "="*70)
    print("TEST 3: add_mat - Matrix addition with scalar: dst = alpha * src + dst")
    print("="*70)
    
    # Test different matrix sizes and alpha values
    test_configs = [
        (4, 4, 2.0),
        (8, 16, 0.5),
        (16, 32, -1.0),
        (32, 64, 3.14),
        (64, 128, 0.001),
    ]
    all_passed = True
    
    for M, N, alpha in test_configs:
        # Create test matrices
        src = torch.randn(M, N, device='cuda', dtype=torch.float32)
        dst_baseline = torch.randn(M, N, device='cuda', dtype=torch.float32)
        dst_triton = dst_baseline.clone()
        
        # Run CuPy baseline
        torch.ops.kaldi.add_mat(dst_baseline, src, alpha)
        
        # Run Triton
        triton_add_mat(dst_triton, src, alpha)
        
        # Compare results
        passed = torch.allclose(dst_baseline, dst_triton, rtol=1e-5, atol=1e-6)
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  M={M:3d}, N={N:3d}, alpha={alpha:6.3f}: {status}")
        
        if not passed:
            all_passed = False
            max_diff = torch.max(torch.abs(dst_baseline - dst_triton)).item()
            print(f"         Max difference: {max_diff:.2e}")
    
    return all_passed


def benchmark_comparison():
    """Simple performance comparison between CuPy baseline and Triton"""
    print("\n" + "="*70)
    print("PERFORMANCE BENCHMARK (1000 iterations)")
    print("="*70)
    
    import time
    
    N = 512
    num_iters = 1000
    
    # Benchmark copy_low_upp
    print(f"\n1. copy_low_upp (N={N})")
    A_baseline = torch.tril(torch.randn(N, N, device='cuda', dtype=torch.float32))
    A_triton = A_baseline.clone()
    
    # Warmup
    for _ in range(10):
        torch.ops.kaldi.copy_low_upp(A_baseline.clone())
        triton_copy_low_upp(A_triton.clone())
    
    # Benchmark CuPy baseline
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        A = torch.tril(torch.randn(N, N, device='cuda', dtype=torch.float32))
        torch.ops.kaldi.copy_low_upp(A)
    torch.cuda.synchronize()
    cupy_time = time.time() - start
    
    # Benchmark Triton
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        A = torch.tril(torch.randn(N, N, device='cuda', dtype=torch.float32))
        triton_copy_low_upp(A)
    torch.cuda.synchronize()
    triton_time = time.time() - start
    
    print(f"   CuPy baseline: {cupy_time:.3f}s ({cupy_time/num_iters*1000:.3f}ms per iter)")
    print(f"   Triton:        {triton_time:.3f}s ({triton_time/num_iters*1000:.3f}ms per iter)")
    print(f"   Speedup:       {cupy_time/triton_time:.2f}x")
    
    # Benchmark add_mat
    print(f"\n2. add_mat (M={N}, N={N})")
    src = torch.randn(N, N, device='cuda', dtype=torch.float32)
    dst_baseline = torch.randn(N, N, device='cuda', dtype=torch.float32)
    dst_triton = dst_baseline.clone()
    alpha = 2.0
    
    # Warmup
    for _ in range(10):
        torch.ops.kaldi.add_mat(dst_baseline.clone(), src, alpha)
        triton_add_mat(dst_triton.clone(), src, alpha)
    
    # Benchmark CuPy baseline
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        dst = torch.randn(N, N, device='cuda', dtype=torch.float32)
        torch.ops.kaldi.add_mat(dst, src, alpha)
    torch.cuda.synchronize()
    cupy_time = time.time() - start
    
    # Benchmark Triton
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        dst = torch.randn(N, N, device='cuda', dtype=torch.float32)
        triton_add_mat(dst, src, alpha)
    torch.cuda.synchronize()
    triton_time = time.time() - start
    
    print(f"   CuPy baseline: {cupy_time:.3f}s ({cupy_time/num_iters*1000:.3f}ms per iter)")
    print(f"   Triton:        {triton_time:.3f}s ({triton_time/num_iters*1000:.3f}ms per iter)")
    print(f"   Speedup:       {cupy_time/triton_time:.2f}x")


def main():
    print("="*70)
    print("K1 CUDA Kernels: CuPy Baseline vs Triton Implementation")
    print("="*70)
    
    # Run correctness tests
    test1_passed = test_copy_low_upp()
    test2_passed = test_copy_upp_low()
    test3_passed = test_add_mat()
    
    # Summary
    print("\n" + "="*70)
    print("CORRECTNESS TEST SUMMARY")
    print("="*70)
    print(f"  copy_low_upp: {'✓ PASS' if test1_passed else '✗ FAIL'}")
    print(f"  copy_upp_low: {'✓ PASS' if test2_passed else '✗ FAIL'}")
    print(f"  add_mat:      {'✓ PASS' if test3_passed else '✗ FAIL'}")
    
    all_passed = test1_passed and test2_passed and test3_passed
    
    if all_passed:
        print("\n✓ All correctness tests PASSED!")
        
        # Run performance benchmark
        benchmark_comparison()
    else:
        print("\n✗ Some tests FAILED!")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("✓ FlagBench K1 Integration: Full Pipeline COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  1. The three kernels are now ready for FlagBench integration")
    print("  2. CuPy baseline (torch.ops.kaldi.*) can be used as ground truth")
    print("  3. Triton implementations are available in triton_kernels_k1/")
    print("  4. You can now generate more complex kernels using LLM")


if __name__ == "__main__":
    main()
