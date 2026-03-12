#!/usr/bin/env python3
"""
Test script for cuBLAS baseline module
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch

def run_all_tests():
    print("=" * 80)
    print("CuBLAS Baseline - Running Tests")
    print("=" * 80)
    print()
    
    # Import compiled module
    from script.cuda_baseline.cublas_batch_builder import build_default_baseline
    module = build_default_baseline()
    
    print(f"Module: {module}")
    print(f"cuBLAS Version: {module.CUBLAS_VERSION}")
    print()
    
    # Test 1: sgemm (Single-precision matrix multiplication)
    print("Test 1: sgemm (float32 matrix multiplication)")
    M, K, N = 64, 32, 64
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)
    C_torch = torch.randn(M, N, device='cuda', dtype=torch.float32)
    C_ref = C_torch.clone()
    
    alpha, beta = 2.0, 0.5
    torch.addmm(C_ref, A, B, beta=beta, alpha=alpha)
    
    C_cublas = C_torch.clone()
    module.sgemm(C_cublas, alpha, A, B, beta)
    
    assert torch.allclose(C_cublas, C_ref, rtol=1e-4), "SGEMM test failed"
    print("✓ sgemm test passed")
    print()
    
    # Test 2: dgemm (Double-precision matrix multiplication)
    print("Test 2: dgemm (float64 matrix multiplication)")
    A64 = torch.randn(M, K, device='cuda', dtype=torch.float64)
    B64 = torch.randn(K, N, device='cuda', dtype=torch.float64)
    C64_torch = torch.randn(M, N, device='cuda', dtype=torch.float64)
    C64_ref = C64_torch.clone()
    
    torch.addmm(C64_ref, A64, B64, beta=beta, alpha=alpha)
    
    C64_cublas = C64_torch.clone()
    module.dgemm(C64_cublas, alpha, A64, B64, beta)
    
    assert torch.allclose(C64_cublas, C64_ref, rtol=1e-8), "DGEMM test failed"
    print("✓ dgemm test passed")
    print()
    
    # Test 3: saxpy (Single-precision vector addition)
    print("Test 3: saxpy (float32 vector addition)")
    n = 1024
    X = torch.randn(n, device='cuda', dtype=torch.float32)
    Y = torch.randn(n, device='cuda', dtype=torch.float32)
    Y_ref = Y.clone()
    
    alpha = 2.0
    Y_ref.add_(X, alpha=alpha)
    
    module.saxpy(Y, alpha, X)
    
    assert torch.allclose(Y, Y_ref, rtol=1e-4), "SAXPY test failed"
    print("✓ saxpy test passed")
    print()
    
    # Test 4: sscal (Single-precision vector scaling)
    print("Test 4: sscal (float32 vector scaling)")
    X = torch.randn(n, device='cuda', dtype=torch.float32)
    X_ref = X.clone()
    
    alpha = 2.0
    X_ref.mul_(alpha)
    
    module.sscal(X, alpha)
    
    assert torch.allclose(X, X_ref, rtol=1e-4), "SSCAL test failed"
    print("✓ sscal test passed")
    print()
    
    # Test 5: sdot (Single-precision dot product)
    print("Test 5: sdot (float32 dot product)")
    X = torch.randn(n, device='cuda', dtype=torch.float32)
    Y = torch.randn(n, device='cuda', dtype=torch.float32)
    
    ref = torch.dot(X, Y)
    result = module.sdot(X, Y)
    
    assert torch.allclose(result, ref, rtol=1e-4), "SDOT test failed"
    print("✓ sdot test passed")
    print()
    
    # Test 6: hgemm (Half-precision matrix multiplication)
    print("Test 6: hgemm (float16 matrix multiplication)")
    A16 = torch.randn(M, K, device='cuda', dtype=torch.float16)
    B16 = torch.randn(K, N, device='cuda', dtype=torch.float16)
    C16_torch = torch.randn(M, N, device='cuda', dtype=torch.float16)
    C16_ref = C16_torch.clone()
    
    # Use float32 for reference computation
    A16_f = A16.float()
    B16_f = B16.float()
    C16_ref_f = C16_ref.float()
    torch.addmm(C16_ref_f, A16_f, B16_f, beta=beta, alpha=alpha)
    C16_ref = C16_ref_f.half()
    
    C16_cublas = C16_torch.clone()
    module.hgemm(C16_cublas, alpha, A16, B16, beta)
    
    assert torch.allclose(C16_cublas, C16_ref, rtol=1e-2, atol=1e-2), "HGEMM test failed"
    print("✓ hgemm test passed")
    print()
    
    print("=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)


if __name__ == '__main__':
    run_all_tests()
