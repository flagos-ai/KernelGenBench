#!/usr/bin/env python3
"""
Temporary script to test a single cuBLAS baseline function.

This script tests sgemm (which passed baseline verification) to ensure
the testing infrastructure works correctly.

Usage:
    python script/test_single_cublas.py
"""

import os
import sys
from pathlib import Path

# Setup environment
os.environ['DISPATCH_TORCH_LIB'] = '0'
os.environ['FLAGBENCH_UPCAST'] = '0'

# Add src to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
# Don't import flagbench main package - it has dependency issues
# import flagbench
import flagbench.baseline
import flagbench.triton
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.register import REGISTERED_OPS

def test_sgemm_simple():
    """Simple test for sgemm: C = alpha * A @ B + beta * C"""
    print("Testing sgemm (float32 matrix multiplication)...")
    
    # Simple test case
    M, N, K = 4, 4, 4
    alpha = 1.0
    beta = 0.0
    transa_baseline = 'N'
    transb_baseline = 'N'
    transa_triton = 0  # Triton expects 0/1 instead of 'N'/'T'
    transb_triton = 0
    dtype = torch.float32
    
    # Create test data
    torch.manual_seed(42)
    A = torch.randn(M, K, dtype=dtype, device='cuda')
    B = torch.randn(K, N, dtype=dtype, device='cuda')
    C_baseline = torch.randn(M, N, dtype=dtype, device='cuda')
    C_triton = C_baseline.clone()
    
    print(f"  Input shapes: A={A.shape}, B={B.shape}, C={C_baseline.shape}")
    print(f"  Parameters: alpha={alpha}, beta={beta}, trans=({transa_baseline}, {transb_baseline})")
    
    # Call baseline (CuPy cuBLAS)
    print("\n  Calling baseline (CuPy cuBLAS)...")
    try:
        # Get function from registry (stored as tuple: (name, func, autograd_info))
        baseline_sgemm = REGISTERED_OPS['baseline']['sgemm'][1]
        ref_out = baseline_sgemm(
            transa_baseline, transb_baseline, M, N, K, alpha, A, K, B, N, beta, C_baseline, N
        )
        print(f"  ✓ Baseline executed successfully")
        print(f"    Output shape: {ref_out.shape}")
        print(f"    Output dtype: {ref_out.dtype}")
        print(f"    Output device: {ref_out.device}")
    except Exception as e:
        print(f"  ✗ Baseline failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Call Triton implementation
    print("\n  Calling Triton kernel...")
    try:
        triton_sgemm = REGISTERED_OPS['triton']['sgemm'][1]
        act_out = triton_sgemm(
            None,  # handle parameter (ignored by Triton)
            transa_triton, transb_triton, M, N, K, alpha, A, K, B, N, beta, C_triton, N
        )
        print(f"  ✓ Triton executed successfully")
        print(f"    Output shape: {act_out.shape}")
        print(f"    Output dtype: {act_out.dtype}")
        print(f"    Output device: {act_out.device}")
    except Exception as e:
        print(f"  ✗ Triton failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Compare results
    print("\n  Comparing results...")
    try:
        assert_close(act_out, ref_out, dtype)
        print(f"  ✓ Results match!")
        
        # Show some statistics
        diff = (act_out - ref_out).abs()
        print(f"    Max absolute difference: {diff.max().item():.2e}")
        print(f"    Mean absolute difference: {diff.mean().item():.2e}")
        
        return True
    except AssertionError as e:
        print(f"  ✗ Results don't match: {e}")
        
        diff = (act_out - ref_out).abs()
        print(f"    Max absolute difference: {diff.max().item():.2e}")
        print(f"    Mean absolute difference: {diff.mean().item():.2e}")
        
        return False


def test_sgemm_various_sizes():
    """Test sgemm with various matrix sizes"""
    print("\n" + "="*70)
    print("Testing sgemm with various sizes...")
    print("="*70)
    
    test_cases = [
        (2, 2, 2),      # Tiny
        (16, 16, 16),   # Small
        (64, 64, 64),   # Medium
        (128, 256, 64), # Non-square
    ]
    
    passed = 0
    failed = 0
    
    for M, N, K in test_cases:
        print(f"\nTest case: M={M}, N={N}, K={K}")
        
        try:
            A = torch.randn(M, K, dtype=torch.float32, device='cuda')
            B = torch.randn(K, N, dtype=torch.float32, device='cuda')
            C_baseline = torch.randn(M, N, dtype=torch.float32, device='cuda')
            C_triton = C_baseline.clone()
            
            baseline_sgemm = REGISTERED_OPS['baseline']['sgemm'][1]
            triton_sgemm = REGISTERED_OPS['triton']['sgemm'][1]
            
            ref_out = baseline_sgemm('N', 'N', M, N, K, 1.0, A, K, B, N, 0.0, C_baseline, N)
            act_out = triton_sgemm(None, 0, 0, M, N, K, 1.0, A, K, B, N, 0.0, C_triton, N)
            
            assert_close(act_out, ref_out, torch.float32)
            print(f"  ✓ PASS")
            passed += 1
        except Exception as e:
            print(f"  ✗ FAIL: {e}")
            failed += 1
    
    print(f"\n{'='*70}")
    print(f"Summary: {passed} passed, {failed} failed")
    print(f"{'='*70}")
    
    return failed == 0


def main():
    print("="*70)
    print("cuBLAS Baseline Smoke Test")
    print("="*70)
    print(f"Testing function: sgemm (verified baseline)")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print("="*70)
    
    # Test 1: Simple case
    print("\n[Test 1/2] Simple 4x4 matrix multiplication")
    success1 = test_sgemm_simple()
    
    # Test 2: Various sizes
    print("\n[Test 2/2] Various matrix sizes")
    success2 = test_sgemm_various_sizes()
    
    # Final summary
    print("\n" + "="*70)
    if success1 and success2:
        print("🎉 ALL TESTS PASSED!")
        print("The testing infrastructure is working correctly.")
        print("="*70)
        return 0
    else:
        print("⚠️  SOME TESTS FAILED")
        print("Please check the errors above.")
        print("="*70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
