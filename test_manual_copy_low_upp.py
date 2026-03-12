"""
Manual test for copy_low_upp kernel - simulating LLM-generated test

This test demonstrates the full flow:
1. Import PyTorch baseline (torch.ops.kaldi.copy_low_upp)
2. Import Triton implementation (via flagbench.use_gems)
3. Compare results
"""

import torch
import sys
from pathlib import Path

# Add src to path for imports
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# Import Kaldi ops (this registers torch.ops.kaldi.* custom ops)
from flagbench.ops import kaldi_ops

print("="*60)
print("Manual Test: copy_low_upp")
print("="*60)

# Test parameters
shape = (4, 4)
dtype = torch.float32

# Create test input
A_ref = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],
    [2.0, 3.0, 0.0, 0.0],
    [4.0, 5.0, 6.0, 0.0],
    [7.0, 8.0, 9.0, 10.0]
], device='cuda', dtype=dtype)

A_test = A_ref.clone()

print("\nInput (lower triangle filled):")
print(A_ref)

# Call PyTorch baseline (CuPy-based)
print("\n[1] Calling PyTorch baseline: torch.ops.kaldi.copy_low_upp")
torch.ops.kaldi.copy_low_upp(A_ref)

print("Reference output:")
print(A_ref)

# TODO: When Triton implementation is available:
# with flagbench.use_gems(REGISTERED_OPS):
#     torch.ops.kaldi.copy_low_upp(A_test)
#
# print("\nTriton output:")
# print(A_test)
#
# if torch.allclose(A_test, A_ref):
#     print("\n✓ Outputs match!")
# else:
#     print("\n✗ Outputs differ!")
#     print(f"Max diff: {torch.max(torch.abs(A_test - A_ref)).item()}")

# For now, just verify baseline works
expected = torch.tensor([
    [1.0, 2.0, 4.0, 7.0],
    [2.0, 3.0, 5.0, 8.0],
    [4.0, 5.0, 6.0, 9.0],
    [7.0, 8.0, 9.0, 10.0]
], device='cuda', dtype=dtype)

if torch.allclose(A_ref, expected):
    print("\n" + "="*60)
    print("✓ Baseline test PASSED!")
    print("="*60)
    print("\nNext steps:")
    print("1. Generate test function using LLM")
    print("2. Implement Triton kernel for copy_low_upp")
    print("3. Run full accuracy test with flagbench.use_gems")
else:
    print("\n✗ Baseline test FAILED!")
    sys.exit(1)
