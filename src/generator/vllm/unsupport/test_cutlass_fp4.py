"""
复现 cutlass_scaled_fp4_mm: 需要 SM >= 100 (Blackwell), 当前 SM80 (A100) 不支持
"""
import torch
from vllm import _custom_ops

a = torch.randn(32, 64, device='cuda', dtype=torch.float16)
b = torch.randint(0, 15, (64, 32), device='cuda', dtype=torch.uint8)
block_scale_a = torch.ones(32, 1, device='cuda', dtype=torch.float32)
block_scale_b = torch.ones(1, 32, device='cuda', dtype=torch.float32)
alpha = torch.tensor(1.0, device='cuda', dtype=torch.float32)

try:
    out = _custom_ops.cutlass_scaled_fp4_mm(a, b, block_scale_a, block_scale_b, alpha, torch.float16)
    print("SUCCESS")
except NotImplementedError as e:
    print(f"FAIL: {e}")
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}")
