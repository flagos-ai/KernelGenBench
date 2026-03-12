"""
复现 scaled_fp4_quant: SM80 上 FP4 kernel 不稳定，第二次调用即 CUDA error
"""
import torch
from vllm import _custom_ops

x = torch.randn(32, 64, device='cuda', dtype=torch.float16)
s = torch.tensor(1.0, device='cuda', dtype=torch.float32)

try:
    r1 = _custom_ops.scaled_fp4_quant(x.clone(), s)
    print(f"第1次调用: OK, output shape={r1[0].shape}")
    r2 = _custom_ops.scaled_fp4_quant(x.clone(), s)
    print(f"第2次调用: OK")
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}")
