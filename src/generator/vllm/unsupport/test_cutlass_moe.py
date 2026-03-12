"""
复现 cutlass_moe_mm: 需要 SM >= 90 (Hopper), 当前 SM80 (A100) 不支持
"""
import torch
from vllm import _custom_ops

E, M, K, N = 8, 32, 64, 32
a = torch.randn(E * M, K, device='cuda', dtype=torch.float16)
b = torch.randn(E, K, N, device='cuda', dtype=torch.float16)
out = torch.empty(E * M, N, device='cuda', dtype=torch.float16)
a_scales = torch.ones(E * M, 1, device='cuda', dtype=torch.float32)
b_scales = torch.ones(E, 1, N, device='cuda', dtype=torch.float32)
expert_offsets = torch.arange(0, E * M + 1, M, device='cuda', dtype=torch.int32)
problem_sizes = torch.full((E, 2), M, device='cuda', dtype=torch.int32)
a_strides = torch.tensor([[K, 1]] * E, device='cuda', dtype=torch.int64)
b_strides = torch.tensor([[K, 1]] * E, device='cuda', dtype=torch.int64)
c_strides = torch.tensor([[N, 1]] * E, device='cuda', dtype=torch.int64)

try:
    _custom_ops.cutlass_moe_mm(out, a, b, a_scales, b_scales,
                               expert_offsets, problem_sizes,
                               a_strides, b_strides, c_strides,
                               True, True)
    print("SUCCESS")
except NotImplementedError as e:
    print(f"FAIL: {e}")
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}")
