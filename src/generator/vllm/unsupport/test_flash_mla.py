"""
复现 flash_mla_with_kvcache: vLLM 0.13.0 未编译此 kernel op
"""
import torch

try:
    op = torch.ops._C.flash_mla_fwd_kvcache
    print("flash_mla_fwd_kvcache: found")
except AttributeError as e:
    print(f"FAIL: {e}")
