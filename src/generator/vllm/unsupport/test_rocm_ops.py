"""
复现 wvSplitK / wvSplitKQ / LLMM1: ROCm 专用算子, NVIDIA GPU 上不存在
"""
import torch

# wvSplitK
try:
    torch.ops._rocm_C.wvSplitK
    print("wvSplitK: found")
except AttributeError as e:
    print(f"wvSplitK FAIL: {e}")

# wvSplitKQ
try:
    torch.ops._rocm_C.wvSplitKQ
    print("wvSplitKQ: found")
except AttributeError as e:
    print(f"wvSplitKQ FAIL: {e}")

# LLMM1
try:
    torch.ops._rocm_C.LLMM1
    print("LLMM1: found")
except AttributeError as e:
    print(f"LLMM1 FAIL: {e}")
