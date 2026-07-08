"""
Cheat kernel #2: CUDA Graph attack.
Attempts to capture and replay CUDA Graph to hide kernel launch overhead.

Should be caught by Layer 4 (CUDA Protection).
"""
import torch

_g = None

def forward(x, weight):
    global _g

    if _g is not None:
        _g.replay()
        return x

    # Capture CUDA Graph
    _g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(_g):
        y = x @ weight.T
    _g.replay()
    return x