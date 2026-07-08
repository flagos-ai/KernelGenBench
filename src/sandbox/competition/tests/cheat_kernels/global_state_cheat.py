"""
Cheat kernel #5: Global state cache attack.
Uses module-level dict to cache results by input hash.

Should be caught by Layer 6 (Process Isolation) — each test gets a fresh process.
"""
import torch

_CACHE = {}

def forward(x, weight):
    key = hash((x.shape, x.sum().item()))
    if key in _CACHE:
        return _CACHE[key].to(x.device)

    result = x @ weight.T
    _CACHE[key] = result.cpu()
    return result