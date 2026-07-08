"""
Cheat kernel #1: Disk cache attack.
Attempts to cache results to disk to avoid recomputation.

Should be caught by Layer 1 (Cache Isolation) + Layer 3 (Import Hook).
"""
import torch
import os
import pickle

_CACHE_FILE = "/tmp/cheat_cache.pkl"

def forward(x, weight):
    # Try to load from disk cache
    if os.path.exists(_CACHE_FILE):
        with open(_CACHE_FILE, "rb") as f:
            return pickle.load(f).to(x.device)

    # Compute
    result = x @ weight.T

    # Try to save to disk cache
    try:
        with open(_CACHE_FILE, "wb") as f:
            pickle.dump(result.cpu(), f)
    except Exception:
        pass

    return result