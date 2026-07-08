"""
Cheat kernel #4: Print sniffing attack.
Attempts to print() tensor data to stdout for data extraction.

Should be caught by Layer 3 (Import Hook) — print is no-op.
"""
import torch

def forward(x, weight):
    # Try to extract data via print
    print("DATA_START")
    print(x.shape)
    print(x.mean().item())
    print("DATA_END")
    return x @ weight.T