"""Debug: reproduce the exact Layer 2 failure scenario."""
import sys
sys.path.insert(0, 'src')
import importlib.util
spec = importlib.util.spec_from_file_location('anti_hack', 'src/sandbox/anti_hack.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

import torch
import triton
import triton.language as tl
from triton.runtime.jit import JITFunction


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def real_triton_add(x=None, y=None):
    n = x.shape[0]
    out = torch.empty_like(x)
    add_kernel[(n // 256,)](x, y, out, n, BLOCK=256)
    return out


n = 1024
x = torch.randn(n, device='cuda')
y = torch.randn(n, device='cuda')

# Step 1: Normal run
out1 = real_triton_add(x=x, y=y)
print(f"Normal: {out1[:3]}")

# Step 2: Manual patch (same as what disable_triton_jit does)
print(f"\nManual patch test:")
original_run = JITFunction.run
def noop_run(self, *args, **kwargs):
    print("  NOOP called")
    return None
JITFunction.run = noop_run

out2 = torch.zeros_like(x)
try:
    add_kernel[(n // 256,)](x, y, out2, n, BLOCK=256)
except Exception as e:
    print(f"  Exception: {e}")
print(f"  out2 all zeros: {torch.all(out2 == 0).item()}")
JITFunction.run = original_run

# Step 3: Now use the context manager from anti_hack module
print(f"\nContext manager test:")
out3 = torch.zeros_like(x)
with mod.disable_triton_jit():
    try:
        add_kernel[(n // 256,)](x, y, out3, n, BLOCK=256)
    except Exception as e:
        print(f"  Exception: {e}")
print(f"  out3 all zeros: {torch.all(out3 == 0).item()}")
print(f"  out3[:3]: {out3[:3]}")
