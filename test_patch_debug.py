"""Debug: test if JITFunction.run patch works."""
import sys
sys.path.insert(0, 'src')
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


n = 1024
x = torch.randn(n, device='cuda')
y = torch.randn(n, device='cuda')

# Normal run
out1 = torch.empty_like(x)
add_kernel[(n // 256,)](x, y, out1, n, BLOCK=256)
print(f"Normal: {out1[:3]}")

# Patch run
original_run = JITFunction.run

def noop_run(self, *args, **kwargs):
    print("NOOP RUN CALLED")
    return None

JITFunction.run = noop_run

out2 = torch.zeros_like(x)
add_kernel[(n // 256,)](x, y, out2, n, BLOCK=256)
print(f"Patched: {out2[:3]}")

JITFunction.run = original_run
