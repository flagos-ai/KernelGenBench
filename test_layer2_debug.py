"""Debug Layer 2: why real triton is detected as hack."""
import sys
sys.path.insert(0, 'src')
import importlib.util
spec = importlib.util.spec_from_file_location('anti_hack', 'src/sandbox/anti_hack.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

import torch
import triton
import triton.language as tl


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

# Normal run
out1 = real_triton_add(x=x, y=y)
print(f"Normal: out[:5] = {out1[:5]}")

# Disabled run
with mod.disable_triton_jit():
    try:
        out2 = real_triton_add(x=x, y=y)
        print(f"Disabled: out[:5] = {out2[:5]}")
    except Exception as e:
        print(f"Disabled: CRASHED with {type(e).__name__}: {e}")
