"""Debug: check if JITFunction.run is a bound method after first call."""
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

# Check before any call
print(f"Before call:")
print(f"  type(add_kernel): {type(add_kernel)}")
print(f"  isinstance JITFunction: {isinstance(add_kernel, JITFunction)}")
print(f"  'run' in add_kernel.__dict__: {'run' in add_kernel.__dict__}")
print(f"  type(add_kernel.run): {type(add_kernel.run)}")

# Normal run
out1 = torch.empty_like(x)
add_kernel[(n // 256,)](x, y, out1, n, BLOCK=256)
print(f"\nAfter first call:")
print(f"  'run' in add_kernel.__dict__: {'run' in add_kernel.__dict__}")

# Check if run is overridden on the instance
for cls in type(add_kernel).__mro__:
    if 'run' in cls.__dict__:
        print(f"  'run' found in {cls.__name__}.__dict__")

# Now patch and test
print(f"\nPatching JITFunction.run...")
original_run = JITFunction.run

call_count = [0]
def noop_run(self, *args, **kwargs):
    call_count[0] += 1
    print(f"  NOOP RUN CALLED (call #{call_count[0]})")
    return None

JITFunction.run = noop_run

# Verify patch is in place
print(f"  JITFunction.run is noop_run: {JITFunction.run is noop_run}")
print(f"  add_kernel.run is noop_run: {add_kernel.run.__func__ is noop_run if hasattr(add_kernel.run, '__func__') else 'N/A'}")

out2 = torch.zeros_like(x)
add_kernel[(n // 256,)](x, y, out2, n, BLOCK=256)
print(f"\nPatched result: {out2[:3]}")
print(f"  noop_run was called {call_count[0]} times")
print(f"  out2 is all zeros: {torch.all(out2 == 0).item()}")

JITFunction.run = original_run
