"""Debug: see what kernel names triton actually produces."""
import sys
sys.path.insert(0, 'src')
import torch
import triton
import triton.language as tl
from torch.profiler import profile, ProfilerActivity


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
out = torch.empty_like(x)

# Warmup
add_kernel[(n // 256,)](x, y, out, n, BLOCK=256)
torch.cuda.synchronize()

# Profile
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    add_kernel[(n // 256,)](x, y, out, n, BLOCK=256)
    torch.cuda.synchronize()

print("All events:")
for event in prof.key_averages():
    print(f"  key={event.key}, device_type={event.device_type}")
