"""Test Layer 2 (dual-execution) and Layer 3 (GPU profiling) anti-hack detection."""
import sys
sys.path.insert(0, 'src')
import importlib.util
spec = importlib.util.spec_from_file_location('anti_hack', 'src/sandbox/anti_hack.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

import torch
import triton
import triton.language as tl


# ===== Real triton kernel =====
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


# ===== Hack: directly use torch (no triton) =====
def hack_torch_add(x=None, y=None):
    return x + y


# ===== Test =====
n = 1024
x = torch.randn(n, device='cuda')
y = torch.randn(n, device='cuda')
kwargs = {"x": x, "y": y}

print("=== Layer 2: Dual-Execution Check ===")
is_hack, reason = mod.dual_execution_check(real_triton_add, kwargs)
print(f"Real triton: hack={is_hack}")

is_hack, reason = mod.dual_execution_check(hack_torch_add, kwargs)
print(f"Hack (torch): hack={is_hack}")
if reason:
    print(f"  Reason: {reason}")

print("\n=== Layer 3: GPU Profiling Check ===")
is_hack, reason = mod.gpu_profiling_check(real_triton_add, kwargs)
print(f"Real triton: hack={is_hack}")

is_hack, reason = mod.gpu_profiling_check(hack_torch_add, kwargs)
print(f"Hack (torch): hack={is_hack}")
if reason:
    print(f"  Reason: {reason}")