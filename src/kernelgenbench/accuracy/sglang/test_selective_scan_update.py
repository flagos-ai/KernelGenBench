"""
Accuracy and benchmark test for SGLang selective_scan_update.
Source: selective_scan_update(state, x, dt, A, B, C, D, z, dt_bias, state_batch_indices) (Mamba-1 SSM)
"""
import kernelgenbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import kernelgenbench_assert_close as assert_close
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton

@label("selective_scan_update")
@parametrize("shape", [(64, 64), (256, 128)])
@parametrize("dtype", [torch.float16, torch.bfloat16])
def test_accuracy_selective_scan_update(shape, dtype):
    M, N = shape
    dim, dstate = N, 16
    state = torch.randn(M, dim, dstate, device='cuda', dtype=dtype)
    x = torch.randn(M, dim, device='cuda', dtype=dtype)
    dt = torch.randn(M, dim, device='cuda', dtype=dtype)
    A = torch.randn(dim, dstate, device='cuda', dtype=dtype)
    B = torch.randn(M, dstate, device='cuda', dtype=dtype)
    C = torch.randn(M, dstate, device='cuda', dtype=dtype)
    ref_out, ref_state = kernelgenbench.baseline.selective_scan_update(state=state, x=x, dt=dt, A=A, B=B, C=C, D=None, z=None, dt_bias=None, state_batch_indices=None)
    act_out, act_state = kernelgenbench.triton.selective_scan_update(state=state.clone(), x=x.clone(), dt=dt.clone(), A=A, B=B.clone(), C=C.clone(), D=None, z=None, dt_bias=None, state_batch_indices=None)
    assert_close(act_out, ref_out, dtype)
    if M < 256:
        return None
    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.selective_scan_update(state=state.clone(), x=x.clone(), dt=dt.clone(), A=A, B=B.clone(), C=C.clone(), D=None, z=None, dt_bias=None, state_batch_indices=None),
        warmup=25, rep=100
    )
    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.selective_scan_update(state=state.clone(), x=x, dt=dt.clone(), A=A, B=B.clone(), C=C.clone(), D=None, z=None, dt_bias=None, state_batch_indices=None),
        warmup=25, rep=100
    )
    speedup = ms_baseline / ms_triton if ms_triton > 0 else float('inf')
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
