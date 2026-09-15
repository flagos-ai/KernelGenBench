"""
Accuracy and benchmark test for SGLang mamba_chunk_scan_combined_fwd.
Source: mamba_chunk_scan_combined_fwd(x [B,T,nheads,D], dt [B,T,nheads], A [nheads], B [B,T,ngroups,dstate], C [B,T,ngroups,dstate]) (Mamba-2 SSD)
"""
import kernelgenbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import kernelgenbench_assert_close as assert_close
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton

@label("mamba_chunk_scan_combined_fwd")
@parametrize("shape", [(64, 64), (256, 128)])
@parametrize("dtype", [torch.float16, torch.bfloat16])
def test_accuracy_mamba_chunk_scan_combined_fwd(shape, dtype):
    M, N = shape
    B, T, nheads, D, dstate, ngroups = 1, 64, 4, 64, 16, 1
    x = torch.randn(B, T, nheads, D, device='cuda', dtype=dtype)
    dt = torch.randn(B, T, nheads, device='cuda', dtype=dtype)
    A = torch.randn(nheads, device='cuda', dtype=dtype)
    Bw = torch.randn(B, T, ngroups, dstate, device='cuda', dtype=dtype)
    Cw = torch.randn(B, T, ngroups, dstate, device='cuda', dtype=dtype)
    ref_out = kernelgenbench.baseline.mamba_chunk_scan_combined_fwd(x, dt, A, Bw, Cw, chunk_size=64, D=None, z=None, dt_bias=None, initial_states=None, cu_seqlens=None)
    act_out = kernelgenbench.triton.mamba_chunk_scan_combined_fwd(x.clone(), dt.clone(), A, Bw.clone(), Cw.clone(), chunk_size=64, D=None, z=None, dt_bias=None, initial_states=None, cu_seqlens=None)
    assert_close(act_out, ref_out, dtype)
    if M < 256:
        return None
    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.mamba_chunk_scan_combined_fwd(x.clone(), dt.clone(), A, Bw.clone(), Cw.clone(), chunk_size=64, D=None, z=None, dt_bias=None, initial_states=None, cu_seqlens=None),
        warmup=25, rep=100
    )
    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.mamba_chunk_scan_combined_fwd(x, dt.clone(), A, Bw.clone(), Cw.clone(), chunk_size=64, D=None, z=None, dt_bias=None, initial_states=None, cu_seqlens=None),
        warmup=25, rep=100
    )
    speedup = ms_baseline / ms_triton if ms_triton > 0 else float('inf')
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
