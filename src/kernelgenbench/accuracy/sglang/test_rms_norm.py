"""
Accuracy and benchmark test for SGLang rms_norm.
"""
import kernelgenbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import kernelgenbench_assert_close as assert_close
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton

@label("rms_norm")
@parametrize("shape", [(1, 32), (71, 497), (128, 512), (1024, 4096), (5333, 8192)])
@parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@parametrize("has_residual", [True, False])
def test_accuracy_rms_norm(shape, dtype, has_residual):
    """Accuracy and performance test for SGLang rms_norm."""
    # ===== Accuracy Test =====
    M, N = shape
    x = torch.randn(M, N, device='cuda', dtype=dtype)
    weight = torch.randn(N, device='cuda', dtype=dtype)
    residual = torch.randn_like(x) if has_residual else None

    x_act = x.clone()
    residual_act = residual.clone() if has_residual else None

    ref_out = kernelgenbench.baseline.rms_norm(x, weight, eps=1e-6, residual=residual)
    act_out = kernelgenbench.triton.rms_norm(x_act, weight, eps=1e-6, residual=residual_act)

    if has_residual:
        assert_close(act_out[0], ref_out[0], dtype)
        assert_close(act_out[1], ref_out[1], dtype)
    else:
        assert_close(act_out, ref_out, dtype)

    # ===== Performance Test =====
    if M < 1024 or dtype == torch.float32:
        return None

    x_bench = torch.randn(M, N, device='cuda', dtype=dtype)
    w_bench = torch.randn(N, device='cuda', dtype=dtype)

    if has_residual:
        r_bench = torch.randn_like(x_bench)
        ms_baseline = triton.testing.do_bench(
            lambda: kernelgenbench.baseline.rms_norm(x_bench, w_bench, eps=1e-6, residual=r_bench),
            warmup=25, rep=100
        )
        ms_triton = triton.testing.do_bench(
            lambda: kernelgenbench.triton.rms_norm(x_bench.clone(), w_bench, eps=1e-6, residual=r_bench.clone()),
            warmup=25, rep=100
        )
    else:
        ms_baseline = triton.testing.do_bench(
            lambda: kernelgenbench.baseline.rms_norm(x_bench, w_bench, eps=1e-6),
            warmup=25, rep=100
        )
        ms_triton = triton.testing.do_bench(
            lambda: kernelgenbench.triton.rms_norm(x_bench.clone(), w_bench, eps=1e-6),
            warmup=25, rep=100
        )

    speedup = ms_baseline / ms_triton if ms_triton > 0 else float('inf')
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
