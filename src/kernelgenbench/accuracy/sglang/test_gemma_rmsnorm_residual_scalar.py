"""
Accuracy and benchmark test for SGLang gemma_rmsnorm_residual_scalar.
Source: gemma_rmsnorm_residual_scalar(x [N,D], weight [D], residual [N,D], scalar [scalar], eps) -> Tensor[N,D]
"""
import kernelgenbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import kernelgenbench_assert_close as assert_close
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton

@label("gemma_rmsnorm_residual_scalar")
@parametrize("shape", [(64, 64), (256, 128)])
@parametrize("dtype", [torch.float16, torch.bfloat16])
def test_accuracy_gemma_rmsnorm_residual_scalar(shape, dtype):
    M, N = shape
    x = torch.randn(M, N, device='cuda', dtype=dtype)
    weight = torch.randn(N, device='cuda', dtype=dtype)
    residual = torch.randn(M, N, device='cuda', dtype=dtype)
    scalar = torch.tensor(0.5, device='cuda', dtype=dtype)
    ref_out = kernelgenbench.baseline.gemma_rmsnorm_residual_scalar(x, weight, residual, scalar, eps=1e-6)
    act_out = kernelgenbench.triton.gemma_rmsnorm_residual_scalar(x.clone(), weight, residual.clone(), scalar, eps=1e-6)
    assert_close(act_out, ref_out, dtype)
    if M < 256:
        return None
    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.gemma_rmsnorm_residual_scalar(x.clone(), weight, residual.clone(), scalar, eps=1e-6),
        warmup=25, rep=100
    )
    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.gemma_rmsnorm_residual_scalar(x.clone(), weight, residual.clone(), scalar, eps=1e-6),
        warmup=25, rep=100
    )
    speedup = ms_baseline / ms_triton if ms_triton > 0 else float('inf')
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
