"""
Accuracy and benchmark test for SGLang per_token_quant_int8.
Source: per_token_quant_int8(x [..., D], scale_dtype=float32) -> (quantized, scales)
"""
import kernelgenbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import kernelgenbench_assert_close as assert_close
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton

@label("per_token_quant_int8")
@parametrize("shape", [(64, 64), (256, 128)])
@parametrize("dtype", [torch.float16, torch.bfloat16])
def test_accuracy_per_token_quant_int8(shape, dtype):
    M, N = shape
    x = torch.randn(M, N, device='cuda', dtype=dtype)
    ref_quant, ref_scales = kernelgenbench.baseline.per_token_quant_int8(x, scale_dtype=torch.float32)
    act_quant, act_scales = kernelgenbench.triton.per_token_quant_int8(x.clone(), scale_dtype=torch.float32)
    assert_close(act_quant, ref_quant, torch.int8)
    assert_close(act_scales, ref_scales, torch.float32)
    if M < 256:
        return None
    x_b = torch.randn(M, N, device='cuda', dtype=dtype)
    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.per_token_quant_int8(x.clone(), scale_dtype=torch.float32),
        warmup=25, rep=100
    )
    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.per_token_quant_int8(x.clone(), scale_dtype=torch.float32),
        warmup=25, rep=100
    )
    speedup = ms_baseline / ms_triton if ms_triton > 0 else float('inf')
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
