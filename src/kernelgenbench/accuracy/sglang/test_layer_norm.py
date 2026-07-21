"""
Accuracy and benchmark test for SGLang LayerNorm.
Source: sglang.srt.layers.layernorm.LayerNorm(hidden_size, eps).forward_cuda(x)
"""
import kernelgenbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import kernelgenbench_assert_close as assert_close
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton

@label("layer_norm")
@parametrize("shape", [(1, 32), (71, 497), (128, 512), (1024, 4096)])
@parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@parametrize("eps", [1e-6, 1e-5])
@parametrize("elementwise_affine", [True, False])
def test_accuracy_layer_norm(shape, dtype, eps, elementwise_affine):
    M, hidden_size = shape
    x = torch.randn(M, hidden_size, device='cuda', dtype=dtype)

    ref_out = kernelgenbench.baseline.layer_norm(x, hidden_size, eps=eps, elementwise_affine=elementwise_affine)
    act_out = kernelgenbench.triton.layer_norm(x.clone(), hidden_size, eps=eps, elementwise_affine=elementwise_affine)

    assert_close(act_out, ref_out, dtype)

    if M < 1024 or dtype == torch.float32:
        return None

    x_bench = torch.randn(M, hidden_size, device='cuda', dtype=dtype)
    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.layer_norm(x_bench, hidden_size, eps=eps, elementwise_affine=elementwise_affine),
        warmup=25, rep=100
    )
    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.layer_norm(x_bench.clone(), hidden_size, eps=eps, elementwise_affine=elementwise_affine),
        warmup=25, rep=100
    )
    speedup = ms_baseline / ms_triton if ms_triton > 0 else float('inf')
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
