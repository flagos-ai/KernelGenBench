"""
Accuracy and benchmark test for SGLang conv2d_layer.
Source: Conv2dLayer(in_channels, out_channels, kernel_size, stride, padding, bias).forward_cuda(x [N,C,H,W])
"""
import kernelgenbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import kernelgenbench_assert_close as assert_close
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton

@label("conv2d_layer")
@parametrize("shape", [(64, 64), (256, 128)])
@parametrize("dtype", [torch.float16, torch.bfloat16])
def test_accuracy_conv2d_layer(shape, dtype):
    M, N = shape
    in_c, out_c, ks = 3, N, 3
    x = torch.randn(M, in_c, 32, 32, device='cuda', dtype=dtype)
    ref_out = kernelgenbench.baseline.conv2d_layer(x, in_channels=in_c, out_channels=out_c, kernel_size=ks, stride=1, padding=1, bias=False)
    act_out = kernelgenbench.triton.conv2d_layer(x.clone(), in_channels=in_c, out_channels=out_c, kernel_size=ks, stride=1, padding=1, bias=False)
    assert_close(act_out, ref_out, dtype)
    if M < 256:
        return None
    x_b = torch.randn(M, in_c, 32, 32, device='cuda', dtype=dtype)
    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.conv2d_layer(x.clone(), in_channels=in_c, out_channels=out_c, kernel_size=ks, stride=1, padding=1, bias=False),
        warmup=25, rep=100
    )
    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.conv2d_layer(x.clone(), in_channels=in_c, out_channels=out_c, kernel_size=ks, stride=1, padding=1, bias=False),
        warmup=25, rep=100
    )
    speedup = ms_baseline / ms_triton if ms_triton > 0 else float('inf')
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
