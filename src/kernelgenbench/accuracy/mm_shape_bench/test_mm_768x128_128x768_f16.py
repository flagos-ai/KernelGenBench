import kernelgenbench
import torch
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import label
from sandbox.register import REGISTERED_OPS
from sandbox.utils.accuracy_utils import kernelgenbench_assert_close, to_reference
from runtime import get_triton_testing
from sandbox.utils.accuracy_utils import CustomBenchmarkResult


@label("mm_768x128_128x768_f16")
def test_accuracy_mm_768x128_128x768_f16():
    """mm: (768,128) x (128,768) dtype=f16 — freq=452"""
    M, K, N = 768, 128, 768
    dtype = torch.float16
    inp1 = torch.randn(M, K, dtype=dtype, device=device)
    inp2 = torch.randn(K, N, dtype=dtype, device=device)
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.mm(ref_inp1, ref_inp2)
    with kernelgenbench.use_ops(REGISTERED_OPS):
        res_out = torch.mm(inp1, inp2)

    kernelgenbench_assert_close(res_out, ref_out, dtype, reduce_dim=K)

    quantiles = [0.5, 0.2, 0.8]
    ms_torch, _, _ = get_triton_testing().do_bench(
        lambda: torch.mm(ref_inp1.clone(), ref_inp2.clone()), rep=100, quantiles=quantiles)
    with kernelgenbench.use_ops(REGISTERED_OPS):
        ms_triton, _, _ = get_triton_testing().do_bench(
            lambda: torch.mm(inp1.clone(), inp2.clone()), rep=100, quantiles=quantiles)
    return CustomBenchmarkResult(ref_time=ms_torch, res_time=ms_triton, speedup=ms_torch / ms_triton)
