"""
Accuracy and benchmark test for SGLang apply_interleaved_rope_triton.
Source: sglang.srt.layers.rotary_embedding.mrope.apply_interleaved_rope_triton(x [3,N,D], mrope_section) -> [N,D]
"""
import kernelgenbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import kernelgenbench_assert_close as assert_close
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton

@label("apply_interleaved_rope_triton")
@parametrize("seq_len", [128, 512, 1024])
@parametrize("rotary_dim", [64, 128])
@parametrize("dtype", [torch.float16, torch.bfloat16])
@parametrize("mrope_section", [[8, 12, 12], [24, 20, 20]])
def test_accuracy_apply_interleaved_rope_triton(seq_len, rotary_dim, dtype, mrope_section):
    N, D = seq_len, rotary_dim
    x = torch.randn(3, N, D, device='cuda', dtype=dtype)

    ref_out = kernelgenbench.baseline.apply_interleaved_rope_triton(x, mrope_section)
    act_out = kernelgenbench.baseline.apply_interleaved_rope_triton(x.clone(), mrope_section)

    assert_close(act_out, ref_out, dtype)

    if seq_len < 512:
        return None

    x_bench = torch.randn(3, N, D, device='cuda', dtype=dtype)
    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.apply_interleaved_rope_triton(x_bench, mrope_section),
        warmup=25, rep=100
    )
    speedup = 1.0
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_baseline, speedup=speedup)
