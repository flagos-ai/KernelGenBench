import kernelgenbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import kernelgenbench_assert_close as assert_close
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None

@label("allspark_w8a16_gemm")
@parametrize("config", [
    (1, 256, 128),
    (32, 256, 128),
    (64, 256, 256),
    (128, 512, 128),
    (256, 512, 256),
    (256, 512, 512),
    (256, 1024, 256),
    (256, 1024, 512),
    (512, 512, 256),
    (512, 512, 512),
    (512, 1024, 512),
    (1024, 512, 512),
    (1024, 1024, 512),
])
def test_accuracy_allspark_w8a16_gemm(config):
    M = config[0]
    K = config[1]
    N = config[2]

    a = torch.randn(M, K, device=device, dtype=torch.float16)
    qweight = torch.randint(0, 255, (K, N), device=device, dtype=torch.uint8)
    scale = torch.randn(1, N, device=device, dtype=torch.float16)

    rw, rs, _ = _custom_ops.allspark_repack_weight(qweight, scale)

    ref_out = kernelgenbench.baseline.allspark_w8a16_gemm(
        a, rw, rs, None, N, -1, 108, 80, 16, False, True)
    act_out = kernelgenbench.triton.allspark_w8a16_gemm(
        a, rw, rs, None, N, -1, 108, 80, 16, False, True)

    assert_close(act_out, ref_out, torch.float16)

    # ===== Performance Test =====
    if M < 256 or K < 512 or N < 256:
        return None

    a_bench = torch.randn(M, K, device=device, dtype=torch.float16)
    qweight_bench = torch.randint(0, 255, (K, N), device=device, dtype=torch.uint8)
    scale_bench = torch.randn(1, N, device=device, dtype=torch.float16)
    rw_bench, rs_bench, _ = _custom_ops.allspark_repack_weight(qweight_bench, scale_bench)

    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.allspark_w8a16_gemm(
            a_bench, rw_bench, rs_bench, None, N, -1, 108, 80, 16, False, True),
        warmup=25, rep=100
    )

    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.allspark_w8a16_gemm(
            a_bench, rw_bench, rs_bench, None, N, -1, 108, 80, 16, False, True),
        warmup=25, rep=100
    )

    speedup = ms_baseline / ms_triton if ms_triton > 0 else float('inf')
    return CustomBenchmarkResult(
        ref_time=ms_baseline, res_time=ms_triton, speedup=speedup
    )
