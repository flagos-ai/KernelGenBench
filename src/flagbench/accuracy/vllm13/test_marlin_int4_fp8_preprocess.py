import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton

@label("marlin_int4_fp8_preprocess")
@parametrize("config", [
    (256, 256),
    (512, 256),
    (256, 512),
    (512, 512),
    (1024, 512),
    (512, 1024),
    (1024, 1024),
    (2048, 1024),
])
def test_accuracy_marlin_int4_fp8_preprocess(config):
    K = config[0]
    N = config[1]
    pack_factor = 8

    qweight = torch.randint(0, 2**31, (K // pack_factor, N), device=device, dtype=torch.int32)

    ref_out = flagbench.baseline.marlin_int4_fp8_preprocess(qweight)
    act_out = flagbench.triton.marlin_int4_fp8_preprocess(qweight)

    assert torch.equal(act_out, ref_out), f"Mismatch: max diff={(act_out - ref_out).abs().max()}"

    # ===== Performance Test =====
    if K < 512 or N < 512:
        return None

    qweight_bench = torch.randint(0, 2**31, (K // pack_factor, N), device=device, dtype=torch.int32)

    ms_baseline = triton.testing.do_bench(
        lambda: flagbench.baseline.marlin_int4_fp8_preprocess(qweight_bench),
        warmup=25, rep=100
    )

    ms_triton = triton.testing.do_bench(
        lambda: flagbench.triton.marlin_int4_fp8_preprocess(qweight_bench),
        warmup=25, rep=100
    )

    speedup = ms_baseline / ms_triton if ms_triton > 0 else float('inf')
    return CustomBenchmarkResult(
        ref_time=ms_baseline, res_time=ms_triton, speedup=speedup
    )
