import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton
from vllm.scalar_type import scalar_types
from vllm.model_executor.layers.quantization.utils.marlin_utils_test_24 import marlin_24_quantize

@label("gptq_marlin_24_gemm")
@parametrize("config", [
    # (M, K, N)
    (1, 128, 256),
    (64, 128, 256),
    (128, 256, 256),
    (256, 256, 512),
    (128, 512, 256),
    (256, 512, 512),
    (128, 256, 512),
])
def test_accuracy_gptq_marlin_24_gemm(config):
    # ===== Accuracy Test =====
    M, K, N = config[0], config[1], config[2]
    gs = 128
    quant_type = scalar_types.uint4b8

    w = torch.randn(K, N, device=device, dtype=torch.float16)
    w_ref, q_w, meta, s = marlin_24_quantize(w, quant_type, gs)
    ws = torch.zeros(N, device=device, dtype=torch.int32)
    a = torch.randn(M, K, device=device, dtype=torch.float16)

    ref_out = flagbench.baseline.gptq_marlin_24_gemm(
        a, q_w, meta, s, ws, quant_type, M, N, K)
    act_out = flagbench.triton.gptq_marlin_24_gemm(
        a, q_w, meta, s, ws, quant_type, M, N, K)

    assert_close(act_out, ref_out, torch.float16)

    # ===== Performance Test =====
    if M < 128:
        return None

    ms_baseline = triton.testing.do_bench(
        lambda: flagbench.baseline.gptq_marlin_24_gemm(
            a, q_w, meta, s, ws, quant_type, M, N, K),
        warmup=25, rep=100)
    ms_triton = triton.testing.do_bench(
        lambda: flagbench.triton.gptq_marlin_24_gemm(
            a, q_w, meta, s, ws, quant_type, M, N, K),
        warmup=25, rep=100)

    speedup = ms_baseline / ms_triton if ms_triton > 0 else float("inf")
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
