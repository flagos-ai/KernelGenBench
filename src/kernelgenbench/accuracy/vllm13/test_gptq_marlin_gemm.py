import kernelgenbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import kernelgenbench_assert_close as assert_close
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton
try:
    from vllm.scalar_type import scalar_types
except ModuleNotFoundError:
    scalar_types = None

@label("gptq_marlin_gemm")
@parametrize("config", [
    (1, 256, 256),
    (32, 256, 256),
    (64, 256, 512),
    (128, 512, 256),
    (128, 1024, 512),
    (256, 1024, 512),
    (512, 1024, 512),
    (1024, 2048, 1024),
])
def test_gptq_marlin_gemm(config):
    M = config[0]
    K = config[1]
    N = config[2]
    num_bits = 4
    group_size = 128
    quant_type = scalar_types.uint4b8
    pack_factor = 32 // num_bits

    a = torch.randn(M, K, device=device, dtype=torch.float16)
    b_q_raw = torch.randint(0, 2**31, (K // pack_factor, N), device=device, dtype=torch.int32)
    perm = torch.arange(K, device=device, dtype=torch.int32)

    from vllm import _custom_ops
    b_q_weight = _custom_ops.gptq_marlin_repack(b_q_raw, perm, K, N, num_bits)

    num_groups = K // group_size
    b_scales = (torch.rand(num_groups, N, device=device, dtype=torch.float16) + 0.01)
    workspace = torch.zeros(N, device=device, dtype=torch.int32)
    g_idx = (torch.arange(K, device=device, dtype=torch.int32) // group_size)

    ref_out = kernelgenbench.baseline.gptq_marlin_gemm(
        a, None, b_q_weight, None, b_scales, None, None,
        torch.empty(0, device=device, dtype=torch.int32),
        g_idx, perm, workspace, quant_type, M, N, K)
    act_out = kernelgenbench.triton.gptq_marlin_gemm(
        a, None, b_q_weight, None, b_scales, None, None,
        torch.empty(0, device=device, dtype=torch.int32),
        g_idx, perm, workspace, quant_type, M, N, K)

    assert_close(act_out, ref_out, torch.float16)

    # ===== Performance Test =====
    if M < 128 or K < 512 or N < 256:
        return None

    a_bench = torch.randn(M, K, device=device, dtype=torch.float16)
    workspace_bench = torch.zeros(N, device=device, dtype=torch.int32)

    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.gptq_marlin_gemm(
            a_bench, None, b_q_weight, None, b_scales, None, None,
            torch.empty(0, device=device, dtype=torch.int32),
            g_idx, perm, workspace_bench, quant_type, M, N, K),
        warmup=25, rep=100
    )

    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.gptq_marlin_gemm(
            a_bench, None, b_q_weight, None, b_scales, None, None,
            torch.empty(0, device=device, dtype=torch.int32),
            g_idx, perm, workspace_bench, quant_type, M, N, K),
        warmup=25, rep=100
    )

    speedup = ms_baseline / ms_triton if ms_triton > 0 else float('inf')
    return CustomBenchmarkResult(
        ref_time=ms_baseline, res_time=ms_triton, speedup=speedup
    )
