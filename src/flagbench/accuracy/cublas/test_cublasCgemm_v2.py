import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
import torch

@label("cublasCgemm_v2")
@parametrize("M, N, K", [
    (1, 1, 1),
    (16, 16, 16),
    (32, 32, 32),
    (32, 64, 16),
    (64, 32, 48),
    (128, 256, 64),
    (256, 128, 128),
    (17, 33, 65),
    (71, 497, 33),
    (128, 128, 128),
    (256, 256, 256),
    (1024, 1024, 256),
    (4096, 1024, 64),
    (1024, 4096, 32),
    (5333, 5333, 71),
    (5333, 71, 33),
    (71, 5333, 17),
])
@parametrize("alpha, beta", [
    (complex(1.0, 0.0), complex(0.0, 0.0)),
    (complex(0.0, 0.0), complex(1.0, 0.0)),
    (complex(0.5, 0.0), complex(0.5, 0.0)),
    (complex(-1.0, 0.0), complex(1.0, 0.0)),
    (complex(2.0, 0.0), complex(-0.5, 0.0)),
    (complex(0.001, 0.0), complex(0.999, 0.0)),
    (complex(-0.999, 0.0), complex(1.0, 0.0)),
    (complex(100.001, 0.0), complex(-0.999, 0.0)),
    (complex(0.5, 0.3), complex(0.5, -0.2)),
    (complex(0.001, 0.001), complex(0.0, 0.0)),
    (complex(1.0, 0.0), complex(0.001, 0.001)),
])
@parametrize("transa, transb", [
    ("N", "N"),
    ("N", "T"),
    ("T", "N"),
    ("T", "T"),
    ("C", "N"),
    ("N", "C"),
    ("C", "C"),
    ("T", "C"),
    ("C", "T"),
])
@parametrize("dtype", [torch.complex64])
def test_accuracy_cublasCgemm_v2(M, N, K, alpha, beta, transa, transb, dtype):
    # Determine storage shapes based on transposition flags (column-major expectations)
    A_rows = M if transa == 'N' else K
    A_cols = K if transa == 'N' else M
    B_rows = K if transb == 'N' else N
    B_cols = N if transb == 'N' else K
    C_rows = M
    C_cols = N

    # Create base row-major tensors
    A_rm = torch.randn(A_rows, A_cols, dtype=dtype, device=device)
    B_rm = torch.randn(B_rows, B_cols, dtype=dtype, device=device)
    C_rm = torch.randn(C_rows, C_cols, dtype=dtype, device=device)

    alpha_c = complex(alpha)
    beta_c = complex(beta)

    # Prepare column-major representations by transposing row-major bases
    A_cm = A_rm.t().contiguous()
    B_cm = B_rm.t().contiguous()
    C_cm_ref = C_rm.t().contiguous()
    C_cm_act = C_rm.t().contiguous()

    lda = A_rows
    ldb = B_rows
    ldc = C_rows

    # Reference (baseline) call
    ref_out_cm = flagbench.baseline.cublasCgemm_v2(
        transa, transb, M, N, K, alpha_c,
        A_cm, lda,
        B_cm, ldb,
        beta_c,
        C_cm_ref, ldc
    )

    # Actual (triton) call
    act_out_cm = flagbench.triton.cublasCgemm_v2(
        transa, transb, M, N, K, alpha_c,
        A_cm, lda,
        B_cm, ldb,
        beta_c,
        C_cm_act, ldc
    )

    # Convert outputs back to row-major for comparison
    ref_out_rm = ref_out_cm.t().contiguous()
    act_out_rm = act_out_cm.t().contiguous()

    assert_close(act_out_rm, ref_out_rm, dtype, reduce_dim=K)

    from sandbox.utils.accuracy_utils import CustomBenchmarkResult
    if M < 128 or N < 128 or K < 64:
        return None

    # Benchmark preparation tensors
    A_bench_rm = torch.randn(A_rows, A_cols, dtype=dtype, device=device)
    B_bench_rm = torch.randn(B_rows, B_cols, dtype=dtype, device=device)
    A_bench_cm = A_bench_rm.t().contiguous()
    B_bench_cm = B_bench_rm.t().contiguous()

    # warmup
    for _ in range(10):
        C_w_rm = torch.randn(C_rows, C_cols, dtype=dtype, device=device)
        C_w_cm1 = C_w_rm.t().contiguous()
        C_w_cm2 = C_w_rm.t().contiguous()
        _ = flagbench.baseline.cublasCgemm_v2(
            transa, transb, M, N, K, alpha_c,
            A_bench_cm, lda,
            B_bench_cm, ldb,
            beta_c,
            C_w_cm1.clone(), ldc
        )
        _ = flagbench.triton.cublasCgemm_v2(
            transa, transb, M, N, K, alpha_c,
            A_bench_cm, lda,
            B_bench_cm, ldb,
            beta_c,
            C_w_cm2.clone(), ldc
        )
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        C_b_rm = torch.randn(C_rows, C_cols, dtype=dtype, device=device)
        C_b_cm = C_b_rm.t().contiguous()
        _ = flagbench.baseline.cublasCgemm_v2(
            transa, transb, M, N, K, alpha_c,
            A_bench_cm, lda,
            B_bench_cm, ldb,
            beta_c,
            C_b_cm.clone(), ldc
        )
    end_event.record()
    torch.cuda.synchronize()
    ms_baseline = start_event.elapsed_time(end_event) / 100

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        C_t_rm = torch.randn(C_rows, C_cols, dtype=dtype, device=device)
        C_t_cm = C_t_rm.t().contiguous()
        _ = flagbench.triton.cublasCgemm_v2(
            transa, transb, M, N, K, alpha_c,
            A_bench_cm, lda,
            B_bench_cm, ldb,
            beta_c,
            C_t_cm.clone(), ldc
        )
    end_event.record()
    torch.cuda.synchronize()
    ms_triton = start_event.elapsed_time(end_event) / 100

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)