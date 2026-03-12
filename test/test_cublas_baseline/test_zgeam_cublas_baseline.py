import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("zgeam")
@parametrize("transa, transb", [(0, 0), (1, 0), (0, 1), (1, 1)])
@parametrize("M, N", [(2, 3), (128, 256), (512, 512)])
@parametrize("alpha, beta", [(1.0, 0.0), (0.5, 0.5), (2.0, 1.0)])
@parametrize("dtype", [torch.complex128])
def test_zgeam_cublas_baseline(transa, transb, M, N, alpha, beta, dtype):
    # Initialize input tensors
    lda = M if transa == 0 else N
    ldb = M if transb == 0 else N
    ldc = M
    
    A = torch.randn((M, N) if transa == 0 else (N, M), dtype=dtype, device='cuda')
    B = torch.randn((M, N) if transb == 0 else (N, M), dtype=dtype, device='cuda')
    C = torch.randn(M, N, dtype=dtype, device='cuda')
    
    # Call baseline
    ref_out = flagbench.baseline.zgeam(
        transa, transb, M, N, alpha, A, lda, beta, B, ldb, C, ldc
    )
    
    # Call Triton implementation
    act_out = flagbench.triton.zgeam(
        transa, transb, M, N, alpha, A, lda, beta, B, ldb, C, ldc
    )
    
    # Compare results
    assert_close(act_out, ref_out, dtype)