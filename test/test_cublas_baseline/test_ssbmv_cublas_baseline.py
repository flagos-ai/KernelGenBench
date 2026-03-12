import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("ssbmv")
@parametrize("n, k", [(16, 2), (128, 16), (512, 32)])
@parametrize("alpha, beta", [(1.0, 0.0), (0.5, 0.5), (2.0, 1.0)])
@parametrize("dtype", [torch.float32])
@parametrize("uplo", ["L", "U"])
def test_ssbmv_cublas_baseline(n, k, alpha, beta, dtype, uplo):
    # Initialize input tensors
    lda = k + 1
    A = torch.randn((lda, n), dtype=dtype, device='cuda')
    x = torch.randn(n, dtype=dtype, device='cuda')
    y = torch.randn(n, dtype=dtype, device='cuda')
    
    # Convert uplo to cublasFillMode_t
    uplo_flag = 0 if uplo == "L" else 1  # 0 for lower, 1 for upper
    
    # Call baseline
    ref_out = flagbench.baseline.ssbmv(
        uplo_flag, n, k, alpha, A, lda, x, 1, beta, y, 1
    )
    
    # Call Triton implementation
    act_out = flagbench.triton.ssbmv(
        uplo_flag, n, k, alpha, A, lda, x, 1, beta, y, 1
    )
    
    # Compare results
    assert_close(act_out, ref_out, dtype)