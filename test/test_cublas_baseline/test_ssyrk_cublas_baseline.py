import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("ssyrk")
@parametrize("n, k", [(2, 3), (128, 256), (512, 512)])
@parametrize("uplo, trans", [("L", "N"), ("U", "T")])
@parametrize("alpha, beta", [(1.0, 0.0), (0.5, 0.5), (2.0, 1.0)])
@parametrize("dtype", [torch.float32])
def test_ssyrk_cublas_baseline(n, k, uplo, trans, alpha, beta, dtype):
    # Initialize input tensors
    lda = k if trans == "N" else n
    A = torch.randn(n, k, dtype=dtype, device='cuda') if trans == "N" else torch.randn(k, n, dtype=dtype, device='cuda')
    C = torch.randn(n, n, dtype=dtype, device='cuda')
    
    # Call baseline
    ref_out = flagbench.baseline.ssyrk(uplo, trans, n, k, alpha, A, lda, beta, C, n)
    
    # Call Triton implementation
    act_out = flagbench.triton.ssyrk(uplo, trans, n, k, alpha, A, lda, beta, C, n)
    
    # Compare results
    assert_close(act_out, ref_out, dtype)