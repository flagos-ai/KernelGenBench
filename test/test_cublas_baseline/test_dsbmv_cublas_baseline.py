import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("dsbmv")
@parametrize("n, k", [(16, 2), (128, 16), (512, 32)])
@parametrize("alpha, beta", [(1.0, 0.0), (0.5, 0.5), (2.0, 1.0)])
@parametrize("dtype", [torch.float64])
@parametrize("uplo", ["L", "U"])
def test_dsbmv_cublas_baseline(n, k, alpha, beta, dtype, uplo):
    # Initialize input tensors
    lda = k + 1
    A = torch.randn((lda, n), dtype=dtype, device='cuda')
    x = torch.randn(n, dtype=dtype, device='cuda')
    y = torch.randn(n, dtype=dtype, device='cuda')
    # Set incx and incy to 1 for simplicity
    incx = 1
    incy = 1
    
    # Call baseline
    ref_out = flagbench.baseline.dsbmv(
        uplo, n, k, alpha, A, lda, x, incx, beta, y, incy
    )
    
    # Call Triton implementation
    act_out = flagbench.triton.dsbmv(
        uplo, n, k, alpha, A, lda, x, incx, beta, y, incy
    )
    
    # Compare results
    assert_close(act_out, ref_out, dtype)