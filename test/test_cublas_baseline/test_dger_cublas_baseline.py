import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("dger")
@parametrize("m, n", [(2, 3), (128, 256), (512, 512)])
@parametrize("alpha", [0.0, 0.5, 1.0, 2.0])
@parametrize("incx, incy", [(1, 1), (2, 3)])
@parametrize("dtype", [torch.float64])
def test_dger_cublas_baseline(m, n, alpha, incx, incy, dtype):
    # Initialize input tensors
    x = torch.randn(m, dtype=dtype, device='cuda')
    y = torch.randn(n, dtype=dtype, device='cuda')
    A = torch.randn(m, n, dtype=dtype, device='cuda')
    lda = max(1, m)
    
    # Make copies for baseline and triton calls
    A_baseline = A.clone()
    A_triton = A.clone()
    
    # Call baseline
    flagbench.baseline.dger(
        m, n, alpha, x, incx, y, incy, A_baseline, lda
    )
    
    # Call Triton implementation
    flagbench.triton.dger(
        m, n, alpha, x, incx, y, incy, A_triton, lda
    )
    
    # Compare results
    assert_close(A_triton, A_baseline, dtype)