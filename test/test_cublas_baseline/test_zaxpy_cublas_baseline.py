import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("zaxpy")
@parametrize("n", [16, 128, 1024])
@parametrize("incx, incy", [(1, 1), (2, 3)])
@parametrize("alpha", [1.0, 0.5, 1.5+2.0j])
@parametrize("dtype", [torch.complex128])
def test_zaxpy_cublas_baseline(n, incx, incy, alpha, dtype):
    # Initialize input tensors
    x = torch.randn(n * incx, dtype=dtype, device='cuda')
    y = torch.randn(n * incy, dtype=dtype, device='cuda')
    
    # Make copies for baseline and triton since operation is in-place
    y_ref = y.clone()
    y_act = y.clone()
    
    # Call baseline
    flagbench.baseline.zaxpy(n, alpha, x, incx, y_ref, incy)
    
    # Call Triton implementation
    flagbench.triton.zaxpy(n, alpha, x, incx, y_act, incy)
    
    # Compare results
    assert_close(y_act, y_ref, dtype)