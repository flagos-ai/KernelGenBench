import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("zgerc")
@parametrize("m, n", [(2, 3), (128, 256), (512, 512)])
@parametrize("alpha", [1.0, 0.5, 2.0])
@parametrize("incx, incy", [(1, 1), (2, 3)])
@parametrize("dtype", [torch.complex128])
def test_zgerc_cublas_baseline(m, n, alpha, incx, incy, dtype):
    # Initialize input tensors
    x = torch.randn(m, dtype=dtype, device='cuda')
    y = torch.randn(n, dtype=dtype, device='cuda')
    A = torch.randn(m, n, dtype=dtype, device='cuda')
    
    # Make copies for baseline and triton calls
    A_ref = A.clone()
    A_act = A.clone()
    
    # Call baseline
    flagbench.baseline.zgerc(
        m, n, 
        alpha, 
        x, incx, 
        y, incy, 
        A_ref, m
    )
    
    # Call Triton implementation
    flagbench.triton.zgerc(
        m, n, 
        alpha, 
        x, incx, 
        y, incy, 
        A_act, m
    )
    
    # Compare results
    assert_close(A_act, A_ref, dtype)