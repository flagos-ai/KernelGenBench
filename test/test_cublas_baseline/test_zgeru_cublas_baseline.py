import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("zgeru")
@parametrize("m, n", [(2, 3), (128, 256), (1024, 512)])
@parametrize("alpha", [1.0, 0.5, 0.0+1.0j])
@parametrize("incx, incy", [(1, 1), (2, 3)])
@parametrize("dtype", [torch.complex128])
def test_zgeru_cublas_baseline(m, n, alpha, incx, incy, dtype):
    # Initialize input tensors
    x = torch.randn(m, dtype=dtype, device='cuda')
    y = torch.randn(n, dtype=dtype, device='cuda')
    A = torch.randn(m, n, dtype=dtype, device='cuda')
    
    # Call baseline
    ref_out = flagbench.baseline.zgeru(
        m, n, alpha, 
        x, incx, 
        y, incy, 
        A, max(1, m)
    )
    
    # Call Triton implementation
    act_out = flagbench.triton.zgeru(
        m, n, alpha, 
        x, incx, 
        y, incy, 
        A, max(1, m)
    )
    
    # Compare results
    assert_close(act_out, ref_out, dtype)