import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("cscal")
@parametrize("n", [16, 128, 1024])
@parametrize("incx", [1, 2, 3])
@parametrize("alpha", [1.0, 0.5, 2.0])
def test_cscal_cublas_baseline(n, incx, alpha):
    dtype = torch.complex64
    alpha = torch.tensor(alpha, dtype=torch.float32, device='cuda').view(torch.complex64)
    x = torch.randn(n, dtype=dtype, device='cuda')
    
    # Make copies for in-place operations
    x_ref = x.clone()
    x_act = x.clone()
    
    # Call baseline
    flagbench.baseline.cscal(n, alpha, x_ref, incx)
    
    # Call Triton implementation
    flagbench.triton.cscal(n, alpha, x_act, incx)
    
    # Compare results
    assert_close(x_act, x_ref, dtype)