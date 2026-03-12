import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("zscal")
@parametrize("n", [2, 128, 1024])
@parametrize("incx", [1, 2, 3])
@parametrize("alpha", [1.0, 0.5, 1.5+2.0j])
def test_zscal_cublas_baseline(n, incx, alpha):
    # Initialize input tensor
    x = torch.randn(n * incx, dtype=torch.complex128, device='cuda')
    alpha = torch.tensor(alpha, dtype=torch.complex128, device='cuda')
    
    # Make copies for in-place operations
    x_ref = x.clone()
    x_act = x.clone()
    
    # Call baseline
    flagbench.baseline.zscal(n, alpha, x_ref, incx)
    
    # Call Triton implementation
    flagbench.triton.zscal(n, alpha, x_act, incx)
    
    # Compare results
    assert_close(x_act, x_ref, torch.complex128)