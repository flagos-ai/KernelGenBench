import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("sscal")
@parametrize("n", [16, 1024, 4096])
@parametrize("incx", [1, 2, 3])
@parametrize("alpha", [1.0, 0.5, 2.0])
@parametrize("dtype", [torch.float32])
def test_sscal_cublas_baseline(n, incx, alpha, dtype):
    # Initialize input tensor
    x = torch.randn(n, dtype=dtype, device='cuda')
    
    # Make copies for baseline and triton since operation is in-place
    x_ref = x.clone()
    x_act = x.clone()
    
    # Call baseline
    flagbench.baseline.sscal(n, alpha, x_ref, incx)
    
    # Call Triton implementation
    flagbench.triton.sscal(n, alpha, x_act, incx)
    
    # Compare results
    assert_close(x_act, x_ref, dtype)