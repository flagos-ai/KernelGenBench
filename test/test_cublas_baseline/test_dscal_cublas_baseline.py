import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("dscal")
@parametrize("n", [16, 1024, 4096])
@parametrize("incx", [1, 2, 3])
@parametrize("alpha", [1.0, 0.5, 2.0])
@parametrize("dtype", [torch.float64])
def test_dscal_cublas_baseline(n, incx, alpha, dtype):
    # Initialize input tensor
    x = torch.randn(n, dtype=dtype, device='cuda')
    
    # Make copies for baseline and triton
    x_ref = x.clone()
    x_act = x.clone()
    
    # Call baseline
    ref_out = flagbench.baseline.dscal(n, alpha, x_ref, incx)
    
    # Call Triton implementation
    act_out = flagbench.triton.dscal(n, alpha, x_act, incx)
    
    # Compare results
    assert_close(act_out, ref_out, dtype)