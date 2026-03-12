import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("dasum")
@parametrize("n, incx", [(10, 1), (1024, 1), (4096, 2), (8192, 4)])
@parametrize("dtype", [torch.float64])
def test_dasum_cublas_baseline(n, incx, dtype):
    # Initialize input tensor
    x = torch.randn(n, dtype=dtype, device='cuda')
    
    # Call baseline
    ref_out = flagbench.baseline.dasum(n, x, incx, torch.zeros(1, dtype=dtype, device='cuda'))
    
    # Call Triton implementation
    act_out = flagbench.triton.dasum(n, x, incx, torch.zeros(1, dtype=dtype, device='cuda'))
    
    # Compare results
    assert_close(act_out, ref_out, dtype)