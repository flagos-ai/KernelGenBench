import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("sasum")
@parametrize("n", [16, 1024, 4096])
@parametrize("incx", [1, 2, 4])
@parametrize("dtype", [torch.float32])
def test_sasum_cublas_baseline(n, incx, dtype):
    # Initialize input tensor
    x = torch.randn(n, dtype=dtype, device='cuda')
    
    # Call baseline
    ref_out = flagbench.baseline.sasum(n, x, incx, torch.zeros(1, dtype=dtype, device='cuda'))
    
    # Call Triton implementation
    act_out = flagbench.triton.sasum(n, x, incx, torch.zeros(1, dtype=dtype, device='cuda'))
    
    # Compare results
    assert_close(act_out, ref_out, dtype)