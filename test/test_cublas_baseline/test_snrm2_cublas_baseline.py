import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("snrm2")
@parametrize("n", [16, 128, 1024, 4096])
@parametrize("incx", [1, 2, 4])
@parametrize("dtype", [torch.float32])
def test_snrm2_cublas_baseline(n, incx, dtype):
    # Initialize input tensor
    x = torch.randn(n, dtype=dtype, device='cuda')
    
    # Call baseline
    ref_out = flagbench.baseline.snrm2(n, x, incx, torch.empty(1, dtype=dtype, device='cuda'))
    
    # Call Triton implementation
    act_out = flagbench.triton.snrm2(n, x, incx, torch.empty(1, dtype=dtype, device='cuda'))
    
    # Compare results
    assert_close(act_out, ref_out, dtype)