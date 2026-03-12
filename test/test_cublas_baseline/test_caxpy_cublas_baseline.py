import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("caxpy")
@parametrize("n", [16, 128, 1024])
@parametrize("incx, incy", [(1, 1), (2, 3)])
@parametrize("alpha", [1.0, 0.5, 2.0])
@parametrize("dtype", [torch.complex64])
def test_caxpy_cublas_baseline(n, incx, incy, alpha, dtype):
    # Initialize input tensors
    x = torch.randn(n, dtype=dtype, device='cuda')
    y = torch.randn(n, dtype=dtype, device='cuda')
    
    # Convert alpha to complex
    alpha = torch.tensor(alpha, dtype=dtype, device='cuda')
    
    # Call baseline
    ref_out = flagbench.baseline.caxpy(n, alpha, x, incx, y, incy)
    
    # Call Triton implementation
    act_out = flagbench.triton.caxpy(n, alpha, x, incx, y, incy)
    
    # Compare results
    assert_close(act_out, ref_out, dtype)