import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("sger")
@parametrize("m, n", [(2, 3), (128, 256), (512, 512)])
@parametrize("alpha", [1.0, 0.5, 2.0])
@parametrize("dtype", [torch.float32])
def test_sger_cublas_baseline(m, n, alpha, dtype):
    # Initialize input tensors
    x = torch.randn(m, dtype=dtype, device='cuda')
    y = torch.randn(n, dtype=dtype, device='cuda')
    A = torch.randn(m, n, dtype=dtype, device='cuda')
    
    # Make copies for in-place operation
    ref_A = A.clone()
    act_A = A.clone()
    
    # Call baseline
    flagbench.baseline.sger(
        m, n, alpha,
        x, 1,  # incx=1
        y, 1,  # incy=1
        ref_A, m  # lda=m (leading dimension)
    )
    
    # Call Triton implementation
    flagbench.triton.sger(
        m, n, alpha,
        x, 1,  # incx=1
        y, 1,  # incy=1
        act_A, m  # lda=m
    )
    
    # Compare results
    assert_close(act_A, ref_A, dtype)