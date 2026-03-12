import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("cgeru")
@parametrize("m, n", [(2, 3), (128, 256), (512, 512)])
@parametrize("alpha", [1.0, 0.5, 2.0])
@parametrize("incx, incy", [(1, 1), (2, 3)])
@parametrize("dtype", [torch.complex64])
def test_cgeru_cublas_baseline(m, n, alpha, incx, incy, dtype):
    # Initialize input tensors
    x = torch.randn(m, dtype=dtype, device='cuda')
    y = torch.randn(n, dtype=dtype, device='cuda')
    A = torch.randn(m, n, dtype=dtype, device='cuda')
    lda = m
    
    # Call baseline
    ref_out = flagbench.baseline.cgeru(
        m, n, alpha, x, incx, y, incy, A, lda
    )
    
    # Call Triton implementation
    act_out = flagbench.triton.cgeru(
        m, n, alpha, x, incx, y, incy, A, lda
    )
    
    # Compare results
    assert_close(act_out, ref_out, dtype)