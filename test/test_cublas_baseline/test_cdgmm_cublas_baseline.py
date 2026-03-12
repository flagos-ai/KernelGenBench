import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("cdgmm")
@parametrize("m, n", [(2, 3), (128, 256), (512, 512)])
@parametrize("mode", ["left", "right"])
@parametrize("dtype", [torch.complex64])
def test_cdgmm_cublas_baseline(m, n, mode, dtype):
    # Initialize input tensors
    A = torch.randn(m, n, dtype=dtype, device='cuda')
    x = torch.randn(n if mode == "right" else m, dtype=dtype, device='cuda')
    C = torch.randn(m, n, dtype=dtype, device='cuda')
    
    # Convert mode to cublasSideMode_t equivalent
    mode_flag = 0 if mode == "left" else 1
    
    # Call baseline
    ref_out = flagbench.baseline.cdgmm(
        mode=mode_flag,
        m=m,
        n=n,
        A=A,
        lda=m,
        x=x,
        incx=1,
        C=C,
        ldc=m
    )
    
    # Call Triton implementation
    act_out = flagbench.triton.cdgmm(
        mode=mode_flag,
        m=m,
        n=n,
        A=A,
        lda=m,
        x=x,
        incx=1,
        C=C,
        ldc=m
    )
    
    # Compare results
    assert_close(act_out, ref_out, dtype)