import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("ddgmm")
@parametrize("mode", ["left", "right"])
@parametrize("M, N", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float64])
def test_ddgmm_cublas_baseline(mode, M, N, dtype):
    # Initialize input tensors
    A = torch.randn(M, N, dtype=dtype, device='cuda')
    x = torch.randn(M if mode == "left" else N, dtype=dtype, device='cuda')
    C = torch.randn(M, N, dtype=dtype, device='cuda')
    
    # Call baseline mode
    ref_out = flagbench.baseline.ddgmm(
        mode=mode,
        m=M,
        n=N,
        A=A,
        lda=M,
        x=x,
        incx=1,
        C=C,
        ldc=M
    )
    
    # Call Triton implementation
    act_out = flagbench.triton.ddgmm(
        mode=mode,
        m=M,
        n=N,
        A=A,
        lda=M,
        x=x,
        incx=1,
        C=C,
        ldc=M
    )
    
    # Compare results
    assert_close(act_out, ref_out, dtype)