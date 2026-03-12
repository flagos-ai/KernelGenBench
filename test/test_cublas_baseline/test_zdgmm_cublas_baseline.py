import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("zdgmm")
@parametrize("M, N", [(2, 3), (128, 256), (512, 512)])
@parametrize("mode", ["left", "right"])
@parametrize("dtype", [torch.complex128])
def test_zdgmm_cublas_baseline(M, N, mode, dtype):
    # Initialize input tensors
    A = torch.randn(M, N, dtype=dtype, device='cuda')
    x = torch.randn(N if mode == "right" else M, dtype=dtype, device='cuda')
    C = torch.empty_like(A)
    
    # Convert mode to cublasSideMode_t equivalent
    mode_flag = 0 if mode == "left" else 1  # 0=CUBLAS_SIDE_LEFT, 1=CUBLAS_SIDE_RIGHT
    
    # Call baseline
    ref_out = flagbench.baseline.zdgmm(
        mode=mode_flag,
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
    act_out = flagbench.triton.zdgmm(
        mode=mode_flag,
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