import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("sdgmm")
@parametrize("M, N", [(2, 3), (128, 256), (512, 512)])
@parametrize("mode", ["left", "right"])
@parametrize("dtype", [torch.float32])
def test_sdgmm_cublas_baseline(M, N, mode, dtype):
    # Initialize input tensors
    A = torch.randn(M, N, dtype=dtype, device='cuda')
    x = torch.randn(M if mode == "left" else N, dtype=dtype, device='cuda')
    C = torch.randn(M, N, dtype=dtype, device='cuda')
    
    # Convert mode to cublasSideMode_t equivalent
    mode_flag = 0 if mode == "left" else 1
    
    # Call baseline
    ref_out = flagbench.baseline.sdgmm(
        mode_flag, M, N, A, M, x, 1, C, M
    )
    
    # Call Triton implementation
    act_out = flagbench.triton.sdgmm(
        mode_flag, M, N, A, M, x, 1, C, M
    )
    
    # Compare results
    assert_close(act_out, ref_out, dtype)