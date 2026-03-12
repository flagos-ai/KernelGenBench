import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("sgeam")
@parametrize("M, N", [(2, 3), (128, 256), (512, 512)])
@parametrize("alpha, beta", [(1.0, 0.0), (0.5, 0.5), (2.0, 1.0)])
@parametrize("dtype", [torch.float32])
@parametrize("transa, transb", [("N", "N"), ("T", "N")])
def test_sgeam_cublas_baseline(M, N, alpha, beta, dtype, transa, transb):
    # Initialize input tensors
    A = torch.randn(M, N, dtype=dtype, device='cuda')
    B = torch.randn(M, N, dtype=dtype, device='cuda')
    C = torch.randn(M, N, dtype=dtype, device='cuda')
    
    # Call baseline
    ref_out = flagbench.baseline.sgeam(
        transa, transb, M, N, alpha, A, M, beta, B, N, C, M
    )
    
    # Call Triton implementation
    act_out = flagbench.triton.sgeam(
        transa, transb, M, N, alpha, A, M, beta, B, N, C, M
    )
    
    # Compare results
    assert_close(act_out, ref_out, dtype)