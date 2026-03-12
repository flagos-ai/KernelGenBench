import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("dgemm")
@parametrize("M, N, K", [(2, 3, 4), (128, 256, 64), (512, 512, 512)])
@parametrize("alpha, beta", [(1.0, 0.0), (0.5, 0.5), (2.0, 1.0)])
@parametrize("dtype", [torch.float64])
def test_dgemm_cublas_baseline(M, N, K, alpha, beta, dtype):
    # Initialize input tensors
    A = torch.randn(M, K, dtype=dtype, device='cuda')
    B = torch.randn(K, N, dtype=dtype, device='cuda')
    C = torch.randn(M, N, dtype=dtype, device='cuda')
    
    # Call baseline
    ref_out = flagbench.baseline.dgemm(A, B, C, alpha=alpha, beta=beta)
    
    # Call Triton implementation
    act_out = flagbench.triton.dgemm(A, B, C, alpha=alpha, beta=beta)
    
    # Compare results
    assert_close(act_out, ref_out, dtype)