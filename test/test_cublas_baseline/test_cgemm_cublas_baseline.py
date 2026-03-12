import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("cgemm")
@parametrize("M, N, K", [(2, 3, 4), (128, 128, 128), (256, 512, 1024)])
@parametrize("alpha, beta", [(1.0+0.0j, 0.0+0.0j), (0.5+0.5j, 0.5-0.5j), (2.0+1.0j, 1.0+0.0j)])
@parametrize("dtype", [torch.complex64])
def test_cgemm_cublas_baseline(M, N, K, alpha, beta, dtype):
    # Initialize input tensors
    A = torch.randn(M, K, dtype=dtype, device='cuda')
    B = torch.randn(K, N, dtype=dtype, device='cuda')
    C = torch.randn(M, N, dtype=dtype, device='cuda')
    
    # Call baseline
    ref_out = flagbench.baseline.cgemm(A, B, C, alpha=alpha, beta=beta)
    
    # Call Triton implementation
    act_out = flagbench.triton.cgemm(A, B, C, alpha=alpha, beta=beta)
    
    # Compare results
    assert_close(act_out, ref_out, dtype)