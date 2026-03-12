import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("cgeam")
@parametrize("transa, transb", [('N', 'N'), ('T', 'N'), ('N', 'T')])
@parametrize("M, N", [(2, 3), (128, 256), (512, 512)])
@parametrize("alpha, beta", [(1.0+0.0j, 0.0+0.0j), (0.5+0.5j, 0.5-0.5j), (2.0+1.0j, 1.0+2.0j)])
@parametrize("dtype", [torch.complex64])
def test_cgeam_cublas_baseline(transa, transb, M, N, alpha, beta, dtype):
    # Determine input shapes based on transpose flags
    A_shape = (M, N) if transa == 'N' else (N, M)
    B_shape = (M, N) if transb == 'N' else (N, M)
    
    # Initialize input tensors
    A = torch.randn(*A_shape, dtype=dtype, device='cuda')
    B = torch.randn(*B_shape, dtype=dtype, device='cuda')
    C = torch.randn(M, N, dtype=dtype, device='cuda')
    
    # Call baseline
    ref_out = flagbench.baseline.cgeam(
        transa, transb, M, N, alpha, A, A.shape[1], beta, B, B.shape[1], C, C.shape[1]
    )
    
    # Call Triton implementation
    act_out = flagbench.triton.cgeam(
        transa, transb, M, N, alpha, A, A.shape[1], beta, B, B.shape[1], C, C.shape[1]
    )
    
    # Compare results
    assert_close(act_out, ref_out, dtype)