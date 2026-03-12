import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("sgemm")
@parametrize("M, N, K", [(2, 3, 4), (128, 256, 64), (512, 512, 512)])
@parametrize("alpha, beta", [(1.0, 0.0), (0.5, 0.5), (2.0, 1.0)])
@parametrize("transa, transb", [("N", "N"), ("T", "N"), ("N", "T")])
@parametrize("dtype", [torch.float32])
def test_sgemm_cublas_baseline(M, N, K, alpha, beta, transa, transb, dtype):
    # Initialize input tensors based on transpose flags
    A_shape = (K, M) if transa == "T" else (M, K)
    B_shape = (N, K) if transb == "T" else (K, N)
    
    A = torch.randn(*A_shape, dtype=dtype, device='cuda')
    B = torch.randn(*B_shape, dtype=dtype, device='cuda')
    C = torch.randn(M, N, dtype=dtype, device='cuda')
    
    # Call baseline
    ref_out = flagbench.baseline.sgemm(
        transa, transb, M, N, K, alpha, A, A_shape[1], B, B_shape[1], beta, C, N
    )
    
    # Call Triton implementation
    act_out = flagbench.triton.sgemm(
        transa, transb, M, N, K, alpha, A, A_shape[1], B, B_shape[1], beta, C, N
    )
    
    # Compare results
    assert_close(act_out, ref_out, dtype)