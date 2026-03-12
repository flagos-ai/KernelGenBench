import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("dgemv")
@parametrize("trans", ["N", "T"])
@parametrize("M, N", [(2, 3), (128, 256), (1024, 512)])
@parametrize("alpha, beta", [(1.0, 0.0), (0.5, 0.5), (2.0, 1.0)])
@parametrize("dtype", [torch.float64])
def test_dgemv_cublas_baseline(trans, M, N, alpha, beta, dtype):
    # Initialize input tensors
    A = torch.randn(M, N, dtype=dtype, device='cuda')
    x = torch.randn(N if trans == "N" else M, dtype=dtype, device='cuda')
    y = torch.randn(M if trans == "N" else N, dtype=dtype, device='cuda')
    
    # Convert trans to cublasOperation_t
    trans_flag = 0 if trans == "N" else 1  # 0 = CUBLAS_OP_N, 1 = CUBLAS_OP_T
    
    # Call baseline
    ref_out = flagbench.baseline.dgemv(
        trans_flag, M, N, alpha, A, M, x, 1, beta, y, 1
    )
    
    # Call Triton implementation
    act_out = flagbench.triton.dgemv(
        trans_flag, M, N, alpha, A, M, x, 1, beta, y, 1
    )
    
    # Compare results
    assert_close(act_out, ref_out, dtype)