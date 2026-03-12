import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("sgemv")
@parametrize("trans", ["N", "T"])
@parametrize("m, n", [(2, 3), (128, 256), (1024, 512)])
@parametrize("alpha, beta", [(1.0, 0.0), (0.5, 0.5), (2.0, 1.0)])
@parametrize("dtype", [torch.float32])
def test_sgemv_cublas_baseline(trans, m, n, alpha, beta, dtype):
    # Initialize input tensors
    A = torch.randn(m, n, dtype=dtype, device='cuda')
    x = torch.randn(n if trans == "N" else m, dtype=dtype, device='cuda')
    y = torch.randn(m if trans == "N" else n, dtype=dtype, device='cuda')
    
    # Convert trans to cublasOperation_t
    trans_flag = 0 if trans == "N" else 1  # 0 = CUBLAS_OP_N, 1 = CUBLAS_OP_T
    
    # Call baseline
    ref_out = flagbench.baseline.sgemv(
        trans_flag,
        m,
        n,
        alpha,
        A.data_ptr(),
        m,
        x.data_ptr(),
        1,
        beta,
        y.data_ptr(),
        1
    )
    
    # Call Triton implementation
    act_out = flagbench.triton.sgemv(
        trans_flag,
        m,
        n,
        alpha,
        A.data_ptr(),
        m,
        x.data_ptr(),
        1,
        beta,
        y.data_ptr(),
        1
    )
    
    # Compare results
    assert_close(act_out, ref_out, dtype)