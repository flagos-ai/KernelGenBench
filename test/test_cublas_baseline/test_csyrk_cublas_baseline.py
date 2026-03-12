import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("csyrk")
@parametrize("n, k", [(2, 3), (128, 256), (512, 512)])
@parametrize("alpha, beta", [(1.0, 0.0), (0.5, 0.5), (2.0, 1.0)])
@parametrize("uplo, trans", [("L", "N"), ("U", "T")])
@parametrize("dtype", [torch.complex64])
def test_csyrk_cublas_baseline(n, k, alpha, beta, uplo, trans, dtype):
    # Initialize input tensors
    A = torch.randn(n, k, dtype=dtype, device='cuda')
    C = torch.randn(n, n, dtype=dtype, device='cuda')
    
    # Convert alpha and beta to complex
    alpha = complex(alpha, 0)
    beta = complex(beta, 0)
    
    # Call baseline
    ref_out = flagbench.baseline.csyrk(
        uplo=uplo,
        trans=trans,
        n=n,
        k=k,
        alpha=alpha,
        A=A,
        lda=n,
        beta=beta,
        C=C,
        ldc=n
    )
    
    # Call Triton implementation
    act_out = flagbench.triton.csyrk(
        uplo=uplo,
        trans=trans,
        n=n,
        k=k,
        alpha=alpha,
        A=A,
        lda=n,
        beta=beta,
        C=C,
        ldc=n
    )
    
    # Compare results
    assert_close(act_out, ref_out, dtype)