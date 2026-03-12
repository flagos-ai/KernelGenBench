import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("saxpy")
@parametrize("n", [
    1,            # Edge case
    32,           # Edge case (minimum valid size)
    33,           # Non-aligned size
    71,           # Non-symmetric
    160,          # Non-symmetric
    256,          # Power-of-2 aligned
    495,          # Non-symmetric
    1024,         # Power-of-2 aligned
    4096,         # Power-of-2 aligned
    4113,         # Non-aligned size
    5333,         # Non-symmetric
])
@parametrize("alpha", [
    1.0,          # Standard
    0.0,          # Standard
    0.001,        # Fractional/small
    -0.999,       # Fractional/negative
    100.001,      # Large
    -111.999,     # Large/negative
    0.5,          # Symmetric
    -0.5,         # Symmetric negative
])
@parametrize("incx, incy", [
    (1, 1),       # Contiguous
    (2, 1),       # Strided x
    (1, 2),       # Strided y
    (3, 4),       # Strided both
    (4, 3),       # Strided both
])
@parametrize("dtype", [torch.float32])
def test_saxpy_cublas_baseline(n, alpha, incx, incy, dtype):
    Lx = (n - 1) * incx + 1
    Ly = (n - 1) * incy + 1
    x = torch.randn((Lx,), dtype=dtype, device="cuda")
    y = torch.randn((Ly,), dtype=dtype, device="cuda")

    y_ref = y.clone()
    y_act = y.clone()

    ref_out = flagbench.baseline.saxpy(n, alpha, x, incx, y_ref, incy)
    act_out = flagbench.triton.saxpy(n, alpha, x, incx, y_act, incy)

    assert_close(act_out, ref_out, dtype, reduce_dim=1)