import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch
import cupy as cp

@label("cupy_ifft")
@parametrize("nx", [16, 64, 256])
@parametrize("dtype", [torch.complex64])
def test_ifft(nx, dtype):
    real_part = torch.randn(nx, dtype=dtype.real_dtype, device="cuda")
    imag_part = torch.randn(nx, dtype=dtype.real_dtype, device="cuda")
    input_tensor = torch.complex(real_part, imag_part)
    
    ref_input = to_reference(input_tensor, True)
    res_input = input_tensor.clone()
    
    ref_input_cp = cp.asarray(ref_input)
    ref_output_cp = cp.fft.ifft(ref_input_cp, n=nx)
    ref_out = torch.as_tensor(ref_output_cp, device=ref_input.device)
    
    res_out = flagbench.triton.ifft(res_input, nx)
    
    assert_close(res_out, ref_out, dtype)

