# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import ctypes

try:
    from ._backend import get_or_create_handle, get_blas_func
except ImportError:
    from kernelgenbench.dataset.baseline.cublas._backend import get_or_create_handle, get_blas_func

# Global variables for caching (initialized once, reused)
_cublas_func = None
_scalar_cache = {}  # Cache GPU tensors for scalar parameters


def _get_cublas_func():
    '''Get BLAS function with signature set (once)'''
    global _cublas_func
    if _cublas_func is None:
        _cublas_func = get_blas_func('cublasDcopy_v2', [
            ctypes.c_void_p,
            ctypes.c_int,  # handle
            ctypes.POINTER(ctypes.c_double),  # n
            ctypes.c_int,  # x
            ctypes.POINTER(ctypes.c_double),  # incx
            ctypes.c_int,  # y
        ])
    return _cublas_func

def cublasDcopy_v2(n, x, incx, y, incy):
    '''ctypes cuBLAS C API baseline for cublasDcopy_v2: y = x'''
    handle = get_or_create_handle()
    func = _get_cublas_func()

    x_ptr = ctypes.c_void_p(x.data_ptr())
    y_ptr = ctypes.c_void_p(y.data_ptr())

    func(
        handle, n,
        ctypes.cast(x_ptr, ctypes.POINTER(ctypes.c_double)), incx,
        ctypes.cast(y_ptr, ctypes.POINTER(ctypes.c_double)), incy
    )

    return y

if __name__ == "__main__":
    torch.manual_seed(0)
    n = 100
    x = torch.randn(n, dtype=torch.float64, device='cuda')
    y = torch.randn(n, dtype=torch.float64, device='cuda')

    result = cublasDcopy_v2(n, x, 1, y, 1)
    torch.testing.assert_close(result, x, rtol=1e-7, atol=1e-7)
    print("✓ cublasDcopy_v2 test passed")
