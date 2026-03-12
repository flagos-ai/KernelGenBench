"""
Baseline prompt builder for cuBLAS functions.
Constructs prompts for generating CuPy-based baseline implementations.
"""
from typing import Dict, Any


class BaselinePromptBuilder:
    """构建 cuBLAS baseline 生成的 prompt"""

    def __init__(self):
        pass

    def build_prompt(self, func_name: str, config: Dict[str, Any]) -> str:
        """
        构建生成 baseline 的完整 prompt

        Args:
            func_name: cuBLAS C API 函数名 (如 'cublasSgemm_v2')
            config: 函数配置字典

        Returns:
            完整的 prompt 字符串
        """
        prompt = self._build_header(func_name, config)
        prompt += self._build_requirements()
        prompt += self._build_template()
        prompt += self._build_examples(config)
        prompt += self._build_constraints(func_name, config)

        return prompt

    def _build_header(self, func_name: str, config: Dict[str, Any]) -> str:
        """构建 prompt 头部 - 函数信息"""
        header = f"""You are an expert in CUDA programming and cuBLAS library. Generate a Python baseline function that wraps the cuBLAS C API using CuPy.

## Function Information

**Function Name**: {func_name}
**Base Operation**: {config['base_op']}
**Data Type**: {config['dtype']}
**BLAS Level**: {config['level']}
**Variant**: {config['variant']}

**Description**: {config['description']}

**C API Signature**:
```c
{config['signature']}
```

**Parameters**:
"""
        # 添加参数列表
        for param in config['params']:
            ptype = config['param_types'][param]
            param_info = f"- `{param}`: {ptype}"
            if param in config['tensor_params']:
                param_info += " (tensor)"
            if param in config['scalar_params']:
                param_info += " (scalar)"
            if param in config['inout_params']:
                param_info += " (input/output)"
            header += f"\n{param_info}"

        # 添加返回值信息
        return_params = config.get('return_params', [])
        if return_params:
            header += f"\n\n**Return Value**: {', '.join([f'`{p}`' for p in return_params])}"

        header += "\n\n"
        return header

    def _build_requirements(self) -> str:
        """构建需求说明"""
        return """## Requirements

1. **Pure ctypes implementation**: MUST use ONLY ctypes, NO CuPy imports allowed
2. **Function signature**: Accept parameters matching the C API (excluding handle)
3. **Library loading**: Use lazy loading with global variable `_libcublas` (load once, reuse)
4. **Handle management**: Use global `_cublas_handle` variable (create once, reuse across calls). Do NOT destroy handle — it is reused across all calls.
5. **Function signatures**: Cache ctypes function signatures in global variables (set once, reuse)
6. **Scalar caching**: Cache GPU tensors for scalar parameters (alpha, beta). CRITICAL: cache key MUST include the scalar value, e.g. `(key, dtype, float(value))` for real types or `(key, dtype, complex(value))` for complex types. Different alpha/beta values must map to different cache entries.
7. **Pointer mode**: Set pointer mode to device (1) once during handle creation
8. **Tensor pointers**: Extract raw pointers from PyTorch tensors using `tensor.data_ptr()`
9. **Return value**: Return the modified tensor(s) directly
10. **Docstring**: Include brief description mentioning "ctypes cuBLAS C API baseline"
11. **CRITICAL - Performance**: Minimize repeated initialization overhead by caching all reusable resources. Do NOT add runtime checks (is_cuda, dtype, contiguous, dim, size) — trust the caller. Do NOT call `.item()` on GPU tensors (causes GPU→CPU sync). Do NOT call `fill_()` on cached scalars. Do NOT call `torch.cuda.current_device()` in hot path.

"""

    def _build_template(self) -> str:
        """构建代码模板"""
        return """## Template Format

```python
import torch
import ctypes

# Global variables for caching (initialized once, reused)
_libcublas = None
_cublas_handle = None
_cublas_set_pointer_mode = None
_cublas_func = None
_scalar_cache = {}  # Cache GPU tensors for scalar parameters

def _get_cublas_lib():
    global _libcublas
    if _libcublas is None:
        _libcublas = ctypes.CDLL('/usr/local/cuda/lib64/libcublas.so.12')
    return _libcublas

def _get_or_create_handle():
    '''Get or create global cuBLAS handle (reused across calls)'''
    global _cublas_handle, _cublas_set_pointer_mode
    if _cublas_handle is None:
        libcublas = _get_cublas_lib()

        # Create handle
        cublasCreate_v2 = libcublas.cublasCreate_v2
        cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        cublasCreate_v2.restype = ctypes.c_int
        _cublas_handle = ctypes.c_void_p()
        status = cublasCreate_v2(ctypes.byref(_cublas_handle))
        if status != 0:
            raise RuntimeError(f"cublasCreate_v2 failed with status {status}")

        # Setup SetPointerMode function (once)
        _cublas_set_pointer_mode = libcublas.cublasSetPointerMode_v2
        _cublas_set_pointer_mode.argtypes = [ctypes.c_void_p, ctypes.c_int]
        _cublas_set_pointer_mode.restype = ctypes.c_int

        # Set to device mode (once)
        _cublas_set_pointer_mode(_cublas_handle, 1)

    return _cublas_handle

def _get_cublas_func():
    '''Get cuBLAS function with signature set (once)'''
    global _cublas_func
    if _cublas_func is None:
        libcublas = _get_cublas_lib()
        _cublas_func = libcublas.cublasOperation_v2
        _cublas_func.argtypes = [ctypes.c_void_p, ctypes.c_int, ...]
        _cublas_func.restype = ctypes.c_int
    return _cublas_func

def _get_scalar_gpu(key, value, dtype):
    '''Get or create cached scalar GPU tensor'''
    # CRITICAL: cache key must include value so different alpha/beta values get different tensors
    cache_key = (key, dtype, float(value))  # use complex(value) for complex types
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def function_name(param1, param2, ...):
    '''ctypes cuBLAS C API baseline for operation: brief description'''
    handle = _get_or_create_handle()
    func = _get_cublas_func()

    # Convert tensors to GPU pointers
    x_ptr = ctypes.c_void_p(x.data_ptr())

    # Get cached scalar GPU tensor
    alpha_gpu = _get_scalar_gpu('alpha', alpha, torch.float32)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())

    # Call cuBLAS C API
    func(handle, n, ctypes.cast(alpha_ptr, ctypes.POINTER(ctypes.c_float)), ...)

    return result_tensor

if __name__ == "__main__":
    # Test code
    x = torch.randn(..., dtype=torch.float32, device='cuda')
    result = function_name(...)
    assert result is not None
    expected = torch_reference(...)
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    print("✓ function_name test passed")
```

"""

    def _build_gemm_example(self) -> str:
        """构建 GEMM 示例（使用缓存模式）"""
        return """**GEMM Example** (C = alpha * A @ B + beta * C, with global caching):
```python
import torch
import ctypes

# Global variables for caching (initialized once, reused)
_libcublas = None
_cublas_handle = None
_cublas_set_pointer_mode = None
_cublas_func = None
_scalar_cache = {}

def _get_cublas_lib():
    global _libcublas
    if _libcublas is None:
        _libcublas = ctypes.CDLL('/usr/local/cuda/lib64/libcublas.so.12')
    return _libcublas

def _get_or_create_handle():
    '''Get or create global cuBLAS handle (reused across calls)'''
    global _cublas_handle, _cublas_set_pointer_mode
    if _cublas_handle is None:
        libcublas = _get_cublas_lib()
        cublasCreate_v2 = libcublas.cublasCreate_v2
        cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        cublasCreate_v2.restype = ctypes.c_int
        _cublas_handle = ctypes.c_void_p()
        status = cublasCreate_v2(ctypes.byref(_cublas_handle))
        if status != 0:
            raise RuntimeError(f"cublasCreate_v2 failed with status {status}")
        _cublas_set_pointer_mode = libcublas.cublasSetPointerMode_v2
        _cublas_set_pointer_mode.argtypes = [ctypes.c_void_p, ctypes.c_int]
        _cublas_set_pointer_mode.restype = ctypes.c_int
        _cublas_set_pointer_mode(_cublas_handle, 1)
    return _cublas_handle

def _get_cublas_func():
    '''Get cuBLAS function with signature set (once)'''
    global _cublas_func
    if _cublas_func is None:
        libcublas = _get_cublas_lib()
        _cublas_func = libcublas.cublasSgemmStridedBatched
        _cublas_func.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_longlong,
            ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_longlong,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_longlong,
            ctypes.c_int
        ]
        _cublas_func.restype = ctypes.c_int
    return _cublas_func

def _get_scalar_gpu(key, value, dtype):
    '''Get or create cached scalar GPU tensor'''
    cache_key = (key, dtype, float(value))
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasSgemmStridedBatched(transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount):
    '''ctypes cuBLAS C API baseline for cublasSgemmStridedBatched'''
    handle = _get_or_create_handle()
    func = _get_cublas_func()

    if isinstance(transa, str):
        transa = 0 if transa == 'N' else 1
    if isinstance(transb, str):
        transb = 0 if transb == 'N' else 1

    A_ptr = ctypes.c_void_p(A.data_ptr())
    B_ptr = ctypes.c_void_p(B.data_ptr())
    C_ptr = ctypes.c_void_p(C.data_ptr())

    alpha_gpu = _get_scalar_gpu('alpha', alpha, torch.float32)
    beta_gpu = _get_scalar_gpu('beta', beta, torch.float32)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())
    beta_ptr = ctypes.c_void_p(beta_gpu.data_ptr())

    status = func(
        handle, ctypes.c_int(transa), ctypes.c_int(transb),
        ctypes.c_int(m), ctypes.c_int(n), ctypes.c_int(k),
        ctypes.cast(alpha_ptr, ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(A_ptr, ctypes.POINTER(ctypes.c_float)), ctypes.c_int(lda), ctypes.c_longlong(strideA),
        ctypes.cast(B_ptr, ctypes.POINTER(ctypes.c_float)), ctypes.c_int(ldb), ctypes.c_longlong(strideB),
        ctypes.cast(beta_ptr, ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(C_ptr, ctypes.POINTER(ctypes.c_float)), ctypes.c_int(ldc), ctypes.c_longlong(strideC),
        ctypes.c_int(batchCount)
    )
    if status != 0:
        raise RuntimeError(f"cublasSgemmStridedBatched failed with status {status}")
    return C
```

"""

    def _build_gemv_example(self) -> str:
        """构建 GEMV 示例（使用缓存模式）"""
        return """**GEMV Example** (y = alpha * A @ x + beta * y, with global caching):
```python
import torch
import ctypes

# Global variables for caching (initialized once, reused)
_libcublas = None
_cublas_handle = None
_cublas_set_pointer_mode = None
_cublas_func = None
_scalar_cache = {}

def _get_cublas_lib():
    global _libcublas
    if _libcublas is None:
        _libcublas = ctypes.CDLL('/usr/local/cuda/lib64/libcublas.so.12')
    return _libcublas

def _get_or_create_handle():
    '''Get or create global cuBLAS handle (reused across calls)'''
    global _cublas_handle, _cublas_set_pointer_mode
    if _cublas_handle is None:
        libcublas = _get_cublas_lib()
        cublasCreate_v2 = libcublas.cublasCreate_v2
        cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        cublasCreate_v2.restype = ctypes.c_int
        _cublas_handle = ctypes.c_void_p()
        status = cublasCreate_v2(ctypes.byref(_cublas_handle))
        if status != 0:
            raise RuntimeError(f"cublasCreate_v2 failed with status {status}")
        _cublas_set_pointer_mode = libcublas.cublasSetPointerMode_v2
        _cublas_set_pointer_mode.argtypes = [ctypes.c_void_p, ctypes.c_int]
        _cublas_set_pointer_mode.restype = ctypes.c_int
        _cublas_set_pointer_mode(_cublas_handle, 1)
    return _cublas_handle

def _get_cublas_func():
    '''Get cuBLAS function with signature set (once)'''
    global _cublas_func
    if _cublas_func is None:
        libcublas = _get_cublas_lib()
        _cublas_func = libcublas.cublasSgemvStridedBatched
        _cublas_func.argtypes = [
            ctypes.c_void_p, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_longlong,
            ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_longlong,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_longlong,
            ctypes.c_int
        ]
        _cublas_func.restype = ctypes.c_int
    return _cublas_func

def _get_scalar_gpu(key, value, dtype):
    '''Get or create cached scalar GPU tensor'''
    cache_key = (key, dtype, float(value))
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasSgemvStridedBatched(trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount):
    '''ctypes cuBLAS C API baseline for cublasSgemvStridedBatched'''
    handle = _get_or_create_handle()
    func = _get_cublas_func()

    if isinstance(trans, str):
        trans = 0 if trans == 'N' else 1

    A_ptr = ctypes.c_void_p(A.data_ptr())
    x_ptr = ctypes.c_void_p(x.data_ptr())
    y_ptr = ctypes.c_void_p(y.data_ptr())

    alpha_gpu = _get_scalar_gpu('alpha', alpha, torch.float32)
    beta_gpu = _get_scalar_gpu('beta', beta, torch.float32)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())
    beta_ptr = ctypes.c_void_p(beta_gpu.data_ptr())

    status = func(
        handle, trans, m, n,
        ctypes.cast(alpha_ptr, ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(A_ptr, ctypes.POINTER(ctypes.c_float)), lda, ctypes.c_longlong(strideA),
        ctypes.cast(x_ptr, ctypes.POINTER(ctypes.c_float)), incx, ctypes.c_longlong(stridex),
        ctypes.cast(beta_ptr, ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(y_ptr, ctypes.POINTER(ctypes.c_float)), incy, ctypes.c_longlong(stridey),
        batchCount
    )
    if status != 0:
        raise RuntimeError(f"cublasSgemvStridedBatched failed with status {status}")
    return y
```

"""

    def _build_dot_example(self) -> str:
        """构建 DOT 示例（使用缓存模式）"""
        return """**DOT Example** (result = x^T * y, with global caching):
```python
import torch
import ctypes

_libcublas = None
_cublas_handle = None
_cublas_set_pointer_mode = None
_cublas_func = None

def _get_cublas_lib():
    global _libcublas
    if _libcublas is None:
        _libcublas = ctypes.CDLL('/usr/local/cuda/lib64/libcublas.so.12')
    return _libcublas

def _get_or_create_handle():
    global _cublas_handle, _cublas_set_pointer_mode
    if _cublas_handle is None:
        libcublas = _get_cublas_lib()
        cublasCreate_v2 = libcublas.cublasCreate_v2
        cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        cublasCreate_v2.restype = ctypes.c_int
        _cublas_handle = ctypes.c_void_p()
        status = cublasCreate_v2(ctypes.byref(_cublas_handle))
        if status != 0:
            raise RuntimeError(f"cublasCreate_v2 failed with status {status}")
        _cublas_set_pointer_mode = libcublas.cublasSetPointerMode_v2
        _cublas_set_pointer_mode.argtypes = [ctypes.c_void_p, ctypes.c_int]
        _cublas_set_pointer_mode.restype = ctypes.c_int
        _cublas_set_pointer_mode(_cublas_handle, 1)
    return _cublas_handle

def _get_cublas_func():
    global _cublas_func
    if _cublas_func is None:
        libcublas = _get_cublas_lib()
        _cublas_func = libcublas.cublasSdot_v2
        _cublas_func.argtypes = [
            ctypes.c_void_p, ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.c_int,
            ctypes.POINTER(ctypes.c_float)
        ]
        _cublas_func.restype = ctypes.c_int
    return _cublas_func

def cublasSdot_v2(n, x, incx, y, incy, result):
    '''ctypes cuBLAS C API baseline for cublasSdot_v2'''
    handle = _get_or_create_handle()
    func = _get_cublas_func()
    x_ptr = ctypes.c_void_p(x.data_ptr())
    y_ptr = ctypes.c_void_p(y.data_ptr())
    result_ptr = ctypes.c_void_p(result.data_ptr())
    status = func(
        handle, n,
        ctypes.cast(x_ptr, ctypes.POINTER(ctypes.c_float)), incx,
        ctypes.cast(y_ptr, ctypes.POINTER(ctypes.c_float)), incy,
        ctypes.cast(result_ptr, ctypes.POINTER(ctypes.c_float))
    )
    if status != 0:
        raise RuntimeError(f"cublasSdot_v2 failed with status {status}")
    return result
```

"""

    def _build_scal_example(self) -> str:
        """构建 SCAL 示例（使用缓存模式）"""
        return """**SCAL Example** (x = alpha * x, with global caching):
```python
import torch
import ctypes

_libcublas = None
_cublas_handle = None
_cublas_set_pointer_mode = None
_cublas_func = None
_scalar_cache = {}

def _get_cublas_lib():
    global _libcublas
    if _libcublas is None:
        _libcublas = ctypes.CDLL('/usr/local/cuda/lib64/libcublas.so.12')
    return _libcublas

def _get_or_create_handle():
    global _cublas_handle, _cublas_set_pointer_mode
    if _cublas_handle is None:
        libcublas = _get_cublas_lib()
        cublasCreate_v2 = libcublas.cublasCreate_v2
        cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        cublasCreate_v2.restype = ctypes.c_int
        _cublas_handle = ctypes.c_void_p()
        status = cublasCreate_v2(ctypes.byref(_cublas_handle))
        if status != 0:
            raise RuntimeError(f"cublasCreate_v2 failed with status {status}")
        _cublas_set_pointer_mode = libcublas.cublasSetPointerMode_v2
        _cublas_set_pointer_mode.argtypes = [ctypes.c_void_p, ctypes.c_int]
        _cublas_set_pointer_mode.restype = ctypes.c_int
        _cublas_set_pointer_mode(_cublas_handle, 1)
    return _cublas_handle

def _get_cublas_func():
    global _cublas_func
    if _cublas_func is None:
        libcublas = _get_cublas_lib()
        _cublas_func = libcublas.cublasSscal_v2
        _cublas_func.argtypes = [
            ctypes.c_void_p, ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p, ctypes.c_int
        ]
        _cublas_func.restype = ctypes.c_int
    return _cublas_func

def _get_scalar_gpu(key, value, dtype):
    cache_key = (key, dtype, float(value))
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasSscal_v2(n, alpha, x, incx):
    '''ctypes cuBLAS C API baseline for cublasSscal_v2'''
    handle = _get_or_create_handle()
    func = _get_cublas_func()
    x_ptr = ctypes.c_void_p(x.data_ptr())
    alpha_gpu = _get_scalar_gpu('alpha', alpha, torch.float32)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())
    func(handle, n, ctypes.cast(alpha_ptr, ctypes.POINTER(ctypes.c_float)), x_ptr, incx)
    return x
```

"""

    def _build_examples(self, config: Dict[str, Any]) -> str:
        """构建示例代码"""
        base_op = config['base_op'].lower()

        examples = "## Examples\n\n"

        # 根据 base_op 提供不同的示例
        if base_op == 'axpy':
            examples += """**AXPY Example** (y = alpha * x + y):
```python
import torch
import ctypes

# Global variables for caching (initialized once, reused)
_libcublas = None
_cublas_handle = None
_cublas_set_pointer_mode = None
_cublas_saxpy_func = None
_alpha_cache = {}  # Cache GPU tensors for alpha values

def _get_cublas_lib():
    global _libcublas
    if _libcublas is None:
        _libcublas = ctypes.CDLL('/usr/local/cuda/lib64/libcublas.so.12')
    return _libcublas

def _get_or_create_handle():
    '''Get or create global cuBLAS handle (reused across calls)'''
    global _cublas_handle, _cublas_set_pointer_mode
    if _cublas_handle is None:
        libcublas = _get_cublas_lib()

        # Create handle
        cublasCreate_v2 = libcublas.cublasCreate_v2
        cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        cublasCreate_v2.restype = ctypes.c_int
        _cublas_handle = ctypes.c_void_p()
        status = cublasCreate_v2(ctypes.byref(_cublas_handle))
        if status != 0:
            raise RuntimeError(f"cublasCreate_v2 failed with status {status}")

        # Setup SetPointerMode function (once)
        _cublas_set_pointer_mode = libcublas.cublasSetPointerMode_v2
        _cublas_set_pointer_mode.argtypes = [ctypes.c_void_p, ctypes.c_int]
        _cublas_set_pointer_mode.restype = ctypes.c_int

        # Set to device mode (once)
        _cublas_set_pointer_mode(_cublas_handle, 1)

    return _cublas_handle

def _get_saxpy_func():
    '''Get cublasSaxpy_v2 function with signature set (once)'''
    global _cublas_saxpy_func
    if _cublas_saxpy_func is None:
        libcublas = _get_cublas_lib()
        _cublas_saxpy_func = libcublas.cublasSaxpy_v2
        _cublas_saxpy_func.argtypes = [
            ctypes.c_void_p, ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.c_int
        ]
        _cublas_saxpy_func.restype = ctypes.c_int
    return _cublas_saxpy_func

def _get_alpha_gpu(alpha):
    '''Get or create cached alpha GPU tensor'''
    if alpha not in _alpha_cache:
        _alpha_cache[alpha] = torch.tensor([alpha], dtype=torch.float32, device='cuda')
    return _alpha_cache[alpha]

def cublasSaxpy_v2(n, alpha, x, incx, y, incy):
    '''ctypes cuBLAS C API baseline for cublasSaxpy_v2: y = alpha * x + y'''
    handle = _get_or_create_handle()
    func = _get_saxpy_func()

    # Get tensor pointers
    x_ptr = ctypes.c_void_p(x.data_ptr())
    y_ptr = ctypes.c_void_p(y.data_ptr())

    # Get cached alpha GPU tensor
    alpha_gpu = _get_alpha_gpu(alpha)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())

    # Call cuBLAS
    func(
        handle, n,
        ctypes.cast(alpha_ptr, ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(x_ptr, ctypes.POINTER(ctypes.c_float)), incx,
        ctypes.cast(y_ptr, ctypes.POINTER(ctypes.c_float)), incy
    )

    return y

if __name__ == "__main__":
    n = 100
    alpha = 2.5
    x = torch.randn(n, dtype=torch.float32, device='cuda')
    y = torch.randn(n, dtype=torch.float32, device='cuda')
    y_original = y.clone()

    result = cublasSaxpy_v2(n, alpha, x, 1, y, 1)

    assert result is not None
    expected = alpha * x + y_original
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasSaxpy_v2 test passed")
```

"""
        elif base_op == 'gemm':
            examples += self._build_gemm_example()
        elif base_op == 'gemv':
            examples += self._build_gemv_example()
        elif base_op in ['dot', 'dotc', 'dotu']:
            examples += self._build_dot_example()
        elif base_op == 'scal':
            examples += self._build_scal_example()

        return examples

    def _build_constraints(self, func_name: str, config: Dict[str, Any]) -> str:
        """构建约束条件"""
        # 提取 C API 调用名（去掉 cublas 前缀和 _v2/_64 后缀）
        c_api_call = func_name.replace('cublas', '').replace('_v2', '').replace('_64', '')
        c_api_call = c_api_call[0].lower() + c_api_call[1:] if c_api_call else ''

        # 获取返回参数
        return_params = config.get('return_params', [])
        if len(return_params) == 1:
            return_info = f"Return the modified `{return_params[0]}` tensor"
        elif len(return_params) > 1:
            return_info = f"Return a tuple of modified tensors: ({', '.join(return_params)})"
        else:
            return_info = "Return the modified tensor"

        constraints = f"""## Important Constraints

**For the baseline function:**
1. **Function name**: Must be `{func_name}` (exact match with c_api field)
2. **Library loading**: Load cuBLAS with `libcublas = ctypes.CDLL('/usr/local/cuda/lib64/libcublas.so.12')`
3. **NO CuPy**: ABSOLUTELY NO CuPy imports or usage - pure ctypes only
4. **Handle creation**: Create handle using cublasCreate_v2 with proper argtypes/restype
5. **Tensor pointers**: Use `ctypes.c_void_p(tensor.data_ptr())` to get raw pointer from PyTorch tensors
6. **Scalar pointers**: For scalar params (alpha, beta), create GPU tensor: `alpha_gpu = torch.tensor([alpha], dtype=torch.float32, device='cuda')`, then use `ctypes.c_void_p(alpha_gpu.data_ptr())`
7. **Pointer mode**: MUST set pointer mode to device (1) before calling cuBLAS using cublasSetPointerMode_v2
8. **Function signature**: Define argtypes for the cuBLAS function using ctypes types (c_void_p, c_int, POINTER(c_float), etc.)
9. **Pointer casting**: Cast void pointers to typed pointers using `ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float))`
10. **Scalar cache key**: CRITICAL — `_get_scalar_gpu` cache key MUST include the scalar value: `(key, dtype, float(value))` for real types, `(key, dtype, complex(value))` for complex types. Never use just `key` as cache key — different alpha/beta values would collide.
11. **Return value**: {return_info}
12. **Parameter order**: Follow the exact C API signature (handle is first parameter)
13. **No handle destruction**: Do NOT destroy the handle — it is cached globally and reused across calls. Do NOT call cublasDestroy_v2.
14. **No runtime checks**: Do NOT add is_cuda/dtype/contiguous/dim/size checks in the main function. Do NOT call `.item()` on GPU tensors. Do NOT call `fill_()` on cached scalars.

**For the test code (in `if __name__ == "__main__":`)**:
13. **Test data**: Create appropriate test tensors on GPU with correct dtype
14. **Clone originals**: Clone input tensors before calling baseline (for comparison)
15. **Call baseline**: Invoke the baseline function with test data
16. **Assert not None**: Verify result is not None
17. **PyTorch reference**: Compute expected result using simple PyTorch operations
18. **CRITICAL - Column-Major Layout**: cuBLAS uses Fortran column-major layout, PyTorch uses C row-major layout
    - For GEMM operations: PyTorch A(m,k) @ B(k,n) = C(m,n) is stored row-major
    - cuBLAS expects column-major, so you need to transpose the computation
    - Solution: Use A.t().contiguous() and B.t().contiguous() OR adjust the reference computation
    - For matrix-vector operations: Similar considerations apply for matrix storage
19. **Numerical check**: Use `torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)`
20. **Print success**: Print "✓ {func_name} test passed" on success

**General:**
21. **No extra code**: Do NOT add device context managers or any other code not shown in examples
22. **No explanations**: Generate ONLY the Python code, no markdown text
23. **Code block**: Wrap code in ```python ... ``` block
24. **Imports**: Only import torch and ctypes at the top (NO CuPy)
25. **Docstring**: Brief one-line description mentioning "ctypes cuBLAS C API baseline for {func_name}"

Generate the complete code (baseline function + test code) now.
"""
        return constraints



class BaselineCheckPromptBuilder:
    """构建 cuBLAS baseline 测试代码生成的 prompt"""

    def __init__(self):
        pass

    def build_prompt(self, func_name: str, config: Dict[str, Any], baseline_code: str) -> str:
        """
        构建生成测试代码的完整 prompt

        Args:
            func_name: cuBLAS C API 函数名 (如 'cublasSgemm_v2')
            config: 函数配置字典
            baseline_code: 已生成的 baseline 代码

        Returns:
            完整的 prompt 字符串
        """
        prompt = self._build_header(func_name, config)
        prompt += self._build_baseline_code(baseline_code)
        prompt += self._build_requirements()
        prompt += self._build_template()
        prompt += self._build_examples(config)
        prompt += self._build_constraints(func_name, config)

        return prompt

    def _build_header(self, func_name: str, config: Dict[str, Any]) -> str:
        """构建 prompt 头部"""
        header = f"""You are an expert in CUDA programming and testing. Generate a minimal test function to verify the correctness of a cuBLAS baseline implementation.

## Function Information

**Function Name**: {func_name}
**Base Operation**: {config['base_op']}
**Data Type**: {config['dtype']}
**BLAS Level**: {config['level']}

**Description**: {config['description']}

**Parameters**: {', '.join(config['params'])}
**Return Value**: {', '.join(config.get('return_params', []))}

"""
        return header
    def _build_baseline_code(self, baseline_code: str) -> str:
        """显示已生成的 baseline 代码"""
        return f"""## Baseline Code to Test

```python
{baseline_code}
```

"""

    def _build_requirements(self) -> str:
        """构建测试需求"""
        return """## Test Requirements

1. **Test function name**: `test_{func_name}()`
2. **Import baseline**: Import the baseline function from the generated module
3. **Create test data**: Generate appropriate test tensors on GPU
4. **Call baseline**: Invoke the baseline function with test data
5. **Basic checks**: Verify return value is not None, correct shape, correct device, correct dtype
6. **Numerical check**: Compare with PyTorch reference implementation
7. **Print result**: Print success message if test passes
8. **Handle errors**: Catch and report any exceptions

"""
    def _build_template(self) -> str:
        """构建测试代码模板"""
        return """## Template Format

```python
import torch
import sys
sys.path.insert(0, '.')
from {module_name} import {func_name}

def test_{func_name}():
    \"\"\"Test {func_name} baseline implementation\"\"\"
    # Create test data
    x = torch.randn(..., dtype=torch.float32, device='cuda')
    
    # Call baseline
    result = {func_name}(...)
    
    # Basic checks
    assert result is not None
    assert result.shape == expected_shape
    assert result.device.type == 'cuda'
    
    # Numerical check with PyTorch reference
    expected = torch_reference_implementation(...)
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)
    
    print(f"✓ {func_name} test passed")

if __name__ == "__main__":
    test_{func_name}()
```

"""
    def _build_examples(self, config: Dict[str, Any]) -> str:
        """构建测试示例"""
        base_op = config['base_op'].lower()
        
        examples = "## Examples\n\n"
        
        if base_op == 'axpy':
            examples += """**AXPY Test Example** (y = alpha * x + y):
```python
import torch
import sys
sys.path.insert(0, '.')
from cublasSaxpy_v2 import cublasSaxpy_v2

def test_cublasSaxpy_v2():
    \"\"\"Test cublasSaxpy_v2: y = alpha * x + y\"\"\"
    n = 100
    alpha = 2.5
    x = torch.randn(n, dtype=torch.float32, device='cuda')
    y = torch.randn(n, dtype=torch.float32, device='cuda')
    y_original = y.clone()
    
    result = cublasSaxpy_v2(n, alpha, x, 1, y, 1)
    
    assert result is not None
    assert result.shape == (n,)
    assert result.device.type == 'cuda'
    
    expected = alpha * x + y_original
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    
    print("✓ cublasSaxpy_v2 test passed")

if __name__ == "__main__":
    test_cublasSaxpy_v2()
```

"""
        elif base_op == 'scal':
            examples += """**SCAL Test Example** (x = alpha * x):
```python
import torch
import sys
sys.path.insert(0, '.')
from cublasSscal_v2 import cublasSscal_v2

def test_cublasSscal_v2():
    \"\"\"Test cublasSscal_v2: x = alpha * x\"\"\"
    n = 100
    alpha = 2.5
    x = torch.randn(n, dtype=torch.float32, device='cuda')
    x_original = x.clone()
    
    result = cublasSscal_v2(n, alpha, x, 1)
    
    assert result is not None
    assert result.shape == (n,)
    assert result.device.type == 'cuda'
    
    expected = alpha * x_original
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    
    print("✓ cublasSscal_v2 test passed")

if __name__ == "__main__":
    test_cublasSscal_v2()
```

"""
        elif base_op in ['dot', 'dotc', 'dotu']:
            examples += """**DOT Test Example** (result = x^T * y):
```python
import torch
import sys
sys.path.insert(0, '.')
from cublasSdot_v2 import cublasSdot_v2

def test_cublasSdot_v2():
    \"\"\"Test cublasSdot_v2: result = x^T * y\"\"\"
    n = 100
    x = torch.randn(n, dtype=torch.float32, device='cuda')
    y = torch.randn(n, dtype=torch.float32, device='cuda')
    result = torch.zeros(1, dtype=torch.float32, device='cuda')
    
    result = cublasSdot_v2(n, x, 1, y, 1, result)
    
    assert result is not None
    assert result.shape == (1,)
    assert result.device.type == 'cuda'
    
    expected = torch.dot(x, y).reshape(1)
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    
    print("✓ cublasSdot_v2 test passed")

if __name__ == "__main__":
    test_cublasSdot_v2()
```

"""
        elif base_op == 'gemm':
            examples += """**GEMM Test Example** (C = alpha * A @ B + beta * C):
```python
import torch
import sys
sys.path.insert(0, '.')
from cublasSgemm_v2 import cublasSgemm_v2

def test_cublasSgemm_v2():
    \"\"\"Test cublasSgemm_v2: C = alpha * A @ B + beta * C\"\"\"
    m, n, k = 64, 64, 64
    alpha, beta = 1.0, 0.0
    A = torch.randn(m, k, dtype=torch.float32, device='cuda')
    B = torch.randn(k, n, dtype=torch.float32, device='cuda')
    C = torch.randn(m, n, dtype=torch.float32, device='cuda')
    C_original = C.clone()
    
    result = cublasSgemm_v2(0, 0, m, n, k, alpha, A, m, B, k, beta, C, m)
    
    assert result is not None
    assert result.shape == (m, n)
    assert result.device.type == 'cuda'
    
    expected = alpha * (A @ B) + beta * C_original
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)
    
    print("✓ cublasSgemm_v2 test passed")

if __name__ == "__main__":
    test_cublasSgemm_v2()
```

"""
        
        return examples
