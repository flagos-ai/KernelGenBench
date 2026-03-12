"""
cuBLAS C API 配置文件
自动生成，每个词条对应一个 cuBLAS C API 函数

总计: 284 个函数
- Level 1: 52 个
- Level 2: 124 个
- Level 3: 108 个
"""

CUBLAS_C_API_CONFIG = {

    # ======================================================================
    # Level 1
    # ======================================================================

    'cublasCdotc_v2': {
        'base_op': 'dotc',
        'dtype': 'complex64',
        'level': 1,
        'variant': 'base',
        'description': 'The `cublasCdotc_v2` function computes the conjugated dot product of two complex single-precision vectors, returning the result in a complex variable. It supports arbitrary stride increments for both input vectors and requires a cuBLAS handle for execution context.',

        # C API 信息
        'c_api': 'cublasCdotc_v2',
        'params': ['n', 'x', 'incx', 'y', 'incy', 'result'],
        'param_types': {
            'n': 'int',
            'x': 'const cuComplex*',
            'incx': 'int',
            'y': 'const cuComplex*',
            'incy': 'int',
            'result': 'cuComplex*'
        },

        # 参数分类
        'tensor_params': ['x', 'y', 'result'],
        'scalar_params': [],
        'inout_params': ['result'],
        'return_params': ['result'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCdotc_v2( cublasHandle_t handle, int n, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* result);',
    },
    'cublasCdotc_v2_64': {
        'base_op': 'dotc',
        'dtype': 'complex64',
        'level': 1,
        'variant': '_64',
        'description': 'The function `cublasCdotc_v2_64` computes the conjugated dot product of two complex single-precision vectors using 64-bit indexing, storing the result in a complex variable. It supports arbitrary stride increments for both input vectors and requires a cuBLAS handle for execution context.',

        # C API 信息
        'c_api': 'cublasCdotc_v2_64',
        'params': ['n', 'x', 'incx', 'y', 'incy', 'result'],
        'param_types': {
            'n': 'int64_t',
            'x': 'const cuComplex*',
            'incx': 'int64_t',
            'y': 'const cuComplex*',
            'incy': 'int64_t',
            'result': 'cuComplex*'
        },

        # 参数分类
        'tensor_params': ['x', 'y', 'result'],
        'scalar_params': [],
        'inout_params': ['result'],
        'return_params': ['result'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCdotc_v2_64(cublasHandle_t handle, int64_t n, const cuComplex* x, int64_t incx, const cuComplex* y, int64_t incy, cuComplex* result);',
    },
    'cublasCdotu_v2': {
        'base_op': 'dotu',
        'dtype': 'complex64',
        'level': 1,
        'variant': 'base',
        'description': 'The `cublasCdotu_v2` function computes the unconjugated dot product of two complex single-precision vectors `x` and `y`, storing the result in `result`. It supports arbitrary stride values (`incx`, `incy`) for accessing elements in the input vectors.',

        # C API 信息
        'c_api': 'cublasCdotu_v2',
        'params': ['n', 'x', 'incx', 'y', 'incy', 'result'],
        'param_types': {
            'n': 'int',
            'x': 'const cuComplex*',
            'incx': 'int',
            'y': 'const cuComplex*',
            'incy': 'int',
            'result': 'cuComplex*'
        },

        # 参数分类
        'tensor_params': ['x', 'y', 'result'],
        'scalar_params': [],
        'inout_params': ['result'],
        'return_params': ['result'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCdotu_v2( cublasHandle_t handle, int n, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* result);',
    },
    'cublasCdotu_v2_64': {
        'base_op': 'dotu',
        'dtype': 'complex64',
        'level': 1,
        'variant': '_64',
        'description': 'The `cublasCdotu_v2_64` function computes the unconjugated dot product of two complex single-precision vectors using 64-bit indexing, storing the result in a complex variable. It supports arbitrary stride increments for both input vectors and operates on the GPU via the cuBLAS handle.',

        # C API 信息
        'c_api': 'cublasCdotu_v2_64',
        'params': ['n', 'x', 'incx', 'y', 'incy', 'result'],
        'param_types': {
            'n': 'int64_t',
            'x': 'const cuComplex*',
            'incx': 'int64_t',
            'y': 'const cuComplex*',
            'incy': 'int64_t',
            'result': 'cuComplex*'
        },

        # 参数分类
        'tensor_params': ['x', 'y', 'result'],
        'scalar_params': [],
        'inout_params': ['result'],
        'return_params': ['result'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCdotu_v2_64(cublasHandle_t handle, int64_t n, const cuComplex* x, int64_t incx, const cuComplex* y, int64_t incy, cuComplex* result);',
    },
    'cublasDdot_v2': {
        'base_op': 'dot',
        'dtype': 'float64',
        'level': 1,
        'variant': 'base',
        'description': 'The `cublasDdot_v2` function computes the double-precision dot product of two vectors `x` and `y` with specified increments `incx` and `incy`, storing the result in `result`. It operates on vectors of length `n` and requires a cuBLAS handle for execution.',

        # C API 信息
        'c_api': 'cublasDdot_v2',
        'params': ['n', 'x', 'incx', 'y', 'incy', 'result'],
        'param_types': {
            'n': 'int',
            'x': 'const double*',
            'incx': 'int',
            'y': 'const double*',
            'incy': 'int',
            'result': 'double*'
        },

        # 参数分类
        'tensor_params': ['x', 'y', 'result'],
        'scalar_params': [],
        'inout_params': ['result'],
        'return_params': ['result'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDdot_v2(cublasHandle_t handle, int n, const double* x, int incx, const double* y, int incy, double* result);',
    },
    'cublasDdot_v2_64': {
        'base_op': 'dot',
        'dtype': 'float64',
        'level': 1,
        'variant': '_64',
        'description': 'The `cublasDdot_v2_64` function computes the double-precision dot product of two vectors `x` and `y` with 64-bit integer parameters for large-scale operations, storing the result in `result`. It supports arbitrary stride values (`incx`, `incy`) for vector elements and requires a cuBLAS handle for execution.',

        # C API 信息
        'c_api': 'cublasDdot_v2_64',
        'params': ['n', 'x', 'incx', 'y', 'incy', 'result'],
        'param_types': {
            'n': 'int64_t',
            'x': 'const double*',
            'incx': 'int64_t',
            'y': 'const double*',
            'incy': 'int64_t',
            'result': 'double*'
        },

        # 参数分类
        'tensor_params': ['x', 'y', 'result'],
        'scalar_params': [],
        'inout_params': ['result'],
        'return_params': ['result'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDdot_v2_64( cublasHandle_t handle, int64_t n, const double* x, int64_t incx, const double* y, int64_t incy, double* result);',
    },
    'cublasSdot_v2': {
        'base_op': 'dot',
        'dtype': 'float32',
        'level': 1,
        'variant': 'base',
        'description': 'The `cublasSdot_v2` function computes the single-precision (float32) dot product of two vectors `x` and `y` with optional stride increments, storing the result in `result`. It operates on vectors of length `n` with strides `incx` and `incy` for `x` and `y` respectively.',

        # C API 信息
        'c_api': 'cublasSdot_v2',
        'params': ['n', 'x', 'incx', 'y', 'incy', 'result'],
        'param_types': {
            'n': 'int',
            'x': 'const float*',
            'incx': 'int',
            'y': 'const float*',
            'incy': 'int',
            'result': 'float*'
        },

        # 参数分类
        'tensor_params': ['x', 'y', 'result'],
        'scalar_params': [],
        'inout_params': ['result'],
        'return_params': ['result'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSdot_v2(cublasHandle_t handle, int n, const float* x, int incx, const float* y, int incy, float* result);',
    },
    'cublasSdot_v2_64': {
        'base_op': 'dot',
        'dtype': 'float32',
        'level': 1,
        'variant': '_64',
        'description': 'Computes the single-precision dot product of two vectors x and y with 64-bit integer parameters, storing the result in *result. The function supports arbitrary stride values (incx, incy) for accessing elements in the input vectors.',

        # C API 信息
        'c_api': 'cublasSdot_v2_64',
        'params': ['n', 'x', 'incx', 'y', 'incy', 'result'],
        'param_types': {
            'n': 'int64_t',
            'x': 'const float*',
            'incx': 'int64_t',
            'y': 'const float*',
            'incy': 'int64_t',
            'result': 'float*'
        },

        # 参数分类
        'tensor_params': ['x', 'y', 'result'],
        'scalar_params': [],
        'inout_params': ['result'],
        'return_params': ['result'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSdot_v2_64( cublasHandle_t handle, int64_t n, const float* x, int64_t incx, const float* y, int64_t incy, float* result);',
    },
    'cublasZdotc_v2': {
        'base_op': 'dotc',
        'dtype': 'complex128',
        'level': 1,
        'variant': 'base',
        'description': 'The `cublasZdotc_v2` function computes the conjugated dot product of two double-precision complex vectors, storing the result in a complex variable. It supports arbitrary stride increments for both input vectors and operates on the GPU using the provided cuBLAS handle.',

        # C API 信息
        'c_api': 'cublasZdotc_v2',
        'params': ['n', 'x', 'incx', 'y', 'incy', 'result'],
        'param_types': {
            'n': 'int',
            'x': 'const cuDoubleComplex*',
            'incx': 'int',
            'y': 'const cuDoubleComplex*',
            'incy': 'int',
            'result': 'cuDoubleComplex*'
        },

        # 参数分类
        'tensor_params': ['x', 'y', 'result'],
        'scalar_params': [],
        'inout_params': ['result'],
        'return_params': ['result'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZdotc_v2(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* result);',
    },
    'cublasZdotc_v2_64': {
        'base_op': 'dotc',
        'dtype': 'complex128',
        'level': 1,
        'variant': '_64',
        'description': 'The `cublasZdotc_v2_64` function computes the conjugated dot product of two double-precision complex vectors using 64-bit integers for large array support, storing the result in a complex variable. It takes input vectors `x` and `y` with specified strides (`incx`, `incy`) and length `n`, and returns the result through the `result` parameter.',

        # C API 信息
        'c_api': 'cublasZdotc_v2_64',
        'params': ['n', 'x', 'incx', 'y', 'incy', 'result'],
        'param_types': {
            'n': 'int64_t',
            'x': 'const cuDoubleComplex*',
            'incx': 'int64_t',
            'y': 'const cuDoubleComplex*',
            'incy': 'int64_t',
            'result': 'cuDoubleComplex*'
        },

        # 参数分类
        'tensor_params': ['x', 'y', 'result'],
        'scalar_params': [],
        'inout_params': ['result'],
        'return_params': ['result'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZdotc_v2_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* y, int64_t incy, cuDoubleComplex* result);',
    },
    'cublasZdotu_v2': {
        'base_op': 'dotu',
        'dtype': 'complex128',
        'level': 1,
        'variant': 'base',
        'description': 'The `cublasZdotu_v2` function computes the unconjugated dot product of two double-precision complex vectors `x` and `y` of length `n`, with specified increments `incx` and `incy`, storing the result in `result`. It operates under the provided cuBLAS handle for GPU execution.',

        # C API 信息
        'c_api': 'cublasZdotu_v2',
        'params': ['n', 'x', 'incx', 'y', 'incy', 'result'],
        'param_types': {
            'n': 'int',
            'x': 'const cuDoubleComplex*',
            'incx': 'int',
            'y': 'const cuDoubleComplex*',
            'incy': 'int',
            'result': 'cuDoubleComplex*'
        },

        # 参数分类
        'tensor_params': ['x', 'y', 'result'],
        'scalar_params': [],
        'inout_params': ['result'],
        'return_params': ['result'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZdotu_v2(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* result);',
    },
    'cublasZdotu_v2_64': {
        'base_op': 'dotu',
        'dtype': 'complex128',
        'level': 1,
        'variant': '_64',
        'description': 'The function `cublasZdotu_v2_64` computes the unconjugated dot product of two double-precision complex vectors using 64-bit integers for indexing, storing the result in a complex variable. It supports arbitrary stride increments for both input vectors.',

        # C API 信息
        'c_api': 'cublasZdotu_v2_64',
        'params': ['n', 'x', 'incx', 'y', 'incy', 'result'],
        'param_types': {
            'n': 'int64_t',
            'x': 'const cuDoubleComplex*',
            'incx': 'int64_t',
            'y': 'const cuDoubleComplex*',
            'incy': 'int64_t',
            'result': 'cuDoubleComplex*'
        },

        # 参数分类
        'tensor_params': ['x', 'y', 'result'],
        'scalar_params': [],
        'inout_params': ['result'],
        'return_params': ['result'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZdotu_v2_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* y, int64_t incy, cuDoubleComplex* result);',
    },
    'cublasDasum_v2': {
        'base_op': 'asum',
        'dtype': 'float64',
        'level': 1,
        'variant': 'base',
        'description': 'The `cublasDasum_v2` function computes the sum of absolute values of double-precision floating-point elements in vector `x` with stride `incx`, storing the result in `result`. It is a Level 1 BLAS operation that requires a cuBLAS handle for execution.',

        # C API 信息
        'c_api': 'cublasDasum_v2',
        'params': ['n', 'x', 'incx', 'result'],
        'param_types': {
            'n': 'int',
            'x': 'const double*',
            'incx': 'int',
            'result': 'double*'
        },

        # 参数分类
        'tensor_params': ['x', 'result'],
        'scalar_params': [],
        'inout_params': ['result'],
        'return_params': ['result'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDasum_v2(cublasHandle_t handle, int n, const double* x, int incx, double* result);',
    },
    'cublasDasum_v2_64': {
        'base_op': 'asum',
        'dtype': 'float64',
        'level': 1,
        'variant': '_64',
        'description': 'The `cublasDasum_v2_64` function computes the sum of absolute values of double-precision elements in vector `x` with stride `incx` and stores the result, supporting 64-bit integers for large vector sizes. It is a Level 1 BLAS operation designed for use with a cuBLAS handle.',

        # C API 信息
        'c_api': 'cublasDasum_v2_64',
        'params': ['n', 'x', 'incx', 'result'],
        'param_types': {
            'n': 'int64_t',
            'x': 'const double*',
            'incx': 'int64_t',
            'result': 'double*'
        },

        # 参数分类
        'tensor_params': ['x', 'result'],
        'scalar_params': [],
        'inout_params': ['result'],
        'return_params': ['result'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDasum_v2_64(cublasHandle_t handle, int64_t n, const double* x, int64_t incx, double* result);',
    },
    'cublasSasum_v2': {
        'base_op': 'asum',
        'dtype': 'float32',
        'level': 1,
        'variant': 'base',
        'description': 'This function computes the sum of absolute values of single-precision floating-point elements in vector x, storing the result. It supports a stride between elements specified by incx.',

        # C API 信息
        'c_api': 'cublasSasum_v2',
        'params': ['n', 'x', 'incx', 'result'],
        'param_types': {
            'n': 'int',
            'x': 'const float*',
            'incx': 'int',
            'result': 'float*'
        },

        # 参数分类
        'tensor_params': ['x', 'result'],
        'scalar_params': [],
        'inout_params': ['result'],
        'return_params': ['result'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSasum_v2(cublasHandle_t handle, int n, const float* x, int incx, float* result);',
    },
    'cublasSasum_v2_64': {
        'base_op': 'asum',
        'dtype': 'float32',
        'level': 1,
        'variant': '_64',
        'description': 'This function computes the sum of absolute values of single-precision floating-point elements in vector x with stride incx, using 64-bit integers for large array support, and stores the result. It is a BLAS Level 1 operation that handles large datasets through its 64-bit parameters.',

        # C API 信息
        'c_api': 'cublasSasum_v2_64',
        'params': ['n', 'x', 'incx', 'result'],
        'param_types': {
            'n': 'int64_t',
            'x': 'const float*',
            'incx': 'int64_t',
            'result': 'float*'
        },

        # 参数分类
        'tensor_params': ['x', 'result'],
        'scalar_params': [],
        'inout_params': ['result'],
        'return_params': ['result'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSasum_v2_64(cublasHandle_t handle, int64_t n, const float* x, int64_t incx, float* result);',
    },
    'cublasCaxpy_v2': {
        'base_op': 'axpy',
        'dtype': 'complex64',
        'level': 1,
        'variant': 'base',
        'description': 'The `cublasCaxpy_v2` function performs the vector-vector operation `y = alpha * x + y` for single-precision complex numbers, where `x` and `y` are vectors with specified increments. It is a Level 1 BLAS operation optimized for CUDA-enabled GPUs.',

        # C API 信息
        'c_api': 'cublasCaxpy_v2',
        'params': ['n', 'alpha', 'x', 'incx', 'y', 'incy'],
        'param_types': {
            'n': 'int',
            'alpha': 'const cuComplex*',
            'x': 'const cuComplex*',
            'incx': 'int',
            'y': 'cuComplex*',
            'incy': 'int'
        },

        # 参数分类
        'tensor_params': ['x', 'y'],
        'scalar_params': ['alpha'],
        'inout_params': ['y'],
        'return_params': ['y'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCaxpy_v2( cublasHandle_t handle, int n, const cuComplex* alpha, const cuComplex* x, int incx, cuComplex* y, int incy);',
    },
    'cublasCaxpy_v2_64': {
        'base_op': 'axpy',
        'dtype': 'complex64',
        'level': 1,
        'variant': '_64',
        'description': 'The `cublasCaxpy_v2_64` function performs a 64-bit complex single-precision vector addition, multiplying vector `x` by scalar `alpha` and adding the result to vector `y` with specified increments. It is a BLAS Level 1 operation supporting large arrays via 64-bit integer parameters.',

        # C API 信息
        'c_api': 'cublasCaxpy_v2_64',
        'params': ['n', 'alpha', 'x', 'incx', 'y', 'incy'],
        'param_types': {
            'n': 'int64_t',
            'alpha': 'const cuComplex*',
            'x': 'const cuComplex*',
            'incx': 'int64_t',
            'y': 'cuComplex*',
            'incy': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['x', 'y'],
        'scalar_params': ['alpha'],
        'inout_params': ['y'],
        'return_params': ['y'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCaxpy_v2_64(cublasHandle_t handle, int64_t n, const cuComplex* alpha, const cuComplex* x, int64_t incx, cuComplex* y, int64_t incy);',
    },
    'cublasCcopy_v2': {
        'base_op': 'copy',
        'dtype': 'complex64',
        'level': 1,
        'variant': 'base',
        'description': 'Copies a complex single-precision vector `x` to vector `y` with specified increments `incx` and `incy` for each vector. The operation is performed on the GPU using the provided cuBLAS handle.',

        # C API 信息
        'c_api': 'cublasCcopy_v2',
        'params': ['n', 'x', 'incx', 'y', 'incy'],
        'param_types': {
            'n': 'int',
            'x': 'const cuComplex*',
            'incx': 'int',
            'y': 'cuComplex*',
            'incy': 'int'
        },

        # 参数分类
        'tensor_params': ['x', 'y'],
        'scalar_params': [],
        'inout_params': ['y'],
        'return_params': ['y'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCcopy_v2(cublasHandle_t handle, int n, const cuComplex* x, int incx, cuComplex* y, int incy);',
    },
    'cublasCcopy_v2_64': {
        'base_op': 'copy',
        'dtype': 'complex64',
        'level': 1,
        'variant': '_64',
        'description': 'This function copies a complex single-precision vector `x` to vector `y` with specified increments, supporting 64-bit integers for large arrays. It operates on complex64 data and is part of the BLAS Level 1 routines.',

        # C API 信息
        'c_api': 'cublasCcopy_v2_64',
        'params': ['n', 'x', 'incx', 'y', 'incy'],
        'param_types': {
            'n': 'int64_t',
            'x': 'const cuComplex*',
            'incx': 'int64_t',
            'y': 'cuComplex*',
            'incy': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['x', 'y'],
        'scalar_params': [],
        'inout_params': ['y'],
        'return_params': ['y'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCcopy_v2_64(cublasHandle_t handle, int64_t n, const cuComplex* x, int64_t incx, cuComplex* y, int64_t incy);',
    },
    'cublasCscal_v2': {
        'base_op': 'scal',
        'dtype': 'complex64',
        'level': 1,
        'variant': 'base',
        'description': 'The `cublasCscal_v2` function scales a complex single-precision vector `x` by a complex scalar `alpha`, where `n` specifies the number of elements and `incx` defines the storage spacing between elements. It operates on the GPU using the provided cuBLAS handle for execution management.',

        # C API 信息
        'c_api': 'cublasCscal_v2',
        'params': ['n', 'alpha', 'x', 'incx'],
        'param_types': {
            'n': 'int',
            'alpha': 'const cuComplex*',
            'x': 'cuComplex*',
            'incx': 'int'
        },

        # 参数分类
        'tensor_params': ['x'],
        'scalar_params': ['alpha'],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCscal_v2(cublasHandle_t handle, int n, const cuComplex* alpha, cuComplex* x, int incx);',
    },
    'cublasCscal_v2_64': {
        'base_op': 'scal',
        'dtype': 'complex64',
        'level': 1,
        'variant': '_64',
        'description': 'The `cublasCscal_v2_64` function scales a complex single-precision vector `x` by a complex scalar `alpha`, where `n` specifies the number of elements and `incx` is the storage spacing between elements, using 64-bit integers for large array support. It operates on the GPU using the cuBLAS library and requires a pre-initialized `handle` for execution.',

        # C API 信息
        'c_api': 'cublasCscal_v2_64',
        'params': ['n', 'alpha', 'x', 'incx'],
        'param_types': {
            'n': 'int64_t',
            'alpha': 'const cuComplex*',
            'x': 'cuComplex*',
            'incx': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['x'],
        'scalar_params': ['alpha'],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCscal_v2_64(cublasHandle_t handle, int64_t n, const cuComplex* alpha, cuComplex* x, int64_t incx);',
    },
    'cublasCswap_v2': {
        'base_op': 'swap',
        'dtype': 'complex64',
        'level': 1,
        'variant': 'base',
        'description': 'The `cublasCswap_v2` function swaps complex single-precision vectors `x` and `y` of length `n`, with optional stride values `incx` and `incy` for each vector. It operates on the GPU using the provided cuBLAS handle for execution management.',

        # C API 信息
        'c_api': 'cublasCswap_v2',
        'params': ['n', 'x', 'incx', 'y', 'incy'],
        'param_types': {
            'n': 'int',
            'x': 'cuComplex*',
            'incx': 'int',
            'y': 'cuComplex*',
            'incy': 'int'
        },

        # 参数分类
        'tensor_params': ['x', 'y'],
        'scalar_params': [],
        'inout_params': ['x', 'y'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCswap_v2(cublasHandle_t handle, int n, cuComplex* x, int incx, cuComplex* y, int incy);',
    },
    'cublasCswap_v2_64': {
        'base_op': 'swap',
        'dtype': 'complex64',
        'level': 1,
        'variant': '_64',
        'description': 'The `cublasCswap_v2_64` function swaps `n` complex64 elements between vectors `x` and `y` with strides `incx` and `incy`, respectively, using 64-bit indexing for large arrays. It operates on the GPU via the specified cuBLAS handle.',

        # C API 信息
        'c_api': 'cublasCswap_v2_64',
        'params': ['n', 'x', 'incx', 'y', 'incy'],
        'param_types': {
            'n': 'int64_t',
            'x': 'cuComplex*',
            'incx': 'int64_t',
            'y': 'cuComplex*',
            'incy': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['x', 'y'],
        'scalar_params': [],
        'inout_params': ['x', 'y'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCswap_v2_64(cublasHandle_t handle, int64_t n, cuComplex* x, int64_t incx, cuComplex* y, int64_t incy);',
    },
    'cublasDaxpy_v2': {
        'base_op': 'axpy',
        'dtype': 'float64',
        'level': 1,
        'variant': 'base',
        'description': 'The `cublasDaxpy_v2` function performs the BLAS Level 1 operation `y = alpha * x + y` using double-precision floating-point numbers, where `x` and `y` are vectors with specified strides `incx` and `incy`, and `alpha` is a scalar.',

        # C API 信息
        'c_api': 'cublasDaxpy_v2',
        'params': ['n', 'alpha', 'x', 'incx', 'y', 'incy'],
        'param_types': {
            'n': 'int',
            'alpha': 'const double*',
            'x': 'const double*',
            'incx': 'int',
            'y': 'double*',
            'incy': 'int'
        },

        # 参数分类
        'tensor_params': ['x', 'y'],
        'scalar_params': ['alpha'],
        'inout_params': ['y'],
        'return_params': ['y'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDaxpy_v2(cublasHandle_t handle, int n, const double* alpha, const double* x, int incx, double* y, int incy);',
    },
    'cublasDaxpy_v2_64': {
        'base_op': 'axpy',
        'dtype': 'float64',
        'level': 1,
        'variant': '_64',
        'description': 'The `cublasDaxpy_v2_64` function performs a double-precision vector addition operation (y = alpha * x + y) with 64-bit integer parameters, where `x` and `y` are vectors with specified increments. It is a BLAS Level 1 routine designed for large-scale computations on CUDA-enabled GPUs.',

        # C API 信息
        'c_api': 'cublasDaxpy_v2_64',
        'params': ['n', 'alpha', 'x', 'incx', 'y', 'incy'],
        'param_types': {
            'n': 'int64_t',
            'alpha': 'const double*',
            'x': 'const double*',
            'incx': 'int64_t',
            'y': 'double*',
            'incy': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['x', 'y'],
        'scalar_params': ['alpha'],
        'inout_params': ['y'],
        'return_params': ['y'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDaxpy_v2_64( cublasHandle_t handle, int64_t n, const double* alpha, const double* x, int64_t incx, double* y, int64_t incy);',
    },
    'cublasDcopy_v2': {
        'base_op': 'copy',
        'dtype': 'float64',
        'level': 1,
        'variant': 'base',
        'description': 'The `cublasDcopy_v2` function copies a double-precision vector `x` to vector `y`, with specified increments `incx` and `incy` for each vector\'s elements. It operates on the GPU using the provided cuBLAS handle for efficient execution.',

        # C API 信息
        'c_api': 'cublasDcopy_v2',
        'params': ['n', 'x', 'incx', 'y', 'incy'],
        'param_types': {
            'n': 'int',
            'x': 'const double*',
            'incx': 'int',
            'y': 'double*',
            'incy': 'int'
        },

        # 参数分类
        'tensor_params': ['x', 'y'],
        'scalar_params': [],
        'inout_params': ['y'],
        'return_params': ['y'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDcopy_v2(cublasHandle_t handle, int n, const double* x, int incx, double* y, int incy);',
    },
    'cublasDcopy_v2_64': {
        'base_op': 'copy',
        'dtype': 'float64',
        'level': 1,
        'variant': '_64',
        'description': 'The `cublasDcopy_v2_64` function copies a double-precision vector `x` to vector `y` with specified increments `incx` and `incy`, supporting 64-bit integer parameters for large-scale operations. It is a BLAS Level 1 function designed for use with the cuBLAS library on GPU-accelerated systems.',

        # C API 信息
        'c_api': 'cublasDcopy_v2_64',
        'params': ['n', 'x', 'incx', 'y', 'incy'],
        'param_types': {
            'n': 'int64_t',
            'x': 'const double*',
            'incx': 'int64_t',
            'y': 'double*',
            'incy': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['x', 'y'],
        'scalar_params': [],
        'inout_params': ['y'],
        'return_params': ['y'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDcopy_v2_64(cublasHandle_t handle, int64_t n, const double* x, int64_t incx, double* y, int64_t incy);',
    },
    'cublasDnrm2_v2': {
        'base_op': 'nrm2',
        'dtype': 'float64',
        'level': 1,
        'variant': 'base',
        'description': 'The `cublasDnrm2_v2` function computes the Euclidean norm (L2 norm) of a double-precision vector `x` with stride `incx` and stores the result in `result`. It is a Level 1 BLAS operation that operates on vectors of length `n`.',

        # C API 信息
        'c_api': 'cublasDnrm2_v2',
        'params': ['n', 'x', 'incx', 'result'],
        'param_types': {
            'n': 'int',
            'x': 'const double*',
            'incx': 'int',
            'result': 'double*'
        },

        # 参数分类
        'tensor_params': ['x', 'result'],
        'scalar_params': [],
        'inout_params': ['result'],
        'return_params': ['result'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDnrm2_v2(cublasHandle_t handle, int n, const double* x, int incx, double* result);',
    },
    'cublasDnrm2_v2_64': {
        'base_op': 'nrm2',
        'dtype': 'float64',
        'level': 1,
        'variant': '_64',
        'description': 'The function `cublasDnrm2_v2_64` computes the Euclidean norm (L2 norm) of a double-precision vector `x` of length `n` with stride `incx`, storing the result in `result`, and supports 64-bit integers for large vector sizes. It operates using the provided cuBLAS handle for execution management.',

        # C API 信息
        'c_api': 'cublasDnrm2_v2_64',
        'params': ['n', 'x', 'incx', 'result'],
        'param_types': {
            'n': 'int64_t',
            'x': 'const double*',
            'incx': 'int64_t',
            'result': 'double*'
        },

        # 参数分类
        'tensor_params': ['x', 'result'],
        'scalar_params': [],
        'inout_params': ['result'],
        'return_params': ['result'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDnrm2_v2_64(cublasHandle_t handle, int64_t n, const double* x, int64_t incx, double* result);',
    },
    'cublasDscal_v2': {
        'base_op': 'scal',
        'dtype': 'float64',
        'level': 1,
        'variant': 'base',
        'description': 'The `cublasDscal_v2` function scales a double-precision vector `x` by multiplying each element with a scalar `alpha`, where `n` specifies the number of elements and `incx` defines the storage spacing between elements. It operates on the GPU using the provided cuBLAS handle for execution management.',

        # C API 信息
        'c_api': 'cublasDscal_v2',
        'params': ['n', 'alpha', 'x', 'incx'],
        'param_types': {
            'n': 'int',
            'alpha': 'const double*',
            'x': 'double*',
            'incx': 'int'
        },

        # 参数分类
        'tensor_params': ['x'],
        'scalar_params': ['alpha'],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDscal_v2(cublasHandle_t handle, int n, const double* alpha, double* x, int incx);',
    },
    'cublasDscal_v2_64': {
        'base_op': 'scal',
        'dtype': 'float64',
        'level': 1,
        'variant': '_64',
        'description': 'The `cublasDscal_v2_64` function scales a double-precision vector `x` of length `n` by a constant `alpha`, with stride `incx`, using 64-bit integers for large array support. It operates on the GPU using the provided cuBLAS handle for execution management.',

        # C API 信息
        'c_api': 'cublasDscal_v2_64',
        'params': ['n', 'alpha', 'x', 'incx'],
        'param_types': {
            'n': 'int64_t',
            'alpha': 'const double*',
            'x': 'double*',
            'incx': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['x'],
        'scalar_params': ['alpha'],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDscal_v2_64(cublasHandle_t handle, int64_t n, const double* alpha, double* x, int64_t incx);',
    },
    'cublasDswap_v2': {
        'base_op': 'swap',
        'dtype': 'float64',
        'level': 1,
        'variant': 'base',
        'description': 'The `cublasDswap_v2` function swaps double-precision vectors `x` and `y` of length `n`, with optional stride increments `incx` and `incy` for each vector. It operates on the GPU using the specified cuBLAS handle for execution management.',

        # C API 信息
        'c_api': 'cublasDswap_v2',
        'params': ['n', 'x', 'incx', 'y', 'incy'],
        'param_types': {
            'n': 'int',
            'x': 'double*',
            'incx': 'int',
            'y': 'double*',
            'incy': 'int'
        },

        # 参数分类
        'tensor_params': ['x', 'y'],
        'scalar_params': [],
        'inout_params': ['x', 'y'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDswap_v2(cublasHandle_t handle, int n, double* x, int incx, double* y, int incy);',
    },
    'cublasDswap_v2_64': {
        'base_op': 'swap',
        'dtype': 'float64',
        'level': 1,
        'variant': '_64',
        'description': 'The `cublasDswap_v2_64` function swaps double-precision (float64) vectors `x` and `y` of length `n` with specified increments `incx` and `incy`, using 64-bit integers for large array support. It operates under the provided cuBLAS handle for GPU-accelerated execution.',

        # C API 信息
        'c_api': 'cublasDswap_v2_64',
        'params': ['n', 'x', 'incx', 'y', 'incy'],
        'param_types': {
            'n': 'int64_t',
            'x': 'double*',
            'incx': 'int64_t',
            'y': 'double*',
            'incy': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['x', 'y'],
        'scalar_params': [],
        'inout_params': ['x', 'y'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDswap_v2_64(cublasHandle_t handle, int64_t n, double* x, int64_t incx, double* y, int64_t incy);',
    },
    'cublasSaxpy_v2': {
        'base_op': 'axpy',
        'dtype': 'float32',
        'level': 1,
        'variant': 'base',
        'description': 'The `cublasSaxpy_v2` function performs the single-precision BLAS operation `y = alpha * x + y`, where `x` and `y` are vectors with specified strides `incx` and `incy`, and `alpha` is a scalar multiplier. It is a level 1 operation that requires a cuBLAS handle for execution.',

        # C API 信息
        'c_api': 'cublasSaxpy_v2',
        'params': ['n', 'alpha', 'x', 'incx', 'y', 'incy'],
        'param_types': {
            'n': 'int',
            'alpha': 'const float*',
            'x': 'const float*',
            'incx': 'int',
            'y': 'float*',
            'incy': 'int'
        },

        # 参数分类
        'tensor_params': ['x', 'y'],
        'scalar_params': ['alpha'],
        'inout_params': ['y'],
        'return_params': ['y'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSaxpy_v2(cublasHandle_t handle, int n, const float* alpha, const float* x, int incx, float* y, int incy);',
    },
    'cublasSaxpy_v2_64': {
        'base_op': 'axpy',
        'dtype': 'float32',
        'level': 1,
        'variant': '_64',
        'description': 'The `cublasSaxpy_v2_64` function performs a single-precision vector addition operation (y = alpha * x + y) with 64-bit integer parameters, where `x` and `y` are vectors with specified increments `incx` and `incy`, and `alpha` is a scalar multiplier.',

        # C API 信息
        'c_api': 'cublasSaxpy_v2_64',
        'params': ['n', 'alpha', 'x', 'incx', 'y', 'incy'],
        'param_types': {
            'n': 'int64_t',
            'alpha': 'const float*',
            'x': 'const float*',
            'incx': 'int64_t',
            'y': 'float*',
            'incy': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['x', 'y'],
        'scalar_params': ['alpha'],
        'inout_params': ['y'],
        'return_params': ['y'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSaxpy_v2_64( cublasHandle_t handle, int64_t n, const float* alpha, const float* x, int64_t incx, float* y, int64_t incy);',
    },
    'cublasScopy_v2': {
        'base_op': 'copy',
        'dtype': 'float32',
        'level': 1,
        'variant': 'base',
        'description': 'Copies a vector of single-precision floats from `x` to `y` with specified increments, where `n` elements are transferred from `x` (with stride `incx`) to `y` (with stride `incy`).',

        # C API 信息
        'c_api': 'cublasScopy_v2',
        'params': ['n', 'x', 'incx', 'y', 'incy'],
        'param_types': {
            'n': 'int',
            'x': 'const float*',
            'incx': 'int',
            'y': 'float*',
            'incy': 'int'
        },

        # 参数分类
        'tensor_params': ['x', 'y'],
        'scalar_params': [],
        'inout_params': ['y'],
        'return_params': ['y'],

        # 完整签名
        'signature': 'cublasStatus_t cublasScopy_v2(cublasHandle_t handle, int n, const float* x, int incx, float* y, int incy);',
    },
    'cublasScopy_v2_64': {
        'base_op': 'copy',
        'dtype': 'float32',
        'level': 1,
        'variant': '_64',
        'description': 'The `cublasScopy_v2_64` function copies a vector `x` of `n` single-precision float elements to vector `y` with specified increments `incx` and `incy`, supporting 64-bit integer parameters for large datasets. It operates on the GPU using the provided cuBLAS handle for execution management.',

        # C API 信息
        'c_api': 'cublasScopy_v2_64',
        'params': ['n', 'x', 'incx', 'y', 'incy'],
        'param_types': {
            'n': 'int64_t',
            'x': 'const float*',
            'incx': 'int64_t',
            'y': 'float*',
            'incy': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['x', 'y'],
        'scalar_params': [],
        'inout_params': ['y'],
        'return_params': ['y'],

        # 完整签名
        'signature': 'cublasStatus_t cublasScopy_v2_64(cublasHandle_t handle, int64_t n, const float* x, int64_t incx, float* y, int64_t incy);',
    },
    'cublasSnrm2_v2': {
        'base_op': 'nrm2',
        'dtype': 'float32',
        'level': 1,
        'variant': 'base',
        'description': 'The `cublasSnrm2_v2` function computes the Euclidean norm (L2 norm) of a single-precision float vector `x` with `n` elements and stride `incx`, storing the result in `result`. It is a Level 1 BLAS operation optimized for CUDA-enabled GPUs.',

        # C API 信息
        'c_api': 'cublasSnrm2_v2',
        'params': ['n', 'x', 'incx', 'result'],
        'param_types': {
            'n': 'int',
            'x': 'const float*',
            'incx': 'int',
            'result': 'float*'
        },

        # 参数分类
        'tensor_params': ['x', 'result'],
        'scalar_params': [],
        'inout_params': ['result'],
        'return_params': ['result'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSnrm2_v2(cublasHandle_t handle, int n, const float* x, int incx, float* result);',
    },
    'cublasSnrm2_v2_64': {
        'base_op': 'nrm2',
        'dtype': 'float32',
        'level': 1,
        'variant': '_64',
        'description': 'This function computes the Euclidean norm (L2 norm) of a single-precision vector `x` with `n` elements and stride `incx`, storing the result in `result`, using 64-bit integers for large array support. It operates under the provided cuBLAS handle for execution management.',

        # C API 信息
        'c_api': 'cublasSnrm2_v2_64',
        'params': ['n', 'x', 'incx', 'result'],
        'param_types': {
            'n': 'int64_t',
            'x': 'const float*',
            'incx': 'int64_t',
            'result': 'float*'
        },

        # 参数分类
        'tensor_params': ['x', 'result'],
        'scalar_params': [],
        'inout_params': ['result'],
        'return_params': ['result'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSnrm2_v2_64(cublasHandle_t handle, int64_t n, const float* x, int64_t incx, float* result);',
    },
    'cublasSscal_v2': {
        'base_op': 'scal',
        'dtype': 'float32',
        'level': 1,
        'variant': 'base',
        'description': 'The `cublasSscal_v2` function multiplies a single-precision float vector `x` by a scalar `alpha`, scaling each of the `n` elements with the specified stride `incx`. It operates on the GPU using the provided cuBLAS handle for execution management.',

        # C API 信息
        'c_api': 'cublasSscal_v2',
        'params': ['n', 'alpha', 'x', 'incx'],
        'param_types': {
            'n': 'int',
            'alpha': 'const float*',
            'x': 'float*',
            'incx': 'int'
        },

        # 参数分类
        'tensor_params': ['x'],
        'scalar_params': ['alpha'],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSscal_v2(cublasHandle_t handle, int n, const float* alpha, float* x, int incx);',
    },
    'cublasSscal_v2_64': {
        'base_op': 'scal',
        'dtype': 'float32',
        'level': 1,
        'variant': '_64',
        'description': 'The `cublasSscal_v2_64` function scales a single-precision (float32) vector `x` of size `n` by a scalar `alpha`, with stride `incx` between elements, using 64-bit integers for large array support. It operates on the GPU through the cuBLAS handle and is part of BLAS Level 1 operations.',

        # C API 信息
        'c_api': 'cublasSscal_v2_64',
        'params': ['n', 'alpha', 'x', 'incx'],
        'param_types': {
            'n': 'int64_t',
            'alpha': 'const float*',
            'x': 'float*',
            'incx': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['x'],
        'scalar_params': ['alpha'],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSscal_v2_64(cublasHandle_t handle, int64_t n, const float* alpha, float* x, int64_t incx);',
    },
    'cublasSswap_v2': {
        'base_op': 'swap',
        'dtype': 'float32',
        'level': 1,
        'variant': 'base',
        'description': 'The `cublasSswap_v2` function swaps single-precision (float32) vector `x` with vector `y`, where vectors can have arbitrary strides specified by `incx` and `incy`. It operates on `n` elements and requires a cuBLAS handle for execution.',

        # C API 信息
        'c_api': 'cublasSswap_v2',
        'params': ['n', 'x', 'incx', 'y', 'incy'],
        'param_types': {
            'n': 'int',
            'x': 'float*',
            'incx': 'int',
            'y': 'float*',
            'incy': 'int'
        },

        # 参数分类
        'tensor_params': ['x', 'y'],
        'scalar_params': [],
        'inout_params': ['x', 'y'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSswap_v2(cublasHandle_t handle, int n, float* x, int incx, float* y, int incy);',
    },
    'cublasSswap_v2_64': {
        'base_op': 'swap',
        'dtype': 'float32',
        'level': 1,
        'variant': '_64',
        'description': 'The `cublasSswap_v2_64` function swaps single-precision (float32) vectors `x` and `y` of length `n` with strides `incx` and `incy`, respectively, using 64-bit integers for large array support. It operates under a cuBLAS context specified by `handle`.',

        # C API 信息
        'c_api': 'cublasSswap_v2_64',
        'params': ['n', 'x', 'incx', 'y', 'incy'],
        'param_types': {
            'n': 'int64_t',
            'x': 'float*',
            'incx': 'int64_t',
            'y': 'float*',
            'incy': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['x', 'y'],
        'scalar_params': [],
        'inout_params': ['x', 'y'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSswap_v2_64(cublasHandle_t handle, int64_t n, float* x, int64_t incx, float* y, int64_t incy);',
    },
    'cublasZaxpy_v2': {
        'base_op': 'axpy',
        'dtype': 'complex128',
        'level': 1,
        'variant': 'base',
        'description': 'The `cublasZaxpy_v2` function performs the vector addition `y = alpha * x + y` for double-precision complex vectors `x` and `y`, where `alpha` is a scalar and the vectors can have specified storage increments. It is a Level 1 BLAS operation optimized for CUDA-enabled GPUs.',

        # C API 信息
        'c_api': 'cublasZaxpy_v2',
        'params': ['n', 'alpha', 'x', 'incx', 'y', 'incy'],
        'param_types': {
            'n': 'int',
            'alpha': 'const cuDoubleComplex*',
            'x': 'const cuDoubleComplex*',
            'incx': 'int',
            'y': 'cuDoubleComplex*',
            'incy': 'int'
        },

        # 参数分类
        'tensor_params': ['x', 'y'],
        'scalar_params': ['alpha'],
        'inout_params': ['y'],
        'return_params': ['y'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZaxpy_v2(cublasHandle_t handle, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy);',
    },
    'cublasZaxpy_v2_64': {
        'base_op': 'axpy',
        'dtype': 'complex128',
        'level': 1,
        'variant': '_64',
        'description': 'The `cublasZaxpy_v2_64` function performs a 64-bit complex double-precision vector addition, scaling vector `x` by `alpha` and adding it to vector `y` with specified increments. It is a BLAS Level 1 operation supporting 64-bit indices for large arrays.',

        # C API 信息
        'c_api': 'cublasZaxpy_v2_64',
        'params': ['n', 'alpha', 'x', 'incx', 'y', 'incy'],
        'param_types': {
            'n': 'int64_t',
            'alpha': 'const cuDoubleComplex*',
            'x': 'const cuDoubleComplex*',
            'incx': 'int64_t',
            'y': 'cuDoubleComplex*',
            'incy': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['x', 'y'],
        'scalar_params': ['alpha'],
        'inout_params': ['y'],
        'return_params': ['y'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZaxpy_v2_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int64_t incx, cuDoubleComplex* y, int64_t incy);',
    },
    'cublasZcopy_v2': {
        'base_op': 'copy',
        'dtype': 'complex128',
        'level': 1,
        'variant': 'base',
        'description': 'Copies a complex double-precision vector `x` to vector `y`, where elements are spaced by increments `incx` and `incy` respectively. The operation is performed on the GPU using the specified cuBLAS handle.',

        # C API 信息
        'c_api': 'cublasZcopy_v2',
        'params': ['n', 'x', 'incx', 'y', 'incy'],
        'param_types': {
            'n': 'int',
            'x': 'const cuDoubleComplex*',
            'incx': 'int',
            'y': 'cuDoubleComplex*',
            'incy': 'int'
        },

        # 参数分类
        'tensor_params': ['x', 'y'],
        'scalar_params': [],
        'inout_params': ['y'],
        'return_params': ['y'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZcopy_v2(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy);',
    },
    'cublasZcopy_v2_64': {
        'base_op': 'copy',
        'dtype': 'complex128',
        'level': 1,
        'variant': '_64',
        'description': 'The `cublasZcopy_v2_64` function copies a complex double-precision vector `x` to another vector `y` with specified increments (`incx`, `incy`) and supports 64-bit integers for large arrays. It operates on the GPU using the provided cuBLAS handle for execution management.',

        # C API 信息
        'c_api': 'cublasZcopy_v2_64',
        'params': ['n', 'x', 'incx', 'y', 'incy'],
        'param_types': {
            'n': 'int64_t',
            'x': 'const cuDoubleComplex*',
            'incx': 'int64_t',
            'y': 'cuDoubleComplex*',
            'incy': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['x', 'y'],
        'scalar_params': [],
        'inout_params': ['y'],
        'return_params': ['y'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZcopy_v2_64( cublasHandle_t handle, int64_t n, const cuDoubleComplex* x, int64_t incx, cuDoubleComplex* y, int64_t incy);',
    },
    'cublasZscal_v2': {
        'base_op': 'scal',
        'dtype': 'complex128',
        'level': 1,
        'variant': 'base',
        'description': 'The `cublasZscal_v2` function scales a complex double-precision vector by a complex double-precision scalar, multiplying each element of the vector `x` by the scalar `alpha` with stride `incx` between elements. It operates on GPU memory using the specified cuBLAS handle for execution management.',

        # C API 信息
        'c_api': 'cublasZscal_v2',
        'params': ['n', 'alpha', 'x', 'incx'],
        'param_types': {
            'n': 'int',
            'alpha': 'const cuDoubleComplex*',
            'x': 'cuDoubleComplex*',
            'incx': 'int'
        },

        # 参数分类
        'tensor_params': ['x'],
        'scalar_params': ['alpha'],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZscal_v2(cublasHandle_t handle, int n, const cuDoubleComplex* alpha, cuDoubleComplex* x, int incx);',
    },
    'cublasZscal_v2_64': {
        'base_op': 'scal',
        'dtype': 'complex128',
        'level': 1,
        'variant': '_64',
        'description': 'The `cublasZscal_v2_64` function scales a complex double-precision vector `x` by a complex scalar `alpha`, supporting 64-bit integers for large arrays, with optional stride `incx` between elements. It operates on the GPU using the provided cuBLAS handle for execution management.',

        # C API 信息
        'c_api': 'cublasZscal_v2_64',
        'params': ['n', 'alpha', 'x', 'incx'],
        'param_types': {
            'n': 'int64_t',
            'alpha': 'const cuDoubleComplex*',
            'x': 'cuDoubleComplex*',
            'incx': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['x'],
        'scalar_params': ['alpha'],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZscal_v2_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* alpha, cuDoubleComplex* x, int64_t incx);',
    },
    'cublasZswap_v2': {
        'base_op': 'swap',
        'dtype': 'complex128',
        'level': 1,
        'variant': 'base',
        'description': 'The `cublasZswap_v2` function swaps double-precision complex vectors `x` and `y` of length `n`, with optional stride increments `incx` and `incy` for each vector. It operates on the GPU using the specified cuBLAS handle for execution management.',

        # C API 信息
        'c_api': 'cublasZswap_v2',
        'params': ['n', 'x', 'incx', 'y', 'incy'],
        'param_types': {
            'n': 'int',
            'x': 'cuDoubleComplex*',
            'incx': 'int',
            'y': 'cuDoubleComplex*',
            'incy': 'int'
        },

        # 参数分类
        'tensor_params': ['x', 'y'],
        'scalar_params': [],
        'inout_params': ['x', 'y'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZswap_v2(cublasHandle_t handle, int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy);',
    },
    'cublasZswap_v2_64': {
        'base_op': 'swap',
        'dtype': 'complex128',
        'level': 1,
        'variant': '_64',
        'description': 'The `cublasZswap_v2_64` function swaps double-precision complex vectors `x` and `y` of length `n` with specified increments `incx` and `incy`, using 64-bit integer parameters for large data sizes. It operates on GPU memory via the cuBLAS handle for efficient computation.',

        # C API 信息
        'c_api': 'cublasZswap_v2_64',
        'params': ['n', 'x', 'incx', 'y', 'incy'],
        'param_types': {
            'n': 'int64_t',
            'x': 'cuDoubleComplex*',
            'incx': 'int64_t',
            'y': 'cuDoubleComplex*',
            'incy': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['x', 'y'],
        'scalar_params': [],
        'inout_params': ['x', 'y'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZswap_v2_64(cublasHandle_t handle, int64_t n, cuDoubleComplex* x, int64_t incx, cuDoubleComplex* y, int64_t incy);',
    },

    # ======================================================================
    # Level 2
    # ======================================================================

    'cublasCgemvStridedBatched': {
        'base_op': 'gemv',
        'dtype': 'complex64',
        'level': 2,
        'variant': 'StridedBatched',
        'description': 'The `cublasCgemvStridedBatched` function performs a batched matrix-vector multiplication using complex64 matrices, where each matrix and vector in the batch is separated by a fixed stride. It computes `y = alpha * op(A) * x + beta * y` for each batch, with `op(A)` being the matrix `A` or its transpose, and supports configurable strides for input and output arrays.',

        # C API 信息
        'c_api': 'cublasCgemvStridedBatched',
        'params': ['trans', 'm', 'n', 'alpha', 'A', 'lda', 'strideA', 'x', 'incx', 'stridex', 'beta', 'y', 'incy', 'stridey', 'batchCount'],
        'param_types': {
            'trans': 'cublasOperation_t',
            'm': 'int',
            'n': 'int',
            'alpha': 'const cuComplex*',
            'A': 'const cuComplex*',
            'lda': 'int',
            'strideA': 'long long int',
            'x': 'const cuComplex*',
            'incx': 'int',
            'stridex': 'long long int',
            'beta': 'const cuComplex*',
            'y': 'cuComplex*',
            'incy': 'int',
            'stridey': 'long long int',
            'batchCount': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'x', 'y'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['y'],
        'return_params': ['y'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCgemvStridedBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, long long int strideA, const cuComplex* x, int incx, long long int stridex, const cuComplex* beta, cuComplex* y, int incy, long long int stridey, int batchCount);',
    },
    'cublasCgemvStridedBatched_64': {
        'base_op': 'gemv',
        'dtype': 'complex64',
        'level': 2,
        'variant': 'StridedBatched_64',
        'description': 'This function performs a batched matrix-vector multiplication using complex single-precision elements, where each matrix and vector in the batch is separated by a fixed stride. It computes y = α*op(A)*x + β*y for each batch, with op(A) being either A or its transpose, and supports 64-bit integers for large problem sizes.',

        # C API 信息
        'c_api': 'cublasCgemvStridedBatched_64',
        'params': ['trans', 'm', 'n', 'alpha', 'A', 'lda', 'strideA', 'x', 'incx', 'stridex', 'beta', 'y', 'incy', 'stridey', 'batchCount'],
        'param_types': {
            'trans': 'cublasOperation_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'alpha': 'const cuComplex*',
            'A': 'const cuComplex*',
            'lda': 'int64_t',
            'strideA': 'long long int',
            'x': 'const cuComplex*',
            'incx': 'int64_t',
            'stridex': 'long long int',
            'beta': 'const cuComplex*',
            'y': 'cuComplex*',
            'incy': 'int64_t',
            'stridey': 'long long int',
            'batchCount': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'x', 'y'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['y'],
        'return_params': ['y'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCgemvStridedBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, long long int strideA, const cuComplex* x, int64_t incx, long long int stridex, const cuComplex* beta, cuComplex* y, int64_t incy, long long int stridey, int64_t batchCount);',
    },
    'cublasDgemvStridedBatched': {
        'base_op': 'gemv',
        'dtype': 'float64',
        'level': 2,
        'variant': 'StridedBatched',
        'description': 'The `cublasDgemvStridedBatched` function performs a batched matrix-vector multiplication using double-precision floating-point numbers, where each matrix and vector in the batch is separated by a fixed stride. It computes `y = alpha * op(A) * x + beta * y` for each batch, with `op(A)` being the matrix `A` or its transpose, and supports configurable strides for input and output arrays.',

        # C API 信息
        'c_api': 'cublasDgemvStridedBatched',
        'params': ['trans', 'm', 'n', 'alpha', 'A', 'lda', 'strideA', 'x', 'incx', 'stridex', 'beta', 'y', 'incy', 'stridey', 'batchCount'],
        'param_types': {
            'trans': 'cublasOperation_t',
            'm': 'int',
            'n': 'int',
            'alpha': 'const double*',
            'A': 'const double*',
            'lda': 'int',
            'strideA': 'long long int',
            'x': 'const double*',
            'incx': 'int',
            'stridex': 'long long int',
            'beta': 'const double*',
            'y': 'double*',
            'incy': 'int',
            'stridey': 'long long int',
            'batchCount': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'x', 'y'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['y'],
        'return_params': ['y'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDgemvStridedBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double* alpha, const double* A, int lda, long long int strideA, const double* x, int incx, long long int stridex, const double* beta, double* y, int incy, long long int stridey, int batchCount);',
    },
    'cublasDgemvStridedBatched_64': {
        'base_op': 'gemv',
        'dtype': 'float64',
        'level': 2,
        'variant': 'StridedBatched_64',
        'description': 'The `cublasDgemvStridedBatched_64` function performs a batched matrix-vector multiplication using double-precision floating-point numbers, where each matrix and vector in the batch is separated by a fixed stride. It computes `y = alpha * op(A) * x + beta * y` for each batch, with 64-bit integers for large matrix dimensions and stride parameters.',

        # C API 信息
        'c_api': 'cublasDgemvStridedBatched_64',
        'params': ['trans', 'm', 'n', 'alpha', 'A', 'lda', 'strideA', 'x', 'incx', 'stridex', 'beta', 'y', 'incy', 'stridey', 'batchCount'],
        'param_types': {
            'trans': 'cublasOperation_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'alpha': 'const double*',
            'A': 'const double*',
            'lda': 'int64_t',
            'strideA': 'long long int',
            'x': 'const double*',
            'incx': 'int64_t',
            'stridex': 'long long int',
            'beta': 'const double*',
            'y': 'double*',
            'incy': 'int64_t',
            'stridey': 'long long int',
            'batchCount': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'x', 'y'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['y'],
        'return_params': ['y'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDgemvStridedBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const double* alpha, const double* A, int64_t lda, long long int strideA, const double* x, int64_t incx, long long int stridex, const double* beta, double* y, int64_t incy, long long int stridey, int64_t batchCount);',
    },
    'cublasSgemvStridedBatched': {
        'base_op': 'gemv',
        'dtype': 'float32',
        'level': 2,
        'variant': 'StridedBatched',
        'description': 'Performs a batched matrix-vector multiplication with strided inputs for float32 matrices, computing y = α*op(A)*x + β*y for each batch, where op(A) may be a transpose operation, with each matrix and vector separated by fixed strides in memory. The operation is applied to multiple matrix-vector pairs in a single call.',

        # C API 信息
        'c_api': 'cublasSgemvStridedBatched',
        'params': ['trans', 'm', 'n', 'alpha', 'A', 'lda', 'strideA', 'x', 'incx', 'stridex', 'beta', 'y', 'incy', 'stridey', 'batchCount'],
        'param_types': {
            'trans': 'cublasOperation_t',
            'm': 'int',
            'n': 'int',
            'alpha': 'const float*',
            'A': 'const float*',
            'lda': 'int',
            'strideA': 'long long int',
            'x': 'const float*',
            'incx': 'int',
            'stridex': 'long long int',
            'beta': 'const float*',
            'y': 'float*',
            'incy': 'int',
            'stridey': 'long long int',
            'batchCount': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'x', 'y'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['y'],
        'return_params': ['y'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSgemvStridedBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float* alpha, const float* A, int lda, long long int strideA, const float* x, int incx, long long int stridex, const float* beta, float* y, int incy, long long int stridey, int batchCount);',
    },
    'cublasSgemvStridedBatched_64': {
        'base_op': 'gemv',
        'dtype': 'float32',
        'level': 2,
        'variant': 'StridedBatched_64',
        'description': 'The `cublasSgemvStridedBatched_64` function performs a batched matrix-vector multiplication using 32-bit floating-point numbers, where each matrix and vector in the batch is separated by a fixed stride. It computes `y = alpha * op(A) * x + beta * y` for each batch, supporting 64-bit integers for large problem sizes and strided memory access.',

        # C API 信息
        'c_api': 'cublasSgemvStridedBatched_64',
        'params': ['trans', 'm', 'n', 'alpha', 'A', 'lda', 'strideA', 'x', 'incx', 'stridex', 'beta', 'y', 'incy', 'stridey', 'batchCount'],
        'param_types': {
            'trans': 'cublasOperation_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'alpha': 'const float*',
            'A': 'const float*',
            'lda': 'int64_t',
            'strideA': 'long long int',
            'x': 'const float*',
            'incx': 'int64_t',
            'stridex': 'long long int',
            'beta': 'const float*',
            'y': 'float*',
            'incy': 'int64_t',
            'stridey': 'long long int',
            'batchCount': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'x', 'y'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['y'],
        'return_params': ['y'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSgemvStridedBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const float* alpha, const float* A, int64_t lda, long long int strideA, const float* x, int64_t incx, long long int stridex, const float* beta, float* y, int64_t incy, long long int stridey, int64_t batchCount);',
    },
    'cublasZgemvStridedBatched': {
        'base_op': 'gemv',
        'dtype': 'complex128',
        'level': 2,
        'variant': 'StridedBatched',
        'description': 'The `cublasZgemvStridedBatched` function performs a batched matrix-vector multiplication using complex double-precision elements, where each matrix and vector in the batch is separated by a fixed stride. It computes `y = alpha * op(A) * x + beta * y` for each batch, with `op(A)` being the matrix A or its transpose, and supports strided memory access for both input and output arrays.',

        # C API 信息
        'c_api': 'cublasZgemvStridedBatched',
        'params': ['trans', 'm', 'n', 'alpha', 'A', 'lda', 'strideA', 'x', 'incx', 'stridex', 'beta', 'y', 'incy', 'stridey', 'batchCount'],
        'param_types': {
            'trans': 'cublasOperation_t',
            'm': 'int',
            'n': 'int',
            'alpha': 'const cuDoubleComplex*',
            'A': 'const cuDoubleComplex*',
            'lda': 'int',
            'strideA': 'long long int',
            'x': 'const cuDoubleComplex*',
            'incx': 'int',
            'stridex': 'long long int',
            'beta': 'const cuDoubleComplex*',
            'y': 'cuDoubleComplex*',
            'incy': 'int',
            'stridey': 'long long int',
            'batchCount': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'x', 'y'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['y'],
        'return_params': ['y'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZgemvStridedBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, long long int strideA, const cuDoubleComplex* x, int incx, long long int stridex, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy, long long int stridey, int batchCount);',
    },
    'cublasZgemvStridedBatched_64': {
        'base_op': 'gemv',
        'dtype': 'complex128',
        'level': 2,
        'variant': 'StridedBatched_64',
        'description': 'The `cublasZgemvStridedBatched_64` function performs a batched matrix-vector multiplication using complex double-precision elements, where each matrix and vector in the batch is separated by a fixed stride. It computes `y = alpha * op(A) * x + beta * y` for each batch, with 64-bit integers for large problem sizes and supports transposed or non-transposed matrix operations.',

        # C API 信息
        'c_api': 'cublasZgemvStridedBatched_64',
        'params': ['trans', 'm', 'n', 'alpha', 'A', 'lda', 'strideA', 'x', 'incx', 'stridex', 'beta', 'y', 'incy', 'stridey', 'batchCount'],
        'param_types': {
            'trans': 'cublasOperation_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'alpha': 'const cuDoubleComplex*',
            'A': 'const cuDoubleComplex*',
            'lda': 'int64_t',
            'strideA': 'long long int',
            'x': 'const cuDoubleComplex*',
            'incx': 'int64_t',
            'stridex': 'long long int',
            'beta': 'const cuDoubleComplex*',
            'y': 'cuDoubleComplex*',
            'incy': 'int64_t',
            'stridey': 'long long int',
            'batchCount': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'x', 'y'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['y'],
        'return_params': ['y'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZgemvStridedBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, long long int strideA, const cuDoubleComplex* x, int64_t incx, long long int stridex, const cuDoubleComplex* beta, cuDoubleComplex* y, int64_t incy, long long int stridey, int64_t batchCount);',
    },
    'cublasCgemvBatched': {
        'base_op': 'gemv',
        'dtype': 'complex64',
        'level': 2,
        'variant': 'Batched',
        'description': 'Performs a batched matrix-vector multiplication for complex single-precision matrices, computing y = α*op(A)*x + β*y for each matrix A in Aarray and corresponding vectors x and y, where op(A) can be a no-op or transpose operation. The operation is applied to all matrices and vectors in the batch with a single function call.',

        # C API 信息
        'c_api': 'cublasCgemvBatched',
        'params': ['trans', 'm', 'n', 'alpha', 'Aarray', 'lda', 'xarray', 'incx', 'beta', 'yarray', 'incy', 'batchCount'],
        'param_types': {
            'trans': 'cublasOperation_t',
            'm': 'int',
            'n': 'int',
            'alpha': 'const cuComplex*',
            'Aarray': 'const cuComplex* const[]',
            'lda': 'int',
            'xarray': 'const cuComplex* const[]',
            'incx': 'int',
            'beta': 'const cuComplex*',
            'yarray': 'cuComplex* const[]',
            'incy': 'int',
            'batchCount': 'int'
        },

        # 参数分类
        'tensor_params': ['Aarray', 'xarray', 'yarray'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['yarray'],
        'return_params': ['yarray'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCgemvBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuComplex* alpha, const cuComplex* const Aarray[], int lda, const cuComplex* const xarray[], int incx, const cuComplex* beta, cuComplex* const yarray[], int incy, int batchCount);',
    },
    'cublasCgemvBatched_64': {
        'base_op': 'gemv',
        'dtype': 'complex64',
        'level': 2,
        'variant': 'Batched_64',
        'description': 'The `cublasCgemvBatched_64` function performs a batched matrix-vector multiplication for complex 64-bit floating-point numbers, computing `y = alpha * op(A) * x + beta * y` for each batch, where `op(A)` can be a transpose or conjugate transpose operation. It processes multiple operations in a single call, with each batch using separate matrices (`Aarray`), vectors (`xarray`, `yarray`), and leading dimensions.',

        # C API 信息
        'c_api': 'cublasCgemvBatched_64',
        'params': ['trans', 'm', 'n', 'alpha', 'Aarray', 'lda', 'xarray', 'incx', 'beta', 'yarray', 'incy', 'batchCount'],
        'param_types': {
            'trans': 'cublasOperation_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'alpha': 'const cuComplex*',
            'Aarray': 'const cuComplex* const[]',
            'lda': 'int64_t',
            'xarray': 'const cuComplex* const[]',
            'incx': 'int64_t',
            'beta': 'const cuComplex*',
            'yarray': 'cuComplex* const[]',
            'incy': 'int64_t',
            'batchCount': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['Aarray', 'xarray', 'yarray'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['yarray'],
        'return_params': ['yarray'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCgemvBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* const Aarray[], int64_t lda, const cuComplex* const xarray[], int64_t incx, const cuComplex* beta, cuComplex* const yarray[], int64_t incy, int64_t batchCount);',
    },
    'cublasDgemvBatched': {
        'base_op': 'gemv',
        'dtype': 'float64',
        'level': 2,
        'variant': 'Batched',
        'description': 'The `cublasDgemvBatched` function performs batched matrix-vector multiplication using double-precision floating-point numbers, computing `y = alpha * op(A) * x + beta * y` for each batch, where `op(A)` can be a matrix or its transpose. It processes multiple operations in a single call with separate arrays for matrices, vectors, and results, specified by the `batchCount` parameter.',

        # C API 信息
        'c_api': 'cublasDgemvBatched',
        'params': ['trans', 'm', 'n', 'alpha', 'Aarray', 'lda', 'xarray', 'incx', 'beta', 'yarray', 'incy', 'batchCount'],
        'param_types': {
            'trans': 'cublasOperation_t',
            'm': 'int',
            'n': 'int',
            'alpha': 'const double*',
            'Aarray': 'const double* const[]',
            'lda': 'int',
            'xarray': 'const double* const[]',
            'incx': 'int',
            'beta': 'const double*',
            'yarray': 'double* const[]',
            'incy': 'int',
            'batchCount': 'int'
        },

        # 参数分类
        'tensor_params': ['Aarray', 'xarray', 'yarray'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['yarray'],
        'return_params': ['yarray'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDgemvBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double* alpha, const double* const Aarray[], int lda, const double* const xarray[], int incx, const double* beta, double* const yarray[], int incy, int batchCount);',
    },
    'cublasDgemvBatched_64': {
        'base_op': 'gemv',
        'dtype': 'float64',
        'level': 2,
        'variant': 'Batched_64',
        'description': 'The `cublasDgemvBatched_64` function performs batched matrix-vector multiplication for double-precision matrices, computing `y = alpha * op(A) * x + beta * y` for each batch, where `op(A)` can be a transpose or non-transpose operation, and supports 64-bit integers for large problem sizes. It processes multiple independent matrix-vector operations in a single call for improved efficiency.',

        # C API 信息
        'c_api': 'cublasDgemvBatched_64',
        'params': ['trans', 'm', 'n', 'alpha', 'Aarray', 'lda', 'xarray', 'incx', 'beta', 'yarray', 'incy', 'batchCount'],
        'param_types': {
            'trans': 'cublasOperation_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'alpha': 'const double*',
            'Aarray': 'const double* const[]',
            'lda': 'int64_t',
            'xarray': 'const double* const[]',
            'incx': 'int64_t',
            'beta': 'const double*',
            'yarray': 'double* const[]',
            'incy': 'int64_t',
            'batchCount': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['Aarray', 'xarray', 'yarray'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['yarray'],
        'return_params': ['yarray'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDgemvBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const double* alpha, const double* const Aarray[], int64_t lda, const double* const xarray[], int64_t incx, const double* beta, double* const yarray[], int64_t incy, int64_t batchCount);',
    },
    'cublasSgemvBatched': {
        'base_op': 'gemv',
        'dtype': 'float32',
        'level': 2,
        'variant': 'Batched',
        'description': 'The `cublasSgemvBatched` function performs batched matrix-vector multiplication for float32 data, computing `y = alpha * op(A) * x + beta * y` for each batch, where `op(A)` can be a transpose or non-transpose operation, and processes multiple operations in a single call. It handles multiple independent matrices (`Aarray`), vectors (`xarray`, `yarray`), and parameters in parallel, with the batch size specified by `batchCount`.',

        # C API 信息
        'c_api': 'cublasSgemvBatched',
        'params': ['trans', 'm', 'n', 'alpha', 'Aarray', 'lda', 'xarray', 'incx', 'beta', 'yarray', 'incy', 'batchCount'],
        'param_types': {
            'trans': 'cublasOperation_t',
            'm': 'int',
            'n': 'int',
            'alpha': 'const float*',
            'Aarray': 'const float* const[]',
            'lda': 'int',
            'xarray': 'const float* const[]',
            'incx': 'int',
            'beta': 'const float*',
            'yarray': 'float* const[]',
            'incy': 'int',
            'batchCount': 'int'
        },

        # 参数分类
        'tensor_params': ['Aarray', 'xarray', 'yarray'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['yarray'],
        'return_params': ['yarray'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSgemvBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float* alpha, const float* const Aarray[], int lda, const float* const xarray[], int incx, const float* beta, float* const yarray[], int incy, int batchCount);',
    },
    'cublasSgemvBatched_64': {
        'base_op': 'gemv',
        'dtype': 'float32',
        'level': 2,
        'variant': 'Batched_64',
        'description': 'The `cublasSgemvBatched_64` function performs a batched matrix-vector multiplication for 32-bit floating-point matrices, computing `y = alpha * op(A) * x + beta * y` for each batch, where `op(A)` can be a transpose or non-transpose operation, and supports 64-bit integer parameters for large problem sizes. It processes multiple operations in a single call with arrays of pointers to matrices and vectors.',

        # C API 信息
        'c_api': 'cublasSgemvBatched_64',
        'params': ['trans', 'm', 'n', 'alpha', 'Aarray', 'lda', 'xarray', 'incx', 'beta', 'yarray', 'incy', 'batchCount'],
        'param_types': {
            'trans': 'cublasOperation_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'alpha': 'const float*',
            'Aarray': 'const float* const[]',
            'lda': 'int64_t',
            'xarray': 'const float* const[]',
            'incx': 'int64_t',
            'beta': 'const float*',
            'yarray': 'float* const[]',
            'incy': 'int64_t',
            'batchCount': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['Aarray', 'xarray', 'yarray'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['yarray'],
        'return_params': ['yarray'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSgemvBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const float* alpha, const float* const Aarray[], int64_t lda, const float* const xarray[], int64_t incx, const float* beta, float* const yarray[], int64_t incy, int64_t batchCount);',
    },
    'cublasZgemvBatched': {
        'base_op': 'gemv',
        'dtype': 'complex128',
        'level': 2,
        'variant': 'Batched',
        'description': 'The `cublasZgemvBatched` function performs batched matrix-vector multiplication for complex double-precision matrices, computing `y = alpha * op(A) * x + beta * y` for each matrix-vector pair in the batch, where `op(A)` can be a matrix transpose or conjugate transpose operation. It processes multiple operations in a single call for improved efficiency, with each matrix and vector pair specified in separate arrays.',

        # C API 信息
        'c_api': 'cublasZgemvBatched',
        'params': ['trans', 'm', 'n', 'alpha', 'Aarray', 'lda', 'xarray', 'incx', 'beta', 'yarray', 'incy', 'batchCount'],
        'param_types': {
            'trans': 'cublasOperation_t',
            'm': 'int',
            'n': 'int',
            'alpha': 'const cuDoubleComplex*',
            'Aarray': 'const cuDoubleComplex* const[]',
            'lda': 'int',
            'xarray': 'const cuDoubleComplex* const[]',
            'incx': 'int',
            'beta': 'const cuDoubleComplex*',
            'yarray': 'cuDoubleComplex* const[]',
            'incy': 'int',
            'batchCount': 'int'
        },

        # 参数分类
        'tensor_params': ['Aarray', 'xarray', 'yarray'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['yarray'],
        'return_params': ['yarray'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZgemvBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* const Aarray[], int lda, const cuDoubleComplex* const xarray[], int incx, const cuDoubleComplex* beta, cuDoubleComplex* const yarray[], int incy, int batchCount);',
    },
    'cublasZgemvBatched_64': {
        'base_op': 'gemv',
        'dtype': 'complex128',
        'level': 2,
        'variant': 'Batched_64',
        'description': 'The `cublasZgemvBatched_64` function performs batched matrix-vector multiplication for complex double-precision matrices, computing `y = alpha * op(A) * x + beta * y` for each matrix-vector pair in the batch, where `op(A)` can be a transpose or conjugate transpose operation, and supports 64-bit integer parameters for large-scale problems.',

        # C API 信息
        'c_api': 'cublasZgemvBatched_64',
        'params': ['trans', 'm', 'n', 'alpha', 'Aarray', 'lda', 'xarray', 'incx', 'beta', 'yarray', 'incy', 'batchCount'],
        'param_types': {
            'trans': 'cublasOperation_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'alpha': 'const cuDoubleComplex*',
            'Aarray': 'const cuDoubleComplex* const[]',
            'lda': 'int64_t',
            'xarray': 'const cuDoubleComplex* const[]',
            'incx': 'int64_t',
            'beta': 'const cuDoubleComplex*',
            'yarray': 'cuDoubleComplex* const[]',
            'incy': 'int64_t',
            'batchCount': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['Aarray', 'xarray', 'yarray'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['yarray'],
        'return_params': ['yarray'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZgemvBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* const Aarray[], int64_t lda, const cuDoubleComplex* const xarray[], int64_t incx, const cuDoubleComplex* beta, cuDoubleComplex* const yarray[], int64_t incy, int64_t batchCount);',
    },
    'cublasCgemv_v2': {
        'base_op': 'gemv',
        'dtype': 'complex64',
        'level': 2,
        'variant': 'base',
        'description': 'Performs a complex single-precision matrix-vector multiplication (y = α·op(A)·x + β·y), where op(A) can be A, A^T, or A^H, with A being an m×n matrix and x/y being vectors. The function supports custom strides for input/output vectors through incx/incy parameters.',

        # C API 信息
        'c_api': 'cublasCgemv_v2',
        'params': ['trans', 'm', 'n', 'alpha', 'A', 'lda', 'x', 'incx', 'beta', 'y', 'incy'],
        'param_types': {
            'trans': 'cublasOperation_t',
            'm': 'int',
            'n': 'int',
            'alpha': 'const cuComplex*',
            'A': 'const cuComplex*',
            'lda': 'int',
            'x': 'const cuComplex*',
            'incx': 'int',
            'beta': 'const cuComplex*',
            'y': 'cuComplex*',
            'incy': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'x', 'y'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['y'],
        'return_params': ['y'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy);',
    },
    'cublasCgemv_v2_64': {
        'base_op': 'gemv',
        'dtype': 'complex64',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasCgemv_v2_64` function performs a 64-bit matrix-vector multiplication using complex single-precision elements, computing `y = alpha * op(A) * x + beta * y`, where `op(A)` can be the matrix `A` or its transpose/conjugate transpose. It supports large matrices and vectors with 64-bit dimensions and strides.',

        # C API 信息
        'c_api': 'cublasCgemv_v2_64',
        'params': ['trans', 'm', 'n', 'alpha', 'A', 'lda', 'x', 'incx', 'beta', 'y', 'incy'],
        'param_types': {
            'trans': 'cublasOperation_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'alpha': 'const cuComplex*',
            'A': 'const cuComplex*',
            'lda': 'int64_t',
            'x': 'const cuComplex*',
            'incx': 'int64_t',
            'beta': 'const cuComplex*',
            'y': 'cuComplex*',
            'incy': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'x', 'y'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['y'],
        'return_params': ['y'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCgemv_v2_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* x, int64_t incx, const cuComplex* beta, cuComplex* y, int64_t incy);',
    },
    'cublasCgerc_v2': {
        'base_op': 'gerc',
        'dtype': 'complex64',
        'level': 2,
        'variant': 'base',
        'description': 'Performs the rank-1 update of a complex matrix A using conjugated vector y, where A = alpha * x * yᴴ + A, with x as an m-element vector, y as an n-element vector, and A as an m×n matrix.',

        # C API 信息
        'c_api': 'cublasCgerc_v2',
        'params': ['m', 'n', 'alpha', 'x', 'incx', 'y', 'incy', 'A', 'lda'],
        'param_types': {
            'm': 'int',
            'n': 'int',
            'alpha': 'const cuComplex*',
            'x': 'const cuComplex*',
            'incx': 'int',
            'y': 'const cuComplex*',
            'incy': 'int',
            'A': 'cuComplex*',
            'lda': 'int'
        },

        # 参数分类
        'tensor_params': ['x', 'y', 'A'],
        'scalar_params': ['alpha'],
        'inout_params': ['A'],
        'return_params': ['A'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCgerc_v2(cublasHandle_t handle, int m, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* A, int lda);',
    },
    'cublasCgerc_v2_64': {
        'base_op': 'gerc',
        'dtype': 'complex64',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasCgerc_v2_64` function performs a rank-1 update of a complex 64-bit matrix `A` by the conjugate outer product of vectors `x` and `y`, scaled by `alpha`, using 64-bit integers for large matrix dimensions. It is a BLAS Level 2 operation for complex single-precision data.',

        # C API 信息
        'c_api': 'cublasCgerc_v2_64',
        'params': ['m', 'n', 'alpha', 'x', 'incx', 'y', 'incy', 'A', 'lda'],
        'param_types': {
            'm': 'int64_t',
            'n': 'int64_t',
            'alpha': 'const cuComplex*',
            'x': 'const cuComplex*',
            'incx': 'int64_t',
            'y': 'const cuComplex*',
            'incy': 'int64_t',
            'A': 'cuComplex*',
            'lda': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['x', 'y', 'A'],
        'scalar_params': ['alpha'],
        'inout_params': ['A'],
        'return_params': ['A'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCgerc_v2_64(cublasHandle_t handle, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* x, int64_t incx, const cuComplex* y, int64_t incy, cuComplex* A, int64_t lda);',
    },
    'cublasCgeru_v2': {
        'base_op': 'geru',
        'dtype': 'complex64',
        'level': 2,
        'variant': 'base',
        'description': 'Performs the rank-1 update of a complex single-precision matrix A using vectors x and y, where A is overwritten with alpha*x*y^T + A. The operation supports arbitrary strides (increments) for input vectors x and y.',

        # C API 信息
        'c_api': 'cublasCgeru_v2',
        'params': ['m', 'n', 'alpha', 'x', 'incx', 'y', 'incy', 'A', 'lda'],
        'param_types': {
            'm': 'int',
            'n': 'int',
            'alpha': 'const cuComplex*',
            'x': 'const cuComplex*',
            'incx': 'int',
            'y': 'const cuComplex*',
            'incy': 'int',
            'A': 'cuComplex*',
            'lda': 'int'
        },

        # 参数分类
        'tensor_params': ['x', 'y', 'A'],
        'scalar_params': ['alpha'],
        'inout_params': ['A'],
        'return_params': ['A'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCgeru_v2(cublasHandle_t handle, int m, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* A, int lda);',
    },
    'cublasCgeru_v2_64': {
        'base_op': 'geru',
        'dtype': 'complex64',
        'level': 2,
        'variant': '_64',
        'description': 'Performs a 64-bit rank-1 update of a complex64 matrix A using vectors x and y, where A = alpha * x * y^T + A, with alpha being a complex scalar and x, y being complex vectors. This is the 64-bit index variant of the complex general rank-1 update (geru) operation.',

        # C API 信息
        'c_api': 'cublasCgeru_v2_64',
        'params': ['m', 'n', 'alpha', 'x', 'incx', 'y', 'incy', 'A', 'lda'],
        'param_types': {
            'm': 'int64_t',
            'n': 'int64_t',
            'alpha': 'const cuComplex*',
            'x': 'const cuComplex*',
            'incx': 'int64_t',
            'y': 'const cuComplex*',
            'incy': 'int64_t',
            'A': 'cuComplex*',
            'lda': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['x', 'y', 'A'],
        'scalar_params': ['alpha'],
        'inout_params': ['A'],
        'return_params': ['A'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCgeru_v2_64(cublasHandle_t handle, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* x, int64_t incx, const cuComplex* y, int64_t incy, cuComplex* A, int64_t lda);',
    },
    'cublasCsymv_v2': {
        'base_op': 'symv',
        'dtype': 'complex64',
        'level': 2,
        'variant': 'base',
        'description': 'The `cublasCsymv_v2` function performs a symmetric matrix-vector multiplication for complex single-precision matrices, computing `y = alpha * A * x + beta * y`, where `A` is a complex symmetric matrix and `x`, `y` are complex vectors. The `uplo` parameter specifies whether the upper or lower triangular part of `A` is used.',

        # C API 信息
        'c_api': 'cublasCsymv_v2',
        'params': ['uplo', 'n', 'alpha', 'A', 'lda', 'x', 'incx', 'beta', 'y', 'incy'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'n': 'int',
            'alpha': 'const cuComplex*',
            'A': 'const cuComplex*',
            'lda': 'int',
            'x': 'const cuComplex*',
            'incx': 'int',
            'beta': 'const cuComplex*',
            'y': 'cuComplex*',
            'incy': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'x', 'y'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['y'],
        'return_params': ['y'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy);',
    },
    'cublasCsymv_v2_64': {
        'base_op': 'symv',
        'dtype': 'complex64',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasCsymv_v2_64` function computes the matrix-vector product for a complex symmetric matrix using 64-bit integers, multiplying the matrix `A` with vector `x`, scaling by `alpha`, adding the scaled result to vector `y` scaled by `beta`, and storing the output in `y`. The `uplo` parameter specifies whether the upper or lower triangular part of `A` is used.',

        # C API 信息
        'c_api': 'cublasCsymv_v2_64',
        'params': ['uplo', 'n', 'alpha', 'A', 'lda', 'x', 'incx', 'beta', 'y', 'incy'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'n': 'int64_t',
            'alpha': 'const cuComplex*',
            'A': 'const cuComplex*',
            'lda': 'int64_t',
            'x': 'const cuComplex*',
            'incx': 'int64_t',
            'beta': 'const cuComplex*',
            'y': 'cuComplex*',
            'incy': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'x', 'y'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['y'],
        'return_params': ['y'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCsymv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* x, int64_t incx, const cuComplex* beta, cuComplex* y, int64_t incy);',
    },
    'cublasCsyr2_v2': {
        'base_op': 'syr2',
        'dtype': 'complex64',
        'level': 2,
        'variant': 'base',
        'description': 'Performs a symmetric rank-2 update of a complex single-precision matrix A using vectors x and y, where A is stored in either upper or lower triangular form. The operation computes A = alpha*(x*y^T + y*x^T) + A, with alpha as a scalar and x/y as vectors with specified increments.',

        # C API 信息
        'c_api': 'cublasCsyr2_v2',
        'params': ['uplo', 'n', 'alpha', 'x', 'incx', 'y', 'incy', 'A', 'lda'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'n': 'int',
            'alpha': 'const cuComplex*',
            'x': 'const cuComplex*',
            'incx': 'int',
            'y': 'const cuComplex*',
            'incy': 'int',
            'A': 'cuComplex*',
            'lda': 'int'
        },

        # 参数分类
        'tensor_params': ['x', 'y', 'A'],
        'scalar_params': ['alpha'],
        'inout_params': ['A'],
        'return_params': ['A'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* A, int lda);',
    },
    'cublasCsyr2_v2_64': {
        'base_op': 'syr2',
        'dtype': 'complex64',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasCsyr2_v2_64` function performs a symmetric rank-2 update of a complex64 matrix `A` using vectors `x` and `y`, scaling the result by `alpha` and supports 64-bit integers for large problem sizes. It operates on either the upper or lower triangular part of `A` as specified by `uplo`.',

        # C API 信息
        'c_api': 'cublasCsyr2_v2_64',
        'params': ['uplo', 'n', 'alpha', 'x', 'incx', 'y', 'incy', 'A', 'lda'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'n': 'int64_t',
            'alpha': 'const cuComplex*',
            'x': 'const cuComplex*',
            'incx': 'int64_t',
            'y': 'const cuComplex*',
            'incy': 'int64_t',
            'A': 'cuComplex*',
            'lda': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['x', 'y', 'A'],
        'scalar_params': ['alpha'],
        'inout_params': ['A'],
        'return_params': ['A'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCsyr2_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuComplex* alpha, const cuComplex* x, int64_t incx, const cuComplex* y, int64_t incy, cuComplex* A, int64_t lda);',
    },
    'cublasCsyr_v2': {
        'base_op': 'syr',
        'dtype': 'complex64',
        'level': 2,
        'variant': 'base',
        'description': 'The `cublasCsyr_v2` function performs a symmetric rank-1 update of a complex single-precision matrix A using the vector x, where A is a scalar multiplier and the update preserves the matrix\'s symmetry based on the specified upper or lower triangular part.',

        # C API 信息
        'c_api': 'cublasCsyr_v2',
        'params': ['uplo', 'n', 'alpha', 'x', 'incx', 'A', 'lda'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'n': 'int',
            'alpha': 'const cuComplex*',
            'x': 'const cuComplex*',
            'incx': 'int',
            'A': 'cuComplex*',
            'lda': 'int'
        },

        # 参数分类
        'tensor_params': ['x', 'A'],
        'scalar_params': ['alpha'],
        'inout_params': ['A'],
        'return_params': ['A'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* x, int incx, cuComplex* A, int lda);',
    },
    'cublasCsyr_v2_64': {
        'base_op': 'syr',
        'dtype': 'complex64',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasCsyr_v2_64` function performs a symmetric rank-1 update on a complex64 matrix `A` by multiplying vector `x` with its conjugate transpose and scaling by `alpha`, storing the result in `A` with upper or lower fill mode specified by `uplo`. It supports 64-bit integers for large matrix operations.',

        # C API 信息
        'c_api': 'cublasCsyr_v2_64',
        'params': ['uplo', 'n', 'alpha', 'x', 'incx', 'A', 'lda'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'n': 'int64_t',
            'alpha': 'const cuComplex*',
            'x': 'const cuComplex*',
            'incx': 'int64_t',
            'A': 'cuComplex*',
            'lda': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['x', 'A'],
        'scalar_params': ['alpha'],
        'inout_params': ['A'],
        'return_params': ['A'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCsyr_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuComplex* alpha, const cuComplex* x, int64_t incx, cuComplex* A, int64_t lda);',
    },
    'cublasCtbmv_v2': {
        'base_op': 'tbmv',
        'dtype': 'complex64',
        'level': 2,
        'variant': 'base',
        'description': 'The `cublasCtbmv_v2` function performs a complex single-precision triangular banded matrix-vector multiplication, where the matrix can be upper or lower triangular, optionally transposed, and may have a unit diagonal. It operates on a banded matrix `A` of size `n x n` with `k` sub/super-diagonals, multiplying it with vector `x` and storing the result in `x`.',

        # C API 信息
        'c_api': 'cublasCtbmv_v2',
        'params': ['uplo', 'trans', 'diag', 'n', 'k', 'A', 'lda', 'x', 'incx'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'n': 'int',
            'k': 'int',
            'A': 'const cuComplex*',
            'lda': 'int',
            'x': 'cuComplex*',
            'incx': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'x'],
        'scalar_params': [],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCtbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuComplex* A, int lda, cuComplex* x, int incx);',
    },
    'cublasCtbmv_v2_64': {
        'base_op': 'tbmv',
        'dtype': 'complex64',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasCtbmv_v2_64` function performs a banded matrix-vector multiplication using a complex64 triangular band matrix, with support for 64-bit integers. It computes `x = op(A) * x` where `op(A)` can be the matrix, its transpose, or conjugate transpose, with options for upper/lower triangular storage and unit diagonal.',

        # C API 信息
        'c_api': 'cublasCtbmv_v2_64',
        'params': ['uplo', 'trans', 'diag', 'n', 'k', 'A', 'lda', 'x', 'incx'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'n': 'int64_t',
            'k': 'int64_t',
            'A': 'const cuComplex*',
            'lda': 'int64_t',
            'x': 'cuComplex*',
            'incx': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'x'],
        'scalar_params': [],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCtbmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const cuComplex* A, int64_t lda, cuComplex* x, int64_t incx);',
    },
    'cublasCtbsv_v2': {
        'base_op': 'tbsv',
        'dtype': 'complex64',
        'level': 2,
        'variant': 'base',
        'description': 'The `cublasCtbsv_v2` function solves a complex64 triangular banded system of equations with a single right-hand side, using the specified uplo, trans, and diag parameters to determine the matrix structure and operation. It operates on a banded matrix `A` of size `n x n` with `k` sub-/super-diagonals, updating the vector `x` in place.',

        # C API 信息
        'c_api': 'cublasCtbsv_v2',
        'params': ['uplo', 'trans', 'diag', 'n', 'k', 'A', 'lda', 'x', 'incx'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'n': 'int',
            'k': 'int',
            'A': 'const cuComplex*',
            'lda': 'int',
            'x': 'cuComplex*',
            'incx': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'x'],
        'scalar_params': [],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCtbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuComplex* A, int lda, cuComplex* x, int incx);',
    },
    'cublasCtbsv_v2_64': {
        'base_op': 'tbsv',
        'dtype': 'complex64',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasCtbsv_v2_64` function solves a complex64 triangular banded system of equations with 64-bit parameters, where the matrix can be upper or lower triangular, transposed or not, and unit or non-unit diagonal. It operates on vectors with specified strides and is designed for large-scale problems requiring extended precision indexing.',

        # C API 信息
        'c_api': 'cublasCtbsv_v2_64',
        'params': ['uplo', 'trans', 'diag', 'n', 'k', 'A', 'lda', 'x', 'incx'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'n': 'int64_t',
            'k': 'int64_t',
            'A': 'const cuComplex*',
            'lda': 'int64_t',
            'x': 'cuComplex*',
            'incx': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'x'],
        'scalar_params': [],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCtbsv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const cuComplex* A, int64_t lda, cuComplex* x, int64_t incx);',
    },
    'cublasCtpmv_v2': {
        'base_op': 'tpmv',
        'dtype': 'complex64',
        'level': 2,
        'variant': 'base',
        'description': 'The `cublasCtpmv_v2` function performs a complex single-precision triangular matrix-vector multiplication using a packed storage format, where the matrix can be upper or lower triangular, optionally transposed, and optionally unit diagonal. It multiplies the matrix by vector `x` and stores the result in `x`, with support for a specified storage increment.',

        # C API 信息
        'c_api': 'cublasCtpmv_v2',
        'params': ['uplo', 'trans', 'diag', 'n', 'AP', 'x', 'incx'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'n': 'int',
            'AP': 'const cuComplex*',
            'x': 'cuComplex*',
            'incx': 'int'
        },

        # 参数分类
        'tensor_params': ['AP', 'x'],
        'scalar_params': [],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCtpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex* AP, cuComplex* x, int incx);',
    },
    'cublasCtpmv_v2_64': {
        'base_op': 'tpmv',
        'dtype': 'complex64',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasCtpmv_v2_64` function performs a complex single-precision triangular matrix-vector multiplication using a packed storage format, supporting 64-bit integers for large problem sizes. It multiplies a triangular matrix by a vector, with options for matrix uplo (upper/lower), transposition (trans), and unit/non-unit diagonal (diag).',

        # C API 信息
        'c_api': 'cublasCtpmv_v2_64',
        'params': ['uplo', 'trans', 'diag', 'n', 'AP', 'x', 'incx'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'n': 'int64_t',
            'AP': 'const cuComplex*',
            'x': 'cuComplex*',
            'incx': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['AP', 'x'],
        'scalar_params': [],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCtpmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuComplex* AP, cuComplex* x, int64_t incx);',
    },
    'cublasCtpsv_v2': {
        'base_op': 'tpsv',
        'dtype': 'complex64',
        'level': 2,
        'variant': 'base',
        'description': 'The `cublasCtpsv_v2` function solves a complex triangular system of equations with a packed storage format, where the matrix can be upper or lower triangular, optionally transposed, and optionally unit-diagonal. It operates on single-precision complex data with specified stride for the input/output vector.',

        # C API 信息
        'c_api': 'cublasCtpsv_v2',
        'params': ['uplo', 'trans', 'diag', 'n', 'AP', 'x', 'incx'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'n': 'int',
            'AP': 'const cuComplex*',
            'x': 'cuComplex*',
            'incx': 'int'
        },

        # 参数分类
        'tensor_params': ['AP', 'x'],
        'scalar_params': [],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCtpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex* AP, cuComplex* x, int incx);',
    },
    'cublasCtpsv_v2_64': {
        'base_op': 'tpsv',
        'dtype': 'complex64',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasCtpsv_v2_64` function solves a complex64 triangular system of equations using a packed storage format, supporting 64-bit integers for large problem sizes, with options for matrix uplo (upper/lower), transposition (trans), and diagonal type (unit/non-unit). It operates on a packed triangular matrix (AP) and a vector (x) with a specified increment (incx).',

        # C API 信息
        'c_api': 'cublasCtpsv_v2_64',
        'params': ['uplo', 'trans', 'diag', 'n', 'AP', 'x', 'incx'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'n': 'int64_t',
            'AP': 'const cuComplex*',
            'x': 'cuComplex*',
            'incx': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['AP', 'x'],
        'scalar_params': [],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCtpsv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuComplex* AP, cuComplex* x, int64_t incx);',
    },
    'cublasCtrmv_v2': {
        'base_op': 'trmv',
        'dtype': 'complex64',
        'level': 2,
        'variant': 'base',
        'description': 'The `cublasCtrmv_v2` function performs a triangular matrix-vector multiplication using a complex64 matrix, where the matrix can be upper or lower triangular, optionally transposed, and optionally unit triangular. It multiplies the matrix `A` by vector `x` and stores the result in `x`.',

        # C API 信息
        'c_api': 'cublasCtrmv_v2',
        'params': ['uplo', 'trans', 'diag', 'n', 'A', 'lda', 'x', 'incx'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'n': 'int',
            'A': 'const cuComplex*',
            'lda': 'int',
            'x': 'cuComplex*',
            'incx': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'x'],
        'scalar_params': [],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCtrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex* A, int lda, cuComplex* x, int incx);',
    },
    'cublasCtrmv_v2_64': {
        'base_op': 'trmv',
        'dtype': 'complex64',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasCtrmv_v2_64` function performs a triangular matrix-vector multiplication using a complex 64-bit floating-point triangular matrix and vector, supporting 64-bit integers for large problem sizes. It multiplies the matrix by the vector according to the upper/lower triangular specification (`uplo`), transpose operation (`trans`), and unit/non-unit diagonal flag (`diag`).',

        # C API 信息
        'c_api': 'cublasCtrmv_v2_64',
        'params': ['uplo', 'trans', 'diag', 'n', 'A', 'lda', 'x', 'incx'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'n': 'int64_t',
            'A': 'const cuComplex*',
            'lda': 'int64_t',
            'x': 'cuComplex*',
            'incx': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'x'],
        'scalar_params': [],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCtrmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuComplex* A, int64_t lda, cuComplex* x, int64_t incx);',
    },
    'cublasCtrsv_v2': {
        'base_op': 'trsv',
        'dtype': 'complex64',
        'level': 2,
        'variant': 'base',
        'description': 'The `cublasCtrsv_v2` function solves a system of linear equations with a triangular matrix and single-precision complex elements, performing either \( A x = b \) or \( A^T x = b \) where \( A \) is an upper or lower triangular matrix. It takes parameters for matrix shape (`uplo`), operation type (`trans`), diagonal properties (`diag`), matrix size (`n`), matrix data (`A`), leading dimension (`lda`), vector data (`x`), and vector stride (`incx`).',

        # C API 信息
        'c_api': 'cublasCtrsv_v2',
        'params': ['uplo', 'trans', 'diag', 'n', 'A', 'lda', 'x', 'incx'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'n': 'int',
            'A': 'const cuComplex*',
            'lda': 'int',
            'x': 'cuComplex*',
            'incx': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'x'],
        'scalar_params': [],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCtrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex* A, int lda, cuComplex* x, int incx);',
    },
    'cublasCtrsv_v2_64': {
        'base_op': 'trsv',
        'dtype': 'complex64',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasCtrsv_v2_64` function solves a system of linear equations with a triangular coefficient matrix for complex 64-bit floating-point data, where the matrix can be upper or lower triangular, transposed or not, and unit or non-unit diagonal. It supports 64-bit integer parameters for matrix and vector dimensions and strides.',

        # C API 信息
        'c_api': 'cublasCtrsv_v2_64',
        'params': ['uplo', 'trans', 'diag', 'n', 'A', 'lda', 'x', 'incx'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'n': 'int64_t',
            'A': 'const cuComplex*',
            'lda': 'int64_t',
            'x': 'cuComplex*',
            'incx': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'x'],
        'scalar_params': [],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCtrsv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuComplex* A, int64_t lda, cuComplex* x, int64_t incx);',
    },
    'cublasDgemv_v2': {
        'base_op': 'gemv',
        'dtype': 'float64',
        'level': 2,
        'variant': 'base',
        'description': 'The `cublasDgemv_v2` function performs double-precision matrix-vector multiplication, computing `y = alpha * op(A) * x + beta * y`, where `op(A)` can be the matrix `A` or its transpose, and `x` and `y` are vectors. It is a Level 2 BLAS operation for general matrices.',

        # C API 信息
        'c_api': 'cublasDgemv_v2',
        'params': ['trans', 'm', 'n', 'alpha', 'A', 'lda', 'x', 'incx', 'beta', 'y', 'incy'],
        'param_types': {
            'trans': 'cublasOperation_t',
            'm': 'int',
            'n': 'int',
            'alpha': 'const double*',
            'A': 'const double*',
            'lda': 'int',
            'x': 'const double*',
            'incx': 'int',
            'beta': 'const double*',
            'y': 'double*',
            'incy': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'x', 'y'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['y'],
        'return_params': ['y'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double* alpha, const double* A, int lda, const double* x, int incx, const double* beta, double* y, int incy);',
    },
    'cublasDgemv_v2_64': {
        'base_op': 'gemv',
        'dtype': 'float64',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasDgemv_v2_64` function performs double-precision matrix-vector multiplication (y = α·op(A)·x + β·y) using 64-bit integers for large matrix dimensions, where op(A) can be a transpose or non-transpose operation. It is a BLAS Level 2 operation designed for GPU acceleration through cuBLAS.',

        # C API 信息
        'c_api': 'cublasDgemv_v2_64',
        'params': ['trans', 'm', 'n', 'alpha', 'A', 'lda', 'x', 'incx', 'beta', 'y', 'incy'],
        'param_types': {
            'trans': 'cublasOperation_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'alpha': 'const double*',
            'A': 'const double*',
            'lda': 'int64_t',
            'x': 'const double*',
            'incx': 'int64_t',
            'beta': 'const double*',
            'y': 'double*',
            'incy': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'x', 'y'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['y'],
        'return_params': ['y'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDgemv_v2_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const double* alpha, const double* A, int64_t lda, const double* x, int64_t incx, const double* beta, double* y, int64_t incy);',
    },
    'cublasDger_v2': {
        'base_op': 'ger',
        'dtype': 'float64',
        'level': 2,
        'variant': 'base',
        'description': 'The `cublasDger_v2` function performs the rank-1 update of a double-precision matrix `A` by adding the outer product of vectors `xx` and `yy`, scaled by `alpha`, where `xx` is an m-element vector and `yy` is an n-element vector. It is a Level 2 BLAS operation that modifies the matrix in place.',

        # C API 信息
        'c_api': 'cublasDger_v2',
        'params': ['m', 'n', 'alpha', 'x', 'incx', 'y', 'incy', 'A', 'lda'],
        'param_types': {
            'm': 'int',
            'n': 'int',
            'alpha': 'const double*',
            'x': 'const double*',
            'incx': 'int',
            'y': 'const double*',
            'incy': 'int',
            'A': 'double*',
            'lda': 'int'
        },

        # 参数分类
        'tensor_params': ['x', 'y', 'A'],
        'scalar_params': ['alpha'],
        'inout_params': ['A'],
        'return_params': ['A'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDger_v2(cublasHandle_t handle, int m, int n, const double* alpha, const double* x, int incx, const double* y, int incy, double* A, int lda);',
    },
    'cublasDger_v2_64': {
        'base_op': 'ger',
        'dtype': 'float64',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasDger_v2_64` function performs a rank-1 update of a double-precision matrix `A` by adding the outer product of vectors `x` and `y` scaled by `alpha`, using 64-bit integers for large matrix dimensions. It is a BLAS Level 2 operation for general matrices.',

        # C API 信息
        'c_api': 'cublasDger_v2_64',
        'params': ['m', 'n', 'alpha', 'x', 'incx', 'y', 'incy', 'A', 'lda'],
        'param_types': {
            'm': 'int64_t',
            'n': 'int64_t',
            'alpha': 'const double*',
            'x': 'const double*',
            'incx': 'int64_t',
            'y': 'const double*',
            'incy': 'int64_t',
            'A': 'double*',
            'lda': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['x', 'y', 'A'],
        'scalar_params': ['alpha'],
        'inout_params': ['A'],
        'return_params': ['A'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDger_v2_64(cublasHandle_t handle, int64_t m, int64_t n, const double* alpha, const double* x, int64_t incx, const double* y, int64_t incy, double* A, int64_t lda);',
    },
    'cublasDsbmv_v2': {
        'base_op': 'sbmv',
        'dtype': 'float64',
        'level': 2,
        'variant': 'base',
        'description': 'The `cublasDsbmv_v2` function performs a symmetric banded matrix-vector multiplication using double-precision floating-point numbers, computing `y = alpha*A*x + beta*y`, where `A` is a symmetric band matrix with `k` subdiagonals. It supports upper or lower matrix storage modes and allows configurable strides for input and output vectors.',

        # C API 信息
        'c_api': 'cublasDsbmv_v2',
        'params': ['uplo', 'n', 'k', 'alpha', 'A', 'lda', 'x', 'incx', 'beta', 'y', 'incy'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'n': 'int',
            'k': 'int',
            'alpha': 'const double*',
            'A': 'const double*',
            'lda': 'int',
            'x': 'const double*',
            'incx': 'int',
            'beta': 'const double*',
            'y': 'double*',
            'incy': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'x', 'y'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['y'],
        'return_params': ['y'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDsbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const double* alpha, const double* A, int lda, const double* x, int incx, const double* beta, double* y, int incy);',
    },
    'cublasDsbmv_v2_64': {
        'base_op': 'sbmv',
        'dtype': 'float64',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasDsbmv_v2_64` function performs a symmetric banded matrix-vector multiplication using double-precision floating-point numbers, where the matrix is stored in banded format, and supports 64-bit integers for large problem sizes. It computes the operation `y = alpha*A*x + beta*y`, with `A` being a symmetric band matrix defined by its upper or lower triangular part.',

        # C API 信息
        'c_api': 'cublasDsbmv_v2_64',
        'params': ['uplo', 'n', 'k', 'alpha', 'A', 'lda', 'x', 'incx', 'beta', 'y', 'incy'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'n': 'int64_t',
            'k': 'int64_t',
            'alpha': 'const double*',
            'A': 'const double*',
            'lda': 'int64_t',
            'x': 'const double*',
            'incx': 'int64_t',
            'beta': 'const double*',
            'y': 'double*',
            'incy': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'x', 'y'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['y'],
        'return_params': ['y'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDsbmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, int64_t k, const double* alpha, const double* A, int64_t lda, const double* x, int64_t incx, const double* beta, double* y, int64_t incy);',
    },
    'cublasDspmv_v2': {
        'base_op': 'spmv',
        'dtype': 'float64',
        'level': 2,
        'variant': 'base',
        'description': 'The `cublasDspmv_v2` function performs a symmetric packed matrix-vector multiplication using double-precision floating-point numbers, computing `y = alpha * A * x + beta * y`, where `A` is a symmetric matrix stored in packed format. It supports configurable storage modes (upper/lower triangular) and stride parameters for input/output vectors.',

        # C API 信息
        'c_api': 'cublasDspmv_v2',
        'params': ['uplo', 'n', 'alpha', 'AP', 'x', 'incx', 'beta', 'y', 'incy'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'n': 'int',
            'alpha': 'const double*',
            'AP': 'const double*',
            'x': 'const double*',
            'incx': 'int',
            'beta': 'const double*',
            'y': 'double*',
            'incy': 'int'
        },

        # 参数分类
        'tensor_params': ['AP', 'x', 'y'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['y'],
        'return_params': ['y'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDspmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* AP, const double* x, int incx, const double* beta, double* y, int incy);',
    },
    'cublasDspmv_v2_64': {
        'base_op': 'spmv',
        'dtype': 'float64',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasDspmv_v2_64` function performs a symmetric packed matrix-vector multiplication using double-precision floating-point numbers, computing `y = alpha * A * x + beta * y` where `A` is a symmetric matrix stored in packed format, with support for 64-bit integers for large-scale operations. It takes parameters for matrix shape (`uplo`), size (`n`), scaling factors (`alpha`, `beta`), packed matrix (`AP`), input/output vectors (`x`, `y`), and their increments (`incx`, `incy`).',

        # C API 信息
        'c_api': 'cublasDspmv_v2_64',
        'params': ['uplo', 'n', 'alpha', 'AP', 'x', 'incx', 'beta', 'y', 'incy'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'n': 'int64_t',
            'alpha': 'const double*',
            'AP': 'const double*',
            'x': 'const double*',
            'incx': 'int64_t',
            'beta': 'const double*',
            'y': 'double*',
            'incy': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['AP', 'x', 'y'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['y'],
        'return_params': ['y'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDspmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double* alpha, const double* AP, const double* x, int64_t incx, const double* beta, double* y, int64_t incy);',
    },
    'cublasDspr2_v2': {
        'base_op': 'spr2',
        'dtype': 'float64',
        'level': 2,
        'variant': 'base',
        'description': 'The `cublasDspr2_v2` function performs a symmetric packed rank-2 update using double-precision floating-point numbers, adding the scaled outer product of vectors `x` and `y` to a packed symmetric matrix `AP`. It supports upper or lower triangular matrix storage (`uplo`) and allows for configurable stride increments (`incx`, `incy`) for the input vectors.',

        # C API 信息
        'c_api': 'cublasDspr2_v2',
        'params': ['uplo', 'n', 'alpha', 'x', 'incx', 'y', 'incy', 'AP'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'n': 'int',
            'alpha': 'const double*',
            'x': 'const double*',
            'incx': 'int',
            'y': 'const double*',
            'incy': 'int',
            'AP': 'double*'
        },

        # 参数分类
        'tensor_params': ['x', 'y', 'AP'],
        'scalar_params': ['alpha'],
        'inout_params': ['AP'],
        'return_params': ['AP'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDspr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* x, int incx, const double* y, int incy, double* AP);',
    },
    'cublasDspr2_v2_64': {
        'base_op': 'spr2',
        'dtype': 'float64',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasDspr2_v2_64` function performs a symmetric packed rank-2 update using double-precision floating-point numbers, adding the scaled outer product of vectors `x` and `y` to a packed symmetric matrix `AP` in 64-bit precision. It supports configurable storage modes (`uplo`) and stride parameters (`incx`, `incy`) for the input vectors.',

        # C API 信息
        'c_api': 'cublasDspr2_v2_64',
        'params': ['uplo', 'n', 'alpha', 'x', 'incx', 'y', 'incy', 'AP'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'n': 'int64_t',
            'alpha': 'const double*',
            'x': 'const double*',
            'incx': 'int64_t',
            'y': 'const double*',
            'incy': 'int64_t',
            'AP': 'double*'
        },

        # 参数分类
        'tensor_params': ['x', 'y', 'AP'],
        'scalar_params': ['alpha'],
        'inout_params': ['AP'],
        'return_params': ['AP'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDspr2_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double* alpha, const double* x, int64_t incx, const double* y, int64_t incy, double* AP);',
    },
    'cublasDspr_v2': {
        'base_op': 'spr',
        'dtype': 'float64',
        'level': 2,
        'variant': 'base',
        'description': 'The `cublasDspr_v2` function performs a symmetric packed rank-1 update of a double-precision matrix, adding the scaled outer product of a vector with itself to a symmetric matrix stored in packed format, with the fill mode specified by `uplo`. It is a Level 2 BLAS operation that takes a handle, matrix dimensions, scaling factor, input vector, stride, and packed matrix as parameters.',

        # C API 信息
        'c_api': 'cublasDspr_v2',
        'params': ['uplo', 'n', 'alpha', 'x', 'incx', 'AP'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'n': 'int',
            'alpha': 'const double*',
            'x': 'const double*',
            'incx': 'int',
            'AP': 'double*'
        },

        # 参数分类
        'tensor_params': ['x', 'AP'],
        'scalar_params': ['alpha'],
        'inout_params': ['AP'],
        'return_params': ['AP'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDspr_v2( cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* x, int incx, double* AP);',
    },
    'cublasDspr_v2_64': {
        'base_op': 'spr',
        'dtype': 'float64',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasDspr_v2_64` function performs the symmetric packed rank-1 update for double-precision real numbers, adding the scaled outer product of vector `x` with itself to a packed symmetric matrix `AP`, with support for 64-bit integers. The operation is defined by the uplo parameter, which specifies whether the upper or lower triangular part of the matrix is stored.',

        # C API 信息
        'c_api': 'cublasDspr_v2_64',
        'params': ['uplo', 'n', 'alpha', 'x', 'incx', 'AP'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'n': 'int64_t',
            'alpha': 'const double*',
            'x': 'const double*',
            'incx': 'int64_t',
            'AP': 'double*'
        },

        # 参数分类
        'tensor_params': ['x', 'AP'],
        'scalar_params': ['alpha'],
        'inout_params': ['AP'],
        'return_params': ['AP'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDspr_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double* alpha, const double* x, int64_t incx, double* AP);',
    },
    'cublasDsymv_v2': {
        'base_op': 'symv',
        'dtype': 'float64',
        'level': 2,
        'variant': 'base',
        'description': 'The `cublasDsymv_v2` function performs a symmetric matrix-vector multiplication using double-precision floating-point numbers, computing `y = alpha * A * x + beta * y`, where `A` is a symmetric matrix. It supports upper or lower triangular matrix storage via the `uplo` parameter and handles strided vector inputs.',

        # C API 信息
        'c_api': 'cublasDsymv_v2',
        'params': ['uplo', 'n', 'alpha', 'A', 'lda', 'x', 'incx', 'beta', 'y', 'incy'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'n': 'int',
            'alpha': 'const double*',
            'A': 'const double*',
            'lda': 'int',
            'x': 'const double*',
            'incx': 'int',
            'beta': 'const double*',
            'y': 'double*',
            'incy': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'x', 'y'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['y'],
        'return_params': ['y'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* A, int lda, const double* x, int incx, const double* beta, double* y, int incy);',
    },
    'cublasDsymv_v2_64': {
        'base_op': 'symv',
        'dtype': 'float64',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasDsymv_v2_64` function performs a symmetric matrix-vector multiplication using double-precision (float64) elements, computing `y = alpha*A*x + beta*y` where `A` is a symmetric matrix. It supports 64-bit integers for large-scale computations and requires specifying the matrix\'s fill mode (upper/lower triangular part).',

        # C API 信息
        'c_api': 'cublasDsymv_v2_64',
        'params': ['uplo', 'n', 'alpha', 'A', 'lda', 'x', 'incx', 'beta', 'y', 'incy'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'n': 'int64_t',
            'alpha': 'const double*',
            'A': 'const double*',
            'lda': 'int64_t',
            'x': 'const double*',
            'incx': 'int64_t',
            'beta': 'const double*',
            'y': 'double*',
            'incy': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'x', 'y'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['y'],
        'return_params': ['y'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDsymv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double* alpha, const double* A, int64_t lda, const double* x, int64_t incx, const double* beta, double* y, int64_t incy);',
    },
    'cublasDsyr2_v2': {
        'base_op': 'syr2',
        'dtype': 'float64',
        'level': 2,
        'variant': 'base',
        'description': 'The `cublasDsyr2_v2` function performs a symmetric rank-2 update of a double-precision matrix `A` using vectors `x` and `y`, scaling the result by `alpha` and storing it in the upper or lower triangular part of `A` as specified by `uplo`.',

        # C API 信息
        'c_api': 'cublasDsyr2_v2',
        'params': ['uplo', 'n', 'alpha', 'x', 'incx', 'y', 'incy', 'A', 'lda'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'n': 'int',
            'alpha': 'const double*',
            'x': 'const double*',
            'incx': 'int',
            'y': 'const double*',
            'incy': 'int',
            'A': 'double*',
            'lda': 'int'
        },

        # 参数分类
        'tensor_params': ['x', 'y', 'A'],
        'scalar_params': ['alpha'],
        'inout_params': ['A'],
        'return_params': ['A'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* x, int incx, const double* y, int incy, double* A, int lda);',
    },
    'cublasDsyr2_v2_64': {
        'base_op': 'syr2',
        'dtype': 'float64',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasDsyr2_v2_64` function performs a symmetric rank-2 update using double-precision floating-point numbers, adding the scaled outer products of vectors `x` and `y` to a symmetric matrix `A` stored in either the upper or lower triangular part, as specified by `uplo`, with 64-bit integer parameters for large-scale computations.',

        # C API 信息
        'c_api': 'cublasDsyr2_v2_64',
        'params': ['uplo', 'n', 'alpha', 'x', 'incx', 'y', 'incy', 'A', 'lda'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'n': 'int64_t',
            'alpha': 'const double*',
            'x': 'const double*',
            'incx': 'int64_t',
            'y': 'const double*',
            'incy': 'int64_t',
            'A': 'double*',
            'lda': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['x', 'y', 'A'],
        'scalar_params': ['alpha'],
        'inout_params': ['A'],
        'return_params': ['A'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDsyr2_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double* alpha, const double* x, int64_t incx, const double* y, int64_t incy, double* A, int64_t lda);',
    },
    'cublasDsyr_v2': {
        'base_op': 'syr',
        'dtype': 'float64',
        'level': 2,
        'variant': 'base',
        'description': 'The `cublasDsyr_v2` function performs a symmetric rank-1 update of a double-precision matrix `A` by adding the outer product of vector `x` scaled by `alpha`, where `A` is stored in either upper or lower triangular form as specified by `uplo`. It is a cuBLAS handle, matrix dimensions, vector increment, and leading dimension parameters for proper execution.',

        # C API 信息
        'c_api': 'cublasDsyr_v2',
        'params': ['uplo', 'n', 'alpha', 'x', 'incx', 'A', 'lda'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'n': 'int',
            'alpha': 'const double*',
            'x': 'const double*',
            'incx': 'int',
            'A': 'double*',
            'lda': 'int'
        },

        # 参数分类
        'tensor_params': ['x', 'A'],
        'scalar_params': ['alpha'],
        'inout_params': ['A'],
        'return_params': ['A'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* x, int incx, double* A, int lda);',
    },
    'cublasDsyr_v2_64': {
        'base_op': 'syr',
        'dtype': 'float64',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasDsyr_v2_64` function performs a symmetric rank-1 update (A = alpha * x * x^T + A) for double-precision matrices, where A is a symmetric matrix stored in upper or lower storage mode, with 64-bit integer parameters for large-scale computations. It takes a handle, fill mode, matrix size, scalar alpha, vector x, stride, matrix A, and leading dimension as inputs.',

        # C API 信息
        'c_api': 'cublasDsyr_v2_64',
        'params': ['uplo', 'n', 'alpha', 'x', 'incx', 'A', 'lda'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'n': 'int64_t',
            'alpha': 'const double*',
            'x': 'const double*',
            'incx': 'int64_t',
            'A': 'double*',
            'lda': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['x', 'A'],
        'scalar_params': ['alpha'],
        'inout_params': ['A'],
        'return_params': ['A'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDsyr_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double* alpha, const double* x, int64_t incx, double* A, int64_t lda);',
    },
    'cublasDtbmv_v2': {
        'base_op': 'tbmv',
        'dtype': 'float64',
        'level': 2,
        'variant': 'base',
        'description': 'The `cublasDtbmv_v2` function performs a banded matrix-vector multiplication using a double-precision triangular band matrix, with options to specify the matrix\'s fill mode, transpose operation, and diagonal type. It computes `x = A*x` or `x = A^T*x` where `A` is a triangular band matrix of size `n x n` with `k` diagonals.',

        # C API 信息
        'c_api': 'cublasDtbmv_v2',
        'params': ['uplo', 'trans', 'diag', 'n', 'k', 'A', 'lda', 'x', 'incx'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'n': 'int',
            'k': 'int',
            'A': 'const double*',
            'lda': 'int',
            'x': 'double*',
            'incx': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'x'],
        'scalar_params': [],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDtbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const double* A, int lda, double* x, int incx);',
    },
    'cublasDtbmv_v2_64': {
        'base_op': 'tbmv',
        'dtype': 'float64',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasDtbmv_v2_64` function performs a 64-bit banded matrix-vector multiplication using a double-precision triangular band matrix, with options for matrix orientation (`uplo`, `trans`) and diagonal type (`diag`). It computes `x = A*x` or `x = A^T*x` for a band matrix `A` stored in packed format, with dimensions `n x n` and bandwidth `k`.',

        # C API 信息
        'c_api': 'cublasDtbmv_v2_64',
        'params': ['uplo', 'trans', 'diag', 'n', 'k', 'A', 'lda', 'x', 'incx'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'n': 'int64_t',
            'k': 'int64_t',
            'A': 'const double*',
            'lda': 'int64_t',
            'x': 'double*',
            'incx': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'x'],
        'scalar_params': [],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDtbmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const double* A, int64_t lda, double* x, int64_t incx);',
    },
    'cublasDtbsv_v2': {
        'base_op': 'tbsv',
        'dtype': 'float64',
        'level': 2,
        'variant': 'base',
        'description': 'The `cublasDtbsv_v2` function solves a triangular banded system of equations with double-precision floating-point elements, where the matrix can be upper or lower triangular, transposed or not, and unit or non-unit diagonal. It takes a banded matrix `A` and vector `x` as input, overwriting `x` with the solution.',

        # C API 信息
        'c_api': 'cublasDtbsv_v2',
        'params': ['uplo', 'trans', 'diag', 'n', 'k', 'A', 'lda', 'x', 'incx'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'n': 'int',
            'k': 'int',
            'A': 'const double*',
            'lda': 'int',
            'x': 'double*',
            'incx': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'x'],
        'scalar_params': [],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDtbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const double* A, int lda, double* x, int incx);',
    },
    'cublasDtbsv_v2_64': {
        'base_op': 'tbsv',
        'dtype': 'float64',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasDtbsv_v2_64` function solves a triangular banded system of equations with double-precision floating-point elements, where the matrix can be upper or lower triangular, transposed or not, and unit or non-unit diagonal. It operates on 64-bit integer dimensions and strides, supporting large-scale computations.',

        # C API 信息
        'c_api': 'cublasDtbsv_v2_64',
        'params': ['uplo', 'trans', 'diag', 'n', 'k', 'A', 'lda', 'x', 'incx'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'n': 'int64_t',
            'k': 'int64_t',
            'A': 'const double*',
            'lda': 'int64_t',
            'x': 'double*',
            'incx': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'x'],
        'scalar_params': [],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDtbsv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const double* A, int64_t lda, double* x, int64_t incx);',
    },
    'cublasDtpmv_v2': {
        'base_op': 'tpmv',
        'dtype': 'float64',
        'level': 2,
        'variant': 'base',
        'description': 'The `cublasDtpmv_v2` function performs a matrix-vector multiplication using a packed triangular matrix (float64) and a vector, with options for matrix uplo (upper/lower), transposition, and unit/non-unit diagonal. It is a Level 2 BLAS operation designed for efficient triangular matrix operations on GPU-accelerated systems.',

        # C API 信息
        'c_api': 'cublasDtpmv_v2',
        'params': ['uplo', 'trans', 'diag', 'n', 'AP', 'x', 'incx'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'n': 'int',
            'AP': 'const double*',
            'x': 'double*',
            'incx': 'int'
        },

        # 参数分类
        'tensor_params': ['AP', 'x'],
        'scalar_params': [],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDtpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double* AP, double* x, int incx);',
    },
    'cublasDtpmv_v2_64': {
        'base_op': 'tpmv',
        'dtype': 'float64',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasDtpmv_v2_64` function performs a 64-bit triangular matrix-vector multiplication using a packed storage format, operating on double-precision (float64) data. It computes `x = A*x` or `x = A^T*x` for a triangular matrix `A` stored in packed form, with options for upper/lower triangular, transpose operation, and unit/non-unit diagonal.',

        # C API 信息
        'c_api': 'cublasDtpmv_v2_64',
        'params': ['uplo', 'trans', 'diag', 'n', 'AP', 'x', 'incx'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'n': 'int64_t',
            'AP': 'const double*',
            'x': 'double*',
            'incx': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['AP', 'x'],
        'scalar_params': [],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDtpmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const double* AP, double* x, int64_t incx);',
    },
    'cublasDtpsv_v2': {
        'base_op': 'tpsv',
        'dtype': 'float64',
        'level': 2,
        'variant': 'base',
        'description': 'The `cublasDtpsv_v2` function solves a triangular system of equations using a packed double-precision matrix, where the matrix can be upper or lower triangular, transposed or not, and unit or non-unit diagonal. It operates on vectors with a specified stride.',

        # C API 信息
        'c_api': 'cublasDtpsv_v2',
        'params': ['uplo', 'trans', 'diag', 'n', 'AP', 'x', 'incx'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'n': 'int',
            'AP': 'const double*',
            'x': 'double*',
            'incx': 'int'
        },

        # 参数分类
        'tensor_params': ['AP', 'x'],
        'scalar_params': [],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDtpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double* AP, double* x, int incx);',
    },
    'cublasDtpsv_v2_64': {
        'base_op': 'tpsv',
        'dtype': 'float64',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasDtpsv_v2_64` function solves a system of linear equations with a packed triangular matrix (float64) and a vector, supporting 64-bit integers for large problem sizes. It performs the operation \( x = A^{-1}x \) or \( x^T = x^TA^{-1} \), where \( A \) is a triangular matrix stored in packed format.',

        # C API 信息
        'c_api': 'cublasDtpsv_v2_64',
        'params': ['uplo', 'trans', 'diag', 'n', 'AP', 'x', 'incx'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'n': 'int64_t',
            'AP': 'const double*',
            'x': 'double*',
            'incx': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['AP', 'x'],
        'scalar_params': [],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDtpsv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const double* AP, double* x, int64_t incx);',
    },
    'cublasDtrmv_v2': {
        'base_op': 'trmv',
        'dtype': 'float64',
        'level': 2,
        'variant': 'base',
        'description': 'The `cublasDtrmv_v2` function performs a matrix-vector multiplication using a triangular double-precision matrix, either multiplying the vector by the matrix or its transpose, with options for upper/lower triangular storage and unit diagonal handling. It is a Level 2 BLAS operation that updates the input vector in-place with the result.',

        # C API 信息
        'c_api': 'cublasDtrmv_v2',
        'params': ['uplo', 'trans', 'diag', 'n', 'A', 'lda', 'x', 'incx'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'n': 'int',
            'A': 'const double*',
            'lda': 'int',
            'x': 'double*',
            'incx': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'x'],
        'scalar_params': [],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDtrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double* A, int lda, double* x, int incx);',
    },
    'cublasDtrmv_v2_64': {
        'base_op': 'trmv',
        'dtype': 'float64',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasDtrmv_v2_64` function computes the matrix-vector product for a triangular double-precision matrix, either multiplying the vector by the matrix or its transpose, with support for 64-bit integers. It allows specifying whether the matrix is upper or lower triangular and whether the diagonal should be treated as unit or non-unit.',

        # C API 信息
        'c_api': 'cublasDtrmv_v2_64',
        'params': ['uplo', 'trans', 'diag', 'n', 'A', 'lda', 'x', 'incx'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'n': 'int64_t',
            'A': 'const double*',
            'lda': 'int64_t',
            'x': 'double*',
            'incx': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'x'],
        'scalar_params': [],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDtrmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const double* A, int64_t lda, double* x, int64_t incx);',
    },
    'cublasDtrsv_v2': {
        'base_op': 'trsv',
        'dtype': 'float64',
        'level': 2,
        'variant': 'base',
        'description': 'The `cublasDtrsv_v2` function solves a triangular system of equations with a double-precision matrix, either performing forward or backward substitution depending on the matrix\'s upper/lower triangular structure and transpose operation specified. It takes a triangular matrix `A`, a vector `x`, and parameters to control the operation type (`trans`), matrix fill mode (`uplo`), and diagonal handling (`diag`).',

        # C API 信息
        'c_api': 'cublasDtrsv_v2',
        'params': ['uplo', 'trans', 'diag', 'n', 'A', 'lda', 'x', 'incx'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'n': 'int',
            'A': 'const double*',
            'lda': 'int',
            'x': 'double*',
            'incx': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'x'],
        'scalar_params': [],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDtrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double* A, int lda, double* x, int incx);',
    },
    'cublasDtrsv_v2_64': {
        'base_op': 'trsv',
        'dtype': 'float64',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasDtrsv_v2_64` function solves a triangular system of equations with double-precision floating-point elements, where the matrix can be upper or lower triangular, transposed or not, and unit or non-unit diagonal. It operates on 64-bit integer parameters for matrix and vector dimensions, strides, and handles large-scale problems.',

        # C API 信息
        'c_api': 'cublasDtrsv_v2_64',
        'params': ['uplo', 'trans', 'diag', 'n', 'A', 'lda', 'x', 'incx'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'n': 'int64_t',
            'A': 'const double*',
            'lda': 'int64_t',
            'x': 'double*',
            'incx': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'x'],
        'scalar_params': [],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDtrsv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const double* A, int64_t lda, double* x, int64_t incx);',
    },
    'cublasSgemv_v2': {
        'base_op': 'gemv',
        'dtype': 'float32',
        'level': 2,
        'variant': 'base',
        'description': 'The `cublasSgemv_v2` function performs single-precision matrix-vector multiplication, computing `y = alpha * op(A) * x + beta * y`, where `op(A)` can be the matrix `A` or its transpose, and `alpha` and `beta` are scalars. It is a Level 2 BLAS operation for 32-bit floating-point data.',

        # C API 信息
        'c_api': 'cublasSgemv_v2',
        'params': ['trans', 'm', 'n', 'alpha', 'A', 'lda', 'x', 'incx', 'beta', 'y', 'incy'],
        'param_types': {
            'trans': 'cublasOperation_t',
            'm': 'int',
            'n': 'int',
            'alpha': 'const float*',
            'A': 'const float*',
            'lda': 'int',
            'x': 'const float*',
            'incx': 'int',
            'beta': 'const float*',
            'y': 'float*',
            'incy': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'x', 'y'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['y'],
        'return_params': ['y'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float* alpha, const float* A, int lda, const float* x, int incx, const float* beta, float* y, int incy);',
    },
    'cublasSgemv_v2_64': {
        'base_op': 'gemv',
        'dtype': 'float32',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasSgemv_v2_64` function performs a single-precision matrix-vector multiplication (y = α·op(A)·x + β·y) using 64-bit integers for matrix and vector dimensions, where op(A) can be a transpose or non-transpose operation. It is a BLAS Level 2 operation designed for large-scale computations on CUDA-enabled GPUs.',

        # C API 信息
        'c_api': 'cublasSgemv_v2_64',
        'params': ['trans', 'm', 'n', 'alpha', 'A', 'lda', 'x', 'incx', 'beta', 'y', 'incy'],
        'param_types': {
            'trans': 'cublasOperation_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'alpha': 'const float*',
            'A': 'const float*',
            'lda': 'int64_t',
            'x': 'const float*',
            'incx': 'int64_t',
            'beta': 'const float*',
            'y': 'float*',
            'incy': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'x', 'y'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['y'],
        'return_params': ['y'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSgemv_v2_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const float* alpha, const float* A, int64_t lda, const float* x, int64_t incx, const float* beta, float* y, int64_t incy);',
    },
    'cublasSger_v2': {
        'base_op': 'ger',
        'dtype': 'float32',
        'level': 2,
        'variant': 'base',
        'description': 'The `cublasSger_v2` function performs the rank-1 update `A = alpha * x * y^T + A` for single-precision floating-point vectors `x` and `y`, and matrix `A`, where `x` is an `m`-element vector, `y` is an `n`-element vector, and `A` is an `m`-by-`n` matrix.',

        # C API 信息
        'c_api': 'cublasSger_v2',
        'params': ['m', 'n', 'alpha', 'x', 'incx', 'y', 'incy', 'A', 'lda'],
        'param_types': {
            'm': 'int',
            'n': 'int',
            'alpha': 'const float*',
            'x': 'const float*',
            'incx': 'int',
            'y': 'const float*',
            'incy': 'int',
            'A': 'float*',
            'lda': 'int'
        },

        # 参数分类
        'tensor_params': ['x', 'y', 'A'],
        'scalar_params': ['alpha'],
        'inout_params': ['A'],
        'return_params': ['A'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSger_v2(cublasHandle_t handle, int m, int n, const float* alpha, const float* x, int incx, const float* y, int incy, float* A, int lda);',
    },
    'cublasSger_v2_64': {
        'base_op': 'ger',
        'dtype': 'float32',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasSger_v2_64` function performs a rank-1 update of a single-precision matrix `A` by adding the outer product of vectors `x` and `y`, scaled by `alpha`, using 64-bit integers for large problem sizes. It is a BLAS Level 2 operation for general matrix-vector multiplication.',

        # C API 信息
        'c_api': 'cublasSger_v2_64',
        'params': ['m', 'n', 'alpha', 'x', 'incx', 'y', 'incy', 'A', 'lda'],
        'param_types': {
            'm': 'int64_t',
            'n': 'int64_t',
            'alpha': 'const float*',
            'x': 'const float*',
            'incx': 'int64_t',
            'y': 'const float*',
            'incy': 'int64_t',
            'A': 'float*',
            'lda': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['x', 'y', 'A'],
        'scalar_params': ['alpha'],
        'inout_params': ['A'],
        'return_params': ['A'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSger_v2_64(cublasHandle_t handle, int64_t m, int64_t n, const float* alpha, const float* x, int64_t incx, const float* y, int64_t incy, float* A, int64_t lda);',
    },
    'cublasSsbmv_v2': {
        'base_op': 'sbmv',
        'dtype': 'float32',
        'level': 2,
        'variant': 'base',
        'description': 'The `cublasSsbmv_v2` function performs a symmetric banded matrix-vector multiplication using single-precision floats, computing `y = alpha*A*x + beta*y`, where `A` is an `n×n` symmetric band matrix with `k` subdiagonals. It supports upper or lower storage modes and allows configurable strides for vectors `x` and `y`.',

        # C API 信息
        'c_api': 'cublasSsbmv_v2',
        'params': ['uplo', 'n', 'k', 'alpha', 'A', 'lda', 'x', 'incx', 'beta', 'y', 'incy'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'n': 'int',
            'k': 'int',
            'alpha': 'const float*',
            'A': 'const float*',
            'lda': 'int',
            'x': 'const float*',
            'incx': 'int',
            'beta': 'const float*',
            'y': 'float*',
            'incy': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'x', 'y'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['y'],
        'return_params': ['y'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSsbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const float* alpha, const float* A, int lda, const float* x, int incx, const float* beta, float* y, int incy);',
    },
    'cublasSsbmv_v2_64': {
        'base_op': 'sbmv',
        'dtype': 'float32',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasSsbmv_v2_64` function performs a symmetric banded matrix-vector multiplication using single-precision floats, where the matrix is stored in banded format, and supports 64-bit integers for large problem sizes. It computes `y = alpha*A*x + beta*y` with `A` being a `n x n` symmetric band matrix with `k` subdiagonals, and `x` and `y` being vectors.',

        # C API 信息
        'c_api': 'cublasSsbmv_v2_64',
        'params': ['uplo', 'n', 'k', 'alpha', 'A', 'lda', 'x', 'incx', 'beta', 'y', 'incy'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'n': 'int64_t',
            'k': 'int64_t',
            'alpha': 'const float*',
            'A': 'const float*',
            'lda': 'int64_t',
            'x': 'const float*',
            'incx': 'int64_t',
            'beta': 'const float*',
            'y': 'float*',
            'incy': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'x', 'y'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['y'],
        'return_params': ['y'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSsbmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, int64_t k, const float* alpha, const float* A, int64_t lda, const float* x, int64_t incx, const float* beta, float* y, int64_t incy);',
    },
    'cublasSspmv_v2': {
        'base_op': 'spmv',
        'dtype': 'float32',
        'level': 2,
        'variant': 'base',
        'description': 'The `cublasSspmv_v2` function performs a symmetric packed matrix-vector multiplication for single-precision (float32) data, computing `y = alpha * A * x + beta * y`, where `A` is a symmetric matrix stored in packed format. It supports configurable storage modes (upper/lower triangular) and stride parameters for input/output vectors.',

        # C API 信息
        'c_api': 'cublasSspmv_v2',
        'params': ['uplo', 'n', 'alpha', 'AP', 'x', 'incx', 'beta', 'y', 'incy'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'n': 'int',
            'alpha': 'const float*',
            'AP': 'const float*',
            'x': 'const float*',
            'incx': 'int',
            'beta': 'const float*',
            'y': 'float*',
            'incy': 'int'
        },

        # 参数分类
        'tensor_params': ['AP', 'x', 'y'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['y'],
        'return_params': ['y'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSspmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* AP, const float* x, int incx, const float* beta, float* y, int incy);',
    },
    'cublasSspmv_v2_64': {
        'base_op': 'spmv',
        'dtype': 'float32',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasSspmv_v2_64` function performs a symmetric packed matrix-vector multiplication for 64-bit integers using single-precision floats, computing `y = alpha * A * x + beta * y` where `A` is a symmetric matrix stored in packed format. It supports configurable storage modes (`uplo`) and stride parameters (`incx`, `incy`) for the input and output vectors.',

        # C API 信息
        'c_api': 'cublasSspmv_v2_64',
        'params': ['uplo', 'n', 'alpha', 'AP', 'x', 'incx', 'beta', 'y', 'incy'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'n': 'int64_t',
            'alpha': 'const float*',
            'AP': 'const float*',
            'x': 'const float*',
            'incx': 'int64_t',
            'beta': 'const float*',
            'y': 'float*',
            'incy': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['AP', 'x', 'y'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['y'],
        'return_params': ['y'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSspmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float* alpha, const float* AP, const float* x, int64_t incx, const float* beta, float* y, int64_t incy);',
    },
    'cublasSspr2_v2': {
        'base_op': 'spr2',
        'dtype': 'float32',
        'level': 2,
        'variant': 'base',
        'description': 'The `cublasSspr2_v2` function performs a symmetric packed rank-2 update for single-precision (float32) vectors, adding the scaled outer product of vectors `x` and `y` to a packed symmetric matrix `AP` in upper or lower storage mode. It takes parameters for matrix shape (`uplo`), size (`n`), scaling factor (`alpha`), vector strides (`incx`, `incy`), and handles the operation via the cuBLAS context (`handle`).',

        # C API 信息
        'c_api': 'cublasSspr2_v2',
        'params': ['uplo', 'n', 'alpha', 'x', 'incx', 'y', 'incy', 'AP'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'n': 'int',
            'alpha': 'const float*',
            'x': 'const float*',
            'incx': 'int',
            'y': 'const float*',
            'incy': 'int',
            'AP': 'float*'
        },

        # 参数分类
        'tensor_params': ['x', 'y', 'AP'],
        'scalar_params': ['alpha'],
        'inout_params': ['AP'],
        'return_params': ['AP'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSspr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, const float* y, int incy, float* AP);',
    },
    'cublasSspr2_v2_64': {
        'base_op': 'spr2',
        'dtype': 'float32',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasSspr2_v2_64` function performs a symmetric packed rank-2 update (`spr2`) for 64-bit integers, adding the scaled outer product of single-precision vectors `x` and `y` to a packed symmetric matrix `AP`, with `uplo` specifying whether the upper or lower triangular part is stored.',

        # C API 信息
        'c_api': 'cublasSspr2_v2_64',
        'params': ['uplo', 'n', 'alpha', 'x', 'incx', 'y', 'incy', 'AP'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'n': 'int64_t',
            'alpha': 'const float*',
            'x': 'const float*',
            'incx': 'int64_t',
            'y': 'const float*',
            'incy': 'int64_t',
            'AP': 'float*'
        },

        # 参数分类
        'tensor_params': ['x', 'y', 'AP'],
        'scalar_params': ['alpha'],
        'inout_params': ['AP'],
        'return_params': ['AP'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSspr2_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float* alpha, const float* x, int64_t incx, const float* y, int64_t incy, float* AP);',
    },
    'cublasSspr_v2': {
        'base_op': 'spr',
        'dtype': 'float32',
        'level': 2,
        'variant': 'base',
        'description': 'The `cublasSspr_v2` function performs a symmetric packed rank-1 update of a single-precision real matrix, adding the scaled outer product of vector `x` with itself to the packed symmetric matrix `AP`, where `uplo` specifies whether the upper or lower triangular part is stored.',

        # C API 信息
        'c_api': 'cublasSspr_v2',
        'params': ['uplo', 'n', 'alpha', 'x', 'incx', 'AP'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'n': 'int',
            'alpha': 'const float*',
            'x': 'const float*',
            'incx': 'int',
            'AP': 'float*'
        },

        # 参数分类
        'tensor_params': ['x', 'AP'],
        'scalar_params': ['alpha'],
        'inout_params': ['AP'],
        'return_params': ['AP'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSspr_v2( cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, float* AP);',
    },
    'cublasSspr_v2_64': {
        'base_op': 'spr',
        'dtype': 'float32',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasSspr_v2_64` function computes a symmetric rank-1 update of a packed single-precision matrix `AP` using the vector `x`, scaled by `alpha`, with 64-bit integer support for large problem sizes. It supports upper or lower triangular matrix storage specified by `uplo`.',

        # C API 信息
        'c_api': 'cublasSspr_v2_64',
        'params': ['uplo', 'n', 'alpha', 'x', 'incx', 'AP'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'n': 'int64_t',
            'alpha': 'const float*',
            'x': 'const float*',
            'incx': 'int64_t',
            'AP': 'float*'
        },

        # 参数分类
        'tensor_params': ['x', 'AP'],
        'scalar_params': ['alpha'],
        'inout_params': ['AP'],
        'return_params': ['AP'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSspr_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float* alpha, const float* x, int64_t incx, float* AP);',
    },
    'cublasSsymv_v2': {
        'base_op': 'symv',
        'dtype': 'float32',
        'level': 2,
        'variant': 'base',
        'description': 'The `cublasSsymv_v2` function performs a symmetric matrix-vector multiplication using a single-precision real symmetric matrix, computing `y = alpha * A * x + beta * y`, where `A` is a symmetric matrix stored in upper or lower triangular form. It takes parameters for matrix layout (`uplo`), size (`n`), scaling factors (`alpha`, `beta`), input/output vectors (`x`, `y`), and their increments (`incx`, `incy`).',

        # C API 信息
        'c_api': 'cublasSsymv_v2',
        'params': ['uplo', 'n', 'alpha', 'A', 'lda', 'x', 'incx', 'beta', 'y', 'incy'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'n': 'int',
            'alpha': 'const float*',
            'A': 'const float*',
            'lda': 'int',
            'x': 'const float*',
            'incx': 'int',
            'beta': 'const float*',
            'y': 'float*',
            'incy': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'x', 'y'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['y'],
        'return_params': ['y'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* A, int lda, const float* x, int incx, const float* beta, float* y, int incy);',
    },
    'cublasSsymv_v2_64': {
        'base_op': 'symv',
        'dtype': 'float32',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasSsymv_v2_64` function performs a symmetric matrix-vector multiplication using 64-bit integers, computing `y = alpha * A * x + beta * y` for a single-precision symmetric matrix `A` and vectors `x`, `y`. It supports configurable uplo (upper/lower triangular) and stride parameters for the matrix and vectors.',

        # C API 信息
        'c_api': 'cublasSsymv_v2_64',
        'params': ['uplo', 'n', 'alpha', 'A', 'lda', 'x', 'incx', 'beta', 'y', 'incy'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'n': 'int64_t',
            'alpha': 'const float*',
            'A': 'const float*',
            'lda': 'int64_t',
            'x': 'const float*',
            'incx': 'int64_t',
            'beta': 'const float*',
            'y': 'float*',
            'incy': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'x', 'y'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['y'],
        'return_params': ['y'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSsymv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float* alpha, const float* A, int64_t lda, const float* x, int64_t incx, const float* beta, float* y, int64_t incy);',
    },
    'cublasSsyr2_v2': {
        'base_op': 'syr2',
        'dtype': 'float32',
        'level': 2,
        'variant': 'base',
        'description': 'The `cublasSsyr2_v2` function performs a symmetric rank-2 update of a single-precision real matrix `A` by adding the outer product of vectors `x` and `y` (scaled by `alpha`), where `A` is stored in either upper or lower triangular form as specified by `uplo`.',

        # C API 信息
        'c_api': 'cublasSsyr2_v2',
        'params': ['uplo', 'n', 'alpha', 'x', 'incx', 'y', 'incy', 'A', 'lda'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'n': 'int',
            'alpha': 'const float*',
            'x': 'const float*',
            'incx': 'int',
            'y': 'const float*',
            'incy': 'int',
            'A': 'float*',
            'lda': 'int'
        },

        # 参数分类
        'tensor_params': ['x', 'y', 'A'],
        'scalar_params': ['alpha'],
        'inout_params': ['A'],
        'return_params': ['A'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, const float* y, int incy, float* A, int lda);',
    },
    'cublasSsyr2_v2_64': {
        'base_op': 'syr2',
        'dtype': 'float32',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasSsyr2_v2_64` function performs a symmetric rank-2 update of a single-precision matrix A with vectors x and y, storing the result in A, using 64-bit integers for large problem sizes. It supports upper or lower triangular matrix storage and requires specifying strides for the input vectors.',

        # C API 信息
        'c_api': 'cublasSsyr2_v2_64',
        'params': ['uplo', 'n', 'alpha', 'x', 'incx', 'y', 'incy', 'A', 'lda'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'n': 'int64_t',
            'alpha': 'const float*',
            'x': 'const float*',
            'incx': 'int64_t',
            'y': 'const float*',
            'incy': 'int64_t',
            'A': 'float*',
            'lda': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['x', 'y', 'A'],
        'scalar_params': ['alpha'],
        'inout_params': ['A'],
        'return_params': ['A'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSsyr2_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float* alpha, const float* x, int64_t incx, const float* y, int64_t incy, float* A, int64_t lda);',
    },
    'cublasSsyr_v2': {
        'base_op': 'syr',
        'dtype': 'float32',
        'level': 2,
        'variant': 'base',
        'description': 'The `cublasSsyr_v2` function performs a symmetric rank-1 update of a float32 matrix `A` by adding the product of a vector `x` with its transpose, scaled by `alpha`, where `A` is stored in either the upper or lower triangular part as specified by `uplo`.',

        # C API 信息
        'c_api': 'cublasSsyr_v2',
        'params': ['uplo', 'n', 'alpha', 'x', 'incx', 'A', 'lda'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'n': 'int',
            'alpha': 'const float*',
            'x': 'const float*',
            'incx': 'int',
            'A': 'float*',
            'lda': 'int'
        },

        # 参数分类
        'tensor_params': ['x', 'A'],
        'scalar_params': ['alpha'],
        'inout_params': ['A'],
        'return_params': ['A'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, float* A, int lda);',
    },
    'cublasSsyr_v2_64': {
        'base_op': 'syr',
        'dtype': 'float32',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasSsyr_v2_64` function performs a symmetric rank-1 update (A = alpha * x * x^T + A) for single-precision (float32) matrices, where A is a symmetric matrix stored in upper or lower triangular form, with 64-bit integer parameters for large problem sizes. It takes a handle, fill mode (uplo), matrix size (n), scalar (alpha), vector (x), stride (incx), matrix (A), and leading dimension (lda) as inputs.',

        # C API 信息
        'c_api': 'cublasSsyr_v2_64',
        'params': ['uplo', 'n', 'alpha', 'x', 'incx', 'A', 'lda'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'n': 'int64_t',
            'alpha': 'const float*',
            'x': 'const float*',
            'incx': 'int64_t',
            'A': 'float*',
            'lda': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['x', 'A'],
        'scalar_params': ['alpha'],
        'inout_params': ['A'],
        'return_params': ['A'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSsyr_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float* alpha, const float* x, int64_t incx, float* A, int64_t lda);',
    },
    'cublasStbmv_v2': {
        'base_op': 'tbmv',
        'dtype': 'float32',
        'level': 2,
        'variant': 'base',
        'description': 'The `cublasStbmv_v2` function performs a matrix-vector multiplication using a single-precision triangular band matrix, where the matrix can be upper or lower triangular, transposed or not, and unit or non-unit diagonal. It computes `x = A*x` or `x = A^T*x` based on the specified parameters.',

        # C API 信息
        'c_api': 'cublasStbmv_v2',
        'params': ['uplo', 'trans', 'diag', 'n', 'k', 'A', 'lda', 'x', 'incx'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'n': 'int',
            'k': 'int',
            'A': 'const float*',
            'lda': 'int',
            'x': 'float*',
            'incx': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'x'],
        'scalar_params': [],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasStbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const float* A, int lda, float* x, int incx);',
    },
    'cublasStbmv_v2_64': {
        'base_op': 'tbmv',
        'dtype': 'float32',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasStbmv_v2_64` function performs a banded triangular matrix-vector multiplication using a 32-bit float matrix and vector, supporting 64-bit integers for large problem sizes. It multiplies the matrix `A` (upper or lower triangular banded) by vector `x`, with options for matrix transposition and unit diagonal handling.',

        # C API 信息
        'c_api': 'cublasStbmv_v2_64',
        'params': ['uplo', 'trans', 'diag', 'n', 'k', 'A', 'lda', 'x', 'incx'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'n': 'int64_t',
            'k': 'int64_t',
            'A': 'const float*',
            'lda': 'int64_t',
            'x': 'float*',
            'incx': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'x'],
        'scalar_params': [],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasStbmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const float* A, int64_t lda, float* x, int64_t incx);',
    },
    'cublasStbsv_v2': {
        'base_op': 'tbsv',
        'dtype': 'float32',
        'level': 2,
        'variant': 'base',
        'description': 'The `cublasStbsv_v2` function solves a triangular banded system of equations with a single right-hand side using 32-bit floating-point arithmetic, where the matrix can be upper or lower triangular, transposed or not, and unit or non-unit diagonal. It takes a banded triangular matrix `A`, vector `x`, and parameters for matrix structure and operation type.',

        # C API 信息
        'c_api': 'cublasStbsv_v2',
        'params': ['uplo', 'trans', 'diag', 'n', 'k', 'A', 'lda', 'x', 'incx'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'n': 'int',
            'k': 'int',
            'A': 'const float*',
            'lda': 'int',
            'x': 'float*',
            'incx': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'x'],
        'scalar_params': [],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasStbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const float* A, int lda, float* x, int incx);',
    },
    'cublasStbsv_v2_64': {
        'base_op': 'tbsv',
        'dtype': 'float32',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasStbsv_v2_64` function solves a triangular banded system of equations with a single right-hand side using 32-bit floating-point arithmetic, supporting 64-bit integers for large problem sizes. It takes a banded matrix `A` and vector `x`, performing the operation `x = op(A)^(-1) * x` where `op` can be a transpose or no transpose, with options for matrix upper/lower storage and unit/non-unit diagonal.',

        # C API 信息
        'c_api': 'cublasStbsv_v2_64',
        'params': ['uplo', 'trans', 'diag', 'n', 'k', 'A', 'lda', 'x', 'incx'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'n': 'int64_t',
            'k': 'int64_t',
            'A': 'const float*',
            'lda': 'int64_t',
            'x': 'float*',
            'incx': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'x'],
        'scalar_params': [],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasStbsv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const float* A, int64_t lda, float* x, int64_t incx);',
    },
    'cublasStpmv_v2': {
        'base_op': 'tpmv',
        'dtype': 'float32',
        'level': 2,
        'variant': 'base',
        'description': 'The `cublasStpmv_v2` function performs a matrix-vector multiplication using a packed triangular matrix (float32), computing `x = A*x` or `x = A^T*x` depending on the transpose operation specified, where `A` is an upper or lower triangular matrix stored in packed format. It supports options for matrix fill mode (upper/lower), transpose operation, and diagonal type (unit/non-unit).',

        # C API 信息
        'c_api': 'cublasStpmv_v2',
        'params': ['uplo', 'trans', 'diag', 'n', 'AP', 'x', 'incx'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'n': 'int',
            'AP': 'const float*',
            'x': 'float*',
            'incx': 'int'
        },

        # 参数分类
        'tensor_params': ['AP', 'x'],
        'scalar_params': [],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasStpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float* AP, float* x, int incx);',
    },
    'cublasStpmv_v2_64': {
        'base_op': 'tpmv',
        'dtype': 'float32',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasStpmv_v2_64` function performs a triangular matrix-vector multiplication using a packed single-precision (float32) triangular matrix, with 64-bit integer parameters. It supports configurable matrix fill mode (uplo), transpose operation (trans), diagonal type (diag), and vector stride (incx).',

        # C API 信息
        'c_api': 'cublasStpmv_v2_64',
        'params': ['uplo', 'trans', 'diag', 'n', 'AP', 'x', 'incx'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'n': 'int64_t',
            'AP': 'const float*',
            'x': 'float*',
            'incx': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['AP', 'x'],
        'scalar_params': [],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasStpmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const float* AP, float* x, int64_t incx);',
    },
    'cublasStpsv_v2': {
        'base_op': 'tpsv',
        'dtype': 'float32',
        'level': 2,
        'variant': 'base',
        'description': 'The `cublasStpsv_v2` function solves a system of linear equations with a packed triangular coefficient matrix and single-precision floating-point elements, where the matrix can be upper or lower triangular, transposed or not, and unit or non-unit diagonal. It computes \( x = op(A)^{-1} x \) in place, with the matrix \( A \) stored in packed format.',

        # C API 信息
        'c_api': 'cublasStpsv_v2',
        'params': ['uplo', 'trans', 'diag', 'n', 'AP', 'x', 'incx'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'n': 'int',
            'AP': 'const float*',
            'x': 'float*',
            'incx': 'int'
        },

        # 参数分类
        'tensor_params': ['AP', 'x'],
        'scalar_params': [],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasStpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float* AP, float* x, int incx);',
    },
    'cublasStpsv_v2_64': {
        'base_op': 'tpsv',
        'dtype': 'float32',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasStpsv_v2_64` function solves a system of linear equations with a packed triangular coefficient matrix (float32) and a single right-hand side vector, using 64-bit integers for large problem sizes. It supports configurable matrix storage (upper/lower triangular), transpose operations, and unit/non-unit diagonal options.',

        # C API 信息
        'c_api': 'cublasStpsv_v2_64',
        'params': ['uplo', 'trans', 'diag', 'n', 'AP', 'x', 'incx'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'n': 'int64_t',
            'AP': 'const float*',
            'x': 'float*',
            'incx': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['AP', 'x'],
        'scalar_params': [],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasStpsv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const float* AP, float* x, int64_t incx);',
    },
    'cublasStrmv_v2': {
        'base_op': 'trmv',
        'dtype': 'float32',
        'level': 2,
        'variant': 'base',
        'description': 'The `cublasStrmv_v2` function performs a matrix-vector multiplication using a triangular matrix (float32) and a vector, where the matrix can be upper or lower triangular, optionally transposed, and optionally unit diagonal. It is a Level 2 BLAS operation that computes `x = A*x` or `x = A^T*x` with the specified triangular matrix properties.',

        # C API 信息
        'c_api': 'cublasStrmv_v2',
        'params': ['uplo', 'trans', 'diag', 'n', 'A', 'lda', 'x', 'incx'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'n': 'int',
            'A': 'const float*',
            'lda': 'int',
            'x': 'float*',
            'incx': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'x'],
        'scalar_params': [],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasStrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float* A, int lda, float* x, int incx);',
    },
    'cublasStrmv_v2_64': {
        'base_op': 'trmv',
        'dtype': 'float32',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasStrmv_v2_64` function performs a triangular matrix-vector multiplication using a single-precision floating-point matrix, where the matrix can be upper or lower triangular, transposed or not, and unit or non-unit diagonal, with 64-bit integer parameters for large problem sizes. It computes `x = A*x` or `x = A^T*x` depending on the specified transpose operation.',

        # C API 信息
        'c_api': 'cublasStrmv_v2_64',
        'params': ['uplo', 'trans', 'diag', 'n', 'A', 'lda', 'x', 'incx'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'n': 'int64_t',
            'A': 'const float*',
            'lda': 'int64_t',
            'x': 'float*',
            'incx': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'x'],
        'scalar_params': [],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasStrmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const float* A, int64_t lda, float* x, int64_t incx);',
    },
    'cublasStrsv_v2': {
        'base_op': 'trsv',
        'dtype': 'float32',
        'level': 2,
        'variant': 'base',
        'description': 'The `cublasStrsv_v2` function solves a triangular system of linear equations with a single right-hand side, using a square matrix `A` (float32) and vector `x`, where `A` can be upper or lower triangular and optionally unit-diagonal. The operation can be performed with or without transposition of the matrix.',

        # C API 信息
        'c_api': 'cublasStrsv_v2',
        'params': ['uplo', 'trans', 'diag', 'n', 'A', 'lda', 'x', 'incx'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'n': 'int',
            'A': 'const float*',
            'lda': 'int',
            'x': 'float*',
            'incx': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'x'],
        'scalar_params': [],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasStrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float* A, int lda, float* x, int incx);',
    },
    'cublasStrsv_v2_64': {
        'base_op': 'trsv',
        'dtype': 'float32',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasStrsv_v2_64` function solves a triangular system of equations with a single right-hand side using 32-bit floating-point arithmetic, supporting 64-bit integers for large matrix dimensions. It takes a triangular matrix `A` and vector `x`, applying the specified fill mode (`uplo`), operation (`trans`), and diagonal type (`diag`) to compute the solution in place.',

        # C API 信息
        'c_api': 'cublasStrsv_v2_64',
        'params': ['uplo', 'trans', 'diag', 'n', 'A', 'lda', 'x', 'incx'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'n': 'int64_t',
            'A': 'const float*',
            'lda': 'int64_t',
            'x': 'float*',
            'incx': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'x'],
        'scalar_params': [],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasStrsv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const float* A, int64_t lda, float* x, int64_t incx);',
    },
    'cublasZgemv_v2': {
        'base_op': 'gemv',
        'dtype': 'complex128',
        'level': 2,
        'variant': 'base',
        'description': 'The cublasZgemv_v2 function performs matrix-vector multiplication using a general complex double-precision matrix, computing y = α·op(A)·x + β·y, where op(A) is either A or its transpose/conjugate transpose. It supports configurable matrix dimensions, strides, and scaling factors for the input and output vectors.',

        # C API 信息
        'c_api': 'cublasZgemv_v2',
        'params': ['trans', 'm', 'n', 'alpha', 'A', 'lda', 'x', 'incx', 'beta', 'y', 'incy'],
        'param_types': {
            'trans': 'cublasOperation_t',
            'm': 'int',
            'n': 'int',
            'alpha': 'const cuDoubleComplex*',
            'A': 'const cuDoubleComplex*',
            'lda': 'int',
            'x': 'const cuDoubleComplex*',
            'incx': 'int',
            'beta': 'const cuDoubleComplex*',
            'y': 'cuDoubleComplex*',
            'incy': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'x', 'y'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['y'],
        'return_params': ['y'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy);',
    },
    'cublasZgemv_v2_64': {
        'base_op': 'gemv',
        'dtype': 'complex128',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasZgemv_v2_64` function performs a 64-bit matrix-vector multiplication using double-precision complex numbers, computing y = α·op(A)·x + β·y, where op(A) can be the matrix A or its transpose, and α, β are scalars. It supports large matrices and vectors with 64-bit integer dimensions and strides.',

        # C API 信息
        'c_api': 'cublasZgemv_v2_64',
        'params': ['trans', 'm', 'n', 'alpha', 'A', 'lda', 'x', 'incx', 'beta', 'y', 'incy'],
        'param_types': {
            'trans': 'cublasOperation_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'alpha': 'const cuDoubleComplex*',
            'A': 'const cuDoubleComplex*',
            'lda': 'int64_t',
            'x': 'const cuDoubleComplex*',
            'incx': 'int64_t',
            'beta': 'const cuDoubleComplex*',
            'y': 'cuDoubleComplex*',
            'incy': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'x', 'y'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['y'],
        'return_params': ['y'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZgemv_v2_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int64_t incy);',
    },
    'cublasZgerc_v2': {
        'base_op': 'gerc',
        'dtype': 'complex128',
        'level': 2,
        'variant': 'base',
        'description': 'Performs the rank-1 update of a complex double-precision matrix A using conjugated vector y, adding alpha * x * yᴴ to A, where yᴴ is the conjugate transpose of y. The operation supports arbitrary strides (incx, incy) for vectors x and y.',

        # C API 信息
        'c_api': 'cublasZgerc_v2',
        'params': ['m', 'n', 'alpha', 'x', 'incx', 'y', 'incy', 'A', 'lda'],
        'param_types': {
            'm': 'int',
            'n': 'int',
            'alpha': 'const cuDoubleComplex*',
            'x': 'const cuDoubleComplex*',
            'incx': 'int',
            'y': 'const cuDoubleComplex*',
            'incy': 'int',
            'A': 'cuDoubleComplex*',
            'lda': 'int'
        },

        # 参数分类
        'tensor_params': ['x', 'y', 'A'],
        'scalar_params': ['alpha'],
        'inout_params': ['A'],
        'return_params': ['A'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZgerc_v2(cublasHandle_t handle, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* A, int lda);',
    },
    'cublasZgerc_v2_64': {
        'base_op': 'gerc',
        'dtype': 'complex128',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasZgerc_v2_64` function performs the rank-1 update of a complex double-precision matrix `A` using the conjugate of vector `x` and vector `y`, scaled by `alpha`, with 64-bit integer parameters for large matrix support. It computes `A = alpha * x * y^H + A`, where `y^H` is the conjugate transpose of `y`.',

        # C API 信息
        'c_api': 'cublasZgerc_v2_64',
        'params': ['m', 'n', 'alpha', 'x', 'incx', 'y', 'incy', 'A', 'lda'],
        'param_types': {
            'm': 'int64_t',
            'n': 'int64_t',
            'alpha': 'const cuDoubleComplex*',
            'x': 'const cuDoubleComplex*',
            'incx': 'int64_t',
            'y': 'const cuDoubleComplex*',
            'incy': 'int64_t',
            'A': 'cuDoubleComplex*',
            'lda': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['x', 'y', 'A'],
        'scalar_params': ['alpha'],
        'inout_params': ['A'],
        'return_params': ['A'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZgerc_v2_64(cublasHandle_t handle, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* y, int64_t incy, cuDoubleComplex* A, int64_t lda);',
    },
    'cublasZgeru_v2': {
        'base_op': 'geru',
        'dtype': 'complex128',
        'level': 2,
        'variant': 'base',
        'description': 'The `cublasZgeru_v2` function performs the rank-1 update of a complex double-precision matrix `A` using the outer product of vectors `x` and `y`, scaled by `alpha`, following the operation `A = alpha * x * y^T + A`. It supports general (non-conjugated) complex arithmetic and requires leading dimension `lda` for matrix storage.',

        # C API 信息
        'c_api': 'cublasZgeru_v2',
        'params': ['m', 'n', 'alpha', 'x', 'incx', 'y', 'incy', 'A', 'lda'],
        'param_types': {
            'm': 'int',
            'n': 'int',
            'alpha': 'const cuDoubleComplex*',
            'x': 'const cuDoubleComplex*',
            'incx': 'int',
            'y': 'const cuDoubleComplex*',
            'incy': 'int',
            'A': 'cuDoubleComplex*',
            'lda': 'int'
        },

        # 参数分类
        'tensor_params': ['x', 'y', 'A'],
        'scalar_params': ['alpha'],
        'inout_params': ['A'],
        'return_params': ['A'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZgeru_v2(cublasHandle_t handle, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* A, int lda);',
    },
    'cublasZgeru_v2_64': {
        'base_op': 'geru',
        'dtype': 'complex128',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasZgeru_v2_64` function performs a rank-1 update of a complex double-precision matrix `A` using vectors `x` and `y`, storing the result as `A = alpha * x * y^T + A`, where `alpha` is a complex scalar and `x` and `y` are complex vectors with 64-bit integer strides.',

        # C API 信息
        'c_api': 'cublasZgeru_v2_64',
        'params': ['m', 'n', 'alpha', 'x', 'incx', 'y', 'incy', 'A', 'lda'],
        'param_types': {
            'm': 'int64_t',
            'n': 'int64_t',
            'alpha': 'const cuDoubleComplex*',
            'x': 'const cuDoubleComplex*',
            'incx': 'int64_t',
            'y': 'const cuDoubleComplex*',
            'incy': 'int64_t',
            'A': 'cuDoubleComplex*',
            'lda': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['x', 'y', 'A'],
        'scalar_params': ['alpha'],
        'inout_params': ['A'],
        'return_params': ['A'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZgeru_v2_64(cublasHandle_t handle, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* y, int64_t incy, cuDoubleComplex* A, int64_t lda);',
    },
    'cublasZsymv_v2': {
        'base_op': 'symv',
        'dtype': 'complex128',
        'level': 2,
        'variant': 'base',
        'description': 'The `cublasZsymv_v2` function performs a symmetric matrix-vector multiplication using a complex double-precision matrix, adding the result to a scaled vector. It computes \( y = \alpha \cdot A \cdot x + \beta \cdot y \), where \( A \) is symmetric and only the specified triangular part (upper or lower) is accessed.',

        # C API 信息
        'c_api': 'cublasZsymv_v2',
        'params': ['uplo', 'n', 'alpha', 'A', 'lda', 'x', 'incx', 'beta', 'y', 'incy'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'n': 'int',
            'alpha': 'const cuDoubleComplex*',
            'A': 'const cuDoubleComplex*',
            'lda': 'int',
            'x': 'const cuDoubleComplex*',
            'incx': 'int',
            'beta': 'const cuDoubleComplex*',
            'y': 'cuDoubleComplex*',
            'incy': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'x', 'y'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['y'],
        'return_params': ['y'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy);',
    },
    'cublasZsymv_v2_64': {
        'base_op': 'symv',
        'dtype': 'complex128',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasZsymv_v2_64` function performs a symmetric matrix-vector multiplication using a complex double-precision matrix, scaling the input vector and adding it to the output vector. It supports 64-bit integers for large matrix operations and allows specifying the matrix\'s triangular part (upper or lower) to be used.',

        # C API 信息
        'c_api': 'cublasZsymv_v2_64',
        'params': ['uplo', 'n', 'alpha', 'A', 'lda', 'x', 'incx', 'beta', 'y', 'incy'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'n': 'int64_t',
            'alpha': 'const cuDoubleComplex*',
            'A': 'const cuDoubleComplex*',
            'lda': 'int64_t',
            'x': 'const cuDoubleComplex*',
            'incx': 'int64_t',
            'beta': 'const cuDoubleComplex*',
            'y': 'cuDoubleComplex*',
            'incy': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'x', 'y'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['y'],
        'return_params': ['y'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZsymv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int64_t incy);',
    },
    'cublasZsyr2_v2': {
        'base_op': 'syr2',
        'dtype': 'complex128',
        'level': 2,
        'variant': 'base',
        'description': 'The `cublasZsyr2_v2` function performs a symmetric rank-2 update of a complex double-precision matrix `A` using vectors `x` and `y`, where `A` is stored in either the upper or lower triangular part as specified by `uplo`. It computes `A = alpha*x*y^T + alpha*y*x^T + A`, with `alpha` as a scalar multiplier.',

        # C API 信息
        'c_api': 'cublasZsyr2_v2',
        'params': ['uplo', 'n', 'alpha', 'x', 'incx', 'y', 'incy', 'A', 'lda'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'n': 'int',
            'alpha': 'const cuDoubleComplex*',
            'x': 'const cuDoubleComplex*',
            'incx': 'int',
            'y': 'const cuDoubleComplex*',
            'incy': 'int',
            'A': 'cuDoubleComplex*',
            'lda': 'int'
        },

        # 参数分类
        'tensor_params': ['x', 'y', 'A'],
        'scalar_params': ['alpha'],
        'inout_params': ['A'],
        'return_params': ['A'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* A, int lda);',
    },
    'cublasZsyr2_v2_64': {
        'base_op': 'syr2',
        'dtype': 'complex128',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasZsyr2_v2_64` function performs a symmetric rank-2 update using double-precision complex vectors, adding the outer products of vectors `x` and `y` to a symmetric matrix `A` stored in either the upper or lower triangular part, as specified by `uplo`. This 64-bit variant supports large matrices and vectors with extended index ranges.',

        # C API 信息
        'c_api': 'cublasZsyr2_v2_64',
        'params': ['uplo', 'n', 'alpha', 'x', 'incx', 'y', 'incy', 'A', 'lda'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'n': 'int64_t',
            'alpha': 'const cuDoubleComplex*',
            'x': 'const cuDoubleComplex*',
            'incx': 'int64_t',
            'y': 'const cuDoubleComplex*',
            'incy': 'int64_t',
            'A': 'cuDoubleComplex*',
            'lda': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['x', 'y', 'A'],
        'scalar_params': ['alpha'],
        'inout_params': ['A'],
        'return_params': ['A'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZsyr2_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* y, int64_t incy, cuDoubleComplex* A, int64_t lda);',
    },
    'cublasZsyr_v2': {
        'base_op': 'syr',
        'dtype': 'complex128',
        'level': 2,
        'variant': 'base',
        'description': 'The function `cublasZsyr_v2` performs a symmetric rank-1 update of a complex double-precision matrix `A` by adding the outer product of vector `x` with itself, scaled by `alpha`, where `A` is stored in either upper or lower triangular form as specified by `uplo`.',

        # C API 信息
        'c_api': 'cublasZsyr_v2',
        'params': ['uplo', 'n', 'alpha', 'x', 'incx', 'A', 'lda'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'n': 'int',
            'alpha': 'const cuDoubleComplex*',
            'x': 'const cuDoubleComplex*',
            'incx': 'int',
            'A': 'cuDoubleComplex*',
            'lda': 'int'
        },

        # 参数分类
        'tensor_params': ['x', 'A'],
        'scalar_params': ['alpha'],
        'inout_params': ['A'],
        'return_params': ['A'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, cuDoubleComplex* A, int lda);',
    },
    'cublasZsyr_v2_64': {
        'base_op': 'syr',
        'dtype': 'complex128',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasZsyr_v2_64` function performs a symmetric rank-1 update on a 64-bit complex double-precision matrix `A` by multiplying a vector `x` with its conjugate transpose and scaling by `alpha`, storing the result in `A` matrix according to the specified fill mode (`uplo`). It supports 64-bit integer parameters for large-scale computations.',

        # C API 信息
        'c_api': 'cublasZsyr_v2_64',
        'params': ['uplo', 'n', 'alpha', 'x', 'incx', 'A', 'lda'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'n': 'int64_t',
            'alpha': 'const cuDoubleComplex*',
            'x': 'const cuDoubleComplex*',
            'incx': 'int64_t',
            'A': 'cuDoubleComplex*',
            'lda': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['x', 'A'],
        'scalar_params': ['alpha'],
        'inout_params': ['A'],
        'return_params': ['A'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZsyr_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int64_t incx, cuDoubleComplex* A, int64_t lda);',
    },
    'cublasZtbmv_v2': {
        'base_op': 'tbmv',
        'dtype': 'complex128',
        'level': 2,
        'variant': 'base',
        'description': 'The `cublasZtbmv_v2` function performs a banded matrix-vector multiplication using a complex double-precision triangular band matrix, optionally applying a transpose operation and accounting for unit or non-unit diagonal elements. It computes `x = op(A) * x` where `op(A)` can be the matrix, its transpose, or conjugate transpose, with `A` stored in banded format.',

        # C API 信息
        'c_api': 'cublasZtbmv_v2',
        'params': ['uplo', 'trans', 'diag', 'n', 'k', 'A', 'lda', 'x', 'incx'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'n': 'int',
            'k': 'int',
            'A': 'const cuDoubleComplex*',
            'lda': 'int',
            'x': 'cuDoubleComplex*',
            'incx': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'x'],
        'scalar_params': [],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZtbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx);',
    },
    'cublasZtbmv_v2_64': {
        'base_op': 'tbmv',
        'dtype': 'complex128',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasZtbmv_v2_64` function performs a banded matrix-vector multiplication using a complex double-precision triangular band matrix, with support for 64-bit integers, where the matrix can be upper or lower triangular, transposed or not, and optionally treated as unit triangular. It computes `x = A*x` or `x = A^T*x` or `x = A^H*x` depending on the specified operation.',

        # C API 信息
        'c_api': 'cublasZtbmv_v2_64',
        'params': ['uplo', 'trans', 'diag', 'n', 'k', 'A', 'lda', 'x', 'incx'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'n': 'int64_t',
            'k': 'int64_t',
            'A': 'const cuDoubleComplex*',
            'lda': 'int64_t',
            'x': 'cuDoubleComplex*',
            'incx': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'x'],
        'scalar_params': [],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZtbmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const cuDoubleComplex* A, int64_t lda, cuDoubleComplex* x, int64_t incx);',
    },
    'cublasZtbsv_v2': {
        'base_op': 'tbsv',
        'dtype': 'complex128',
        'level': 2,
        'variant': 'base',
        'description': 'The `cublasZtbsv_v2` function solves a complex double-precision triangular banded system of equations with a single right-hand side, using the specified uplo, trans, and diag parameters to determine the matrix structure and operation. It operates on a banded matrix `A` of size `n x n` with `k` sub-/super-diagonals and updates the vector `x` in place.',

        # C API 信息
        'c_api': 'cublasZtbsv_v2',
        'params': ['uplo', 'trans', 'diag', 'n', 'k', 'A', 'lda', 'x', 'incx'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'n': 'int',
            'k': 'int',
            'A': 'const cuDoubleComplex*',
            'lda': 'int',
            'x': 'cuDoubleComplex*',
            'incx': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'x'],
        'scalar_params': [],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZtbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx);',
    },
    'cublasZtbsv_v2_64': {
        'base_op': 'tbsv',
        'dtype': 'complex128',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasZtbsv_v2_64` function solves a system of linear equations with a complex double-precision triangular band matrix, using the specified uplo, trans, and diag parameters, and supports 64-bit integers for large problem sizes. It operates on vectors with a given increment and stores the result in place.',

        # C API 信息
        'c_api': 'cublasZtbsv_v2_64',
        'params': ['uplo', 'trans', 'diag', 'n', 'k', 'A', 'lda', 'x', 'incx'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'n': 'int64_t',
            'k': 'int64_t',
            'A': 'const cuDoubleComplex*',
            'lda': 'int64_t',
            'x': 'cuDoubleComplex*',
            'incx': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'x'],
        'scalar_params': [],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZtbsv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const cuDoubleComplex* A, int64_t lda, cuDoubleComplex* x, int64_t incx);',
    },
    'cublasZtpmv_v2': {
        'base_op': 'tpmv',
        'dtype': 'complex128',
        'level': 2,
        'variant': 'base',
        'description': 'The `cublasZtpmv_v2` function performs a matrix-vector multiplication using a packed triangular matrix and a complex double-precision vector, with options for matrix uplo (upper/lower), transposition, and unit/non-unit diagonal. It is a Level 2 BLAS operation that computes `x = op(A) * x`, where `A` is stored in packed format.',

        # C API 信息
        'c_api': 'cublasZtpmv_v2',
        'params': ['uplo', 'trans', 'diag', 'n', 'AP', 'x', 'incx'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'n': 'int',
            'AP': 'const cuDoubleComplex*',
            'x': 'cuDoubleComplex*',
            'incx': 'int'
        },

        # 参数分类
        'tensor_params': ['AP', 'x'],
        'scalar_params': [],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZtpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex* AP, cuDoubleComplex* x, int incx);',
    },
    'cublasZtpmv_v2_64': {
        'base_op': 'tpmv',
        'dtype': 'complex128',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasZtpmv_v2_64` function performs a complex double-precision matrix-vector multiplication using a triangular packed matrix, supporting 64-bit integers for large-scale computations. It multiplies the matrix by vector `x` according to the specified uplo, trans, and diag parameters, storing the result in `x`.',

        # C API 信息
        'c_api': 'cublasZtpmv_v2_64',
        'params': ['uplo', 'trans', 'diag', 'n', 'AP', 'x', 'incx'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'n': 'int64_t',
            'AP': 'const cuDoubleComplex*',
            'x': 'cuDoubleComplex*',
            'incx': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['AP', 'x'],
        'scalar_params': [],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZtpmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuDoubleComplex* AP, cuDoubleComplex* x, int64_t incx);',
    },
    'cublasZtpsv_v2': {
        'base_op': 'tpsv',
        'dtype': 'complex128',
        'level': 2,
        'variant': 'base',
        'description': 'The `cublasZtpsv_v2` function solves a system of linear equations with a complex double-precision triangular packed matrix, either in upper or lower triangular form, and a complex double-precision vector, optionally applying a transpose operation and assuming unit diagonal elements if specified.',

        # C API 信息
        'c_api': 'cublasZtpsv_v2',
        'params': ['uplo', 'trans', 'diag', 'n', 'AP', 'x', 'incx'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'n': 'int',
            'AP': 'const cuDoubleComplex*',
            'x': 'cuDoubleComplex*',
            'incx': 'int'
        },

        # 参数分类
        'tensor_params': ['AP', 'x'],
        'scalar_params': [],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZtpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex* AP, cuDoubleComplex* x, int incx);',
    },
    'cublasZtpsv_v2_64': {
        'base_op': 'tpsv',
        'dtype': 'complex128',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasZtpsv_v2_64` function solves a system of linear equations with a packed triangular matrix of complex double-precision elements, where the matrix can be upper or lower triangular, transposed or not, and unit or non-unit diagonal. It operates on 64-bit integer dimensions and increments, making it suitable for large-scale computations.',

        # C API 信息
        'c_api': 'cublasZtpsv_v2_64',
        'params': ['uplo', 'trans', 'diag', 'n', 'AP', 'x', 'incx'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'n': 'int64_t',
            'AP': 'const cuDoubleComplex*',
            'x': 'cuDoubleComplex*',
            'incx': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['AP', 'x'],
        'scalar_params': [],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZtpsv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuDoubleComplex* AP, cuDoubleComplex* x, int64_t incx);',
    },
    'cublasZtrmv_v2': {
        'base_op': 'trmv',
        'dtype': 'complex128',
        'level': 2,
        'variant': 'base',
        'description': 'The `cublasZtrmv_v2` function performs a matrix-vector multiplication using a complex double-precision triangular matrix, either multiplying the vector by the matrix, its transpose, or its conjugate transpose, with options for upper/lower triangular storage and unit diagonal handling. It is a Level 2 BLAS operation designed for triangular matrix operations on GPU-accelerated systems.',

        # C API 信息
        'c_api': 'cublasZtrmv_v2',
        'params': ['uplo', 'trans', 'diag', 'n', 'A', 'lda', 'x', 'incx'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'n': 'int',
            'A': 'const cuDoubleComplex*',
            'lda': 'int',
            'x': 'cuDoubleComplex*',
            'incx': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'x'],
        'scalar_params': [],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZtrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx);',
    },
    'cublasZtrmv_v2_64': {
        'base_op': 'trmv',
        'dtype': 'complex128',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasZtrmv_v2_64` function performs a triangular matrix-vector multiplication using a complex double-precision matrix, where the matrix can be upper or lower triangular and optionally transposed or conjugated. It supports 64-bit integers for large matrix dimensions and strides.',

        # C API 信息
        'c_api': 'cublasZtrmv_v2_64',
        'params': ['uplo', 'trans', 'diag', 'n', 'A', 'lda', 'x', 'incx'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'n': 'int64_t',
            'A': 'const cuDoubleComplex*',
            'lda': 'int64_t',
            'x': 'cuDoubleComplex*',
            'incx': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'x'],
        'scalar_params': [],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZtrmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuDoubleComplex* A, int64_t lda, cuDoubleComplex* x, int64_t incx);',
    },
    'cublasZtrsv_v2': {
        'base_op': 'trsv',
        'dtype': 'complex128',
        'level': 2,
        'variant': 'base',
        'description': 'The `cublasZtrsv_v2` function solves a system of linear equations with a complex double-precision triangular matrix, either computing \( A \cdot x = b \) or \( A^T \cdot x = b \), where \( A \) is an upper or lower triangular matrix and \( x \) is overwritten with the solution. It supports configurable matrix storage (uplo), transpose operation (trans), and diagonal type (unit/non-unit).',

        # C API 信息
        'c_api': 'cublasZtrsv_v2',
        'params': ['uplo', 'trans', 'diag', 'n', 'A', 'lda', 'x', 'incx'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'n': 'int',
            'A': 'const cuDoubleComplex*',
            'lda': 'int',
            'x': 'cuDoubleComplex*',
            'incx': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'x'],
        'scalar_params': [],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZtrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx);',
    },
    'cublasZtrsv_v2_64': {
        'base_op': 'trsv',
        'dtype': 'complex128',
        'level': 2,
        'variant': '_64',
        'description': 'The `cublasZtrsv_v2_64` function solves a system of linear equations with a complex double-precision triangular matrix, computing `x = op(A)^(-1) * x`, where `op(A)` can be the matrix, its transpose, or conjugate transpose. It supports 64-bit integers for large matrix operations and allows specification of matrix storage (upper/lower triangular) and diagonal type (unit/non-unit).',

        # C API 信息
        'c_api': 'cublasZtrsv_v2_64',
        'params': ['uplo', 'trans', 'diag', 'n', 'A', 'lda', 'x', 'incx'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'n': 'int64_t',
            'A': 'const cuDoubleComplex*',
            'lda': 'int64_t',
            'x': 'cuDoubleComplex*',
            'incx': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'x'],
        'scalar_params': [],
        'inout_params': ['x'],
        'return_params': ['x'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZtrsv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuDoubleComplex* A, int64_t lda, cuDoubleComplex* x, int64_t incx);',
    },

    # ======================================================================
    # Level 3
    # ======================================================================

    'cublasSgemmStridedBatched': {
        'base_op': 'gemm',
        'dtype': 'float32',
        'level': 3,
        'variant': 'StridedBatched',
        'description': 'The `cublasSgemmStridedBatched` function performs batched matrix-matrix multiplication for single-precision matrices, where each batch uses strided memory access for input and output matrices. It computes C = α * op(A) * op(B) + β * C for each batch, with op() optionally transposing the matrices.',

        # C API 信息
        'c_api': 'cublasSgemmStridedBatched',
        'params': ['transa', 'transb', 'm', 'n', 'k', 'alpha', 'A', 'lda', 'strideA', 'B', 'ldb', 'strideB', 'beta', 'C', 'ldc', 'strideC', 'batchCount'],
        'param_types': {
            'transa': 'cublasOperation_t',
            'transb': 'cublasOperation_t',
            'm': 'int',
            'n': 'int',
            'k': 'int',
            'alpha': 'const float*',
            'A': 'const float*',
            'lda': 'int',
            'strideA': 'long long int',
            'B': 'const float*',
            'ldb': 'int',
            'strideB': 'long long int',
            'beta': 'const float*',
            'C': 'float*',
            'ldc': 'int',
            'strideC': 'long long int',
            'batchCount': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float* beta, float* , int ldc, long long int strideC, int batchCount);',
    },
    'cublasSgemmStridedBatched_64': {
        'base_op': 'gemm',
        'dtype': 'float32',
        'level': 3,
        'variant': 'StridedBatched_64',
        'description': 'This function performs batched matrix-matrix multiplication for float32 matrices with 64-bit integers, where each batch uses strided memory access. It computes C = alpha * op(A) * op(B) + beta * C for each matrix in the batch, with configurable transpositions and fixed strides between matrices.',

        # C API 信息
        'c_api': 'cublasSgemmStridedBatched_64',
        'params': ['transa', 'transb', 'm', 'n', 'k', 'alpha', 'A', 'lda', 'strideA', 'B', 'ldb', 'strideB', 'beta', 'C', 'ldc', 'strideC', 'batchCount'],
        'param_types': {
            'transa': 'cublasOperation_t',
            'transb': 'cublasOperation_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'k': 'int64_t',
            'alpha': 'const float*',
            'A': 'const float*',
            'lda': 'int64_t',
            'strideA': 'long long int',
            'B': 'const float*',
            'ldb': 'int64_t',
            'strideB': 'long long int',
            'beta': 'const float*',
            'C': 'float*',
            'ldc': 'int64_t',
            'strideC': 'long long int',
            'batchCount': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSgemmStridedBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const float* alpha, const float* A, int64_t lda, long long int strideA, const float* B, int64_t ldb, long long int strideB, const float* beta, float* , int64_t ldc, long long int strideC, int64_t batchCount);',
    },
    'cublasCgemmStridedBatched': {
        'base_op': 'gemm',
        'dtype': 'complex64',
        'level': 3,
        'variant': 'StridedBatched',
        'description': 'Performs batched complex matrix-matrix multiplication with strided inputs, computing C = α * op(A) * op(B) + β * C for each batch, where op(X) is optionally transposed and each matrix in the batch is accessed with fixed strides.',

        # C API 信息
        'c_api': 'cublasCgemmStridedBatched',
        'params': ['transa', 'transb', 'm', 'n', 'k', 'alpha', 'A', 'lda', 'strideA', 'B', 'ldb', 'strideB', 'beta', 'C', 'ldc', 'strideC', 'batchCount'],
        'param_types': {
            'transa': 'cublasOperation_t',
            'transb': 'cublasOperation_t',
            'm': 'int',
            'n': 'int',
            'k': 'int',
            'alpha': 'const cuComplex*',
            'A': 'const cuComplex*',
            'lda': 'int',
            'strideA': 'long long int',
            'B': 'const cuComplex*',
            'ldb': 'int',
            'strideB': 'long long int',
            'beta': 'const cuComplex*',
            'C': 'cuComplex*',
            'ldc': 'int',
            'strideC': 'long long int',
            'batchCount': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, long long int strideA, const cuComplex* B, int ldb, long long int strideB, const cuComplex* beta, cuComplex* , int ldc, long long int strideC, int batchCount);',
    },
    'cublasCgemmStridedBatched_64': {
        'base_op': 'gemm',
        'dtype': 'complex64',
        'level': 3,
        'variant': 'StridedBatched_64',
        'description': 'The `cublasCgemmStridedBatched_64` function performs batched complex matrix-matrix multiplication (GEMM) with 64-bit integers, where each matrix in the batch is separated by a fixed stride. It computes the operation `C = alpha * op(A) * op(B) + beta * C` for each matrix in the batch, supporting transposition options for A and B.',

        # C API 信息
        'c_api': 'cublasCgemmStridedBatched_64',
        'params': ['transa', 'transb', 'm', 'n', 'k', 'alpha', 'A', 'lda', 'strideA', 'B', 'ldb', 'strideB', 'beta', 'C', 'ldc', 'strideC', 'batchCount'],
        'param_types': {
            'transa': 'cublasOperation_t',
            'transb': 'cublasOperation_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'k': 'int64_t',
            'alpha': 'const cuComplex*',
            'A': 'const cuComplex*',
            'lda': 'int64_t',
            'strideA': 'long long int',
            'B': 'const cuComplex*',
            'ldb': 'int64_t',
            'strideB': 'long long int',
            'beta': 'const cuComplex*',
            'C': 'cuComplex*',
            'ldc': 'int64_t',
            'strideC': 'long long int',
            'batchCount': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCgemmStridedBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, long long int strideA, const cuComplex* B, int64_t ldb, long long int strideB, const cuComplex* beta, cuComplex* , int64_t ldc, long long int strideC, int64_t batchCount);',
    },
    'cublasDgemmStridedBatched': {
        'base_op': 'gemm',
        'dtype': 'float64',
        'level': 3,
        'variant': 'StridedBatched',
        'description': 'The `cublasDgemmStridedBatched` function performs batched matrix-matrix multiplication for double-precision matrices, where each batch uses strided memory access. It computes C = α * op(A) * op(B) + β * C for each batch, with separate strides for A, B, and C matrices.',

        # C API 信息
        'c_api': 'cublasDgemmStridedBatched',
        'params': ['transa', 'transb', 'm', 'n', 'k', 'alpha', 'A', 'lda', 'strideA', 'B', 'ldb', 'strideB', 'beta', 'C', 'ldc', 'strideC', 'batchCount'],
        'param_types': {
            'transa': 'cublasOperation_t',
            'transb': 'cublasOperation_t',
            'm': 'int',
            'n': 'int',
            'k': 'int',
            'alpha': 'const double*',
            'A': 'const double*',
            'lda': 'int',
            'strideA': 'long long int',
            'B': 'const double*',
            'ldb': 'int',
            'strideB': 'long long int',
            'beta': 'const double*',
            'C': 'double*',
            'ldc': 'int',
            'strideC': 'long long int',
            'batchCount': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double* alpha, const double* A, int lda, long long int strideA, const double* B, int ldb, long long int strideB, const double* beta, double* , int ldc, long long int strideC, int batchCount);',
    },
    'cublasDgemmStridedBatched_64': {
        'base_op': 'gemm',
        'dtype': 'float64',
        'level': 3,
        'variant': 'StridedBatched_64',
        'description': 'The `cublasDgemmStridedBatched_64` function performs batched double-precision matrix-matrix multiplication (GEMM) with 64-bit integers, where each batch uses strided input and output matrices. It computes `C = alpha * op(A) * op(B) + beta * C` for each batch, with matrices separated by fixed strides in memory.',

        # C API 信息
        'c_api': 'cublasDgemmStridedBatched_64',
        'params': ['transa', 'transb', 'm', 'n', 'k', 'alpha', 'A', 'lda', 'strideA', 'B', 'ldb', 'strideB', 'beta', 'C', 'ldc', 'strideC', 'batchCount'],
        'param_types': {
            'transa': 'cublasOperation_t',
            'transb': 'cublasOperation_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'k': 'int64_t',
            'alpha': 'const double*',
            'A': 'const double*',
            'lda': 'int64_t',
            'strideA': 'long long int',
            'B': 'const double*',
            'ldb': 'int64_t',
            'strideB': 'long long int',
            'beta': 'const double*',
            'C': 'double*',
            'ldc': 'int64_t',
            'strideC': 'long long int',
            'batchCount': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDgemmStridedBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const double* alpha, const double* A, int64_t lda, long long int strideA, const double* B, int64_t ldb, long long int strideB, const double* beta, double* , int64_t ldc, long long int strideC, int64_t batchCount);',
    },
    'cublasHgemmStridedBatched': {
        'base_op': 'gemm',
        'dtype': 'float16',
        'level': 3,
        'variant': 'StridedBatched',
        'description': 'Performs a batched matrix-matrix multiplication using strided arrays of half-precision (float16) matrices, computing C = alpha * op(A) * op(B) + beta * C for each batch, where op(X) is either X or its transpose. The function processes multiple matrix multiplications with consistent dimensions and strides between batches.',

        # C API 信息
        'c_api': 'cublasHgemmStridedBatched',
        'params': ['transa', 'transb', 'm', 'n', 'k', 'alpha', 'A', 'lda', 'strideA', 'B', 'ldb', 'strideB', 'beta', 'C', 'ldc', 'strideC', 'batchCount'],
        'param_types': {
            'transa': 'cublasOperation_t',
            'transb': 'cublasOperation_t',
            'm': 'int',
            'n': 'int',
            'k': 'int',
            'alpha': 'const __half*',
            'A': 'const __half*',
            'lda': 'int',
            'strideA': 'long long int',
            'B': 'const __half*',
            'ldb': 'int',
            'strideB': 'long long int',
            'beta': 'const __half*',
            'C': '__half*',
            'ldc': 'int',
            'strideC': 'long long int',
            'batchCount': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasHgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const __half* alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half* beta, __half* , int ldc, long long int strideC, int batchCount);',
    },
    'cublasHgemmStridedBatched_64': {
        'base_op': 'gemm',
        'dtype': 'float16',
        'level': 3,
        'variant': 'StridedBatched_64',
        'description': 'This function performs batched matrix-matrix multiplication (GEMM) operations using 16-bit floating-point (half precision) matrices with 64-bit integers for large problem sizes, where each matrix in the batch is separated by a fixed stride. It computes C = α * op(A) * op(B) + β * C for each matrix in the batch, supporting transposition options and strided memory access.',

        # C API 信息
        'c_api': 'cublasHgemmStridedBatched_64',
        'params': ['transa', 'transb', 'm', 'n', 'k', 'alpha', 'A', 'lda', 'strideA', 'B', 'ldb', 'strideB', 'beta', 'C', 'ldc', 'strideC', 'batchCount'],
        'param_types': {
            'transa': 'cublasOperation_t',
            'transb': 'cublasOperation_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'k': 'int64_t',
            'alpha': 'const __half*',
            'A': 'const __half*',
            'lda': 'int64_t',
            'strideA': 'long long int',
            'B': 'const __half*',
            'ldb': 'int64_t',
            'strideB': 'long long int',
            'beta': 'const __half*',
            'C': '__half*',
            'ldc': 'int64_t',
            'strideC': 'long long int',
            'batchCount': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasHgemmStridedBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const __half* alpha, const __half* A, int64_t lda, long long int strideA, const __half* B, int64_t ldb, long long int strideB, const __half* beta, __half* , int64_t ldc, long long int strideC, int64_t batchCount);',
    },
    'cublasZgemmStridedBatched': {
        'base_op': 'gemm',
        'dtype': 'complex128',
        'level': 3,
        'variant': 'StridedBatched',
        'description': 'Performs batched complex double-precision matrix-matrix multiplication with strided inputs, computing C = alpha * op(A) * op(B) + beta * C for each batch, where op(X) is either X or X^T/X^H, with matrices stored at fixed strides in memory.',

        # C API 信息
        'c_api': 'cublasZgemmStridedBatched',
        'params': ['transa', 'transb', 'm', 'n', 'k', 'alpha', 'A', 'lda', 'strideA', 'B', 'ldb', 'strideB', 'beta', 'C', 'ldc', 'strideC', 'batchCount'],
        'param_types': {
            'transa': 'cublasOperation_t',
            'transb': 'cublasOperation_t',
            'm': 'int',
            'n': 'int',
            'k': 'int',
            'alpha': 'const cuDoubleComplex*',
            'A': 'const cuDoubleComplex*',
            'lda': 'int',
            'strideA': 'long long int',
            'B': 'const cuDoubleComplex*',
            'ldb': 'int',
            'strideB': 'long long int',
            'beta': 'const cuDoubleComplex*',
            'C': 'cuDoubleComplex*',
            'ldc': 'int',
            'strideC': 'long long int',
            'batchCount': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, long long int strideA, const cuDoubleComplex* B, int ldb, long long int strideB, const cuDoubleComplex* beta, cuDoubleComplex* , int ldc, long long int strideC, int batchCount);',
    },
    'cublasZgemmStridedBatched_64': {
        'base_op': 'gemm',
        'dtype': 'complex128',
        'level': 3,
        'variant': 'StridedBatched_64',
        'description': 'Performs batched complex double-precision matrix-matrix multiplication with 64-bit integers, where each matrix in the batch is separated by a fixed stride, supporting transposition of input matrices and scaling factors. This function is optimized for processing multiple matrix operations efficiently in a single call.',

        # C API 信息
        'c_api': 'cublasZgemmStridedBatched_64',
        'params': ['transa', 'transb', 'm', 'n', 'k', 'alpha', 'A', 'lda', 'strideA', 'B', 'ldb', 'strideB', 'beta', 'C', 'ldc', 'strideC', 'batchCount'],
        'param_types': {
            'transa': 'cublasOperation_t',
            'transb': 'cublasOperation_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'k': 'int64_t',
            'alpha': 'const cuDoubleComplex*',
            'A': 'const cuDoubleComplex*',
            'lda': 'int64_t',
            'strideA': 'long long int',
            'B': 'const cuDoubleComplex*',
            'ldb': 'int64_t',
            'strideB': 'long long int',
            'beta': 'const cuDoubleComplex*',
            'C': 'cuDoubleComplex*',
            'ldc': 'int64_t',
            'strideC': 'long long int',
            'batchCount': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZgemmStridedBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, long long int strideA, const cuDoubleComplex* B, int64_t ldb, long long int strideB, const cuDoubleComplex* beta, cuDoubleComplex* , int64_t ldc, long long int strideC, int64_t batchCount);',
    },
    'cublasSgemmGroupedBatched': {
        'base_op': 'gemm',
        'dtype': 'float32',
        'level': 3,
        'variant': 'GroupedBatched',
        'description': 'The `cublasSgemmGroupedBatched` function performs batched matrix-matrix multiplication for grouped sets of float32 matrices, where each group can have different sizes and transpose operations, computing C = alpha * op(A) * op(B) + beta * C for each matrix in the batch. It processes multiple groups of matrices in a single call, with each group having its own parameters and batch size.',

        # C API 信息
        'c_api': 'cublasSgemmGroupedBatched',
        'params': ['transa_array', 'transb_array', 'm_array', 'n_array', 'k_array', 'alpha_array', 'Aarray', 'lda_array', 'Barray', 'ldb_array', 'beta_array', 'Carray', 'ldc_array', 'group_count', 'group_size'],
        'param_types': {
            'transa_array': 'const cublasOperation_t[]',
            'transb_array': 'const cublasOperation_t[]',
            'm_array': 'const int[]',
            'n_array': 'const int[]',
            'k_array': 'const int[]',
            'alpha_array': 'const float[]',
            'Aarray': 'const float* const[]',
            'lda_array': 'const int[]',
            'Barray': 'const float* const[]',
            'ldb_array': 'const int[]',
            'beta_array': 'const float[]',
            'Carray': 'float* const[]',
            'ldc_array': 'const int[]',
            'group_count': 'int',
            'group_size': 'const int[]'
        },

        # 参数分类
        'tensor_params': ['transa_array', 'transb_array', 'm_array', 'n_array', 'k_array', 'Aarray', 'lda_array', 'Barray', 'ldb_array', 'Carray', 'ldc_array', 'group_size'],
        'scalar_params': ['alpha_array', 'beta_array'],
        'inout_params': ['Carray'],
        'return_params': ['Carray'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSgemmGroupedBatched(cublasHandle_t handle, const cublasOperation_t transa_array[], const cublasOperation_t transb_array[], const int m_array[], const int n_array[], const int k_array[], const float alpha_array[], const float* const Aarray[], const int lda_array[], const float* const Barray[], const int ldb_array[], const float beta_array[], float* const Carray[], const int ldc_array[], int group_count, const int group_size[]);',
    },
    'cublasSgemmGroupedBatched_64': {
        'base_op': 'gemm',
        'dtype': 'float32',
        'level': 3,
        'variant': 'GroupedBatched_64',
        'description': 'The `cublasSgemmGroupedBatched_64` function performs grouped batched matrix-matrix multiplication for float32 matrices, where each group can have different sizes, transposition options, and scaling factors, with 64-bit integers for large matrix dimensions. It processes multiple matrix multiplications in batches, organized into groups with specified sizes, allowing efficient execution of diverse GEMM operations in a single call.',

        # C API 信息
        'c_api': 'cublasSgemmGroupedBatched_64',
        'params': ['transa_array', 'transb_array', 'm_array', 'n_array', 'k_array', 'alpha_array', 'Aarray', 'lda_array', 'Barray', 'ldb_array', 'beta_array', 'Carray', 'ldc_array', 'group_count', 'group_size'],
        'param_types': {
            'transa_array': 'const cublasOperation_t[]',
            'transb_array': 'const cublasOperation_t[]',
            'm_array': 'const int64_t[]',
            'n_array': 'const int64_t[]',
            'k_array': 'const int64_t[]',
            'alpha_array': 'const float[]',
            'Aarray': 'const float* const[]',
            'lda_array': 'const int64_t[]',
            'Barray': 'const float* const[]',
            'ldb_array': 'const int64_t[]',
            'beta_array': 'const float[]',
            'Carray': 'float* const[]',
            'ldc_array': 'const int64_t[]',
            'group_count': 'int64_t',
            'group_size': 'const int64_t[]'
        },

        # 参数分类
        'tensor_params': ['transa_array', 'transb_array', 'm_array', 'n_array', 'k_array', 'Aarray', 'lda_array', 'Barray', 'ldb_array', 'Carray', 'ldc_array', 'group_size'],
        'scalar_params': ['alpha_array', 'beta_array'],
        'inout_params': ['Carray'],
        'return_params': ['Carray'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSgemmGroupedBatched_64(cublasHandle_t handle, const cublasOperation_t transa_array[], const cublasOperation_t transb_array[], const int64_t m_array[], const int64_t n_array[], const int64_t k_array[], const float alpha_array[], const float* const Aarray[], const int64_t lda_array[], const float* const Barray[], const int64_t ldb_array[], const float beta_array[], float* const Carray[], const int64_t ldc_array[], int64_t group_count, const int64_t group_size[]);',
    },
    'cublasDgemmGroupedBatched': {
        'base_op': 'gemm',
        'dtype': 'float64',
        'level': 3,
        'variant': 'GroupedBatched',
        'description': 'The `cublasDgemmGroupedBatched` function performs batched double-precision matrix-matrix multiplication (GEMM) operations on grouped sets of matrices, where each group can have different sizes, transpositions, and scaling factors. It efficiently processes multiple GEMM operations in parallel by organizing them into groups with similar parameters.',

        # C API 信息
        'c_api': 'cublasDgemmGroupedBatched',
        'params': ['transa_array', 'transb_array', 'm_array', 'n_array', 'k_array', 'alpha_array', 'Aarray', 'lda_array', 'Barray', 'ldb_array', 'beta_array', 'Carray', 'ldc_array', 'group_count', 'group_size'],
        'param_types': {
            'transa_array': 'const cublasOperation_t[]',
            'transb_array': 'const cublasOperation_t[]',
            'm_array': 'const int[]',
            'n_array': 'const int[]',
            'k_array': 'const int[]',
            'alpha_array': 'const double[]',
            'Aarray': 'const double* const[]',
            'lda_array': 'const int[]',
            'Barray': 'const double* const[]',
            'ldb_array': 'const int[]',
            'beta_array': 'const double[]',
            'Carray': 'double* const[]',
            'ldc_array': 'const int[]',
            'group_count': 'int',
            'group_size': 'const int[]'
        },

        # 参数分类
        'tensor_params': ['transa_array', 'transb_array', 'm_array', 'n_array', 'k_array', 'Aarray', 'lda_array', 'Barray', 'ldb_array', 'Carray', 'ldc_array', 'group_size'],
        'scalar_params': ['alpha_array', 'beta_array'],
        'inout_params': ['Carray'],
        'return_params': ['Carray'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDgemmGroupedBatched(cublasHandle_t handle, const cublasOperation_t transa_array[], const cublasOperation_t transb_array[], const int m_array[], const int n_array[], const int k_array[], const double alpha_array[], const double* const Aarray[], const int lda_array[], const double* const Barray[], const int ldb_array[], const double beta_array[], double* const Carray[], const int ldc_array[], int group_count, const int group_size[]);',
    },
    'cublasDgemmGroupedBatched_64': {
        'base_op': 'gemm',
        'dtype': 'float64',
        'level': 3,
        'variant': 'GroupedBatched_64',
        'description': 'The `cublasDgemmGroupedBatched_64` function performs grouped batched double-precision matrix-matrix multiplication (GEMM) operations, allowing different parameters (transpose, dimensions, scaling factors, and leading dimensions) for each group of matrices while supporting 64-bit integers for large problem sizes. It processes multiple matrix multiplications in batches grouped by shared parameters, improving efficiency for heterogeneous operations.',

        # C API 信息
        'c_api': 'cublasDgemmGroupedBatched_64',
        'params': ['transa_array', 'transb_array', 'm_array', 'n_array', 'k_array', 'alpha_array', 'Aarray', 'lda_array', 'Barray', 'ldb_array', 'beta_array', 'Carray', 'ldc_array', 'group_count', 'group_size'],
        'param_types': {
            'transa_array': 'const cublasOperation_t[]',
            'transb_array': 'const cublasOperation_t[]',
            'm_array': 'const int64_t[]',
            'n_array': 'const int64_t[]',
            'k_array': 'const int64_t[]',
            'alpha_array': 'const double[]',
            'Aarray': 'const double* const[]',
            'lda_array': 'const int64_t[]',
            'Barray': 'const double* const[]',
            'ldb_array': 'const int64_t[]',
            'beta_array': 'const double[]',
            'Carray': 'double* const[]',
            'ldc_array': 'const int64_t[]',
            'group_count': 'int64_t',
            'group_size': 'const int64_t[]'
        },

        # 参数分类
        'tensor_params': ['transa_array', 'transb_array', 'm_array', 'n_array', 'k_array', 'Aarray', 'lda_array', 'Barray', 'ldb_array', 'Carray', 'ldc_array', 'group_size'],
        'scalar_params': ['alpha_array', 'beta_array'],
        'inout_params': ['Carray'],
        'return_params': ['Carray'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDgemmGroupedBatched_64(cublasHandle_t handle, const cublasOperation_t transa_array[], const cublasOperation_t transb_array[], const int64_t m_array[], const int64_t n_array[], const int64_t k_array[], const double alpha_array[], const double* const Aarray[], const int64_t lda_array[], const double* const Barray[], const int64_t ldb_array[], const double beta_array[], double* const Carray[], const int64_t ldc_array[], int64_t group_count, const int64_t group_size[]);',
    },
    'cublasSgemmBatched': {
        'base_op': 'gemm',
        'dtype': 'float32',
        'level': 3,
        'variant': 'Batched',
        'description': 'The `cublasSgemmBatched` function performs batched matrix-matrix multiplication for single-precision floating-point matrices, computing C = alpha * op(A) * op(B) + beta * C for each matrix in the input arrays, where op(X) is either X or its transpose. It processes multiple independent matrix multiplications in a single call, specified by the batchCount parameter.',

        # C API 信息
        'c_api': 'cublasSgemmBatched',
        'params': ['transa', 'transb', 'm', 'n', 'k', 'alpha', 'Aarray', 'lda', 'Barray', 'ldb', 'beta', 'Carray', 'ldc', 'batchCount'],
        'param_types': {
            'transa': 'cublasOperation_t',
            'transb': 'cublasOperation_t',
            'm': 'int',
            'n': 'int',
            'k': 'int',
            'alpha': 'const float*',
            'Aarray': 'const float* const[]',
            'lda': 'int',
            'Barray': 'const float* const[]',
            'ldb': 'int',
            'beta': 'const float*',
            'Carray': 'float* const[]',
            'ldc': 'int',
            'batchCount': 'int'
        },

        # 参数分类
        'tensor_params': ['Aarray', 'Barray', 'Carray'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['Carray'],
        'return_params': ['Carray'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha, const float* const Aarray[], int lda, const float* const Barray[], int ldb, const float* beta, float* const Carray[], int ldc, int batchCount);',
    },
    'cublasSgemmBatched_64': {
        'base_op': 'gemm',
        'dtype': 'float32',
        'level': 3,
        'variant': 'Batched_64',
        'description': 'The `cublasSgemmBatched_64` function performs batched matrix-matrix multiplication for 32-bit floating-point matrices, computing `C = alpha * op(A) * op(B) + beta * C` for each matrix in the batch, where `op` denotes optional transposition, with support for 64-bit integer dimensions. It processes multiple matrix operations in a single call, improving efficiency for large batches.',

        # C API 信息
        'c_api': 'cublasSgemmBatched_64',
        'params': ['transa', 'transb', 'm', 'n', 'k', 'alpha', 'Aarray', 'lda', 'Barray', 'ldb', 'beta', 'Carray', 'ldc', 'batchCount'],
        'param_types': {
            'transa': 'cublasOperation_t',
            'transb': 'cublasOperation_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'k': 'int64_t',
            'alpha': 'const float*',
            'Aarray': 'const float* const[]',
            'lda': 'int64_t',
            'Barray': 'const float* const[]',
            'ldb': 'int64_t',
            'beta': 'const float*',
            'Carray': 'float* const[]',
            'ldc': 'int64_t',
            'batchCount': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['Aarray', 'Barray', 'Carray'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['Carray'],
        'return_params': ['Carray'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSgemmBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const float* alpha, const float* const Aarray[], int64_t lda, const float* const Barray[], int64_t ldb, const float* beta, float* const Carray[], int64_t ldc, int64_t batchCount);',
    },
    'cublasSgemmEx': {
        'base_op': 'gemm',
        'dtype': 'float32',
        'level': 3,
        'variant': 'Ex',
        'description': 'The `cublasSgemmEx` function performs a mixed-precision matrix-matrix multiplication (GEMM) operation using float32 computation, where input matrices A and B can have different data types (specified by Atype and Btype) and the result is stored in matrix C with its own data type (Ctype). It supports optional transposition of input matrices and scales the result by alpha and beta factors.',

        # C API 信息
        'c_api': 'cublasSgemmEx',
        'params': ['transa', 'transb', 'm', 'n', 'k', 'alpha', 'A', 'Atype', 'lda', 'B', 'Btype', 'ldb', 'beta', 'C', 'Ctype', 'ldc'],
        'param_types': {
            'transa': 'cublasOperation_t',
            'transb': 'cublasOperation_t',
            'm': 'int',
            'n': 'int',
            'k': 'int',
            'alpha': 'const float*',
            'A': 'const void*',
            'Atype': 'cudaDataType',
            'lda': 'int',
            'B': 'const void*',
            'Btype': 'cudaDataType',
            'ldb': 'int',
            'beta': 'const float*',
            'C': 'void*',
            'Ctype': 'cudaDataType',
            'ldc': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSgemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha, const void* A, cudaDataType Atype, int lda, const void* B, cudaDataType Btype, int ldb, const float* beta, void* , cudaDataType Ctype, int ldc);',
    },
    'cublasSgemmEx_64': {
        'base_op': 'gemm',
        'dtype': 'float32',
        'level': 3,
        'variant': 'Ex_64',
        'description': 'The `cublasSgemmEx_64` function performs 64-bit integer-based matrix-matrix multiplication using single-precision floating-point values, supporting mixed input/output data types (A, B, and C) as specified by their respective type parameters. It computes C = alpha * op(A) * op(B) + beta * C, where op denotes optional matrix transposition, and handles large matrices with 64-bit dimensions and leading dimensions.',

        # C API 信息
        'c_api': 'cublasSgemmEx_64',
        'params': ['transa', 'transb', 'm', 'n', 'k', 'alpha', 'A', 'Atype', 'lda', 'B', 'Btype', 'ldb', 'beta', 'C', 'Ctype', 'ldc'],
        'param_types': {
            'transa': 'cublasOperation_t',
            'transb': 'cublasOperation_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'k': 'int64_t',
            'alpha': 'const float*',
            'A': 'const void*',
            'Atype': 'cudaDataType',
            'lda': 'int64_t',
            'B': 'const void*',
            'Btype': 'cudaDataType',
            'ldb': 'int64_t',
            'beta': 'const float*',
            'C': 'void*',
            'Ctype': 'cudaDataType',
            'ldc': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSgemmEx_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const float* alpha, const void* A, cudaDataType Atype, int64_t lda, const void* B, cudaDataType Btype, int64_t ldb, const float* beta, void* , cudaDataType Ctype, int64_t ldc);',
    },
    'cublasSgemm_v2': {
        'base_op': 'gemm',
        'dtype': 'float32',
        'level': 3,
        'variant': 'base',
        'description': 'The `cublasSgemm_v2` function performs single-precision matrix-matrix multiplication (C = α * op(A) * op(B) + β * C), where op(X) can be the matrix X or its transpose, with options to specify matrix dimensions, leading dimensions, and scaling factors. It is a Level 3 BLAS operation optimized for GPU execution via the cuBLAS library.',

        # C API 信息
        'c_api': 'cublasSgemm_v2',
        'params': ['transa', 'transb', 'm', 'n', 'k', 'alpha', 'A', 'lda', 'B', 'ldb', 'beta', 'C', 'ldc'],
        'param_types': {
            'transa': 'cublasOperation_t',
            'transb': 'cublasOperation_t',
            'm': 'int',
            'n': 'int',
            'k': 'int',
            'alpha': 'const float*',
            'A': 'const float*',
            'lda': 'int',
            'B': 'const float*',
            'ldb': 'int',
            'beta': 'const float*',
            'C': 'float*',
            'ldc': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* , int ldc);',
    },
    'cublasSgemm_v2_64': {
        'base_op': 'gemm',
        'dtype': 'float32',
        'level': 3,
        'variant': '_64',
        'description': 'The `cublasSgemm_v2_64` function performs 64-bit integer precision matrix-matrix multiplication for single-precision floating-point matrices, computing C = alpha * op(A) * op(B) + beta * C, where op(X) can be a transpose or no-transpose operation. It is a BLAS Level 3 operation supporting large matrices through 64-bit parameters for dimensions and leading dimensions.',

        # C API 信息
        'c_api': 'cublasSgemm_v2_64',
        'params': ['transa', 'transb', 'm', 'n', 'k', 'alpha', 'A', 'lda', 'B', 'ldb', 'beta', 'C', 'ldc'],
        'param_types': {
            'transa': 'cublasOperation_t',
            'transb': 'cublasOperation_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'k': 'int64_t',
            'alpha': 'const float*',
            'A': 'const float*',
            'lda': 'int64_t',
            'B': 'const float*',
            'ldb': 'int64_t',
            'beta': 'const float*',
            'C': 'float*',
            'ldc': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSgemm_v2_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const float* alpha, const float* A, int64_t lda, const float* B, int64_t ldb, const float* beta, float* , int64_t ldc);',
    },
    'cublasCgemmBatched': {
        'base_op': 'gemm',
        'dtype': 'complex64',
        'level': 3,
        'variant': 'Batched',
        'description': 'The `cublasCgemmBatched` function performs batched matrix-matrix multiplication for complex single-precision matrices, computing `C = alpha * op(A) * op(B) + beta * C` for each matrix pair in the input arrays, where `op` denotes optional transposition. It processes multiple matrix operations in a single call, specified by the `batchCount` parameter, improving efficiency for small matrices.',

        # C API 信息
        'c_api': 'cublasCgemmBatched',
        'params': ['transa', 'transb', 'm', 'n', 'k', 'alpha', 'Aarray', 'lda', 'Barray', 'ldb', 'beta', 'Carray', 'ldc', 'batchCount'],
        'param_types': {
            'transa': 'cublasOperation_t',
            'transb': 'cublasOperation_t',
            'm': 'int',
            'n': 'int',
            'k': 'int',
            'alpha': 'const cuComplex*',
            'Aarray': 'const cuComplex* const[]',
            'lda': 'int',
            'Barray': 'const cuComplex* const[]',
            'ldb': 'int',
            'beta': 'const cuComplex*',
            'Carray': 'cuComplex* const[]',
            'ldc': 'int',
            'batchCount': 'int'
        },

        # 参数分类
        'tensor_params': ['Aarray', 'Barray', 'Carray'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['Carray'],
        'return_params': ['Carray'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* const Aarray[], int lda, const cuComplex* const Barray[], int ldb, const cuComplex* beta, cuComplex* const Carray[], int ldc, int batchCount);',
    },
    'cublasCgemmBatched_64': {
        'base_op': 'gemm',
        'dtype': 'complex64',
        'level': 3,
        'variant': 'Batched_64',
        'description': 'The `cublasCgemmBatched_64` function performs batched matrix-matrix multiplication for complex 64-bit floating-point matrices, computing `C = alpha * op(A) * op(B) + beta * C` for each matrix in the batch, where `op` denotes optional transposition. It supports 64-bit integer dimensions and batch processing for improved performance on large sets of small matrices.',

        # C API 信息
        'c_api': 'cublasCgemmBatched_64',
        'params': ['transa', 'transb', 'm', 'n', 'k', 'alpha', 'Aarray', 'lda', 'Barray', 'ldb', 'beta', 'Carray', 'ldc', 'batchCount'],
        'param_types': {
            'transa': 'cublasOperation_t',
            'transb': 'cublasOperation_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'k': 'int64_t',
            'alpha': 'const cuComplex*',
            'Aarray': 'const cuComplex* const[]',
            'lda': 'int64_t',
            'Barray': 'const cuComplex* const[]',
            'ldb': 'int64_t',
            'beta': 'const cuComplex*',
            'Carray': 'cuComplex* const[]',
            'ldc': 'int64_t',
            'batchCount': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['Aarray', 'Barray', 'Carray'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['Carray'],
        'return_params': ['Carray'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCgemmBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* const Aarray[], int64_t lda, const cuComplex* const Barray[], int64_t ldb, const cuComplex* beta, cuComplex* const Carray[], int64_t ldc, int64_t batchCount);',
    },
    'cublasDgemmBatched': {
        'base_op': 'gemm',
        'dtype': 'float64',
        'level': 3,
        'variant': 'Batched',
        'description': 'The `cublasDgemmBatched` function performs batched matrix-matrix multiplication for double-precision matrices, computing `C = alpha * op(A) * op(B) + beta * C` for each matrix in the input arrays, where `op` denotes optional transposition. It processes multiple matrix multiplications in a single call, specified by the `batchCount` parameter, with separate pointers for each matrix in the batch.',

        # C API 信息
        'c_api': 'cublasDgemmBatched',
        'params': ['transa', 'transb', 'm', 'n', 'k', 'alpha', 'Aarray', 'lda', 'Barray', 'ldb', 'beta', 'Carray', 'ldc', 'batchCount'],
        'param_types': {
            'transa': 'cublasOperation_t',
            'transb': 'cublasOperation_t',
            'm': 'int',
            'n': 'int',
            'k': 'int',
            'alpha': 'const double*',
            'Aarray': 'const double* const[]',
            'lda': 'int',
            'Barray': 'const double* const[]',
            'ldb': 'int',
            'beta': 'const double*',
            'Carray': 'double* const[]',
            'ldc': 'int',
            'batchCount': 'int'
        },

        # 参数分类
        'tensor_params': ['Aarray', 'Barray', 'Carray'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['Carray'],
        'return_params': ['Carray'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double* alpha, const double* const Aarray[], int lda, const double* const Barray[], int ldb, const double* beta, double* const Carray[], int ldc, int batchCount);',
    },
    'cublasDgemmBatched_64': {
        'base_op': 'gemm',
        'dtype': 'float64',
        'level': 3,
        'variant': 'Batched_64',
        'description': 'The `cublasDgemmBatched_64` function performs batched double-precision matrix-matrix multiplication (GEMM) operations, computing C = α·op(A)·op(B) + β·C for each matrix in the input arrays, where op(X) can be a transpose or no-op, and supports 64-bit integer parameters for large-scale computations. It processes multiple independent matrix multiplications in a single call, improving performance for batch operations.',

        # C API 信息
        'c_api': 'cublasDgemmBatched_64',
        'params': ['transa', 'transb', 'm', 'n', 'k', 'alpha', 'Aarray', 'lda', 'Barray', 'ldb', 'beta', 'Carray', 'ldc', 'batchCount'],
        'param_types': {
            'transa': 'cublasOperation_t',
            'transb': 'cublasOperation_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'k': 'int64_t',
            'alpha': 'const double*',
            'Aarray': 'const double* const[]',
            'lda': 'int64_t',
            'Barray': 'const double* const[]',
            'ldb': 'int64_t',
            'beta': 'const double*',
            'Carray': 'double* const[]',
            'ldc': 'int64_t',
            'batchCount': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['Aarray', 'Barray', 'Carray'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['Carray'],
        'return_params': ['Carray'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDgemmBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const double* alpha, const double* const Aarray[], int64_t lda, const double* const Barray[], int64_t ldb, const double* beta, double* const Carray[], int64_t ldc, int64_t batchCount);',
    },
    'cublasHgemmBatched': {
        'base_op': 'gemm',
        'dtype': 'float16',
        'level': 3,
        'variant': 'Batched',
        'description': 'The `cublasHgemmBatched` function performs batched matrix-matrix multiplication for float16 matrices, computing C = α * op(A) * op(B) + β * C for each matrix in the batch, where op denotes optional transposition. It processes multiple matrix multiplications in a single call, with each batch entry using separate pointers for A, B, and C matrices.',

        # C API 信息
        'c_api': 'cublasHgemmBatched',
        'params': ['transa', 'transb', 'm', 'n', 'k', 'alpha', 'Aarray', 'lda', 'Barray', 'ldb', 'beta', 'Carray', 'ldc', 'batchCount'],
        'param_types': {
            'transa': 'cublasOperation_t',
            'transb': 'cublasOperation_t',
            'm': 'int',
            'n': 'int',
            'k': 'int',
            'alpha': 'const __half*',
            'Aarray': 'const __half* const[]',
            'lda': 'int',
            'Barray': 'const __half* const[]',
            'ldb': 'int',
            'beta': 'const __half*',
            'Carray': '__half* const[]',
            'ldc': 'int',
            'batchCount': 'int'
        },

        # 参数分类
        'tensor_params': ['Aarray', 'Barray', 'Carray'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['Carray'],
        'return_params': ['Carray'],

        # 完整签名
        'signature': 'cublasStatus_t cublasHgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const __half* alpha, const __half* const Aarray[], int lda, const __half* const Barray[], int ldb, const __half* beta, __half* const Carray[], int ldc, int batchCount);',
    },
    'cublasHgemmBatched_64': {
        'base_op': 'gemm',
        'dtype': 'float16',
        'level': 3,
        'variant': 'Batched_64',
        'description': 'The `cublasHgemmBatched_64` function performs batched matrix-matrix multiplication for 64-bit integer dimensions using half-precision (float16) matrices, computing C = α·op(A)·op(B) + β·C for each batch, where op(X) can be a transpose or no-transpose operation. It processes multiple matrix multiplications in a single call, with each batch using separate pointers for A, B, and C matrices.',

        # C API 信息
        'c_api': 'cublasHgemmBatched_64',
        'params': ['transa', 'transb', 'm', 'n', 'k', 'alpha', 'Aarray', 'lda', 'Barray', 'ldb', 'beta', 'Carray', 'ldc', 'batchCount'],
        'param_types': {
            'transa': 'cublasOperation_t',
            'transb': 'cublasOperation_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'k': 'int64_t',
            'alpha': 'const __half*',
            'Aarray': 'const __half* const[]',
            'lda': 'int64_t',
            'Barray': 'const __half* const[]',
            'ldb': 'int64_t',
            'beta': 'const __half*',
            'Carray': '__half* const[]',
            'ldc': 'int64_t',
            'batchCount': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['Aarray', 'Barray', 'Carray'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['Carray'],
        'return_params': ['Carray'],

        # 完整签名
        'signature': 'cublasStatus_t cublasHgemmBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const __half* alpha, const __half* const Aarray[], int64_t lda, const __half* const Barray[], int64_t ldb, const __half* beta, __half* const Carray[], int64_t ldc, int64_t batchCount);',
    },
    'cublasZgemmBatched': {
        'base_op': 'gemm',
        'dtype': 'complex128',
        'level': 3,
        'variant': 'Batched',
        'description': 'Performs batched matrix-matrix multiplication for complex double-precision matrices, computing C = alpha * op(A) * op(B) + beta * C for each matrix in the batch, where op(X) can be a transpose or conjugate transpose operation. The function processes multiple matrix multiplications in a single call with arrays of pointers to input and output matrices.',

        # C API 信息
        'c_api': 'cublasZgemmBatched',
        'params': ['transa', 'transb', 'm', 'n', 'k', 'alpha', 'Aarray', 'lda', 'Barray', 'ldb', 'beta', 'Carray', 'ldc', 'batchCount'],
        'param_types': {
            'transa': 'cublasOperation_t',
            'transb': 'cublasOperation_t',
            'm': 'int',
            'n': 'int',
            'k': 'int',
            'alpha': 'const cuDoubleComplex*',
            'Aarray': 'const cuDoubleComplex* const[]',
            'lda': 'int',
            'Barray': 'const cuDoubleComplex* const[]',
            'ldb': 'int',
            'beta': 'const cuDoubleComplex*',
            'Carray': 'cuDoubleComplex* const[]',
            'ldc': 'int',
            'batchCount': 'int'
        },

        # 参数分类
        'tensor_params': ['Aarray', 'Barray', 'Carray'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['Carray'],
        'return_params': ['Carray'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* const Aarray[], int lda, const cuDoubleComplex* const Barray[], int ldb, const cuDoubleComplex* beta, cuDoubleComplex* const Carray[], int ldc, int batchCount);',
    },
    'cublasZgemmBatched_64': {
        'base_op': 'gemm',
        'dtype': 'complex128',
        'level': 3,
        'variant': 'Batched_64',
        'description': 'The `cublasZgemmBatched_64` function performs batched matrix-matrix multiplication for complex double-precision matrices, processing multiple independent GEMM (General Matrix Multiply) operations with 64-bit integers for large problem sizes. It computes C = α * op(A) * op(B) + β * C for each matrix in the input arrays, where op(X) can be a transpose or conjugate transpose operation.',

        # C API 信息
        'c_api': 'cublasZgemmBatched_64',
        'params': ['transa', 'transb', 'm', 'n', 'k', 'alpha', 'Aarray', 'lda', 'Barray', 'ldb', 'beta', 'Carray', 'ldc', 'batchCount'],
        'param_types': {
            'transa': 'cublasOperation_t',
            'transb': 'cublasOperation_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'k': 'int64_t',
            'alpha': 'const cuDoubleComplex*',
            'Aarray': 'const cuDoubleComplex* const[]',
            'lda': 'int64_t',
            'Barray': 'const cuDoubleComplex* const[]',
            'ldb': 'int64_t',
            'beta': 'const cuDoubleComplex*',
            'Carray': 'cuDoubleComplex* const[]',
            'ldc': 'int64_t',
            'batchCount': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['Aarray', 'Barray', 'Carray'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['Carray'],
        'return_params': ['Carray'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZgemmBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* const Aarray[], int64_t lda, const cuDoubleComplex* const Barray[], int64_t ldb, const cuDoubleComplex* beta, cuDoubleComplex* const Carray[], int64_t ldc, int64_t batchCount);',
    },
    'cublasCgemm3m': {
        'base_op': 'gemm',
        'dtype': 'complex64',
        'level': 3,
        'variant': '3m',
        'description': 'The `cublasCgemm3m` function performs complex matrix-matrix multiplication using the 3m algorithm, computing C = alpha * op(A) * op(B) + beta * C for complex 32-bit floating-point matrices, where op(X) can be a transpose or conjugate transpose operation. It is optimized for complex arithmetic efficiency while maintaining BLAS Level 3 performance.',

        # C API 信息
        'c_api': 'cublasCgemm3m',
        'params': ['transa', 'transb', 'm', 'n', 'k', 'alpha', 'A', 'lda', 'B', 'ldb', 'beta', 'C', 'ldc'],
        'param_types': {
            'transa': 'cublasOperation_t',
            'transb': 'cublasOperation_t',
            'm': 'int',
            'n': 'int',
            'k': 'int',
            'alpha': 'const cuComplex*',
            'A': 'const cuComplex*',
            'lda': 'int',
            'B': 'const cuComplex*',
            'ldb': 'int',
            'beta': 'const cuComplex*',
            'C': 'cuComplex*',
            'ldc': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCgemm3m(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* , int ldc);',
    },
    'cublasCgemm3m_64': {
        'base_op': 'gemm',
        'dtype': 'complex64',
        'level': 3,
        'variant': '3m_64',
        'description': 'The `cublasCgemm3m_64` function performs 64-bit complex matrix-matrix operations (GEMM) using the 3M method, computing C = alpha * op(A) * op(B) + beta * C, where op(X) is optionally transposed, and A, B, and C are complex matrices with 64-bit integer dimensions. It is optimized for complex single-precision arithmetic on CUDA-enabled GPUs.',

        # C API 信息
        'c_api': 'cublasCgemm3m_64',
        'params': ['transa', 'transb', 'm', 'n', 'k', 'alpha', 'A', 'lda', 'B', 'ldb', 'beta', 'C', 'ldc'],
        'param_types': {
            'transa': 'cublasOperation_t',
            'transb': 'cublasOperation_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'k': 'int64_t',
            'alpha': 'const cuComplex*',
            'A': 'const cuComplex*',
            'lda': 'int64_t',
            'B': 'const cuComplex*',
            'ldb': 'int64_t',
            'beta': 'const cuComplex*',
            'C': 'cuComplex*',
            'ldc': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCgemm3m_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, const cuComplex* beta, cuComplex* , int64_t ldc);',
    },
    'cublasCgemmEx': {
        'base_op': 'gemm',
        'dtype': 'complex64',
        'level': 3,
        'variant': 'Ex',
        'description': 'The `cublasCgemmEx` function performs a complex matrix-matrix multiplication (GEMM) operation with extended support for mixed-precision inputs, computing C = alpha * op(A) * op(B) + beta * C, where op(X) can be a transpose or conjugate transpose, and A, B, and C can have different data types. It is a Level 3 BLAS operation optimized for CUDA-enabled GPUs.',

        # C API 信息
        'c_api': 'cublasCgemmEx',
        'params': ['transa', 'transb', 'm', 'n', 'k', 'alpha', 'A', 'Atype', 'lda', 'B', 'Btype', 'ldb', 'beta', 'C', 'Ctype', 'ldc'],
        'param_types': {
            'transa': 'cublasOperation_t',
            'transb': 'cublasOperation_t',
            'm': 'int',
            'n': 'int',
            'k': 'int',
            'alpha': 'const cuComplex*',
            'A': 'const void*',
            'Atype': 'cudaDataType',
            'lda': 'int',
            'B': 'const void*',
            'Btype': 'cudaDataType',
            'ldb': 'int',
            'beta': 'const cuComplex*',
            'C': 'void*',
            'Ctype': 'cudaDataType',
            'ldc': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCgemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const void* A, cudaDataType Atype, int lda, const void* B, cudaDataType Btype, int ldb, const cuComplex* beta, void* , cudaDataType Ctype, int ldc);',
    },
    'cublasCgemmEx_64': {
        'base_op': 'gemm',
        'dtype': 'complex64',
        'level': 3,
        'variant': 'Ex_64',
        'description': 'The `cublasCgemmEx_64` function performs 64-bit index matrix multiplication for complex single-precision matrices, supporting mixed data types for inputs and output, and allows specifying transpose operations for matrices A and B. It is a BLAS Level 3 operation that computes C = alpha * op(A) * op(B) + beta * C, where op denotes optional transposition.',

        # C API 信息
        'c_api': 'cublasCgemmEx_64',
        'params': ['transa', 'transb', 'm', 'n', 'k', 'alpha', 'A', 'Atype', 'lda', 'B', 'Btype', 'ldb', 'beta', 'C', 'Ctype', 'ldc'],
        'param_types': {
            'transa': 'cublasOperation_t',
            'transb': 'cublasOperation_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'k': 'int64_t',
            'alpha': 'const cuComplex*',
            'A': 'const void*',
            'Atype': 'cudaDataType',
            'lda': 'int64_t',
            'B': 'const void*',
            'Btype': 'cudaDataType',
            'ldb': 'int64_t',
            'beta': 'const cuComplex*',
            'C': 'void*',
            'Ctype': 'cudaDataType',
            'ldc': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCgemmEx_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuComplex* alpha, const void* A, cudaDataType Atype, int64_t lda, const void* B, cudaDataType Btype, int64_t ldb, const cuComplex* beta, void* , cudaDataType Ctype, int64_t ldc);',
    },
    'cublasCgemm_v2': {
        'base_op': 'gemm',
        'dtype': 'complex64',
        'level': 3,
        'variant': 'base',
        'description': 'The `cublasCgemm_v2` function performs complex single-precision matrix-matrix multiplication (C = α * op(A) * op(B) + β * C), where op(X) can be a transpose or conjugate transpose operation, and A, B, C are complex matrices with specified leading dimensions. It is a Level 3 BLAS operation optimized for GPU execution via the cuBLAS library.',

        # C API 信息
        'c_api': 'cublasCgemm_v2',
        'params': ['transa', 'transb', 'm', 'n', 'k', 'alpha', 'A', 'lda', 'B', 'ldb', 'beta', 'C', 'ldc'],
        'param_types': {
            'transa': 'cublasOperation_t',
            'transb': 'cublasOperation_t',
            'm': 'int',
            'n': 'int',
            'k': 'int',
            'alpha': 'const cuComplex*',
            'A': 'const cuComplex*',
            'lda': 'int',
            'B': 'const cuComplex*',
            'ldb': 'int',
            'beta': 'const cuComplex*',
            'C': 'cuComplex*',
            'ldc': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* , int ldc);',
    },
    'cublasCgemm_v2_64': {
        'base_op': 'gemm',
        'dtype': 'complex64',
        'level': 3,
        'variant': '_64',
        'description': 'The `cublasCgemm_v2_64` function performs 64-bit matrix-matrix multiplication for complex single-precision matrices, computing C = alpha * op(A) * op(B) + beta * C, where op(X) can be a transpose or conjugate transpose operation. It supports large matrices with 64-bit integer dimensions and strides.',

        # C API 信息
        'c_api': 'cublasCgemm_v2_64',
        'params': ['transa', 'transb', 'm', 'n', 'k', 'alpha', 'A', 'lda', 'B', 'ldb', 'beta', 'C', 'ldc'],
        'param_types': {
            'transa': 'cublasOperation_t',
            'transb': 'cublasOperation_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'k': 'int64_t',
            'alpha': 'const cuComplex*',
            'A': 'const cuComplex*',
            'lda': 'int64_t',
            'B': 'const cuComplex*',
            'ldb': 'int64_t',
            'beta': 'const cuComplex*',
            'C': 'cuComplex*',
            'ldc': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCgemm_v2_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, const cuComplex* beta, cuComplex* , int64_t ldc);',
    },
    'cublasDgemm_v2': {
        'base_op': 'gemm',
        'dtype': 'float64',
        'level': 3,
        'variant': 'base',
        'description': 'The `cublasDgemm_v2` function performs double-precision matrix-matrix multiplication (C = α * op(A) * op(B) + β * C), where op(X) can be the matrix X or its transpose, with specified dimensions and leading dimensions for input and output matrices. It is a Level 3 BLAS operation optimized for GPU execution through the cuBLAS library.',

        # C API 信息
        'c_api': 'cublasDgemm_v2',
        'params': ['transa', 'transb', 'm', 'n', 'k', 'alpha', 'A', 'lda', 'B', 'ldb', 'beta', 'C', 'ldc'],
        'param_types': {
            'transa': 'cublasOperation_t',
            'transb': 'cublasOperation_t',
            'm': 'int',
            'n': 'int',
            'k': 'int',
            'alpha': 'const double*',
            'A': 'const double*',
            'lda': 'int',
            'B': 'const double*',
            'ldb': 'int',
            'beta': 'const double*',
            'C': 'double*',
            'ldc': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* , int ldc);',
    },
    'cublasDgemm_v2_64': {
        'base_op': 'gemm',
        'dtype': 'float64',
        'level': 3,
        'variant': '_64',
        'description': 'The `cublasDgemm_v2_64` function performs 64-bit integer precision double-precision matrix-matrix multiplication (GEMM) with optional matrix transposition, computing C = α·op(A)·op(B) + β·C, where op(X) is either X or X^T. It supports large matrices through 64-bit parameters for dimensions and leading dimensions.',

        # C API 信息
        'c_api': 'cublasDgemm_v2_64',
        'params': ['transa', 'transb', 'm', 'n', 'k', 'alpha', 'A', 'lda', 'B', 'ldb', 'beta', 'C', 'ldc'],
        'param_types': {
            'transa': 'cublasOperation_t',
            'transb': 'cublasOperation_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'k': 'int64_t',
            'alpha': 'const double*',
            'A': 'const double*',
            'lda': 'int64_t',
            'B': 'const double*',
            'ldb': 'int64_t',
            'beta': 'const double*',
            'C': 'double*',
            'ldc': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDgemm_v2_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const double* alpha, const double* A, int64_t lda, const double* B, int64_t ldb, const double* beta, double* , int64_t ldc);',
    },
    'cublasHgemm': {
        'base_op': 'gemm',
        'dtype': 'float16',
        'level': 3,
        'variant': 'base',
        'description': 'The `cublasHgemm` function performs matrix-matrix multiplication using half-precision (float16) matrices, computing C = alpha * op(A) * op(B) + beta * C, where op(X) can be a transpose or no-transpose operation. It is a Level 3 BLAS operation optimized for GPU execution via the cuBLAS library.',

        # C API 信息
        'c_api': 'cublasHgemm',
        'params': ['transa', 'transb', 'm', 'n', 'k', 'alpha', 'A', 'lda', 'B', 'ldb', 'beta', 'C', 'ldc'],
        'param_types': {
            'transa': 'cublasOperation_t',
            'transb': 'cublasOperation_t',
            'm': 'int',
            'n': 'int',
            'k': 'int',
            'alpha': 'const __half*',
            'A': 'const __half*',
            'lda': 'int',
            'B': 'const __half*',
            'ldb': 'int',
            'beta': 'const __half*',
            'C': '__half*',
            'ldc': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasHgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const __half* alpha, const __half* A, int lda, const __half* B, int ldb, const __half* beta, __half* , int ldc);',
    },
    'cublasHgemm_64': {
        'base_op': 'gemm',
        'dtype': 'float16',
        'level': 3,
        'variant': '_64',
        'description': 'The `cublasHgemm_64` function computes the matrix-matrix multiplication of two half-precision (float16) matrices with 64-bit integer dimensions, optionally applying transposes to the input matrices and scaling the result. It performs the operation C = alpha * op(A) * op(B) + beta * C, where op(X) can be the matrix X or its transpose.',

        # C API 信息
        'c_api': 'cublasHgemm_64',
        'params': ['transa', 'transb', 'm', 'n', 'k', 'alpha', 'A', 'lda', 'B', 'ldb', 'beta', 'C', 'ldc'],
        'param_types': {
            'transa': 'cublasOperation_t',
            'transb': 'cublasOperation_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'k': 'int64_t',
            'alpha': 'const __half*',
            'A': 'const __half*',
            'lda': 'int64_t',
            'B': 'const __half*',
            'ldb': 'int64_t',
            'beta': 'const __half*',
            'C': '__half*',
            'ldc': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasHgemm_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const __half* alpha, const __half* A, int64_t lda, const __half* B, int64_t ldb, const __half* beta, __half* , int64_t ldc);',
    },
    'cublasZgemm3m': {
        'base_op': 'gemm',
        'dtype': 'complex128',
        'level': 3,
        'variant': '3m',
        'description': 'Performs complex double-precision matrix-matrix multiplication using a 3M algorithm, computing C = alpha * op(A) * op(B) + beta * C, where op(X) is either X, X^T, or X^H, with optimized performance for certain matrix sizes.',

        # C API 信息
        'c_api': 'cublasZgemm3m',
        'params': ['transa', 'transb', 'm', 'n', 'k', 'alpha', 'A', 'lda', 'B', 'ldb', 'beta', 'C', 'ldc'],
        'param_types': {
            'transa': 'cublasOperation_t',
            'transb': 'cublasOperation_t',
            'm': 'int',
            'n': 'int',
            'k': 'int',
            'alpha': 'const cuDoubleComplex*',
            'A': 'const cuDoubleComplex*',
            'lda': 'int',
            'B': 'const cuDoubleComplex*',
            'ldb': 'int',
            'beta': 'const cuDoubleComplex*',
            'C': 'cuDoubleComplex*',
            'ldc': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZgemm3m(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* , int ldc);',
    },
    'cublasZgemm3m_64': {
        'base_op': 'gemm',
        'dtype': 'complex128',
        'level': 3,
        'variant': '3m_64',
        'description': 'The `cublasZgemm3m_64` function performs 64-bit complex double-precision matrix-matrix multiplication using the 3M algorithm, combining the input matrices A and B according to specified transpose operations and scaling factors alpha and beta. It is designed for large-scale computations with 64-bit integer dimensions and strides.',

        # C API 信息
        'c_api': 'cublasZgemm3m_64',
        'params': ['transa', 'transb', 'm', 'n', 'k', 'alpha', 'A', 'lda', 'B', 'ldb', 'beta', 'C', 'ldc'],
        'param_types': {
            'transa': 'cublasOperation_t',
            'transb': 'cublasOperation_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'k': 'int64_t',
            'alpha': 'const cuDoubleComplex*',
            'A': 'const cuDoubleComplex*',
            'lda': 'int64_t',
            'B': 'const cuDoubleComplex*',
            'ldb': 'int64_t',
            'beta': 'const cuDoubleComplex*',
            'C': 'cuDoubleComplex*',
            'ldc': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZgemm3m_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, const cuDoubleComplex* beta, cuDoubleComplex* , int64_t ldc);',
    },
    'cublasZgemm_v2': {
        'base_op': 'gemm',
        'dtype': 'complex128',
        'level': 3,
        'variant': 'base',
        'description': 'Performs complex double-precision matrix-matrix multiplication (C = α * op(A) * op(B) + β * C), where op(X) can be the matrix X, its transpose, or conjugate transpose, with matrices A, B, and C stored in column-major format.',

        # C API 信息
        'c_api': 'cublasZgemm_v2',
        'params': ['transa', 'transb', 'm', 'n', 'k', 'alpha', 'A', 'lda', 'B', 'ldb', 'beta', 'C', 'ldc'],
        'param_types': {
            'transa': 'cublasOperation_t',
            'transb': 'cublasOperation_t',
            'm': 'int',
            'n': 'int',
            'k': 'int',
            'alpha': 'const cuDoubleComplex*',
            'A': 'const cuDoubleComplex*',
            'lda': 'int',
            'B': 'const cuDoubleComplex*',
            'ldb': 'int',
            'beta': 'const cuDoubleComplex*',
            'C': 'cuDoubleComplex*',
            'ldc': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* , int ldc);',
    },
    'cublasZgemm_v2_64': {
        'base_op': 'gemm',
        'dtype': 'complex128',
        'level': 3,
        'variant': '_64',
        'description': 'The `cublasZgemm_v2_64` function performs 64-bit complex double-precision matrix-matrix multiplication, computing C = alpha * op(A) * op(B) + beta * C, where op(X) can be a transpose or conjugate transpose operation, and supports large matrices with 64-bit dimensions. It is part of the cuBLAS Level 3 BLAS operations for GPU-accelerated linear algebra.',

        # C API 信息
        'c_api': 'cublasZgemm_v2_64',
        'params': ['transa', 'transb', 'm', 'n', 'k', 'alpha', 'A', 'lda', 'B', 'ldb', 'beta', 'C', 'ldc'],
        'param_types': {
            'transa': 'cublasOperation_t',
            'transb': 'cublasOperation_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'k': 'int64_t',
            'alpha': 'const cuDoubleComplex*',
            'A': 'const cuDoubleComplex*',
            'lda': 'int64_t',
            'B': 'const cuDoubleComplex*',
            'ldb': 'int64_t',
            'beta': 'const cuDoubleComplex*',
            'C': 'cuDoubleComplex*',
            'ldc': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZgemm_v2_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, const cuDoubleComplex* beta, cuDoubleComplex* , int64_t ldc);',
    },
    'cublasCtrsmBatched': {
        'base_op': 'trsm',
        'dtype': 'complex64',
        'level': 3,
        'variant': 'Batched',
        'description': 'The `cublasCtrsmBatched` function performs batched triangular matrix solve operations with multiple complex64 matrices, computing `B = alpha * op(A)^{-1} * B` or `B = alpha * B * op(A)^{-1}` for each batch, where `A` is a triangular matrix and `B` is a general matrix. It supports configurable side, fill mode, transpose operation, and diagonal type for each matrix in the batch.',

        # C API 信息
        'c_api': 'cublasCtrsmBatched',
        'params': ['side', 'uplo', 'trans', 'diag', 'm', 'n', 'alpha', 'A', 'lda', 'B', 'ldb', 'batchCount'],
        'param_types': {
            'side': 'cublasSideMode_t',
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'm': 'int',
            'n': 'int',
            'alpha': 'const cuComplex*',
            'A': 'const cuComplex* const[]',
            'lda': 'int',
            'B': 'cuComplex* const[]',
            'ldb': 'int',
            'batchCount': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'B'],
        'scalar_params': ['alpha'],
        'inout_params': ['B'],
        'return_params': ['B'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex* alpha, const cuComplex* const A[], int lda, cuComplex* const B[], int ldb, int batchCount);',
    },
    'cublasCtrsmBatched_64': {
        'base_op': 'trsm',
        'dtype': 'complex64',
        'level': 3,
        'variant': 'Batched_64',
        'description': 'The `cublasCtrsmBatched_64` function performs batched triangular matrix solve operations with 64-bit indices, solving multiple complex64 systems of the form `op(A) * X = alpha * B` or `X * op(A) = alpha * B`, where `A` is a triangular matrix and `X` and `B` are rectangular matrices. It supports different side, uplo, transposition, and diagonal modes for each matrix in the batch.',

        # C API 信息
        'c_api': 'cublasCtrsmBatched_64',
        'params': ['side', 'uplo', 'trans', 'diag', 'm', 'n', 'alpha', 'A', 'lda', 'B', 'ldb', 'batchCount'],
        'param_types': {
            'side': 'cublasSideMode_t',
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'alpha': 'const cuComplex*',
            'A': 'const cuComplex* const[]',
            'lda': 'int64_t',
            'B': 'cuComplex* const[]',
            'ldb': 'int64_t',
            'batchCount': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'B'],
        'scalar_params': ['alpha'],
        'inout_params': ['B'],
        'return_params': ['B'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCtrsmBatched_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* const A[], int64_t lda, cuComplex* const B[], int64_t ldb, int64_t batchCount);',
    },
    'cublasDtrsmBatched': {
        'base_op': 'trsm',
        'dtype': 'float64',
        'level': 3,
        'variant': 'Batched',
        'description': 'The `cublasDtrsmBatched` function performs batched triangular matrix solve operations with double precision, computing `B = alpha * op(A)^{-1} * B` or `B = alpha * B * op(A)^{-1}` for each batch, where `A` is a triangular matrix and `B` is a general matrix. It processes multiple matrices in a single call, specified by `batchCount`, with options for matrix side, fill mode, transpose operation, and diagonal type.',

        # C API 信息
        'c_api': 'cublasDtrsmBatched',
        'params': ['side', 'uplo', 'trans', 'diag', 'm', 'n', 'alpha', 'A', 'lda', 'B', 'ldb', 'batchCount'],
        'param_types': {
            'side': 'cublasSideMode_t',
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'm': 'int',
            'n': 'int',
            'alpha': 'const double*',
            'A': 'const double* const[]',
            'lda': 'int',
            'B': 'double* const[]',
            'ldb': 'int',
            'batchCount': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'B'],
        'scalar_params': ['alpha'],
        'inout_params': ['B'],
        'return_params': ['B'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double* alpha, const double* const A[], int lda, double* const B[], int ldb, int batchCount);',
    },
    'cublasDtrsmBatched_64': {
        'base_op': 'trsm',
        'dtype': 'float64',
        'level': 3,
        'variant': 'Batched_64',
        'description': 'The `cublasDtrsmBatched_64` function performs batched triangular matrix-matrix operations of the form `B = alpha * op(A)^{-1} * B` or `B = alpha * B * op(A)^{-1}`, where `A` is a triangular matrix and `B` is a general matrix, using double-precision floating-point arithmetic for 64-bit problem sizes. It processes multiple matrices in a single call, with each batch handling separate `A` and `B` arrays.',

        # C API 信息
        'c_api': 'cublasDtrsmBatched_64',
        'params': ['side', 'uplo', 'trans', 'diag', 'm', 'n', 'alpha', 'A', 'lda', 'B', 'ldb', 'batchCount'],
        'param_types': {
            'side': 'cublasSideMode_t',
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'alpha': 'const double*',
            'A': 'const double* const[]',
            'lda': 'int64_t',
            'B': 'double* const[]',
            'ldb': 'int64_t',
            'batchCount': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'B'],
        'scalar_params': ['alpha'],
        'inout_params': ['B'],
        'return_params': ['B'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDtrsmBatched_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const double* alpha, const double* const A[], int64_t lda, double* const B[], int64_t ldb, int64_t batchCount);',
    },
    'cublasStrsmBatched': {
        'base_op': 'trsm',
        'dtype': 'float32',
        'level': 3,
        'variant': 'Batched',
        'description': 'The `cublasStrsmBatched` function performs batched triangular matrix solve operations with multiple right-hand sides, computing `B = alpha * op(A)^{-1} * B` or `B = alpha * B * op(A)^{-1}` for each batch of float32 matrices A and B, where A is triangular and op(A) may be transposed. It supports different triangular storage modes (upper/lower), transpose operations, and diagonal types (unit/non-unit) across the batch.',

        # C API 信息
        'c_api': 'cublasStrsmBatched',
        'params': ['side', 'uplo', 'trans', 'diag', 'm', 'n', 'alpha', 'A', 'lda', 'B', 'ldb', 'batchCount'],
        'param_types': {
            'side': 'cublasSideMode_t',
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'm': 'int',
            'n': 'int',
            'alpha': 'const float*',
            'A': 'const float* const[]',
            'lda': 'int',
            'B': 'float* const[]',
            'ldb': 'int',
            'batchCount': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'B'],
        'scalar_params': ['alpha'],
        'inout_params': ['B'],
        'return_params': ['B'],

        # 完整签名
        'signature': 'cublasStatus_t cublasStrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float* alpha, const float* const A[], int lda, float* const B[], int ldb, int batchCount);',
    },
    'cublasStrsmBatched_64': {
        'base_op': 'trsm',
        'dtype': 'float32',
        'level': 3,
        'variant': 'Batched_64',
        'description': 'This function performs a batched triangular matrix solve (float32 precision) for multiple systems of equations, where each system is of the form `alpha * op(A) * X = B` or `alpha * X * op(A) = B`, with A being a triangular matrix and X/B being rectangular matrices. It supports 64-bit integers for large matrix dimensions and batch processing.',

        # C API 信息
        'c_api': 'cublasStrsmBatched_64',
        'params': ['side', 'uplo', 'trans', 'diag', 'm', 'n', 'alpha', 'A', 'lda', 'B', 'ldb', 'batchCount'],
        'param_types': {
            'side': 'cublasSideMode_t',
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'alpha': 'const float*',
            'A': 'const float* const[]',
            'lda': 'int64_t',
            'B': 'float* const[]',
            'ldb': 'int64_t',
            'batchCount': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'B'],
        'scalar_params': ['alpha'],
        'inout_params': ['B'],
        'return_params': ['B'],

        # 完整签名
        'signature': 'cublasStatus_t cublasStrsmBatched_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const float* alpha, const float* const A[], int64_t lda, float* const B[], int64_t ldb, int64_t batchCount);',
    },
    'cublasZtrsmBatched': {
        'base_op': 'trsm',
        'dtype': 'complex128',
        'level': 3,
        'variant': 'Batched',
        'description': 'The `cublasZtrsmBatched` function performs batched triangular matrix solve operations with multiple right-hand sides, solving the equation `op(A) * X = alpha * B` or `X * op(A) = alpha * B` for complex double-precision matrices, where `A` is a triangular matrix and `X` and `B` are rectangular matrices, across multiple independent problems in a single call.',

        # C API 信息
        'c_api': 'cublasZtrsmBatched',
        'params': ['side', 'uplo', 'trans', 'diag', 'm', 'n', 'alpha', 'A', 'lda', 'B', 'ldb', 'batchCount'],
        'param_types': {
            'side': 'cublasSideMode_t',
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'm': 'int',
            'n': 'int',
            'alpha': 'const cuDoubleComplex*',
            'A': 'const cuDoubleComplex* const[]',
            'lda': 'int',
            'B': 'cuDoubleComplex* const[]',
            'ldb': 'int',
            'batchCount': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'B'],
        'scalar_params': ['alpha'],
        'inout_params': ['B'],
        'return_params': ['B'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* const A[], int lda, cuDoubleComplex* const B[], int ldb, int batchCount);',
    },
    'cublasZtrsmBatched_64': {
        'base_op': 'trsm',
        'dtype': 'complex128',
        'level': 3,
        'variant': 'Batched_64',
        'description': 'The `cublasZtrsmBatched_64` function performs batched triangular matrix solve operations with 64-bit integers, solving multiple complex double-precision systems of the form `op(A) * X = alpha * B` or `X * op(A) = alpha * B`, where `A` is a triangular matrix and `X` and `B` are rectangular matrices. It processes multiple matrices in a single call, specified by the batch count, with each matrix pair (A, B) stored in separate arrays.',

        # C API 信息
        'c_api': 'cublasZtrsmBatched_64',
        'params': ['side', 'uplo', 'trans', 'diag', 'm', 'n', 'alpha', 'A', 'lda', 'B', 'ldb', 'batchCount'],
        'param_types': {
            'side': 'cublasSideMode_t',
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'alpha': 'const cuDoubleComplex*',
            'A': 'const cuDoubleComplex* const[]',
            'lda': 'int64_t',
            'B': 'cuDoubleComplex* const[]',
            'ldb': 'int64_t',
            'batchCount': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'B'],
        'scalar_params': ['alpha'],
        'inout_params': ['B'],
        'return_params': ['B'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZtrsmBatched_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* const A[], int64_t lda, cuDoubleComplex* const B[], int64_t ldb, int64_t batchCount);',
    },
    'cublasCdgmm': {
        'base_op': 'dgmm',
        'dtype': 'complex64',
        'level': 3,
        'variant': 'base',
        'description': 'The `cublasCdgmm` function performs a diagonal matrix-matrix multiplication with complex single-precision elements, multiplying each column or row of matrix A by a vector x depending on the specified side mode, and stores the result in matrix C. It is a Level 3 BLAS operation that supports different matrix layouts through leading dimensions (lda, ldc) and vector stride (incx).',

        # C API 信息
        'c_api': 'cublasCdgmm',
        'params': ['mode', 'm', 'n', 'A', 'lda', 'x', 'incx', 'C', 'ldc'],
        'param_types': {
            'mode': 'cublasSideMode_t',
            'm': 'int',
            'n': 'int',
            'A': 'const cuComplex*',
            'lda': 'int',
            'x': 'const cuComplex*',
            'incx': 'int',
            'C': 'cuComplex*',
            'ldc': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'x', 'C'],
        'scalar_params': [],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const cuComplex* A, int lda, const cuComplex* x, int incx, cuComplex* , int ldc);',
    },
    'cublasCdgmm_64': {
        'base_op': 'dgmm',
        'dtype': 'complex64',
        'level': 3,
        'variant': '_64',
        'description': 'The `cublasCdgmm_64` function performs a diagonal matrix-matrix multiplication for complex 64-bit floating-point numbers, where matrix A is multiplied by a diagonal matrix formed from vector x, storing the result in matrix C. It supports 64-bit integers for large matrix dimensions and allows specifying the side of the multiplication (left or right) via the mode parameter.',

        # C API 信息
        'c_api': 'cublasCdgmm_64',
        'params': ['mode', 'm', 'n', 'A', 'lda', 'x', 'incx', 'C', 'ldc'],
        'param_types': {
            'mode': 'cublasSideMode_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'A': 'const cuComplex*',
            'lda': 'int64_t',
            'x': 'const cuComplex*',
            'incx': 'int64_t',
            'C': 'cuComplex*',
            'ldc': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'x', 'C'],
        'scalar_params': [],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCdgmm_64(cublasHandle_t handle, cublasSideMode_t mode, int64_t m, int64_t n, const cuComplex* A, int64_t lda, const cuComplex* x, int64_t incx, cuComplex* , int64_t ldc);',
    },
    'cublasCgeam': {
        'base_op': 'geam',
        'dtype': 'complex64',
        'level': 3,
        'variant': 'base',
        'description': 'The `cublasCgeam` function performs a matrix-matrix operation for complex single-precision matrices, computing `C = alpha * op(A) + beta * op(B)`, where `op` denotes optional transposition of matrices A and B. It is a Level 3 BLAS operation that combines scaled and optionally transposed input matrices into an output matrix.',

        # C API 信息
        'c_api': 'cublasCgeam',
        'params': ['transa', 'transb', 'm', 'n', 'alpha', 'A', 'lda', 'beta', 'B', 'ldb', 'C', 'ldc'],
        'param_types': {
            'transa': 'cublasOperation_t',
            'transb': 'cublasOperation_t',
            'm': 'int',
            'n': 'int',
            'alpha': 'const cuComplex*',
            'A': 'const cuComplex*',
            'lda': 'int',
            'beta': 'const cuComplex*',
            'B': 'const cuComplex*',
            'ldb': 'int',
            'C': 'cuComplex*',
            'ldc': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* beta, const cuComplex* B, int ldb, cuComplex* , int ldc);',
    },
    'cublasCgeam_64': {
        'base_op': 'geam',
        'dtype': 'complex64',
        'level': 3,
        'variant': '_64',
        'description': 'The `cublasCgeam_64` function performs a 64-bit matrix-matrix operation for complex single-precision matrices, computing `C = alpha * op(A) + beta * op(B)`, where `op` denotes optional transposition of matrices A and B. It is a BLAS Level 3 routine supporting large-scale operations with 64-bit integer parameters.',

        # C API 信息
        'c_api': 'cublasCgeam_64',
        'params': ['transa', 'transb', 'm', 'n', 'alpha', 'A', 'lda', 'beta', 'B', 'ldb', 'C', 'ldc'],
        'param_types': {
            'transa': 'cublasOperation_t',
            'transb': 'cublasOperation_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'alpha': 'const cuComplex*',
            'A': 'const cuComplex*',
            'lda': 'int64_t',
            'beta': 'const cuComplex*',
            'B': 'const cuComplex*',
            'ldb': 'int64_t',
            'C': 'cuComplex*',
            'ldc': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCgeam_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* beta, const cuComplex* B, int64_t ldb, cuComplex* , int64_t ldc);',
    },
    'cublasCsymm_v2': {
        'base_op': 'symm',
        'dtype': 'complex64',
        'level': 3,
        'variant': 'base',
        'description': 'The `cublasCsymm_v2` function performs a complex symmetric matrix-matrix multiplication, computing C = α*A*B + β*C or C = α*B*A + β*C, where A is symmetric and B, C are general matrices. It supports both left and right multiplication and can use either the upper or lower triangular part of matrix A.',

        # C API 信息
        'c_api': 'cublasCsymm_v2',
        'params': ['side', 'uplo', 'm', 'n', 'alpha', 'A', 'lda', 'B', 'ldb', 'beta', 'C', 'ldc'],
        'param_types': {
            'side': 'cublasSideMode_t',
            'uplo': 'cublasFillMode_t',
            'm': 'int',
            'n': 'int',
            'alpha': 'const cuComplex*',
            'A': 'const cuComplex*',
            'lda': 'int',
            'B': 'const cuComplex*',
            'ldb': 'int',
            'beta': 'const cuComplex*',
            'C': 'cuComplex*',
            'ldc': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCsymm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* , int ldc);',
    },
    'cublasCsymm_v2_64': {
        'base_op': 'symm',
        'dtype': 'complex64',
        'level': 3,
        'variant': '_64',
        'description': 'Performs 64-bit symmetric matrix-matrix multiplication with complex single-precision elements, computing C = α*A*B + β*C or C = α*B*A + β*C, where A is symmetric and B, C are general matrices.',

        # C API 信息
        'c_api': 'cublasCsymm_v2_64',
        'params': ['side', 'uplo', 'm', 'n', 'alpha', 'A', 'lda', 'B', 'ldb', 'beta', 'C', 'ldc'],
        'param_types': {
            'side': 'cublasSideMode_t',
            'uplo': 'cublasFillMode_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'alpha': 'const cuComplex*',
            'A': 'const cuComplex*',
            'lda': 'int64_t',
            'B': 'const cuComplex*',
            'ldb': 'int64_t',
            'beta': 'const cuComplex*',
            'C': 'cuComplex*',
            'ldc': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCsymm_v2_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, const cuComplex* beta, cuComplex* , int64_t ldc);',
    },
    'cublasCsyr2k_v2': {
        'base_op': 'syr2k',
        'dtype': 'complex64',
        'level': 3,
        'variant': 'base',
        'description': 'The `cublasCsyr2k_v2` function performs a symmetric rank-2k update for complex single-precision matrices, computing `alpha*(A*B^T + B*A^T) + beta*C` or `alpha*(A^T*B + B^T*A) + beta*C` and storing the result in symmetric matrix C. It supports upper or lower triangular storage and optional matrix transposition.',

        # C API 信息
        'c_api': 'cublasCsyr2k_v2',
        'params': ['uplo', 'trans', 'n', 'k', 'alpha', 'A', 'lda', 'B', 'ldb', 'beta', 'C', 'ldc'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'n': 'int',
            'k': 'int',
            'alpha': 'const cuComplex*',
            'A': 'const cuComplex*',
            'lda': 'int',
            'B': 'const cuComplex*',
            'ldb': 'int',
            'beta': 'const cuComplex*',
            'C': 'cuComplex*',
            'ldc': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* , int ldc);',
    },
    'cublasCsyr2k_v2_64': {
        'base_op': 'syr2k',
        'dtype': 'complex64',
        'level': 3,
        'variant': '_64',
        'description': 'The `cublasCsyr2k_v2_64` function performs a symmetric rank-2k update for complex 64-bit matrices, computing `C = alpha*(A*B^T + B*A^T) + beta*C` or `C = alpha*(A^T*B + B^T*A) + beta*C`, where `C` is symmetric and stored in upper or lower triangular form. It supports 64-bit integers for large matrix dimensions and operates on complex single-precision data.',

        # C API 信息
        'c_api': 'cublasCsyr2k_v2_64',
        'params': ['uplo', 'trans', 'n', 'k', 'alpha', 'A', 'lda', 'B', 'ldb', 'beta', 'C', 'ldc'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'n': 'int64_t',
            'k': 'int64_t',
            'alpha': 'const cuComplex*',
            'A': 'const cuComplex*',
            'lda': 'int64_t',
            'B': 'const cuComplex*',
            'ldb': 'int64_t',
            'beta': 'const cuComplex*',
            'C': 'cuComplex*',
            'ldc': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCsyr2k_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, const cuComplex* beta, cuComplex* , int64_t ldc);',
    },
    'cublasCsyrkEx': {
        'base_op': 'syrk',
        'dtype': 'complex64',
        'level': 3,
        'variant': 'Ex',
        'description': 'The `cublasCsyrkEx` function performs a complex symmetric rank-k update, computing `C = alpha*A*A^T + beta*C` or `C = alpha*A^T*A + beta*C` for complex matrices, supporting mixed data types via the `Ex` interface. It operates on BLAS Level 3 with configurable fill mode (`uplo`) and transpose operation (`trans`), using user-specified data types for input (`Atype`) and output (`Ctype`).',

        # C API 信息
        'c_api': 'cublasCsyrkEx',
        'params': ['uplo', 'trans', 'n', 'k', 'alpha', 'A', 'Atype', 'lda', 'beta', 'C', 'Ctype', 'ldc'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'n': 'int',
            'k': 'int',
            'alpha': 'const cuComplex*',
            'A': 'const void*',
            'Atype': 'cudaDataType',
            'lda': 'int',
            'beta': 'const cuComplex*',
            'C': 'void*',
            'Ctype': 'cudaDataType',
            'ldc': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCsyrkEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const void* A, cudaDataType Atype, int lda, const cuComplex* beta, void* , cudaDataType Ctype, int ldc);',
    },
    'cublasCsyrkEx_64': {
        'base_op': 'syrk',
        'dtype': 'complex64',
        'level': 3,
        'variant': 'Ex_64',
        'description': 'The `cublasCsyrkEx_64` function performs a 64-bit complex single-precision symmetric rank-k update, multiplying a complex matrix by its transpose and adding the result to a symmetric matrix, with support for mixed data types. It operates on 64-bit integer dimensions and allows specifying the fill mode (upper/lower) and transpose operation for the input matrix.',

        # C API 信息
        'c_api': 'cublasCsyrkEx_64',
        'params': ['uplo', 'trans', 'n', 'k', 'alpha', 'A', 'Atype', 'lda', 'beta', 'C', 'Ctype', 'ldc'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'n': 'int64_t',
            'k': 'int64_t',
            'alpha': 'const cuComplex*',
            'A': 'const void*',
            'Atype': 'cudaDataType',
            'lda': 'int64_t',
            'beta': 'const cuComplex*',
            'C': 'void*',
            'Ctype': 'cudaDataType',
            'ldc': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCsyrkEx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuComplex* alpha, const void* A, cudaDataType Atype, int64_t lda, const cuComplex* beta, void* , cudaDataType Ctype, int64_t ldc);',
    },
    'cublasCsyrk_v2': {
        'base_op': 'syrk',
        'dtype': 'complex64',
        'level': 3,
        'variant': 'base',
        'description': 'The `cublasCsyrk_v2` function performs a symmetric rank-k update for complex single-precision matrices, computing either `C = alpha*A*A^T + beta*C` or `C = alpha*A^T*A + beta*C`, where `C` is a symmetric matrix stored in upper or lower storage mode. It is a Level 3 BLAS operation supporting matrix-matrix multiplication with symmetric output.',

        # C API 信息
        'c_api': 'cublasCsyrk_v2',
        'params': ['uplo', 'trans', 'n', 'k', 'alpha', 'A', 'lda', 'beta', 'C', 'ldc'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'n': 'int',
            'k': 'int',
            'alpha': 'const cuComplex*',
            'A': 'const cuComplex*',
            'lda': 'int',
            'beta': 'const cuComplex*',
            'C': 'cuComplex*',
            'ldc': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* beta, cuComplex* , int ldc);',
    },
    'cublasCsyrk_v2_64': {
        'base_op': 'syrk',
        'dtype': 'complex64',
        'level': 3,
        'variant': '_64',
        'description': 'The `cublasCsyrk_v2_64` function performs a rank-k update of a complex 64-bit symmetric matrix C using a general matrix A, with support for 64-bit integers. It computes C = α·A·Aᵀ + β·C or C = α·Aᵀ·A + β·C, depending on the transpose operation and fill mode specified.',

        # C API 信息
        'c_api': 'cublasCsyrk_v2_64',
        'params': ['uplo', 'trans', 'n', 'k', 'alpha', 'A', 'lda', 'beta', 'C', 'ldc'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'n': 'int64_t',
            'k': 'int64_t',
            'alpha': 'const cuComplex*',
            'A': 'const cuComplex*',
            'lda': 'int64_t',
            'beta': 'const cuComplex*',
            'C': 'cuComplex*',
            'ldc': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCsyrk_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* beta, cuComplex* , int64_t ldc);',
    },
    'cublasCtrmm_v2': {
        'base_op': 'trmm',
        'dtype': 'complex64',
        'level': 3,
        'variant': 'base',
        'description': 'The `cublasCtrmm_v2` function performs a complex64 triangular matrix-matrix multiplication, where one of the input matrices is triangular, and stores the result in a separate matrix. It supports options for matrix side (left/right), triangle type (upper/lower), transpose operation, and diagonal handling.',

        # C API 信息
        'c_api': 'cublasCtrmm_v2',
        'params': ['side', 'uplo', 'trans', 'diag', 'm', 'n', 'alpha', 'A', 'lda', 'B', 'ldb', 'C', 'ldc'],
        'param_types': {
            'side': 'cublasSideMode_t',
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'm': 'int',
            'n': 'int',
            'alpha': 'const cuComplex*',
            'A': 'const cuComplex*',
            'lda': 'int',
            'B': 'const cuComplex*',
            'ldb': 'int',
            'C': 'cuComplex*',
            'ldc': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCtrmm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, cuComplex* , int ldc);',
    },
    'cublasCtrmm_v2_64': {
        'base_op': 'trmm',
        'dtype': 'complex64',
        'level': 3,
        'variant': '_64',
        'description': 'The `cublasCtrmm_v2_64` function performs a 64-bit triangular matrix-matrix multiplication with complex single-precision elements, where one input matrix is triangular, and stores the result in a separate output matrix. It supports configurable side (left/right), uplo (upper/lower), trans (transpose), and diag (unit/non-unit) parameters for flexible matrix operations.',

        # C API 信息
        'c_api': 'cublasCtrmm_v2_64',
        'params': ['side', 'uplo', 'trans', 'diag', 'm', 'n', 'alpha', 'A', 'lda', 'B', 'ldb', 'C', 'ldc'],
        'param_types': {
            'side': 'cublasSideMode_t',
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'alpha': 'const cuComplex*',
            'A': 'const cuComplex*',
            'lda': 'int64_t',
            'B': 'const cuComplex*',
            'ldb': 'int64_t',
            'C': 'cuComplex*',
            'ldc': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCtrmm_v2_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, cuComplex* , int64_t ldc);',
    },
    'cublasCtrsm_v2': {
        'base_op': 'trsm',
        'dtype': 'complex64',
        'level': 3,
        'variant': 'base',
        'description': 'The `cublasCtrsm_v2` function solves a complex64 triangular matrix equation, either AX = αB or XA = αB, where A is a triangular matrix and X and B are m×n matrices, with options for matrix side, fill mode, transpose operation, and diagonal type. It is a Level 3 BLAS operation that performs in-place computation on the input matrix B.',

        # C API 信息
        'c_api': 'cublasCtrsm_v2',
        'params': ['side', 'uplo', 'trans', 'diag', 'm', 'n', 'alpha', 'A', 'lda', 'B', 'ldb'],
        'param_types': {
            'side': 'cublasSideMode_t',
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'm': 'int',
            'n': 'int',
            'alpha': 'const cuComplex*',
            'A': 'const cuComplex*',
            'lda': 'int',
            'B': 'cuComplex*',
            'ldb': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'B'],
        'scalar_params': ['alpha'],
        'inout_params': ['B'],
        'return_params': ['B'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCtrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, cuComplex* B, int ldb);',
    },
    'cublasCtrsm_v2_64': {
        'base_op': 'trsm',
        'dtype': 'complex64',
        'level': 3,
        'variant': '_64',
        'description': 'The `cublasCtrsm_v2_64` function solves a complex64 triangular linear system with multiple right-hand sides, where the matrix can be upper or lower triangular and optionally transposed, using 64-bit integers for large problem sizes. It scales the solution by a scalar `alpha` and operates on matrices stored in column-major format.',

        # C API 信息
        'c_api': 'cublasCtrsm_v2_64',
        'params': ['side', 'uplo', 'trans', 'diag', 'm', 'n', 'alpha', 'A', 'lda', 'B', 'ldb'],
        'param_types': {
            'side': 'cublasSideMode_t',
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'alpha': 'const cuComplex*',
            'A': 'const cuComplex*',
            'lda': 'int64_t',
            'B': 'cuComplex*',
            'ldb': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'B'],
        'scalar_params': ['alpha'],
        'inout_params': ['B'],
        'return_params': ['B'],

        # 完整签名
        'signature': 'cublasStatus_t cublasCtrsm_v2_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, cuComplex* B, int64_t ldb);',
    },
    'cublasDdgmm': {
        'base_op': 'dgmm',
        'dtype': 'float64',
        'level': 3,
        'variant': 'base',
        'description': 'The `cublasDdgmm` function performs a diagonal matrix-matrix multiplication, where a double-precision matrix `A` is multiplied by a diagonal matrix formed from a double-precision vector `x`, producing the result matrix `C`. The operation can be performed with the diagonal on either the left or right side of `A`, as specified by the `mode` parameter.',

        # C API 信息
        'c_api': 'cublasDdgmm',
        'params': ['mode', 'm', 'n', 'A', 'lda', 'x', 'incx', 'C', 'ldc'],
        'param_types': {
            'mode': 'cublasSideMode_t',
            'm': 'int',
            'n': 'int',
            'A': 'const double*',
            'lda': 'int',
            'x': 'const double*',
            'incx': 'int',
            'C': 'double*',
            'ldc': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'x', 'C'],
        'scalar_params': [],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const double* A, int lda, const double* x, int incx, double* , int ldc);',
    },
    'cublasDdgmm_64': {
        'base_op': 'dgmm',
        'dtype': 'float64',
        'level': 3,
        'variant': '_64',
        'description': 'The `cublasDdgmm_64` function performs a double-precision matrix-matrix multiplication where one matrix is a diagonal matrix constructed from a vector, either left or right multiplying the input matrix based on the specified mode. It supports 64-bit integers for matrix dimensions and strides, operating on double-precision floating-point data.',

        # C API 信息
        'c_api': 'cublasDdgmm_64',
        'params': ['mode', 'm', 'n', 'A', 'lda', 'x', 'incx', 'C', 'ldc'],
        'param_types': {
            'mode': 'cublasSideMode_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'A': 'const double*',
            'lda': 'int64_t',
            'x': 'const double*',
            'incx': 'int64_t',
            'C': 'double*',
            'ldc': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'x', 'C'],
        'scalar_params': [],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDdgmm_64(cublasHandle_t handle, cublasSideMode_t mode, int64_t m, int64_t n, const double* A, int64_t lda, const double* x, int64_t incx, double* , int64_t ldc);',
    },
    'cublasDgeam': {
        'base_op': 'geam',
        'dtype': 'float64',
        'level': 3,
        'variant': 'base',
        'description': 'The `cublasDgeam` function performs a double-precision matrix-matrix operation that combines scaling and addition, computing `alpha * op(A) + beta * op(B)` where `op` denotes optional matrix transposition. It operates on double-precision matrices with dimensions `m x n` and supports custom leading dimensions.',

        # C API 信息
        'c_api': 'cublasDgeam',
        'params': ['transa', 'transb', 'm', 'n', 'alpha', 'A', 'lda', 'beta', 'B', 'ldb', 'C', 'ldc'],
        'param_types': {
            'transa': 'cublasOperation_t',
            'transb': 'cublasOperation_t',
            'm': 'int',
            'n': 'int',
            'alpha': 'const double*',
            'A': 'const double*',
            'lda': 'int',
            'beta': 'const double*',
            'B': 'const double*',
            'ldb': 'int',
            'C': 'double*',
            'ldc': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const double* alpha, const double* A, int lda, const double* beta, const double* B, int ldb, double* , int ldc);',
    },
    'cublasDgeam_64': {
        'base_op': 'geam',
        'dtype': 'float64',
        'level': 3,
        'variant': '_64',
        'description': 'The `cublasDgeam_64` function performs a 64-bit integer GEneral Matrix Addition and Multiplication (GEAM) operation, computing `C = alpha * op(A) + beta * op(B)` where `op` denotes optional matrix transposition, with double-precision floating-point elements. It supports large matrices using 64-bit dimensions and strides.',

        # C API 信息
        'c_api': 'cublasDgeam_64',
        'params': ['transa', 'transb', 'm', 'n', 'alpha', 'A', 'lda', 'beta', 'B', 'ldb', 'C', 'ldc'],
        'param_types': {
            'transa': 'cublasOperation_t',
            'transb': 'cublasOperation_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'alpha': 'const double*',
            'A': 'const double*',
            'lda': 'int64_t',
            'beta': 'const double*',
            'B': 'const double*',
            'ldb': 'int64_t',
            'C': 'double*',
            'ldc': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDgeam_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, const double* alpha, const double* A, int64_t lda, const double* beta, const double* B, int64_t ldb, double* , int64_t ldc);',
    },
    'cublasDsymm_v2': {
        'base_op': 'symm',
        'dtype': 'float64',
        'level': 3,
        'variant': 'base',
        'description': 'The `cublasDsymm_v2` function performs double-precision symmetric matrix-matrix multiplication, computing C = αAB + βC or C = αBA + βC, where A is symmetric and B, C are general matrices, with options for side (left/right) and triangle (upper/lower) selection.',

        # C API 信息
        'c_api': 'cublasDsymm_v2',
        'params': ['side', 'uplo', 'm', 'n', 'alpha', 'A', 'lda', 'B', 'ldb', 'beta', 'C', 'ldc'],
        'param_types': {
            'side': 'cublasSideMode_t',
            'uplo': 'cublasFillMode_t',
            'm': 'int',
            'n': 'int',
            'alpha': 'const double*',
            'A': 'const double*',
            'lda': 'int',
            'B': 'const double*',
            'ldb': 'int',
            'beta': 'const double*',
            'C': 'double*',
            'ldc': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDsymm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* , int ldc);',
    },
    'cublasDsymm_v2_64': {
        'base_op': 'symm',
        'dtype': 'float64',
        'level': 3,
        'variant': '_64',
        'description': 'The `cublasDsymm_v2_64` function performs double-precision symmetric matrix-matrix multiplication, computing C = alpha * A * B + beta * C or C = alpha * B * A + beta * C, where A` is symmetric and `B`, `C` are general matrices, using 64-bit integers for large matrix dimensions.',

        # C API 信息
        'c_api': 'cublasDsymm_v2_64',
        'params': ['side', 'uplo', 'm', 'n', 'alpha', 'A', 'lda', 'B', 'ldb', 'beta', 'C', 'ldc'],
        'param_types': {
            'side': 'cublasSideMode_t',
            'uplo': 'cublasFillMode_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'alpha': 'const double*',
            'A': 'const double*',
            'lda': 'int64_t',
            'B': 'const double*',
            'ldb': 'int64_t',
            'beta': 'const double*',
            'C': 'double*',
            'ldc': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDsymm_v2_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int64_t m, int64_t n, const double* alpha, const double* A, int64_t lda, const double* B, int64_t ldb, const double* beta, double* , int64_t ldc);',
    },
    'cublasDsyr2k_v2': {
        'base_op': 'syr2k',
        'dtype': 'float64',
        'level': 3,
        'variant': 'base',
        'description': 'The `cublasDsyr2k_v2` function performs a symmetric rank-2k update using double-precision floating-point numbers, computing `C = alpha*(A*B^T + B*A^T) + beta*C` or `C = alpha*(A^T*B + B^T*A) + beta*C`, where `C` is a symmetric matrix. It supports upper or lower triangular storage and matrix transposition options.',

        # C API 信息
        'c_api': 'cublasDsyr2k_v2',
        'params': ['uplo', 'trans', 'n', 'k', 'alpha', 'A', 'lda', 'B', 'ldb', 'beta', 'C', 'ldc'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'n': 'int',
            'k': 'int',
            'alpha': 'const double*',
            'A': 'const double*',
            'lda': 'int',
            'B': 'const double*',
            'ldb': 'int',
            'beta': 'const double*',
            'C': 'double*',
            'ldc': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* , int ldc);',
    },
    'cublasDsyr2k_v2_64': {
        'base_op': 'syr2k',
        'dtype': 'float64',
        'level': 3,
        'variant': '_64',
        'description': 'The `cublasDsyr2k_v2_64` function performs a symmetric rank-2k update using double-precision floating-point numbers, computing `alpha*(A*B^T + B*A^T) + beta*C` or `alpha*(A^T*B + B^T*A) + beta*C` and storing the result in symmetric matrix C, with 64-bit integer parameters for large matrix support.',

        # C API 信息
        'c_api': 'cublasDsyr2k_v2_64',
        'params': ['uplo', 'trans', 'n', 'k', 'alpha', 'A', 'lda', 'B', 'ldb', 'beta', 'C', 'ldc'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'n': 'int64_t',
            'k': 'int64_t',
            'alpha': 'const double*',
            'A': 'const double*',
            'lda': 'int64_t',
            'B': 'const double*',
            'ldb': 'int64_t',
            'beta': 'const double*',
            'C': 'double*',
            'ldc': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDsyr2k_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const double* alpha, const double* A, int64_t lda, const double* B, int64_t ldb, const double* beta, double* , int64_t ldc);',
    },
    'cublasDsyrk_v2': {
        'base_op': 'syrk',
        'dtype': 'float64',
        'level': 3,
        'variant': 'base',
        'description': 'The `cublasDsyrk_v2` function performs a symmetric rank-k update using double-precision floating-point numbers, computing either `C = alpha*A*A^T + beta*C` or `C = alpha*A^T*A + beta*C` depending on the transpose operation, where `C` is a symmetric matrix stored in upper or lower triangular form.',

        # C API 信息
        'c_api': 'cublasDsyrk_v2',
        'params': ['uplo', 'trans', 'n', 'k', 'alpha', 'A', 'lda', 'beta', 'C', 'ldc'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'n': 'int',
            'k': 'int',
            'alpha': 'const double*',
            'A': 'const double*',
            'lda': 'int',
            'beta': 'const double*',
            'C': 'double*',
            'ldc': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double* alpha, const double* A, int lda, const double* beta, double* , int ldc);',
    },
    'cublasDsyrk_v2_64': {
        'base_op': 'syrk',
        'dtype': 'float64',
        'level': 3,
        'variant': '_64',
        'description': 'The `cublasDsyrk_v2_64` function performs a symmetric rank-k update using double-precision floating-point numbers, computing `C = alpha*A*A^T + beta*C` or `C = alpha*A^T*A + beta*C` depending on the transpose operation, where `C` is a symmetric matrix stored in upper or lower storage mode. It supports 64-bit integers for matrix dimensions and strides.',

        # C API 信息
        'c_api': 'cublasDsyrk_v2_64',
        'params': ['uplo', 'trans', 'n', 'k', 'alpha', 'A', 'lda', 'beta', 'C', 'ldc'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'n': 'int64_t',
            'k': 'int64_t',
            'alpha': 'const double*',
            'A': 'const double*',
            'lda': 'int64_t',
            'beta': 'const double*',
            'C': 'double*',
            'ldc': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDsyrk_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const double* alpha, const double* A, int64_t lda, const double* beta, double* , int64_t ldc);',
    },
    'cublasDtrmm_v2': {
        'base_op': 'trmm',
        'dtype': 'float64',
        'level': 3,
        'variant': 'base',
        'description': 'The `cublasDtrmm_v2` function performs a double-precision triangular matrix-matrix multiplication, where one input matrix is triangular, and stores the result in another matrix. It supports options for matrix side, uplo (upper/lower triangular), transpose operation, and diagonal type.',

        # C API 信息
        'c_api': 'cublasDtrmm_v2',
        'params': ['side', 'uplo', 'trans', 'diag', 'm', 'n', 'alpha', 'A', 'lda', 'B', 'ldb', 'C', 'ldc'],
        'param_types': {
            'side': 'cublasSideMode_t',
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'm': 'int',
            'n': 'int',
            'alpha': 'const double*',
            'A': 'const double*',
            'lda': 'int',
            'B': 'const double*',
            'ldb': 'int',
            'C': 'double*',
            'ldc': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDtrmm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double* alpha, const double* A, int lda, const double* B, int ldb, double* , int ldc);',
    },
    'cublasDtrmm_v2_64': {
        'base_op': 'trmm',
        'dtype': 'float64',
        'level': 3,
        'variant': '_64',
        'description': 'The `cublasDtrmm_v2_64` function performs a double-precision triangular matrix-matrix multiplication, where one input matrix is triangular, and stores the result in another matrix, supporting 64-bit integers for large problem sizes. It allows specifying the triangular matrix\'s side (left/right), storage (upper/lower), transpose operation, and diagonal type.',

        # C API 信息
        'c_api': 'cublasDtrmm_v2_64',
        'params': ['side', 'uplo', 'trans', 'diag', 'm', 'n', 'alpha', 'A', 'lda', 'B', 'ldb', 'C', 'ldc'],
        'param_types': {
            'side': 'cublasSideMode_t',
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'alpha': 'const double*',
            'A': 'const double*',
            'lda': 'int64_t',
            'B': 'const double*',
            'ldb': 'int64_t',
            'C': 'double*',
            'ldc': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDtrmm_v2_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const double* alpha, const double* A, int64_t lda, const double* B, int64_t ldb, double* , int64_t ldc);',
    },
    'cublasDtrsm_v2': {
        'base_op': 'trsm',
        'dtype': 'float64',
        'level': 3,
        'variant': 'base',
        'description': 'The `cublasDtrsm_v2` function solves a triangular matrix equation with double-precision floating-point elements, performing the operation `B = alpha * op(A)^{-1} * B` or `B = alpha * B * op(A)^{-1}`, where `A` is a triangular matrix and `B` is a general matrix. It supports configurable side, uplo, trans, and diag parameters to control the matrix shape, operation, and diagonal handling.',

        # C API 信息
        'c_api': 'cublasDtrsm_v2',
        'params': ['side', 'uplo', 'trans', 'diag', 'm', 'n', 'alpha', 'A', 'lda', 'B', 'ldb'],
        'param_types': {
            'side': 'cublasSideMode_t',
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'm': 'int',
            'n': 'int',
            'alpha': 'const double*',
            'A': 'const double*',
            'lda': 'int',
            'B': 'double*',
            'ldb': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'B'],
        'scalar_params': ['alpha'],
        'inout_params': ['B'],
        'return_params': ['B'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDtrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double* alpha, const double* A, int lda, double* B, int ldb);',
    },
    'cublasDtrsm_v2_64': {
        'base_op': 'trsm',
        'dtype': 'float64',
        'level': 3,
        'variant': '_64',
        'description': 'The `cublasDtrsm_v2_64` function solves a triangular matrix equation with double-precision floating-point elements, performing the operation `B = alpha * op(A)^-1 * B` or `B = alpha * B * op(A)^-1`, where `A` is a triangular matrix and `B` is a general matrix, supporting 64-bit integer parameters for large-scale computations.',

        # C API 信息
        'c_api': 'cublasDtrsm_v2_64',
        'params': ['side', 'uplo', 'trans', 'diag', 'm', 'n', 'alpha', 'A', 'lda', 'B', 'ldb'],
        'param_types': {
            'side': 'cublasSideMode_t',
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'alpha': 'const double*',
            'A': 'const double*',
            'lda': 'int64_t',
            'B': 'double*',
            'ldb': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'B'],
        'scalar_params': ['alpha'],
        'inout_params': ['B'],
        'return_params': ['B'],

        # 完整签名
        'signature': 'cublasStatus_t cublasDtrsm_v2_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const double* alpha, const double* A, int64_t lda, double* B, int64_t ldb);',
    },
    'cublasSdgmm': {
        'base_op': 'dgmm',
        'dtype': 'float32',
        'level': 3,
        'variant': 'base',
        'description': 'The `cublasSdgmm` function performs a matrix-matrix multiplication where one input matrix is a diagonal matrix represented by a vector, using single-precision floating-point (float32) values. It supports left or right multiplication of the diagonal matrix (specified by `mode`) and stores the result in the output matrix.',

        # C API 信息
        'c_api': 'cublasSdgmm',
        'params': ['mode', 'm', 'n', 'A', 'lda', 'x', 'incx', 'C', 'ldc'],
        'param_types': {
            'mode': 'cublasSideMode_t',
            'm': 'int',
            'n': 'int',
            'A': 'const float*',
            'lda': 'int',
            'x': 'const float*',
            'incx': 'int',
            'C': 'float*',
            'ldc': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'x', 'C'],
        'scalar_params': [],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const float* A, int lda, const float* x, int incx, float* , int ldc);',
    },
    'cublasSdgmm_64': {
        'base_op': 'dgmm',
        'dtype': 'float32',
        'level': 3,
        'variant': '_64',
        'description': 'The `cublasSdgmm_64` function performs a double-precision general matrix-matrix multiplication (DGMM) operation on 32-bit floating-point matrices, where one matrix is multiplied by a diagonal matrix formed from a vector, with 64-bit integer parameters for large matrix support. It supports left or right multiplication modes and stores the result in a separate output matrix.',

        # C API 信息
        'c_api': 'cublasSdgmm_64',
        'params': ['mode', 'm', 'n', 'A', 'lda', 'x', 'incx', 'C', 'ldc'],
        'param_types': {
            'mode': 'cublasSideMode_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'A': 'const float*',
            'lda': 'int64_t',
            'x': 'const float*',
            'incx': 'int64_t',
            'C': 'float*',
            'ldc': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'x', 'C'],
        'scalar_params': [],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSdgmm_64(cublasHandle_t handle, cublasSideMode_t mode, int64_t m, int64_t n, const float* A, int64_t lda, const float* x, int64_t incx, float* , int64_t ldc);',
    },
    'cublasSgeam': {
        'base_op': 'geam',
        'dtype': 'float32',
        'level': 3,
        'variant': 'base',
        'description': 'The `cublasSgeam` function performs a matrix-matrix addition and scaling operation, computing C = alpha * op(A) + beta * op(B), where op(X) can be the matrix X or its transpose, and A, B, C are single-precision floating-point matrices.',

        # C API 信息
        'c_api': 'cublasSgeam',
        'params': ['transa', 'transb', 'm', 'n', 'alpha', 'A', 'lda', 'beta', 'B', 'ldb', 'C', 'ldc'],
        'param_types': {
            'transa': 'cublasOperation_t',
            'transb': 'cublasOperation_t',
            'm': 'int',
            'n': 'int',
            'alpha': 'const float*',
            'A': 'const float*',
            'lda': 'int',
            'beta': 'const float*',
            'B': 'const float*',
            'ldb': 'int',
            'C': 'float*',
            'ldc': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const float* alpha, const float* A, int lda, const float* beta, const float* B, int ldb, float* , int ldc);',
    },
    'cublasSgeam_64': {
        'base_op': 'geam',
        'dtype': 'float32',
        'level': 3,
        'variant': '_64',
        'description': 'The `cublasSgeam_64` function performs a 64-bit integer matrix-matrix operation for single-precision floating-point data, computing C = alpha * op(A) + beta * op(B), where op(X) can be the matrix X or its transpose. It supports large matrices with 64-bit dimensions and strides (m, n, lda, ldb, ldc).',

        # C API 信息
        'c_api': 'cublasSgeam_64',
        'params': ['transa', 'transb', 'm', 'n', 'alpha', 'A', 'lda', 'beta', 'B', 'ldb', 'C', 'ldc'],
        'param_types': {
            'transa': 'cublasOperation_t',
            'transb': 'cublasOperation_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'alpha': 'const float*',
            'A': 'const float*',
            'lda': 'int64_t',
            'beta': 'const float*',
            'B': 'const float*',
            'ldb': 'int64_t',
            'C': 'float*',
            'ldc': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSgeam_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, const float* alpha, const float* A, int64_t lda, const float* beta, const float* B, int64_t ldb, float* , int64_t ldc);',
    },
    'cublasSsymm_v2': {
        'base_op': 'symm',
        'dtype': 'float32',
        'level': 3,
        'variant': 'base',
        'description': 'The `cublasSsymm_v2` function performs symmetric matrix-matrix multiplication using single-precision floating-point numbers, computing C = α*A*B + β*C or C = α*B*A + β*C, where A is a symmetric matrix and B, C are general matrices. The operation side (left/right) and triangle (upper/lower) of A are specified by the `side` and `uplo` parameters.',

        # C API 信息
        'c_api': 'cublasSsymm_v2',
        'params': ['side', 'uplo', 'm', 'n', 'alpha', 'A', 'lda', 'B', 'ldb', 'beta', 'C', 'ldc'],
        'param_types': {
            'side': 'cublasSideMode_t',
            'uplo': 'cublasFillMode_t',
            'm': 'int',
            'n': 'int',
            'alpha': 'const float*',
            'A': 'const float*',
            'lda': 'int',
            'B': 'const float*',
            'ldb': 'int',
            'beta': 'const float*',
            'C': 'float*',
            'ldc': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSsymm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* , int ldc);',
    },
    'cublasSsymm_v2_64': {
        'base_op': 'symm',
        'dtype': 'float32',
        'level': 3,
        'variant': '_64',
        'description': 'The `cublasSsymm_v2_64` function performs 64-bit symmetric matrix-matrix multiplication using single-precision floats, computing `C = alpha*A*B + beta*C` or `C = alpha*B*A + beta*C` depending on the side parameter, where `A` is symmetric.',

        # C API 信息
        'c_api': 'cublasSsymm_v2_64',
        'params': ['side', 'uplo', 'm', 'n', 'alpha', 'A', 'lda', 'B', 'ldb', 'beta', 'C', 'ldc'],
        'param_types': {
            'side': 'cublasSideMode_t',
            'uplo': 'cublasFillMode_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'alpha': 'const float*',
            'A': 'const float*',
            'lda': 'int64_t',
            'B': 'const float*',
            'ldb': 'int64_t',
            'beta': 'const float*',
            'C': 'float*',
            'ldc': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSsymm_v2_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int64_t m, int64_t n, const float* alpha, const float* A, int64_t lda, const float* B, int64_t ldb, const float* beta, float* , int64_t ldc);',
    },
    'cublasSsyr2k_v2': {
        'base_op': 'syr2k',
        'dtype': 'float32',
        'level': 3,
        'variant': 'base',
        'description': 'The `cublasSsyr2k_v2` function performs a symmetric rank-2k update, computing `C = alpha*(A*B^T + B*A^T) + beta*C` or `C = alpha*(A^T*B + B^T*A) + beta*C` for single-precision matrices, where `C` is symmetric. It supports upper or lower storage modes and optional matrix transposition.',

        # C API 信息
        'c_api': 'cublasSsyr2k_v2',
        'params': ['uplo', 'trans', 'n', 'k', 'alpha', 'A', 'lda', 'B', 'ldb', 'beta', 'C', 'ldc'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'n': 'int',
            'k': 'int',
            'alpha': 'const float*',
            'A': 'const float*',
            'lda': 'int',
            'B': 'const float*',
            'ldb': 'int',
            'beta': 'const float*',
            'C': 'float*',
            'ldc': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* , int ldc);',
    },
    'cublasSsyr2k_v2_64': {
        'base_op': 'syr2k',
        'dtype': 'float32',
        'level': 3,
        'variant': '_64',
        'description': 'The `cublasSsyr2k_v2_64` function performs a symmetric rank-2k update using single-precision floating-point numbers, computing `alpha*(A*B^T + B*A^T) + beta*C` or `alpha*(A^T*B + B^T*A) + beta*C` and storing the result in symmetric matrix C, with 64-bit integer parameters for large matrix dimensions.',

        # C API 信息
        'c_api': 'cublasSsyr2k_v2_64',
        'params': ['uplo', 'trans', 'n', 'k', 'alpha', 'A', 'lda', 'B', 'ldb', 'beta', 'C', 'ldc'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'n': 'int64_t',
            'k': 'int64_t',
            'alpha': 'const float*',
            'A': 'const float*',
            'lda': 'int64_t',
            'B': 'const float*',
            'ldb': 'int64_t',
            'beta': 'const float*',
            'C': 'float*',
            'ldc': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSsyr2k_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const float* alpha, const float* A, int64_t lda, const float* B, int64_t ldb, const float* beta, float* , int64_t ldc);',
    },
    'cublasSsyrk_v2': {
        'base_op': 'syrk',
        'dtype': 'float32',
        'level': 3,
        'variant': 'base',
        'description': 'The `cublasSsyrk_v2` function performs a symmetric rank-k update for single-precision matrices, computing `C = alpha*A*A^T + beta*C` or `C = alpha*A^T*A + beta*C` depending on the transpose operation, where `C` is a symmetric matrix stored in upper or lower triangular form. It is a Level 3 BLAS operation for efficient symmetric matrix updates.',

        # C API 信息
        'c_api': 'cublasSsyrk_v2',
        'params': ['uplo', 'trans', 'n', 'k', 'alpha', 'A', 'lda', 'beta', 'C', 'ldc'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'n': 'int',
            'k': 'int',
            'alpha': 'const float*',
            'A': 'const float*',
            'lda': 'int',
            'beta': 'const float*',
            'C': 'float*',
            'ldc': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float* alpha, const float* A, int lda, const float* beta, float* , int ldc);',
    },
    'cublasSsyrk_v2_64': {
        'base_op': 'syrk',
        'dtype': 'float32',
        'level': 3,
        'variant': '_64',
        'description': 'The `cublasSsyrk_v2_64` function performs a symmetric rank-k update for 64-bit integers, multiplying a single-precision matrix by its transpose and adding it to a symmetric matrix, with options for upper or lower triangular storage. It supports 64-bit dimensions and strides, and scales the operation using scalar coefficients alpha and beta.',

        # C API 信息
        'c_api': 'cublasSsyrk_v2_64',
        'params': ['uplo', 'trans', 'n', 'k', 'alpha', 'A', 'lda', 'beta', 'C', 'ldc'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'n': 'int64_t',
            'k': 'int64_t',
            'alpha': 'const float*',
            'A': 'const float*',
            'lda': 'int64_t',
            'beta': 'const float*',
            'C': 'float*',
            'ldc': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasSsyrk_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const float* alpha, const float* A, int64_t lda, const float* beta, float* , int64_t ldc);',
    },
    'cublasStrmm_v2': {
        'base_op': 'trmm',
        'dtype': 'float32',
        'level': 3,
        'variant': 'base',
        'description': 'The `cublasStrmm_v2` function performs a triangular matrix-matrix multiplication with float32 precision, computing either `C = alpha * op(A) * B` or `C = alpha * B * op(A)`, where `A` is a triangular matrix and `op` denotes a possible transpose operation. It supports configurable side (left/right multiplication), uplo (upper/lower triangular), trans (transpose options), and diag (unit/non-unit diagonal) parameters.',

        # C API 信息
        'c_api': 'cublasStrmm_v2',
        'params': ['side', 'uplo', 'trans', 'diag', 'm', 'n', 'alpha', 'A', 'lda', 'B', 'ldb', 'C', 'ldc'],
        'param_types': {
            'side': 'cublasSideMode_t',
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'm': 'int',
            'n': 'int',
            'alpha': 'const float*',
            'A': 'const float*',
            'lda': 'int',
            'B': 'const float*',
            'ldb': 'int',
            'C': 'float*',
            'ldc': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasStrmm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float* alpha, const float* A, int lda, const float* B, int ldb, float* , int ldc);',
    },
    'cublasStrmm_v2_64': {
        'base_op': 'trmm',
        'dtype': 'float32',
        'level': 3,
        'variant': '_64',
        'description': 'The `cublasStrmm_v2_64` function performs 64-bit triangular matrix-matrix multiplication with float32 precision, computing `C = alpha * op(A) * B` or `C = alpha * B * op(A)`, where `A` is a triangular matrix and `op` denotes optional transposition. It supports configurable side, fill mode, operation type, and diagonal handling for matrix `A`.',

        # C API 信息
        'c_api': 'cublasStrmm_v2_64',
        'params': ['side', 'uplo', 'trans', 'diag', 'm', 'n', 'alpha', 'A', 'lda', 'B', 'ldb', 'C', 'ldc'],
        'param_types': {
            'side': 'cublasSideMode_t',
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'alpha': 'const float*',
            'A': 'const float*',
            'lda': 'int64_t',
            'B': 'const float*',
            'ldb': 'int64_t',
            'C': 'float*',
            'ldc': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasStrmm_v2_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const float* alpha, const float* A, int64_t lda, const float* B, int64_t ldb, float* , int64_t ldc);',
    },
    'cublasStrsm_v2': {
        'base_op': 'trsm',
        'dtype': 'float32',
        'level': 3,
        'variant': 'base',
        'description': 'The `cublasStrsm_v2` function solves a triangular system of equations with multiple right-hand sides, where the matrix and vectors are single-precision floats. It computes \( B = \alpha \cdot op(A)^{-1} \cdot B \) or \( B = \alpha \cdot B \cdot op(A)^{-1} \), with options for matrix side, uplo, transposition, and diagonal type.',

        # C API 信息
        'c_api': 'cublasStrsm_v2',
        'params': ['side', 'uplo', 'trans', 'diag', 'm', 'n', 'alpha', 'A', 'lda', 'B', 'ldb'],
        'param_types': {
            'side': 'cublasSideMode_t',
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'm': 'int',
            'n': 'int',
            'alpha': 'const float*',
            'A': 'const float*',
            'lda': 'int',
            'B': 'float*',
            'ldb': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'B'],
        'scalar_params': ['alpha'],
        'inout_params': ['B'],
        'return_params': ['B'],

        # 完整签名
        'signature': 'cublasStatus_t cublasStrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float* alpha, const float* A, int lda, float* B, int ldb);',
    },
    'cublasStrsm_v2_64': {
        'base_op': 'trsm',
        'dtype': 'float32',
        'level': 3,
        'variant': '_64',
        'description': 'The `cublasStrsm_v2_64` function solves a triangular system of equations with multiple right-hand sides using 32-bit floating-point arithmetic, supporting 64-bit integers for large matrix dimensions. It computes `B = alpha * op(A)^-1 * B` or `B = alpha * B * op(A)^-1`, where `A` is a triangular matrix and `B` is a general matrix.',

        # C API 信息
        'c_api': 'cublasStrsm_v2_64',
        'params': ['side', 'uplo', 'trans', 'diag', 'm', 'n', 'alpha', 'A', 'lda', 'B', 'ldb'],
        'param_types': {
            'side': 'cublasSideMode_t',
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'alpha': 'const float*',
            'A': 'const float*',
            'lda': 'int64_t',
            'B': 'float*',
            'ldb': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'B'],
        'scalar_params': ['alpha'],
        'inout_params': ['B'],
        'return_params': ['B'],

        # 完整签名
        'signature': 'cublasStatus_t cublasStrsm_v2_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const float* alpha, const float* A, int64_t lda, float* B, int64_t ldb);',
    },
    'cublasZdgmm': {
        'base_op': 'dgmm',
        'dtype': 'complex128',
        'level': 3,
        'variant': 'base',
        'description': 'The `cublasZdgmm` function performs a diagonal matrix-matrix multiplication using a double-precision complex diagonal matrix, either left-multiplying (if mode is `CUBLAS_SIDE_LEFT`) or right-multiplying (if mode is `CUBLAS_SIDE_RIGHT`) the input matrix `A` by the diagonal elements stored in vector `x`, storing the result in matrix `C`.',

        # C API 信息
        'c_api': 'cublasZdgmm',
        'params': ['mode', 'm', 'n', 'A', 'lda', 'x', 'incx', 'C', 'ldc'],
        'param_types': {
            'mode': 'cublasSideMode_t',
            'm': 'int',
            'n': 'int',
            'A': 'const cuDoubleComplex*',
            'lda': 'int',
            'x': 'const cuDoubleComplex*',
            'incx': 'int',
            'C': 'cuDoubleComplex*',
            'ldc': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'x', 'C'],
        'scalar_params': [],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, cuDoubleComplex* , int ldc);',
    },
    'cublasZdgmm_64': {
        'base_op': 'dgmm',
        'dtype': 'complex128',
        'level': 3,
        'variant': '_64',
        'description': 'The `cublasZdgmm_64` function performs a diagonal matrix-matrix multiplication using a complex64 diagonal matrix, either left-multiplying or right-multiplying matrix A with diagonal elements from vector x, storing the result in matrix C. It supports 64-bit integers for large matrix dimensions and strides.',

        # C API 信息
        'c_api': 'cublasZdgmm_64',
        'params': ['mode', 'm', 'n', 'A', 'lda', 'x', 'incx', 'C', 'ldc'],
        'param_types': {
            'mode': 'cublasSideMode_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'A': 'const cuDoubleComplex*',
            'lda': 'int64_t',
            'x': 'const cuDoubleComplex*',
            'incx': 'int64_t',
            'C': 'cuDoubleComplex*',
            'ldc': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'x', 'C'],
        'scalar_params': [],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZdgmm_64(cublasHandle_t handle, cublasSideMode_t mode, int64_t m, int64_t n, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* x, int64_t incx, cuDoubleComplex* , int64_t ldc);',
    },
    'cublasZgeam': {
        'base_op': 'geam',
        'dtype': 'complex128',
        'level': 3,
        'variant': 'base',
        'description': 'The `cublasZgeam` function performs a matrix-matrix operation for complex double-precision matrices, computing `C = alpha * op(A) + beta * op(B)`, where `op` denotes optional transposition of matrices A and B. It is a BLAS Level 3 function supporting general matrix operations with configurable scaling and transposition parameters.',

        # C API 信息
        'c_api': 'cublasZgeam',
        'params': ['transa', 'transb', 'm', 'n', 'alpha', 'A', 'lda', 'beta', 'B', 'ldb', 'C', 'ldc'],
        'param_types': {
            'transa': 'cublasOperation_t',
            'transb': 'cublasOperation_t',
            'm': 'int',
            'n': 'int',
            'alpha': 'const cuDoubleComplex*',
            'A': 'const cuDoubleComplex*',
            'lda': 'int',
            'beta': 'const cuDoubleComplex*',
            'B': 'const cuDoubleComplex*',
            'ldb': 'int',
            'C': 'cuDoubleComplex*',
            'ldc': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* beta, const cuDoubleComplex* B, int ldb, cuDoubleComplex* , int ldc);',
    },
    'cublasZgeam_64': {
        'base_op': 'geam',
        'dtype': 'complex128',
        'level': 3,
        'variant': '_64',
        'description': 'The `cublasZgeam_64` function performs a 64-bit integer matrix-matrix operation for complex double-precision matrices, computing `C = alpha * op(A) + beta * op(B)`, where `op` denotes optional matrix transposition. It is a BLAS Level 3 routine supporting large matrices with 64-bit dimensions.',

        # C API 信息
        'c_api': 'cublasZgeam_64',
        'params': ['transa', 'transb', 'm', 'n', 'alpha', 'A', 'lda', 'beta', 'B', 'ldb', 'C', 'ldc'],
        'param_types': {
            'transa': 'cublasOperation_t',
            'transb': 'cublasOperation_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'alpha': 'const cuDoubleComplex*',
            'A': 'const cuDoubleComplex*',
            'lda': 'int64_t',
            'beta': 'const cuDoubleComplex*',
            'B': 'const cuDoubleComplex*',
            'ldb': 'int64_t',
            'C': 'cuDoubleComplex*',
            'ldc': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZgeam_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* beta, const cuDoubleComplex* B, int64_t ldb, cuDoubleComplex* , int64_t ldc);',
    },
    'cublasZsymm_v2': {
        'base_op': 'symm',
        'dtype': 'complex128',
        'level': 3,
        'variant': 'base',
        'description': 'The `cublasZsymm_v2` function performs a complex double-precision symmetric matrix-matrix multiplication, computing `C = alpha * A * B + beta * C` or `C = alpha * B * A + beta * C`, where `A` is symmetric and `B`, `C` are general matrices. The operation side (left/right) and triangle (upper/lower) of `A` are specified by the `side` and `uplo` parameters.',

        # C API 信息
        'c_api': 'cublasZsymm_v2',
        'params': ['side', 'uplo', 'm', 'n', 'alpha', 'A', 'lda', 'B', 'ldb', 'beta', 'C', 'ldc'],
        'param_types': {
            'side': 'cublasSideMode_t',
            'uplo': 'cublasFillMode_t',
            'm': 'int',
            'n': 'int',
            'alpha': 'const cuDoubleComplex*',
            'A': 'const cuDoubleComplex*',
            'lda': 'int',
            'B': 'const cuDoubleComplex*',
            'ldb': 'int',
            'beta': 'const cuDoubleComplex*',
            'C': 'cuDoubleComplex*',
            'ldc': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZsymm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* , int ldc);',
    },
    'cublasZsymm_v2_64': {
        'base_op': 'symm',
        'dtype': 'complex128',
        'level': 3,
        'variant': '_64',
        'description': 'The `cublasZsymm_v2_64` function performs a complex double-precision symmetric matrix-matrix multiplication, computing C = αAB + βC or C = αBA + βC, where A is symmetric and B, C are general matrices, with 64-bit integer parameters for large matrix support. It supports both left and right multiplication and can handle upper or lower triangular storage of the symmetric matrix A.',

        # C API 信息
        'c_api': 'cublasZsymm_v2_64',
        'params': ['side', 'uplo', 'm', 'n', 'alpha', 'A', 'lda', 'B', 'ldb', 'beta', 'C', 'ldc'],
        'param_types': {
            'side': 'cublasSideMode_t',
            'uplo': 'cublasFillMode_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'alpha': 'const cuDoubleComplex*',
            'A': 'const cuDoubleComplex*',
            'lda': 'int64_t',
            'B': 'const cuDoubleComplex*',
            'ldb': 'int64_t',
            'beta': 'const cuDoubleComplex*',
            'C': 'cuDoubleComplex*',
            'ldc': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZsymm_v2_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, const cuDoubleComplex* beta, cuDoubleComplex* , int64_t ldc);',
    },
    'cublasZsyr2k_v2': {
        'base_op': 'syr2k',
        'dtype': 'complex128',
        'level': 3,
        'variant': 'base',
        'description': 'The `cublasZsyr2k_v2` function performs a symmetric rank-2k update using complex double-precision matrices, computing `C = alpha*(A*B^T + B*A^T) + beta*C` or `C = alpha*(A^T*B + B^T*A) + beta*C` depending on the transpose operation specified. It operates on symmetric matrix `C` stored in either upper or lower triangular form.',

        # C API 信息
        'c_api': 'cublasZsyr2k_v2',
        'params': ['uplo', 'trans', 'n', 'k', 'alpha', 'A', 'lda', 'B', 'ldb', 'beta', 'C', 'ldc'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'n': 'int',
            'k': 'int',
            'alpha': 'const cuDoubleComplex*',
            'A': 'const cuDoubleComplex*',
            'lda': 'int',
            'B': 'const cuDoubleComplex*',
            'ldb': 'int',
            'beta': 'const cuDoubleComplex*',
            'C': 'cuDoubleComplex*',
            'ldc': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* , int ldc);',
    },
    'cublasZsyr2k_v2_64': {
        'base_op': 'syr2k',
        'dtype': 'complex128',
        'level': 3,
        'variant': '_64',
        'description': 'The `cublasZsyr2k_v2_64` function performs a symmetric rank-2k update using double-precision complex matrices, computing `C = alpha*(A*B^T + B*A^T) + beta*C` or `C = alpha*(A^T*B + B^T*A) + beta*C`, where `C` is symmetric and stored in upper or lower triangular form. It supports 64-bit integers for large matrix dimensions and is part of the Level 3 BLAS operations.',

        # C API 信息
        'c_api': 'cublasZsyr2k_v2_64',
        'params': ['uplo', 'trans', 'n', 'k', 'alpha', 'A', 'lda', 'B', 'ldb', 'beta', 'C', 'ldc'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'n': 'int64_t',
            'k': 'int64_t',
            'alpha': 'const cuDoubleComplex*',
            'A': 'const cuDoubleComplex*',
            'lda': 'int64_t',
            'B': 'const cuDoubleComplex*',
            'ldb': 'int64_t',
            'beta': 'const cuDoubleComplex*',
            'C': 'cuDoubleComplex*',
            'ldc': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZsyr2k_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, const cuDoubleComplex* beta, cuDoubleComplex* , int64_t ldc);',
    },
    'cublasZsyrk_v2': {
        'base_op': 'syrk',
        'dtype': 'complex128',
        'level': 3,
        'variant': 'base',
        'description': 'The `cublasZsyrk_v2` function performs a symmetric rank-k update for complex double-precision matrices, computing `C = alpha*A*A^T + beta*C` or `C = alpha*A^T*A + beta*C`, where `C` is a symmetric matrix. It supports upper or lower triangular storage and matrix transposition.',

        # C API 信息
        'c_api': 'cublasZsyrk_v2',
        'params': ['uplo', 'trans', 'n', 'k', 'alpha', 'A', 'lda', 'beta', 'C', 'ldc'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'n': 'int',
            'k': 'int',
            'alpha': 'const cuDoubleComplex*',
            'A': 'const cuDoubleComplex*',
            'lda': 'int',
            'beta': 'const cuDoubleComplex*',
            'C': 'cuDoubleComplex*',
            'ldc': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* beta, cuDoubleComplex* , int ldc);',
    },
    'cublasZsyrk_v2_64': {
        'base_op': 'syrk',
        'dtype': 'complex128',
        'level': 3,
        'variant': '_64',
        'description': 'The `cublasZsyrk_v2_64` function performs a rank-k update of a complex double-precision symmetric matrix, computing `C = alpha*A*A^T + beta*C` or `C = alpha*A^T*A + beta*C`, where `C` is symmetric and stored in upper or lower triangular form. It supports 64-bit integers for large matrix dimensions and operates on complex double-precision data.',

        # C API 信息
        'c_api': 'cublasZsyrk_v2_64',
        'params': ['uplo', 'trans', 'n', 'k', 'alpha', 'A', 'lda', 'beta', 'C', 'ldc'],
        'param_types': {
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'n': 'int64_t',
            'k': 'int64_t',
            'alpha': 'const cuDoubleComplex*',
            'A': 'const cuDoubleComplex*',
            'lda': 'int64_t',
            'beta': 'const cuDoubleComplex*',
            'C': 'cuDoubleComplex*',
            'ldc': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'C'],
        'scalar_params': ['alpha', 'beta'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZsyrk_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* beta, cuDoubleComplex* , int64_t ldc);',
    },
    'cublasZtrmm_v2': {
        'base_op': 'trmm',
        'dtype': 'complex128',
        'level': 3,
        'variant': 'base',
        'description': 'The `cublasZtrmm_v2` function performs a triangular matrix-matrix multiplication with complex double precision elements, where one input matrix is triangular. It computes the product of a triangular matrix with another matrix, optionally applying a side operation, transpose, and diagonal specification, and scales the result by a complex scalar.',

        # C API 信息
        'c_api': 'cublasZtrmm_v2',
        'params': ['side', 'uplo', 'trans', 'diag', 'm', 'n', 'alpha', 'A', 'lda', 'B', 'ldb', 'C', 'ldc'],
        'param_types': {
            'side': 'cublasSideMode_t',
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'm': 'int',
            'n': 'int',
            'alpha': 'const cuDoubleComplex*',
            'A': 'const cuDoubleComplex*',
            'lda': 'int',
            'B': 'const cuDoubleComplex*',
            'ldb': 'int',
            'C': 'cuDoubleComplex*',
            'ldc': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZtrmm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, cuDoubleComplex* , int ldc);',
    },
    'cublasZtrmm_v2_64': {
        'base_op': 'trmm',
        'dtype': 'complex128',
        'level': 3,
        'variant': '_64',
        'description': 'The `cublasZtrmm_v2_64` function performs a triangular matrix-matrix multiplication with 64-bit integers, where one input matrix is triangular, using double-precision complex numbers. It computes the operation `C = alpha * op(A) * B` or `C = alpha * B * op(A)`, with options for matrix side, fill mode, transpose operation, and diagonal type.',

        # C API 信息
        'c_api': 'cublasZtrmm_v2_64',
        'params': ['side', 'uplo', 'trans', 'diag', 'm', 'n', 'alpha', 'A', 'lda', 'B', 'ldb', 'C', 'ldc'],
        'param_types': {
            'side': 'cublasSideMode_t',
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'alpha': 'const cuDoubleComplex*',
            'A': 'const cuDoubleComplex*',
            'lda': 'int64_t',
            'B': 'const cuDoubleComplex*',
            'ldb': 'int64_t',
            'C': 'cuDoubleComplex*',
            'ldc': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'B', 'C'],
        'scalar_params': ['alpha'],
        'inout_params': ['C'],
        'return_params': ['C'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZtrmm_v2_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, cuDoubleComplex* , int64_t ldc);',
    },
    'cublasZtrsm_v2': {
        'base_op': 'trsm',
        'dtype': 'complex128',
        'level': 3,
        'variant': 'base',
        'description': 'The `cublasZtrsm_v2` function solves a complex double-precision triangular matrix equation, either AX = αB or XA = αB, where A is a triangular matrix and X and B are general matrices. It supports configurable side (left/right), uplo (upper/lower), transposition, and diagonal unit options.',

        # C API 信息
        'c_api': 'cublasZtrsm_v2',
        'params': ['side', 'uplo', 'trans', 'diag', 'm', 'n', 'alpha', 'A', 'lda', 'B', 'ldb'],
        'param_types': {
            'side': 'cublasSideMode_t',
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'm': 'int',
            'n': 'int',
            'alpha': 'const cuDoubleComplex*',
            'A': 'const cuDoubleComplex*',
            'lda': 'int',
            'B': 'cuDoubleComplex*',
            'ldb': 'int'
        },

        # 参数分类
        'tensor_params': ['A', 'B'],
        'scalar_params': ['alpha'],
        'inout_params': ['B'],
        'return_params': ['B'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZtrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, cuDoubleComplex* B, int ldb);',
    },
    'cublasZtrsm_v2_64': {
        'base_op': 'trsm',
        'dtype': 'complex128',
        'level': 3,
        'variant': '_64',
        'description': 'The `cublasZtrsm_v2_64` function solves a complex double-precision triangular system of equations with 64-bit integer parameters, performing the operation B = α * op(A)^{-1} * B or B = α * B * op(A)^{-1}, where A is a triangular matrix and B is a general matrix. It supports configurable side, uplo, transposition, and diagonal options for matrix A.',

        # C API 信息
        'c_api': 'cublasZtrsm_v2_64',
        'params': ['side', 'uplo', 'trans', 'diag', 'm', 'n', 'alpha', 'A', 'lda', 'B', 'ldb'],
        'param_types': {
            'side': 'cublasSideMode_t',
            'uplo': 'cublasFillMode_t',
            'trans': 'cublasOperation_t',
            'diag': 'cublasDiagType_t',
            'm': 'int64_t',
            'n': 'int64_t',
            'alpha': 'const cuDoubleComplex*',
            'A': 'const cuDoubleComplex*',
            'lda': 'int64_t',
            'B': 'cuDoubleComplex*',
            'ldb': 'int64_t'
        },

        # 参数分类
        'tensor_params': ['A', 'B'],
        'scalar_params': ['alpha'],
        'inout_params': ['B'],
        'return_params': ['B'],

        # 完整签名
        'signature': 'cublasStatus_t cublasZtrsm_v2_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, cuDoubleComplex* B, int64_t ldb);',
    },
}
