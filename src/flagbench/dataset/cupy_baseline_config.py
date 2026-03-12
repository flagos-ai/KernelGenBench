"""
CuPy Baseline Configuration
包含 110 个 CuPy 可以实现的 cuBLAS 函数配置

生成时间: 2026-01-29
基于: cublas_ops.md 和 CuPy cuBLAS API
"""

# 数据类型映射
DTYPE_MAP = {
    's': 'float32',
    'd': 'float64',
    'c': 'complex64',
    'z': 'complex128',
    'h': 'float16',
}

# BLAS Level 分类
BLAS_LEVEL = {
    # Level 1 - 向量操作
    'axpy': 1, 'scal': 1, 'dot': 1, 'dotu': 1, 'dotc': 1,
    'nrm2': 1, 'asum': 1,

    # Level 2 - 矩阵-向量操作
    'gemv': 2, 'ger': 2, 'geru': 2, 'gerc': 2, 'sbmv': 2,

    # Level 3 - 矩阵-矩阵操作
    'gemm': 3, 'syrk': 3, 'geam': 3, 'dgmm': 3,
}

# 主配置字典
# 结构: {op_name: config}
BASELINE_CONFIG = {

    # ========================================================================
    # Level 1 - 向量操作
    # ========================================================================

    'saxpy': {
        'level': 1,
        'dtype': 'float32',
        'description': 'Constant times a vector plus a vector: y = alpha * x + y',
        'cupy_api': 'cublas.saxpy',
        'params': ['n', 'alpha', 'x', 'incx', 'y', 'incy'],
        'variants': {
            'base': True,      # cublasSaxpy_v2
            '_64': True,       # cublasSaxpy_v2_64
        }
    },

    'daxpy': {
        'level': 1,
        'dtype': 'float64',
        'description': 'Constant times a vector plus a vector: y = alpha * x + y',
        'cupy_api': 'cublas.daxpy',
        'params': ['n', 'alpha', 'x', 'incx', 'y', 'incy'],
        'variants': {
            'base': True,
            '_64': True,
        }
    },

    'caxpy': {
        'level': 1,
        'dtype': 'complex64',
        'description': 'Constant times a vector plus a vector: y = alpha * x + y',
        'cupy_api': 'cublas.caxpy',
        'params': ['n', 'alpha', 'x', 'incx', 'y', 'incy'],
        'variants': {
            'base': True,
            '_64': True,
        }
    },

    'zaxpy': {
        'level': 1,
        'dtype': 'complex128',
        'description': 'Constant times a vector plus a vector: y = alpha * x + y',
        'cupy_api': 'cublas.zaxpy',
        'params': ['n', 'alpha', 'x', 'incx', 'y', 'incy'],
        'variants': {
            'base': True,
            '_64': True,
        }
    },

    'sasum': {
        'level': 1,
        'dtype': 'float32',
        'description': 'Computes sum of absolute values: result = sum(|x[i]|)',
        'cupy_api': 'cublas.sasum',
        'params': ['n', 'x', 'incx', 'result'],
        'variants': {
            'base': True,
            '_64': True,
        }
    },

    'dasum': {
        'level': 1,
        'dtype': 'float64',
        'description': 'Computes sum of absolute values: result = sum(|x[i]|)',
        'cupy_api': 'cublas.dasum',
        'params': ['n', 'x', 'incx', 'result'],
        'variants': {
            'base': True,
            '_64': True,
        }
    },

    'snrm2': {
        'level': 1,
        'dtype': 'float32',
        'description': 'Computes Euclidean norm: result = sqrt(sum(x[i]^2))',
        'cupy_api': 'cublas.snrm2',
        'params': ['n', 'x', 'incx', 'result'],
        'variants': {
            'base': True,
            '_64': True,
        }
    },

    'dnrm2': {
        'level': 1,
        'dtype': 'float64',
        'description': 'Computes Euclidean norm: result = sqrt(sum(x[i]^2))',
        'cupy_api': 'cublas.dnrm2',
        'params': ['n', 'x', 'incx', 'result'],
        'variants': {
            'base': True,
            '_64': True,
        }
    },

    'sscal': {
        'level': 1,
        'dtype': 'float32',
        'description': 'Scales a vector by a scalar: x = alpha * x',
        'cupy_api': 'cublas.sscal',
        'params': ['n', 'alpha', 'x', 'incx'],
        'variants': {
            'base': True,
            '_64': True,
        }
    },

    'dscal': {
        'level': 1,
        'dtype': 'float64',
        'description': 'Scales a vector by a scalar: x = alpha * x',
        'cupy_api': 'cublas.dscal',
        'params': ['n', 'alpha', 'x', 'incx'],
        'variants': {
            'base': True,
            '_64': True,
        }
    },

    'cscal': {
        'level': 1,
        'dtype': 'complex64',
        'description': 'Scales a vector by a scalar: x = alpha * x',
        'cupy_api': 'cublas.cscal',
        'params': ['n', 'alpha', 'x', 'incx'],
        'variants': {
            'base': True,
            '_64': True,
        }
    },

    'zscal': {
        'level': 1,
        'dtype': 'complex128',
        'description': 'Scales a vector by a scalar: x = alpha * x',
        'cupy_api': 'cublas.zscal',
        'params': ['n', 'alpha', 'x', 'incx'],
        'variants': {
            'base': True,
            '_64': True,
        }
    },

    'sdot': {
        'level': 1,
        'dtype': 'float32',
        'description': 'Computes dot product: result = sum(x[i] * y[i])',
        'cupy_api': 'cublas.sdot',
        'params': ['n', 'x', 'incx', 'y', 'incy', 'result'],
        'variants': {
            'base': True,
            '_64': True,
        }
    },

    'ddot': {
        'level': 1,
        'dtype': 'float64',
        'description': 'Computes dot product: result = sum(x[i] * y[i])',
        'cupy_api': 'cublas.ddot',
        'params': ['n', 'x', 'incx', 'y', 'incy', 'result'],
        'variants': {
            'base': True,
            '_64': True,
        }
    },

    'cdotu': {
        'level': 1,
        'dtype': 'complex64',
        'description': 'Computes complex dot product (unconjugated): result = sum(x[i] * y[i])',
        'cupy_api': 'cublas.cdotu',
        'params': ['n', 'x', 'incx', 'y', 'incy', 'result'],
        'variants': {
            'base': True,
            '_64': True,
        }
    },

    'cdotc': {
        'level': 1,
        'dtype': 'complex64',
        'description': 'Computes complex dot product (conjugated): result = sum(conj(x[i]) * y[i])',
        'cupy_api': 'cublas.cdotc',
        'params': ['n', 'x', 'incx', 'y', 'incy', 'result'],
        'variants': {
            'base': True,
            '_64': True,
        }
    },

    'zdotu': {
        'level': 1,
        'dtype': 'complex128',
        'description': 'Computes complex dot product (unconjugated): result = sum(x[i] * y[i])',
        'cupy_api': 'cublas.zdotu',
        'params': ['n', 'x', 'incx', 'y', 'incy', 'result'],
        'variants': {
            'base': True,
            '_64': True,
        }
    },

    'zdotc': {
        'level': 1,
        'dtype': 'complex128',
        'description': 'Computes complex dot product (conjugated): result = sum(conj(x[i]) * y[i])',
        'cupy_api': 'cublas.zdotc',
        'params': ['n', 'x', 'incx', 'y', 'incy', 'result'],
        'variants': {
            'base': True,
            '_64': True,
        }
    },

    # ========================================================================
    # Level 2 - 矩阵-向量操作
    # ========================================================================

    'sgemv': {
        'level': 2,
        'dtype': 'float32',
        'description': 'Matrix-vector multiplication: y = alpha * op(A) * x + beta * y',
        'cupy_api': 'cublas.sgemv',
        'params': ['trans', 'm', 'n', 'alpha', 'A', 'lda', 'x', 'incx', 'beta', 'y', 'incy'],
        'variants': {
            'base': True,
            '_64': True,
        }
    },

    'dgemv': {
        'level': 2,
        'dtype': 'float64',
        'description': 'Matrix-vector multiplication: y = alpha * op(A) * x + beta * y',
        'cupy_api': 'cublas.dgemv',
        'params': ['trans', 'm', 'n', 'alpha', 'A', 'lda', 'x', 'incx', 'beta', 'y', 'incy'],
        'variants': {
            'base': True,
            '_64': True,
        }
    },

    'cgemv': {
        'level': 2,
        'dtype': 'complex64',
        'description': 'Matrix-vector multiplication: y = alpha * op(A) * x + beta * y',
        'cupy_api': 'cublas.cgemv',
        'params': ['trans', 'm', 'n', 'alpha', 'A', 'lda', 'x', 'incx', 'beta', 'y', 'incy'],
        'variants': {
            'base': True,
            '_64': True,
        }
    },

    'zgemv': {
        'level': 2,
        'dtype': 'complex128',
        'description': 'Matrix-vector multiplication: y = alpha * op(A) * x + beta * y',
        'cupy_api': 'cublas.zgemv',
        'params': ['trans', 'm', 'n', 'alpha', 'A', 'lda', 'x', 'incx', 'beta', 'y', 'incy'],
        'variants': {
            'base': True,
            '_64': True,
        }
    },

    'sger': {
        'level': 2,
        'dtype': 'float32',
        'description': 'Rank-1 update: A = alpha * x * y^T + A',
        'cupy_api': 'cublas.sger',
        'params': ['m', 'n', 'alpha', 'x', 'incx', 'y', 'incy', 'A', 'lda'],
        'variants': {
            'base': True,
            '_64': True,
        }
    },

    'dger': {
        'level': 2,
        'dtype': 'float64',
        'description': 'Rank-1 update: A = alpha * x * y^T + A',
        'cupy_api': 'cublas.dger',
        'params': ['m', 'n', 'alpha', 'x', 'incx', 'y', 'incy', 'A', 'lda'],
        'variants': {
            'base': True,
            '_64': True,
        }
    },

    'cgeru': {
        'level': 2,
        'dtype': 'complex64',
        'description': 'Complex rank-1 update (unconjugated): A = alpha * x * y^T + A',
        'cupy_api': 'cublas.cgeru',
        'params': ['m', 'n', 'alpha', 'x', 'incx', 'y', 'incy', 'A', 'lda'],
        'variants': {
            'base': True,
            '_64': True,
        }
    },

    'cgerc': {
        'level': 2,
        'dtype': 'complex64',
        'description': 'Complex rank-1 update (conjugated): A = alpha * x * conj(y)^T + A',
        'cupy_api': 'cublas.cgerc',
        'params': ['m', 'n', 'alpha', 'x', 'incx', 'y', 'incy', 'A', 'lda'],
        'variants': {
            'base': True,
            '_64': True,
        }
    },

    'zgeru': {
        'level': 2,
        'dtype': 'complex128',
        'description': 'Complex rank-1 update (unconjugated): A = alpha * x * y^T + A',
        'cupy_api': 'cublas.zgeru',
        'params': ['m', 'n', 'alpha', 'x', 'incx', 'y', 'incy', 'A', 'lda'],
        'variants': {
            'base': True,
            '_64': True,
        }
    },

    'zgerc': {
        'level': 2,
        'dtype': 'complex128',
        'description': 'Complex rank-1 update (conjugated): A = alpha * x * conj(y)^T + A',
        'cupy_api': 'cublas.zgerc',
        'params': ['m', 'n', 'alpha', 'x', 'incx', 'y', 'incy', 'A', 'lda'],
        'variants': {
            'base': True,
            '_64': True,
        }
    },

    'ssbmv': {
        'level': 2,
        'dtype': 'float32',
        'description': 'Symmetric banded matrix-vector multiplication: y = alpha * A * x + beta * y',
        'cupy_api': 'cublas.ssbmv',
        'params': ['uplo', 'n', 'k', 'alpha', 'A', 'lda', 'x', 'incx', 'beta', 'y', 'incy'],
        'variants': {
            'base': True,
            '_64': True,
        }
    },

    'dsbmv': {
        'level': 2,
        'dtype': 'float64',
        'description': 'Symmetric banded matrix-vector multiplication: y = alpha * A * x + beta * y',
        'cupy_api': 'cublas.dsbmv',
        'params': ['uplo', 'n', 'k', 'alpha', 'A', 'lda', 'x', 'incx', 'beta', 'y', 'incy'],
        'variants': {
            'base': True,
            '_64': True,
        }
    },


    # ========================================================================
    # Level 3 - 矩阵-矩阵操作 (GEMM 系列)
    # ========================================================================

    'sgemm': {
        'level': 3,
        'dtype': 'float32',
        'description': 'General matrix-matrix multiplication: C = alpha * op(A) * op(B) + beta * C',
        'cupy_api': 'cublas.gemm',
        'cupy_batched_api': 'cublas.sgemmBatched',
        'cupy_strided_batched_api': 'cublas.sgemmStridedBatched',
        'params': ['transa', 'transb', 'm', 'n', 'k', 'alpha', 'A', 'lda', 'B', 'ldb', 'beta', 'C', 'ldc'],
        'variants': {
            'base': True,              # cublasSgemm_v2
            '_64': True,               # cublasSgemm_v2_64
            'Batched': True,           # cublasSgemmBatched
            'Batched_64': True,        # cublasSgemmBatched_64
            'StridedBatched': True,    # cublasSgemmStridedBatched
            'StridedBatched_64': True, # cublasSgemmStridedBatched_64
        }
    },

    'dgemm': {
        'level': 3,
        'dtype': 'float64',
        'description': 'General matrix-matrix multiplication: C = alpha * op(A) * op(B) + beta * C',
        'cupy_api': 'cublas.gemm',
        'cupy_batched_api': 'cublas.dgemmBatched',
        'cupy_strided_batched_api': 'cublas.dgemmStridedBatched',
        'params': ['transa', 'transb', 'm', 'n', 'k', 'alpha', 'A', 'lda', 'B', 'ldb', 'beta', 'C', 'ldc'],
        'variants': {
            'base': True,
            '_64': True,
            'Batched': True,
            'Batched_64': True,
            'StridedBatched': True,
            'StridedBatched_64': True,
        }
    },

    'cgemm': {
        'level': 3,
        'dtype': 'complex64',
        'description': 'General matrix-matrix multiplication: C = alpha * op(A) * op(B) + beta * C',
        'cupy_api': 'cublas.gemm',
        'cupy_batched_api': 'cublas.cgemmBatched',
        'cupy_strided_batched_api': 'cublas.cgemmStridedBatched',
        'params': ['transa', 'transb', 'm', 'n', 'k', 'alpha', 'A', 'lda', 'B', 'ldb', 'beta', 'C', 'ldc'],
        'variants': {
            'base': True,
            '_64': True,
            'Batched': True,
            'Batched_64': True,
            'StridedBatched': True,
            'StridedBatched_64': True,
        }
    },

    'zgemm': {
        'level': 3,
        'dtype': 'complex128',
        'description': 'General matrix-matrix multiplication: C = alpha * op(A) * op(B) + beta * C',
        'cupy_api': 'cublas.gemm',
        'cupy_batched_api': 'cublas.zgemmBatched',
        'cupy_strided_batched_api': 'cublas.zgemmStridedBatched',
        'params': ['transa', 'transb', 'm', 'n', 'k', 'alpha', 'A', 'lda', 'B', 'ldb', 'beta', 'C', 'ldc'],
        'variants': {
            'base': True,
            '_64': True,
            'Batched': True,
            'Batched_64': True,
            'StridedBatched': True,
            'StridedBatched_64': True,
        }
    },


    # ========================================================================
    # Level 3 - 矩阵-矩阵操作 (SYRK 系列)
    # ========================================================================

    'ssyrk': {
        'level': 3,
        'dtype': 'float32',
        'description': 'Symmetric rank-k update: C = alpha * A * A^T + beta * C',
        'cupy_api': 'cublas.ssyrk',
        'params': ['uplo', 'trans', 'n', 'k', 'alpha', 'A', 'lda', 'beta', 'C', 'ldc'],
        'variants': {
            'base': True,
            '_64': True,
        }
    },

    'dsyrk': {
        'level': 3,
        'dtype': 'float64',
        'description': 'Symmetric rank-k update: C = alpha * A * A^T + beta * C',
        'cupy_api': 'cublas.dsyrk',
        'params': ['uplo', 'trans', 'n', 'k', 'alpha', 'A', 'lda', 'beta', 'C', 'ldc'],
        'variants': {
            'base': True,
            '_64': True,
        }
    },

    'csyrk': {
        'level': 3,
        'dtype': 'complex64',
        'description': 'Symmetric rank-k update: C = alpha * A * A^T + beta * C',
        'cupy_api': 'cublas.csyrk',
        'params': ['uplo', 'trans', 'n', 'k', 'alpha', 'A', 'lda', 'beta', 'C', 'ldc'],
        'variants': {
            'base': True,
            '_64': True,
        }
    },

    'zsyrk': {
        'level': 3,
        'dtype': 'complex128',
        'description': 'Symmetric rank-k update: C = alpha * A * A^T + beta * C',
        'cupy_api': 'cublas.zsyrk',
        'params': ['uplo', 'trans', 'n', 'k', 'alpha', 'A', 'lda', 'beta', 'C', 'ldc'],
        'variants': {
            'base': True,
            '_64': True,
        }
    },


    # ========================================================================
    # Level 3 - 矩阵-矩阵操作 (GEAM 系列)
    # ========================================================================

    'sgeam': {
        'level': 3,
        'dtype': 'float32',
        'description': 'Matrix-matrix addition: C = alpha * op(A) + beta * op(B)',
        'cupy_api': 'cublas.sgeam',
        'params': ['transa', 'transb', 'm', 'n', 'alpha', 'A', 'lda', 'beta', 'B', 'ldb', 'C', 'ldc'],
        'variants': {
            'base': True,
            '_64': True,
        }
    },

    'dgeam': {
        'level': 3,
        'dtype': 'float64',
        'description': 'Matrix-matrix addition: C = alpha * op(A) + beta * op(B)',
        'cupy_api': 'cublas.dgeam',
        'params': ['transa', 'transb', 'm', 'n', 'alpha', 'A', 'lda', 'beta', 'B', 'ldb', 'C', 'ldc'],
        'variants': {
            'base': True,
            '_64': True,
        }
    },

    'cgeam': {
        'level': 3,
        'dtype': 'complex64',
        'description': 'Matrix-matrix addition: C = alpha * op(A) + beta * op(B)',
        'cupy_api': 'cublas.cgeam',
        'params': ['transa', 'transb', 'm', 'n', 'alpha', 'A', 'lda', 'beta', 'B', 'ldb', 'C', 'ldc'],
        'variants': {
            'base': True,
            '_64': True,
        }
    },

    'zgeam': {
        'level': 3,
        'dtype': 'complex128',
        'description': 'Matrix-matrix addition: C = alpha * op(A) + beta * op(B)',
        'cupy_api': 'cublas.zgeam',
        'params': ['transa', 'transb', 'm', 'n', 'alpha', 'A', 'lda', 'beta', 'B', 'ldb', 'C', 'ldc'],
        'variants': {
            'base': True,
            '_64': True,
        }
    },


    # ========================================================================
    # Level 3 - 矩阵-矩阵操作 (DGMM 系列)
    # ========================================================================

    'sdgmm': {
        'level': 3,
        'dtype': 'float32',
        'description': 'Diagonal matrix multiplication: C = A * diag(x) or C = diag(x) * A',
        'cupy_api': 'cublas.sdgmm',
        'params': ['mode', 'm', 'n', 'A', 'lda', 'x', 'incx', 'C', 'ldc'],
        'variants': {
            'base': True,
            '_64': True,
        }
    },

    'ddgmm': {
        'level': 3,
        'dtype': 'float64',
        'description': 'Diagonal matrix multiplication: C = A * diag(x) or C = diag(x) * A',
        'cupy_api': 'cublas.ddgmm',
        'params': ['mode', 'm', 'n', 'A', 'lda', 'x', 'incx', 'C', 'ldc'],
        'variants': {
            'base': True,
            '_64': True,
        }
    },

    'cdgmm': {
        'level': 3,
        'dtype': 'complex64',
        'description': 'Diagonal matrix multiplication: C = A * diag(x) or C = diag(x) * A',
        'cupy_api': 'cublas.cdgmm',
        'params': ['mode', 'm', 'n', 'A', 'lda', 'x', 'incx', 'C', 'ldc'],
        'variants': {
            'base': True,
            '_64': True,
        }
    },

    'zdgmm': {
        'level': 3,
        'dtype': 'complex128',
        'description': 'Diagonal matrix multiplication: C = A * diag(x) or C = diag(x) * A',
        'cupy_api': 'cublas.zdgmm',
        'params': ['mode', 'm', 'n', 'A', 'lda', 'x', 'incx', 'C', 'ldc'],
        'variants': {
            'base': True,
            '_64': True,
        }
    },

}


# ============================================================================
# 辅助函数
# ============================================================================

def get_all_ops():
    """获取所有算子名称"""
    return list(BASELINE_CONFIG.keys())

def get_ops_by_level(level):
    """按 BLAS Level 获取算子"""
    return [op for op, cfg in BASELINE_CONFIG.items() if cfg['level'] == level]

def get_ops_with_batched():
    """获取支持 Batched 变体的算子"""
    return [op for op, cfg in BASELINE_CONFIG.items() 
            if cfg.get('variants', {}).get('Batched', False)]

def get_ops_with_strided_batched():
    """获取支持 StridedBatched 变体的算子"""
    return [op for op, cfg in BASELINE_CONFIG.items() 
            if cfg.get('variants', {}).get('StridedBatched', False)]

def get_total_functions():
    """统计总共可以生成多少个 cuBLAS 函数"""
    total = 0
    for op, cfg in BASELINE_CONFIG.items():
        variants = cfg.get('variants', {})
        total += sum(1 for v in variants.values() if v)
    return total

def print_summary():
    """打印配置摘要"""
    print("=" * 70)
    print("CuPy Baseline Configuration Summary")
    print("=" * 70)
    print(f"Total operators: {len(BASELINE_CONFIG)}")
    print(f"Total functions: {get_total_functions()}")
    print()
    print(f"Level 1 operators: {len(get_ops_by_level(1))}")
    print(f"Level 2 operators: {len(get_ops_by_level(2))}")
    print(f"Level 3 operators: {len(get_ops_by_level(3))}")
    print()
    print(f"Operators with Batched: {len(get_ops_with_batched())}")
    print(f"Operators with StridedBatched: {len(get_ops_with_strided_batched())}")
    print("=" * 70)

if __name__ == '__main__':
    print_summary()
    
    # 示例：打印所有支持 StridedBatched 的算子
    print("\nOperators with StridedBatched support:")
    for op in get_ops_with_strided_batched():
        print(f"  - {op}")
