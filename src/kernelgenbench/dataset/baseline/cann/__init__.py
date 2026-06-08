# CANN Baseline Functions (Ascend NPU)
# All 50 baselines: 44 use ctypes CANN aclnn C API, 6 use PyTorch fallback.
try:
    # --- Level 1: Vector operations (9) ---
    from .cublasDcopy_v2 import cublasDcopy_v2
    from .cublasCcopy_v2 import cublasCcopy_v2
    from .cublasSdot_v2 import cublasSdot_v2
    from .cublasCdotu_v2 import cublasCdotu_v2
    from .cublasZdotc_v2 import cublasZdotc_v2
    from .cublasSaxpy_v2 import cublasSaxpy_v2
    from .cublasDaxpy_v2 import cublasDaxpy_v2
    from .cublasSscal_v2 import cublasSscal_v2
    from .cublasDasum_v2 import cublasDasum_v2

    # --- Level 2: Matrix-vector operations (13) ---
    from .cublasSger_v2 import cublasSger_v2
    from .cublasCgeru_v2 import cublasCgeru_v2
    from .cublasZgerc_v2 import cublasZgerc_v2
    from .cublasDgemv_v2 import cublasDgemv_v2
    from .cublasCgemv_v2 import cublasCgemv_v2
    from .cublasSgemvBatched import cublasSgemvBatched
    from .cublasSgemvStridedBatched import cublasSgemvStridedBatched
    from .cublasDgemvBatched import cublasDgemvBatched
    from .cublasDgemvStridedBatched import cublasDgemvStridedBatched
    from .cublasCgemvBatched_64 import cublasCgemvBatched_64
    from .cublasCgemvStridedBatched import cublasCgemvStridedBatched
    from .cublasZgemvBatched import cublasZgemvBatched
    from .cublasZgemvStridedBatched import cublasZgemvStridedBatched

    # --- Level 3: Matrix-matrix operations (22) ---
    from .cublasSgemm_v2 import cublasSgemm_v2
    from .cublasCgemm_v2 import cublasCgemm_v2
    from .cublasSgemmEx import cublasSgemmEx
    from .cublasSgemmStridedBatched import cublasSgemmStridedBatched
    from .cublasDgemmStridedBatched import cublasDgemmStridedBatched
    from .cublasDgemmStridedBatched_64 import cublasDgemmStridedBatched_64
    from .cublasCgemmStridedBatched import cublasCgemmStridedBatched
    from .cublasCgemmStridedBatched_64 import cublasCgemmStridedBatched_64
    from .cublasZgemmStridedBatched import cublasZgemmStridedBatched
    from .cublasHgemmStridedBatched import cublasHgemmStridedBatched
    from .cublasSgemmBatched_64 import cublasSgemmBatched_64
    from .cublasDgemmBatched import cublasDgemmBatched
    from .cublasHgemmBatched import cublasHgemmBatched
    from .cublasZgemmBatched import cublasZgemmBatched
    from .cublasSgeam import cublasSgeam
    from .cublasSdgmm import cublasSdgmm
    from .cublasSsyrk_v2 import cublasSsyrk_v2
    from .cublasCsyrkEx import cublasCsyrkEx
    from .cublasStrsm_v2 import cublasStrsm_v2
    from .cublasStrsv_v2 import cublasStrsv_v2
    from .cublasDtrsmBatched import cublasDtrsmBatched
    from .cublasZtrsmBatched import cublasZtrsmBatched

    # --- PyTorch fallback (6, no CANN C API equivalent) ---
    from .cublasCsymm_v2 import cublasCsymm_v2
    from .cublasCsymv_v2 import cublasCsymv_v2
    from .cublasDsbmv_v2 import cublasDsbmv_v2
    from .cublasDsyr2_v2 import cublasDsyr2_v2
    from .cublasStbmv_v2 import cublasStbmv_v2
    from .cublasZswap_v2 import cublasZswap_v2
except (OSError, ImportError):
    pass
