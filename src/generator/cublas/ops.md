# cuBLAS Operators Test Statistics

Total: 50 operators

| # | Operator | Accuracy Cases | Benchmark Cases |
|---|----------|----------------|-----------------|
| 1 | cublasCcopy_v2 | 99 | 36 |
| 2 | cublasCdotu_v2 | 99 | 36 |
| 3 | cublasCgemmStridedBatched | 576 | 576 |
| 4 | cublasCgemmStridedBatched_64 | 576 | 576 |
| 5 | cublasCgemm_v2 | 960 | 960 |
| 6 | cublasCgemvBatched_64 | 360 | 360 |
| 7 | cublasCgemvStridedBatched | 96 | 96 |
| 8 | cublasCgemv_v2 | 288 | 288 |
| 9 | cublasCgeru_v2 | 60 | 30 |
| 10 | cublasCsymm_v2 | 60 | 24 |
| 11 | cublasCsymv_v2 | 60 | 30 |
| 12 | cublasCsyrkEx | 192 | 96 |
| 13 | cublasDasum_v2 | 18 | 9 |
| 14 | cublasDaxpy_v2 | 144 | 72 |
| 15 | cublasDcopy_v2 | 63 | 27 |
| 16 | cublasDgemmBatched | 576 | 576 |
| 17 | cublasDgemmStridedBatched | 768 | 768 |
| 18 | cublasDgemmStridedBatched_64 | 576 | 576 |
| 19 | cublasDgemvBatched | 360 | 360 |
| 20 | cublasDgemvStridedBatched | 96 | 96 |
| 21 | cublasDgemv_v2 | 120 | 120 |
| 22 | cublasDsbmv_v2 | 60 | 30 |
| 23 | cublasDsyr2_v2 | 60 | 30 |
| 24 | cublasDtrsmBatched | 108 | 108 |
| 25 | cublasHgemmBatched | 576 | 576 |
| 26 | cublasHgemmStridedBatched | 768 | 768 |
| 27 | cublasSaxpy_v2 | 810 | 292 |
| 28 | cublasSdgmm | 18 | 9 |
| 29 | cublasSdot_v2 | 99 | 36 |
| 30 | cublasSgeam | 432 | 432 |
| 31 | cublasSgemmBatched_64 | 576 | 576 |
| 32 | cublasSgemmEx | 288 | 288 |
| 33 | cublasSgemmStridedBatched | 768 | 768 |
| 34 | cublasSgemm_v2 | 480 | 480 |
| 35 | cublasSgemvBatched | 512 | 512 |
| 36 | cublasSgemvStridedBatched | 96 | 96 |
| 37 | cublasSger_v2 | 128 | 64 |
| 38 | cublasSscal_v2 | 216 | 96 |
| 39 | cublasSsyrk_v2 | 60 | 30 |
| 40 | cublasStbmv_v2 | 192 | 96 |
| 41 | cublasStrsm_v2 | 96 | 96 |
| 42 | cublasStrsv_v2 | 96 | 48 |
| 43 | cublasZdotc_v2 | 60 | 30 |
| 44 | cublasZgemmBatched | 576 | 576 |
| 45 | cublasZgemmStridedBatched | 576 | 576 |
| 46 | cublasZgemvBatched | 24 | 24 |
| 47 | cublasZgemvStridedBatched | 96 | 96 |
| 48 | cublasZgerc_v2 | 60 | 30 |
| 49 | cublasZswap_v2 | 36 | 18 |
| 50 | cublasZtrsmBatched | 108 | 108 |

**Total Accuracy Cases: 14,122** (reduced from 183,429, -92.3%)
**Total Benchmark Cases: 12,601** (reduced from 171,573, -92.7%)

## Optimization Summary

22 operators optimized, all now <1000 cases per operator
