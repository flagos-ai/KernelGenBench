# CuPy cuBLAS API Coverage Report (实际测试)

**生成时间**: Mon Jan 19 17:14:21 CST 2026

## 总览

- **总cuBLAS函数数**: 239
- **有CuPy包装**: 47 (19.7%)
- **无CuPy包装**: 192 (80.3%)
- **CuPy实际支持的函数**: 21 个

## CuPy实际支持的函数

```
asum, axpy, batched_gesv, dgmm, dot, dotc, dotu, geam, gemm, gemv, ger, gerc, geru, get_batched_gesv_limit, iamax, iamin, nrm2, sbmv, scal, set_batched_gesv_limit, syrk
```

## 按操作分类

| 操作 | 总数 | 有CuPy | 无CuPy | 覆盖率 | 状态 |
|------|------|--------|--------|--------|------|
| amaxex          |   1 |   0 |   1 |   0.0% | ❌ |
| aminex          |   1 |   0 |   1 |   0.0% | ❌ |
| asum            |   2 |   2 |   0 | 100.0% | ✅ |
| asumex          |   1 |   0 |   1 |   0.0% | ❌ |
| axpy            |   4 |   4 |   0 | 100.0% | ✅ |
| axpyex          |   1 |   0 |   1 |   0.0% | ❌ |
| calex           |   1 |   0 |   1 |   0.0% | ❌ |
| casum           |   1 |   0 |   1 |   0.0% | ❌ |
| cnrm2           |   1 |   0 |   1 |   0.0% | ❌ |
| copy            |   4 |   0 |   4 |   0.0% | ❌ |
| dgmm            |   4 |   4 |   0 | 100.0% | ✅ |
| dot             |   2 |   2 |   0 | 100.0% | ✅ |
| dotc            |   2 |   2 |   0 | 100.0% | ✅ |
| dotu            |   2 |   2 |   0 | 100.0% | ✅ |
| drot            |   1 |   0 |   1 |   0.0% | ❌ |
| dscal           |   1 |   0 |   1 |   0.0% | ❌ |
| etloggercallback |   1 |   0 |   1 |   0.0% | ❌ |
| gbmv            |   4 |   0 |   4 |   0.0% | ❌ |
| geam            |   4 |   4 |   0 | 100.0% | ✅ |
| gemm            |   5 |   5 |   0 | 100.0% | ✅ |
| gemm3m          |   2 |   0 |   2 |   0.0% | ❌ |
| gemm3mbatched   |   1 |   0 |   1 |   0.0% | ❌ |
| gemm3mex        |   1 |   0 |   1 |   0.0% | ❌ |
| gemm3mstridedbatched |   1 |   0 |   1 |   0.0% | ❌ |
| gemmbatched     |   5 |   0 |   5 |   0.0% | ❌ |
| gemmbatchedex   |   1 |   0 |   1 |   0.0% | ❌ |
| gemmex          |   3 |   0 |   3 |   0.0% | ❌ |
| gemmgroupedbatched |   2 |   0 |   2 |   0.0% | ❌ |
| gemmstridedbatched |   5 |   0 |   5 |   0.0% | ❌ |
| gemmstridedbatchedex |   1 |   0 |   1 |   0.0% | ❌ |
| gemv            |   4 |   4 |   0 | 100.0% | ✅ |
| gemvbatched     |   4 |   0 |   4 |   0.0% | ❌ |
| gemvstridedbatched |   4 |   0 |   4 |   0.0% | ❌ |
| ger             |   2 |   2 |   0 | 100.0% | ✅ |
| gerc            |   2 |   2 |   0 | 100.0% | ✅ |
| geru            |   2 |   2 |   0 | 100.0% | ✅ |
| getloggercallback |   1 |   0 |   1 |   0.0% | ❌ |
| getrfbatched    |   4 |   0 |   4 |   0.0% | ❌ |
| getribatched    |   4 |   0 |   4 |   0.0% | ❌ |
| getrsbatched    |   4 |   0 |   4 |   0.0% | ❌ |
| hbmv            |   2 |   0 |   2 |   0.0% | ❌ |
| hemm            |   2 |   0 |   2 |   0.0% | ❌ |
| hemv            |   2 |   0 |   2 |   0.0% | ❌ |
| her             |   2 |   0 |   2 |   0.0% | ❌ |
| her2            |   2 |   0 |   2 |   0.0% | ❌ |
| her2k           |   2 |   0 |   2 |   0.0% | ❌ |
| herk            |   2 |   0 |   2 |   0.0% | ❌ |
| herk3mex        |   1 |   0 |   1 |   0.0% | ❌ |
| herkex          |   1 |   0 |   1 |   0.0% | ❌ |
| herkx           |   2 |   0 |   2 |   0.0% | ❌ |
| hpmv            |   2 |   0 |   2 |   0.0% | ❌ |
| hpr             |   2 |   0 |   2 |   0.0% | ❌ |
| hpr2            |   2 |   0 |   2 |   0.0% | ❌ |
| loggerconfigure |   1 |   0 |   1 |   0.0% | ❌ |
| nrm2            |   2 |   2 |   0 | 100.0% | ✅ |
| nrm2ex          |   1 |   0 |   1 |   0.0% | ❌ |
| opyex           |   1 |   0 |   1 |   0.0% | ❌ |
| otcex           |   1 |   0 |   1 |   0.0% | ❌ |
| otex            |   1 |   0 |   1 |   0.0% | ❌ |
| rot             |   4 |   0 |   4 |   0.0% | ❌ |
| rotex           |   1 |   0 |   1 |   0.0% | ❌ |
| rotg            |   4 |   0 |   4 |   0.0% | ❌ |
| rotgex          |   1 |   0 |   1 |   0.0% | ❌ |
| rotm            |   2 |   0 |   2 |   0.0% | ❌ |
| rotmex          |   1 |   0 |   1 |   0.0% | ❌ |
| rotmg           |   2 |   0 |   2 |   0.0% | ❌ |
| rotmgex         |   1 |   0 |   1 |   0.0% | ❌ |
| sbmv            |   2 |   2 |   0 | 100.0% | ✅ |
| scal            |   4 |   4 |   0 | 100.0% | ✅ |
| shgemvbatched   |   1 |   0 |   1 |   0.0% | ❌ |
| shgemvstridedbatched |   1 |   0 |   1 |   0.0% | ❌ |
| spmv            |   2 |   0 |   2 |   0.0% | ❌ |
| spr             |   2 |   0 |   2 |   0.0% | ❌ |
| spr2            |   2 |   0 |   2 |   0.0% | ❌ |
| srot            |   1 |   0 |   1 |   0.0% | ❌ |
| sscal           |   1 |   0 |   1 |   0.0% | ❌ |
| ssgemvbatched   |   1 |   0 |   1 |   0.0% | ❌ |
| ssgemvstridedbatched |   1 |   0 |   1 |   0.0% | ❌ |
| swap            |   4 |   0 |   4 |   0.0% | ❌ |
| symm            |   4 |   0 |   4 |   0.0% | ❌ |
| symv            |   4 |   0 |   4 |   0.0% | ❌ |
| syr             |   4 |   0 |   4 |   0.0% | ❌ |
| syr2            |   4 |   0 |   4 |   0.0% | ❌ |
| syr2k           |   4 |   0 |   4 |   0.0% | ❌ |
| syrk            |   4 |   4 |   0 | 100.0% | ✅ |
| syrk3mex        |   1 |   0 |   1 |   0.0% | ❌ |
| syrkex          |   1 |   0 |   1 |   0.0% | ❌ |
| syrkx           |   4 |   0 |   4 |   0.0% | ❌ |
| tbmv            |   4 |   0 |   4 |   0.0% | ❌ |
| tbsv            |   4 |   0 |   4 |   0.0% | ❌ |
| tpmv            |   4 |   0 |   4 |   0.0% | ❌ |
| tpsv            |   4 |   0 |   4 |   0.0% | ❌ |
| trmm            |   4 |   0 |   4 |   0.0% | ❌ |
| trmv            |   4 |   0 |   4 |   0.0% | ❌ |
| trsm            |   4 |   0 |   4 |   0.0% | ❌ |
| trsmbatched     |   4 |   0 |   4 |   0.0% | ❌ |
| trsv            |   4 |   0 |   4 |   0.0% | ❌ |
| tssgemvbatched  |   1 |   0 |   1 |   0.0% | ❌ |
| tssgemvstridedbatched |   1 |   0 |   1 |   0.0% | ❌ |
| tstgemvbatched  |   1 |   0 |   1 |   0.0% | ❌ |
| tstgemvstridedbatched |   1 |   0 |   1 |   0.0% | ❌ |
| uint8gemmbias   |   1 |   0 |   1 |   0.0% | ❌ |
| wapex           |   1 |   0 |   1 |   0.0% | ❌ |
| zasum           |   1 |   0 |   1 |   0.0% | ❌ |
| znrm2           |   1 |   0 |   1 |   0.0% | ❌ |

## 有CuPy包装的函数

这些函数可以直接调用 `cupy.cublas.<function_name>()`：

| cuBLAS Function | Operation | Data Type |
|----------------|-----------|-----------|
| `cublasCaxpy_v2` | axpy | complex64 |
| `cublasCdgmm` | dgmm | complex64 |
| `cublasCdotc_v2` | dotc | complex64 |
| `cublasCdotu_v2` | dotu | complex64 |
| `cublasCgeam` | geam | complex64 |
| `cublasCgemm_v2` | gemm | complex64 |
| `cublasCgemv_v2` | gemv | complex64 |
| `cublasCgerc_v2` | gerc | complex64 |
| `cublasCgeru_v2` | geru | complex64 |
| `cublasCscal_v2` | scal | complex64 |
| `cublasCsyrk_v2` | syrk | complex64 |
| `cublasDasum_v2` | asum | float64 |
| `cublasDaxpy_v2` | axpy | float64 |
| `cublasDdgmm` | dgmm | float64 |
| `cublasDdot_v2` | dot | float64 |
| `cublasDgeam` | geam | float64 |
| `cublasDgemm_v2` | gemm | float64 |
| `cublasDgemv_v2` | gemv | float64 |
| `cublasDger_v2` | ger | float64 |
| `cublasDnrm2_v2` | nrm2 | float64 |
| `cublasDsbmv_v2` | sbmv | float64 |
| `cublasDscal_v2` | scal | float64 |
| `cublasDsyrk_v2` | syrk | float64 |
| `cublasHgemm` | gemm | float16 |
| `cublasSasum_v2` | asum | float32 |
| `cublasSaxpy_v2` | axpy | float32 |
| `cublasSdgmm` | dgmm | float32 |
| `cublasSdot_v2` | dot | float32 |
| `cublasSgeam` | geam | float32 |
| `cublasSgemm_v2` | gemm | float32 |
| `cublasSgemv_v2` | gemv | float32 |
| `cublasSger_v2` | ger | float32 |
| `cublasSnrm2_v2` | nrm2 | float32 |
| `cublasSsbmv_v2` | sbmv | float32 |
| `cublasSscal_v2` | scal | float32 |
| `cublasSsyrk_v2` | syrk | float32 |
| `cublasZaxpy_v2` | axpy | complex128 |
| `cublasZdgmm` | dgmm | complex128 |
| `cublasZdotc_v2` | dotc | complex128 |
| `cublasZdotu_v2` | dotu | complex128 |
| `cublasZgeam` | geam | complex128 |
| `cublasZgemm_v2` | gemm | complex128 |
| `cublasZgemv_v2` | gemv | complex128 |
| `cublasZgerc_v2` | gerc | complex128 |
| `cublasZgeru_v2` | geru | complex128 |
| `cublasZscal_v2` | scal | complex128 |
| `cublasZsyrk_v2` | syrk | complex128 |

## 无CuPy包装的函数 (192个)

这些函数需要使用CuPy数组操作手动实现。

### 按操作分组

#### amaxex (1个变体)

- `cublasIamaxEx`

#### aminex (1个变体)

- `cublasIaminEx`

#### asumex (1个变体)

- `cublasAsumEx`

#### axpyex (1个变体)

- `cublasAxpyEx`

#### calex (1个变体)

- `cublasScalEx`

#### casum (1个变体)

- `cublasScasum_v2`

#### cnrm2 (1个变体)

- `cublasScnrm2_v2`

#### copy (4个变体)

- `cublasScopy_v2`
- `cublasDcopy_v2`
- `cublasCcopy_v2`
- `cublasZcopy_v2`

#### drot (1个变体)

- `cublasZdrot_v2`

#### dscal (1个变体)

- `cublasZdscal_v2`

#### etloggercallback (1个变体)

- `cublasSetLoggerCallback`

#### gbmv (4个变体)

- `cublasSgbmv_v2`
- `cublasDgbmv_v2`
- `cublasCgbmv_v2`
- `cublasZgbmv_v2`

#### gemm3m (2个变体)

- `cublasCgemm3m`
- `cublasZgemm3m`

#### gemm3mbatched (1个变体)

- `cublasCgemm3mBatched`

#### gemm3mex (1个变体)

- `cublasCgemm3mEx`

#### gemm3mstridedbatched (1个变体)

- `cublasCgemm3mStridedBatched`

#### gemmbatched (5个变体)

- `cublasHgemmBatched`
- `cublasSgemmBatched`
- `cublasDgemmBatched`
- `cublasCgemmBatched`
- `cublasZgemmBatched`

#### gemmbatchedex (1个变体)

- `cublasGemmBatchedEx`

#### gemmex (3个变体)

- `cublasSgemmEx`
- `cublasGemmEx`
- `cublasCgemmEx`

#### gemmgroupedbatched (2个变体)

- `cublasSgemmGroupedBatched`
- `cublasDgemmGroupedBatched`

#### gemmstridedbatched (5个变体)

- `cublasHgemmStridedBatched`
- `cublasSgemmStridedBatched`
- `cublasDgemmStridedBatched`
- `cublasCgemmStridedBatched`
- `cublasZgemmStridedBatched`

#### gemmstridedbatchedex (1个变体)

- `cublasGemmStridedBatchedEx`

#### gemvbatched (4个变体)

- `cublasSgemvBatched`
- `cublasDgemvBatched`
- `cublasCgemvBatched`
- `cublasZgemvBatched`

#### gemvstridedbatched (4个变体)

- `cublasSgemvStridedBatched`
- `cublasDgemvStridedBatched`
- `cublasCgemvStridedBatched`
- `cublasZgemvStridedBatched`

#### getloggercallback (1个变体)

- `cublasGetLoggerCallback`

#### getrfbatched (4个变体)

- `cublasSgetrfBatched`
- `cublasDgetrfBatched`
- `cublasCgetrfBatched`
- `cublasZgetrfBatched`

#### getribatched (4个变体)

- `cublasSgetriBatched`
- `cublasDgetriBatched`
- `cublasCgetriBatched`
- `cublasZgetriBatched`

#### getrsbatched (4个变体)

- `cublasSgetrsBatched`
- `cublasDgetrsBatched`
- `cublasCgetrsBatched`
- `cublasZgetrsBatched`

#### hbmv (2个变体)

- `cublasChbmv_v2`
- `cublasZhbmv_v2`

#### hemm (2个变体)

- `cublasChemm_v2`
- `cublasZhemm_v2`

#### hemv (2个变体)

- `cublasChemv_v2`
- `cublasZhemv_v2`

#### her (2个变体)

- `cublasCher_v2`
- `cublasZher_v2`

#### her2 (2个变体)

- `cublasCher2_v2`
- `cublasZher2_v2`

#### her2k (2个变体)

- `cublasCher2k_v2`
- `cublasZher2k_v2`

#### herk (2个变体)

- `cublasCherk_v2`
- `cublasZherk_v2`

#### herk3mex (1个变体)

- `cublasCherk3mEx`

#### herkex (1个变体)

- `cublasCherkEx`

#### herkx (2个变体)

- `cublasCherkx`
- `cublasZherkx`

#### hpmv (2个变体)

- `cublasChpmv_v2`
- `cublasZhpmv_v2`

#### hpr (2个变体)

- `cublasChpr_v2`
- `cublasZhpr_v2`

#### hpr2 (2个变体)

- `cublasChpr2_v2`
- `cublasZhpr2_v2`

#### loggerconfigure (1个变体)

- `cublasLoggerConfigure`

#### nrm2ex (1个变体)

- `cublasNrm2Ex`

#### opyex (1个变体)

- `cublasCopyEx`

#### otcex (1个变体)

- `cublasDotcEx`

#### otex (1个变体)

- `cublasDotEx`

#### rot (4个变体)

- `cublasSrot_v2`
- `cublasDrot_v2`
- `cublasCrot_v2`
- `cublasZrot_v2`

#### rotex (1个变体)

- `cublasRotEx`

#### rotg (4个变体)

- `cublasSrotg_v2`
- `cublasDrotg_v2`
- `cublasCrotg_v2`
- `cublasZrotg_v2`

#### rotgex (1个变体)

- `cublasRotgEx`

#### rotm (2个变体)

- `cublasSrotm_v2`
- `cublasDrotm_v2`

#### rotmex (1个变体)

- `cublasRotmEx`

#### rotmg (2个变体)

- `cublasSrotmg_v2`
- `cublasDrotmg_v2`

#### rotmgex (1个变体)

- `cublasRotmgEx`

#### shgemvbatched (1个变体)

- `cublasHSHgemvBatched`

#### shgemvstridedbatched (1个变体)

- `cublasHSHgemvStridedBatched`

#### spmv (2个变体)

- `cublasSspmv_v2`
- `cublasDspmv_v2`

#### spr (2个变体)

- `cublasSspr_v2`
- `cublasDspr_v2`

#### spr2 (2个变体)

- `cublasSspr2_v2`
- `cublasDspr2_v2`

#### srot (1个变体)

- `cublasCsrot_v2`

#### sscal (1个变体)

- `cublasCsscal_v2`

#### ssgemvbatched (1个变体)

- `cublasHSSgemvBatched`

#### ssgemvstridedbatched (1个变体)

- `cublasHSSgemvStridedBatched`

#### swap (4个变体)

- `cublasSswap_v2`
- `cublasDswap_v2`
- `cublasCswap_v2`
- `cublasZswap_v2`

#### symm (4个变体)

- `cublasSsymm_v2`
- `cublasDsymm_v2`
- `cublasCsymm_v2`
- `cublasZsymm_v2`

#### symv (4个变体)

- `cublasSsymv_v2`
- `cublasDsymv_v2`
- `cublasCsymv_v2`
- `cublasZsymv_v2`

#### syr (4个变体)

- `cublasSsyr_v2`
- `cublasDsyr_v2`
- `cublasCsyr_v2`
- `cublasZsyr_v2`

#### syr2 (4个变体)

- `cublasSsyr2_v2`
- `cublasDsyr2_v2`
- `cublasCsyr2_v2`
- `cublasZsyr2_v2`

#### syr2k (4个变体)

- `cublasSsyr2k_v2`
- `cublasDsyr2k_v2`
- `cublasCsyr2k_v2`
- `cublasZsyr2k_v2`

#### syrk3mex (1个变体)

- `cublasCsyrk3mEx`

#### syrkex (1个变体)

- `cublasCsyrkEx`

#### syrkx (4个变体)

- `cublasSsyrkx`
- `cublasDsyrkx`
- `cublasCsyrkx`
- `cublasZsyrkx`

#### tbmv (4个变体)

- `cublasStbmv_v2`
- `cublasDtbmv_v2`
- `cublasCtbmv_v2`
- `cublasZtbmv_v2`

#### tbsv (4个变体)

- `cublasStbsv_v2`
- `cublasDtbsv_v2`
- `cublasCtbsv_v2`
- `cublasZtbsv_v2`

#### tpmv (4个变体)

- `cublasStpmv_v2`
- `cublasDtpmv_v2`
- `cublasCtpmv_v2`
- `cublasZtpmv_v2`

#### tpsv (4个变体)

- `cublasStpsv_v2`
- `cublasDtpsv_v2`
- `cublasCtpsv_v2`
- `cublasZtpsv_v2`

#### trmm (4个变体)

- `cublasStrmm_v2`
- `cublasDtrmm_v2`
- `cublasCtrmm_v2`
- `cublasZtrmm_v2`

#### trmv (4个变体)

- `cublasStrmv_v2`
- `cublasDtrmv_v2`
- `cublasCtrmv_v2`
- `cublasZtrmv_v2`

#### trsm (4个变体)

- `cublasStrsm_v2`
- `cublasDtrsm_v2`
- `cublasCtrsm_v2`
- `cublasZtrsm_v2`

#### trsmbatched (4个变体)

- `cublasStrsmBatched`
- `cublasDtrsmBatched`
- `cublasCtrsmBatched`
- `cublasZtrsmBatched`

#### trsv (4个变体)

- `cublasStrsv_v2`
- `cublasDtrsv_v2`
- `cublasCtrsv_v2`
- `cublasZtrsv_v2`

#### tssgemvbatched (1个变体)

- `cublasTSSgemvBatched`

#### tssgemvstridedbatched (1个变体)

- `cublasTSSgemvStridedBatched`

#### tstgemvbatched (1个变体)

- `cublasTSTgemvBatched`

#### tstgemvstridedbatched (1个变体)

- `cublasTSTgemvStridedBatched`

#### uint8gemmbias (1个变体)

- `cublasUint8gemmBias`

#### wapex (1个变体)

- `cublasSwapEx`

#### zasum (1个变体)

- `cublasDzasum_v2`

#### znrm2 (1个变体)

- `cublasDznrm2_v2`


## 代码生成建议

### 有CuPy包装的操作（直接调用）

```python
from cupy import cublas

# Level 1 BLAS
cublas.axpy(alpha, x, y)           # y = alpha*x + y
cublas.dot(x, y)                   # x.T @ y
cublas.scal(alpha, x)              # x = alpha*x
cublas.nrm2(x)                     # ||x||_2
cublas.asum(x)                     # sum(|x_i|)

# Level 2 BLAS
cublas.gemv(trans, A, x)           # y = alpha*op(A)@x + beta*y
cublas.ger(A, x, y)                # A = alpha*x@y.T + A
cublas.geru(A, x, y)               # A = alpha*x@y.T + A (unconjugated)
cublas.gerc(A, x, y)               # A = alpha*conj(x)@y.T + A

# Level 3 BLAS
cublas.gemm(transa, transb, A, B)  # C = alpha*op(A)@op(B) + beta*C
cublas.syrk(uplo, trans, A)        # C = alpha*A@A.T + beta*C
```

### 无CuPy包装的操作（手动实现）

对于没有CuPy包装的操作，使用CuPy数组操作：

```python
import cupy as cp

# 例如：copy操作
y = x.copy()

# 例如：swap操作  
y[:], x[:] = x.copy(), y.copy()

# 例如：rot操作（Givens旋转）
r = cp.sqrt(x**2 + y**2)
cos_val = x / r
sin_val = y / r
x_new = cos_val * x + sin_val * y
y_new = -sin_val * x + cos_val * y
```

## 重要发现

1. **CuPy只包装了最常用的21个cuBLAS函数**
2. **大量变体（Ex版本、batched版本）没有被包装**
3. **某些操作（如copy, swap, rot）虽然常用，但CuPy未直接包装**
4. **建议策略**：
   - 先生成有CuPy包装的函数（高质量baseline）
   - 再处理无CuPy包装的函数（使用CuPy数组操作）
