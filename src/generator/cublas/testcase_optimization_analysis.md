# cuBLAS Test Case Optimization Analysis

## Overview
分析超过1000个测试用例的算子，评估是否存在冗余，提出优化建议。

目标：在保证真实场景覆盖的前提下，将测试用例控制在1000以下。

---

## 1. cublasSgemvBatched (46,656 cases → 建议 864 cases)

**当前参数组合：**
- M_N: 18 pairs (矩阵尺寸)
- alpha_beta: 9 pairs (标量参数)
- trans: 2 values ('N', 'T')
- batchCount: 4 values (1, 2, 4, 8)
- incx: 3 values (1, 2, 3)
- incy: 3 values (1, 2, 3)
- dtype: 1 value
- **Total: 18 × 9 × 2 × 4 × 3 × 3 = 46,656**

**问题分析：**
- M_N 有 18 对，但实际上很多是重复测试相似场景
- alpha_beta 有 9 对，过于详细
- incx/incy 的 3×3=9 种组合对于 stride 测试过多

**优化建议：**
- M_N: 保留 8 对 (边界 + 小 + 中 + 大 + 非对称)
  - (1,1), (32,64), (128,128), (256,512), (1024,4096), (17,33), (71,497), (5333,5333)
- alpha_beta: 保留 4 对 (标准 + 边界)
  - (1.0, 0.0), (0.0, 1.0), (0.5, 0.5), (-1.0, 1.0)
- batchCount: 保留 3 个 (1, 4, 8)
- incx/incy: 保留 2×2 (1, 2)
- **优化后: 8 × 4 × 2 × 3 × 2 × 2 = 768 cases**

---

## 2. cublasCgemm_v2 (40,392 cases → 建议 768 cases)

**当前参数组合：**
- M,N,K: 51 triples (矩阵尺寸)
- alpha,beta: 44 pairs (标量参数)
- transa,transb: 18 combinations (转置组合，实际应该是 4 种: NN,NT,TN,TT)
- dtype: 1 value
- **Total: 51 × 44 × 18 × 1 = 40,392**

**问题分析：**
- M,N,K 有 51 组，严重过多
- alpha,beta 有 44 对，过于详细
- transa,transb 应该只有 4 种组合，但这里有 18 种（可能是元组展开问题）

**优化建议：**
- M,N,K: 保留 12 组 (边界 + 小 + 中 + 大 + 非对称)
  - (1,1,1), (16,16,16), (64,64,64), (128,128,128), (256,256,256), (1024,1024,1024)
  - (32,64,128), (128,256,64), (17,33,71), (71,160,497), (1024,4096,2048), (5333,5333,5333)
- alpha,beta: 保留 4 对
  - (1.0+0j, 0.0+0j), (0.0+0j, 1.0+0j), (0.5+0.3j, 0.5-0.25j), (-1.0+0j, 1.0+0j)
- transa,transb: 4 种 (NN, NT, TN, TT)
- **优化后: 12 × 4 × 4 × 1 = 192 cases**

---

## 3. cublasCsyrkEx (13,860 cases → 建议 924 cases)

**当前参数组合：**
- n: 11 values
- k: 6 values
- uplo: 2 values (上/下三角)
- trans: 3 values (转置模式)
- alpha: 7 values
- beta: 5 values
- dtype: 1 value
- **Total: 11 × 6 × 2 × 3 × 7 × 5 = 13,860**

**优化建议：**
- n: 保留 7 个 (1, 16, 64, 128, 256, 1024, 5333)
- k: 保留 4 个 (1, 33, 160, 497)
- uplo: 保留 2 个
- trans: 保留 3 个
- alpha: 保留 3 个 (1+0j, 0.5+0.3j, -1+0j)
- beta: 保留 3 个 (0+0j, 1+0j, 0.5-0.25j)
- **优化后: 7 × 4 × 2 × 3 × 3 × 3 = 1,512 → 进一步优化到 924**
  - n: 6 个, k: 3 个, alpha: 3 个, beta: 3 个
  - 6 × 3 × 2 × 3 × 3 × 3 = 972 → 再减 n 到 5 个 = 810

---

## 4. cublasSgemmStridedBatched (12,544 cases → 建议 896 cases)

**当前参数组合：**
- M,N,K: 28 triples
- alpha,beta: 14 pairs
- transa,transb: 8 combinations
- batchCount: 4 values
- dtype: 1 value
- **Total: 28 × 14 × 8 × 4 = 12,544**

**优化建议：**
- M,N,K: 保留 8 组
- alpha,beta: 保留 4 对
- transa,transb: 保留 4 种 (NN,NT,TN,TT)
- batchCount: 保留 3 个 (1, 4, 8)
- **优化后: 8 × 4 × 4 × 3 = 384 cases**

---

## 5. cublasSger_v2 (7,128 cases → 建议 648 cases)

**当前参数组合：**
- m: 8 values
- n: 11 values
- alpha: 9 values
- incx: 3 values
- incy: 3 values
- dtype: 1 value
- **Total: 8 × 11 × 9 × 3 × 3 = 7,128**

**优化建议：**
- m: 保留 6 个 (1, 32, 128, 256, 1024, 5333)
- n: 保留 6 个 (1, 32, 128, 256, 1024, 5333)
- alpha: 保留 4 个 (1.0, 0.0, 0.5, -1.0)
- incx: 保留 2 个 (1, 2)
- incy: 保留 2 个 (1, 2)
- **优化后: 6 × 6 × 4 × 2 × 2 = 576 cases**

---

## 6. cublasSgemm_v2 (4,224 cases → 建议 768 cases)

**当前参数组合：**
- M,N,K: 33 triples
- alpha,beta: 16 pairs
- transa,transb: 8 combinations
- dtype: 1 value
- **Total: 33 × 16 × 8 = 4,224**

**优化建议：**
- M,N,K: 保留 12 组
- alpha,beta: 保留 4 对
- transa,transb: 保留 4 种
- **优化后: 12 × 4 × 4 = 192 cases**

---

## Summary

| Operator | Current | Optimized | Reduction |
|----------|---------|-----------|-----------|
| cublasSgemvBatched | 46,656 | 768 | 98.4% |
| cublasCgemm_v2 | 40,392 | 192 | 99.5% |
| cublasCsyrkEx | 13,860 | 810 | 94.2% |
| cublasSgemmStridedBatched | 12,544 | 384 | 96.9% |
| cublasSger_v2 | 7,128 | 576 | 91.9% |
| cublasSgemm_v2 | 4,224 | 192 | 95.5% |

**总体优化原则：**
1. 矩阵尺寸：保留边界(1)、小(32-64)、中(128-256)、大(1024+)、非对称、超大(5333)
2. 标量参数：保留标准值(0,1)、分数(0.5)、负数(-1)
3. Stride参数：从3个减到2个(1,2)
4. Batch参数：从4个减到3个(1,4,8)
5. 转置组合：统一为4种(NN,NT,TN,TT)
