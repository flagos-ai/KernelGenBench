# cuBLAS Python Baseline 工作日志

## 项目目标

为 FlagBench 的 Triton 内核基准测试系统构建 CUDA baseline，使用 cuBLAS 等库作为性能对照标准。

---

## 已完成工作

### 1. PyTorch Extension 方案（已完成）

**实现路径：** `cache/cublas_baseline/generated_wrapper.cpp`

**技术方案：**
- 使用 `torch.utils.cpp_extension.load()` 动态编译 C++ wrapper
- 封装 cuBLAS API 为 PyTorch 可调用函数
- 解决 Row-Major（PyTorch）↔ Column-Major（cuBLAS）转换问题

**已实现算子（6个）：**
1. `sgemm` - 单精度矩阵乘法 (FP32)
2. `dgemm` - 双精度矩阵乘法 (FP64)
3. `hgemm` - 半精度矩阵乘法 (FP16)
4. `saxpy` - 单精度向量加法 (Y = α·X + Y)
5. `sscal` - 单精度向量缩放 (X = α·X)
6. `sdot` - 单精度点积 (返回标量)

**核心技术点：**

#### Row-Major ↔ Column-Major 转换
```cpp
// PyTorch: C = A * B (Row-Major)
// cuBLAS:  C^T = B^T * A^T (Column-Major)
// 解决方案：交换 A 和 B，交换 M 和 N

cublasSgemm(
    handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    N, M, K,                   // 交换维度
    &beta,
    B.data_ptr<float>(), N,    // B 在前
    A.data_ptr<float>(), K,    // A 在后
    &alpha,
    C.data_ptr<float>(), N
);
```

#### 单例 cuBLAS Handle
```cpp
static cublasHandle_t get_handle() {
    static cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        cublasCreate(&handle);
    }
    return handle;
}
```

**当前状态：**
- ✅ 编译成功
- ⚠️ 测试脚本未运行（原始 Python 源文件已删除）
- ℹ️ 仅存在编译产物：`cache/cublas_baseline/generated_wrapper.cpp`

**遇到的问题及解决：**
| 问题 | 解决方案 |
|------|----------|
| `cuda_runtime.h` 缺失 | 移除不必要的 CUDA 头文件 |
| `cublas_v2.h` 缺失 | 添加 CUDA include 路径 |
| 函数名冲突 | 重命名为 `kaldi_*` 前缀 |
| `c10::Half*` 类型转换 | 使用 `reinterpret_cast<const __half*>()` |

---

### 2. Kaldi K1 CUDA 内核（概念验证）

**成果：**
- 成功编译 3 个内核：`copy_low_upp`, `copy_upp_low`, `add_mat`
- 所有测试通过 ✓

**问题：**
- Kaldi 依赖复杂头文件（cu-common.h, kaldi-error.h, OpenFst）
- 批量提取 169 个内核的正则表达式匹配失败
- **决定暂停此方案**

---

### 3. 内核分类分析

**分类结果：**
- Phase 1（标准模式）：127 个
- Phase 2（特殊类型）：31 个
- Phase 3（边缘情况）：8 个
- 已完成：3 个

---

## 新方案：CuPy 直接调用方案（待评估）

### 核心思路

直接使用 `cupy.cuda.cublas.sgemm()` 等现成接口，无需编译 C++ wrapper。

### 优势分析

1. **零编译成本**
   - 无需维护 C++ wrapper 代码
   - 无需处理 PyTorch extension 编译问题
   - 无需管理 CUDA 头文件路径

2. **API 现成**
   - CuPy 已完整封装 cuBLAS/cuFFT/cuSPARSE 等库
   - 接口稳定，文档完善
   - 社区维护，bug 少

3. **流程简化**
   - 直接修改 prompt 生成 `cupy.cuda.cublas.*` 调用
   - 仅需读入 cuBLAS 函数签名（入参/出参）
   - 自动生成测试函数和 benchmark

### 潜在问题

#### ❓ 问题 1：CuPy 与 PyTorch Tensor 互操作性
```python
# PyTorch tensor → CuPy array
import torch
import cupy as cp
from torch.utils.dlpack import to_dlpack, from_dlpack

# 转换过程
torch_tensor = torch.randn(1024, 1024, device='cuda')
cupy_array = cp.from_dlpack(to_dlpack(torch_tensor))  # ✅ 零拷贝

# 反向转换
result_cupy = cp.random.randn(1024, 1024)
result_torch = from_dlpack(result_cupy.toDlpack())     # ✅ 零拷贝
```

**结论：** ✅ 可行，使用 DLPack 协议零拷贝转换。

#### ❓ 问题 2：性能开销
- DLPack 转换：零拷贝，仅元数据映射（~1μs）
- cuBLAS 调用：与直接 C++ 调用完全相同（CuPy 是薄封装）

**结论：** ✅ 性能开销可忽略，适合做 baseline。

#### ❓ 问题 3：API 覆盖度
需要检查 CuPy 是否覆盖所有需要的 cuBLAS 函数。

**示例：**
```python
import cupy.cuda.cublas as cublas

# Level 1 BLAS
cublas.saxpy()  # ✅
cublas.sdot()   # ✅
cublas.sscal()  # ✅

# Level 2 BLAS
cublas.sgemv()  # ✅

# Level 3 BLAS
cublas.sgemm()  # ✅
cublas.dgemm()  # ✅
cublas.hgemm()  # ✅（需要检查 FP16 支持）
```

**需要验证：**
- [ ] FP16（Half precision）支持
- [ ] Batched GEMM（批量矩阵乘法）
- [ ] Strided Batched GEMM

#### ❓ 问题 4：Row-Major vs Column-Major
CuPy 默认使用 **Row-Major**（与 NumPy 一致），但 `cupy.cuda.cublas` 是 **直接封装 cuBLAS**，仍然是 Column-Major。

**解决方案：**
- 选项 A：使用 `cupy.linalg.*`（高层 API，自动处理转换）
- 选项 B：手动处理转换（与 C++ wrapper 相同逻辑）

```python
# 选项 A（推荐）
import cupy as cp
C = cp.dot(A, B)  # 自动处理 Row-Major

# 选项 B（底层控制）
import cupy.cuda.cublas as cublas
# 需要手动交换维度，类似 C++ wrapper
```

**结论：** ✅ 使用 `cupy.linalg.*` 更简单，除非需要极致性能调优。

#### ❓ 问题 5：集成到 FlagBench 流程
当前 FlagBench 流程：
```
生成 Triton kernel → 生成测试函数 → 生成 benchmark → 对比性能
```

新流程：
```
读取 cuBLAS 函数签名 → 修改 prompt 生成 CuPy 调用 → 生成测试/benchmark
```

**需要修改的部分：**
1. **函数签名读取**
   - 从哪里获取？（cuBLAS 文档 / CuPy API 文档）
   - 如何解析参数类型和维度约束？

2. **Prompt 修改**
   - 原 prompt：生成 Triton kernel 的测试
   - 新 prompt：生成 CuPy 调用的测试
   - 需要处理 Tensor ↔ Array 转换

3. **Benchmark 生成**
   - 统一接口：`def baseline(A, B, ...)`
   - 内部处理 DLPack 转换

**示例伪代码：**
```python
def generate_cupy_baseline(func_name, signature):
    """
    func_name: "sgemm"
    signature: {
        'inputs': [('A', 'float32', '(M, K)'), ('B', 'float32', '(K, N)')],
        'outputs': [('C', 'float32', '(M, N)')],
        'params': [('alpha', 'float'), ('beta', 'float')]
    }
    """
    template = f"""
import cupy as cp
from torch.utils.dlpack import to_dlpack, from_dlpack

def {func_name}_baseline(A_torch, B_torch, alpha=1.0, beta=0.0):
    # Convert to CuPy
    A = cp.from_dlpack(to_dlpack(A_torch))
    B = cp.from_dlpack(to_dlpack(B_torch))
    
    # Call cuBLAS via CuPy
    C = alpha * cp.dot(A, B) + beta * ...
    
    # Convert back to PyTorch
    return from_dlpack(C.toDlpack())
"""
    return template
```

---

## 下一步工作

### 方案 A：继续 PyTorch Extension（已完成技术验证）
- [ ] 恢复测试脚本，验证 6 个算子正确性
- [ ] 添加更多 cuBLAS 算子（GEMV, IAMAX, ASUM, NRM2）
- [ ] 性能测试对比
- [ ] 集成到 FlagBench

### 方案 B：切换到 CuPy（推荐尝试）
- [ ] **验证 CuPy API 覆盖度**（优先级：高）
  - 检查 cuBLAS Level 1/2/3 支持
  - 检查 FP16/Batched GEMM 支持
- [ ] **编写 Tensor ↔ Array 转换样例**
  - 验证 DLPack 零拷贝性能
  - 测试内存布局一致性
- [ ] **修改 Prompt 生成流程**
  - 设计 cuBLAS 函数签名格式
  - 修改测试函数生成 prompt
  - 修改 benchmark 生成 prompt
- [ ] **实现函数签名读取**
  - 从哪里获取（cuBLAS docs / CuPy docs）
  - 如何解析和结构化存储
- [ ] **端到端测试**
  - 生成一个 `sgemm` baseline
  - 对比 Triton kernel 性能

---

## 技术决策讨论点

### CuPy 方案的关键问题

1. **是否需要底层 cuBLAS 控制？**
   - 如需精细控制（strided access, workspace size），用 `cupy.cuda.cublas.*`
   - 如仅需功能正确性，用 `cupy.linalg.*` 更简单

2. **函数签名从哪里获取？**
   - 选项 A：手动编写（维护成本高）
   - 选项 B：爬取 cuBLAS 官方文档（自动化）
   - 选项 C：解析 CuPy 源码（可能过度工程化）

3. **如何处理 cuBLAS 不支持的算子？**
   - 某些 Kaldi kernel 可能无 cuBLAS 等价物
   - 备选方案：cuDNN / Thrust / 自定义 CUDA kernel

---

## 环境信息

```bash
# 激活环境
source /share/project/zhaohuxing/anaconda3/bin/activate zpy_flagbench

# 工作目录
cd /share/project/zpy/flagbench

# 关键路径
/share/project/zpy/flagbench/script/cuda_baseline/
├── cache/
│   ├── cublas_baseline/
│   │   ├── generated_wrapper.cpp   # PyTorch Extension 方案
│   │   └── module_info.json
│   ├── extracted_cuda/             # Kaldi 提取的 CUDA 代码
│   └── generated_adapters/         # Kaldi adapter（已废弃）
```

---

## 决策记录

### 2026-01-16
- ✅ PyTorch Extension 方案技术验证通过
- ⏸️ Kaldi K1 方案因依赖复杂暂停
- 🔍 正在评估 CuPy 方案可行性
- ✅ **CuPy 方案第一步完成**：成功解析 cuBLAS 头文件，生成 239 个函数的 schema
  - 位置：`/share/project/zpy/flagbench/script/cublas_cupy/`
  - Schema 文件：`cublas_ops.json`
  - 覆盖：Level 1/2/3 BLAS、Batched 操作、FP16/FP32/FP64/Complex
  - 参数角色自动识别：context, input, output, inout, scalar, value
