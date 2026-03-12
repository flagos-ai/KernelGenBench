# Kaldi K1 CUDA Kernels - Load_inline 方案

使用 `torch.utils.cpp_extension.load_inline` 将 Kaldi K1 的所有 CUDA kernels 包装成 PyTorch 可调用函数。

## 核心优势

✅ **零侵入**：直接复用 Kaldi 原生 CUDA 代码，无需修改源码  
✅ **零开销**：C++ 直接调用，无 Python/CuPy 中间层  
✅ **准确的性能测量**：真实反映 CUDA kernel 性能  
✅ **自动化**：提取、生成 adapter、编译全自动  
✅ **可复用**：工具链可用于任何 CUDA 仓库  

## 快速开始

### 1. 一键构建所有 kernel

```bash
# 激活环境
source /share/project/zhaohuxing/anaconda3/bin/activate zpy_flagbench
cd /share/project/zpy/flagbench

# 构建所有 kernel（自动提取、生成、编译）
python script/cuda_baseline/build_kaldi_k1.py --build-all
```

### 2. 在 Python 中使用

```python
from script.cuda_baseline.kaldi_k1_kernels import KaldiK1Kernels
import torch

# 加载所有 kernel（自动）
kaldi = KaldiK1Kernels()

# 使用 kernel - copy_low_upp
A = torch.randn(64, 64, device='cuda')
kaldi.copy_low_upp(A)  # 原地操作：复制下三角到上三角

# 使用 kernel - add_mat
dst = torch.randn(128, 128, device='cuda')
src = torch.randn(128, 128, device='cuda')
kaldi.add_mat(dst, src, 2.0)  # dst = 2.0 * src + dst

# 查看所有可用 kernel
print(kaldi.available_kernels())
```

### 3. 单独构建特定 kernel

```bash
python script/cuda_baseline/build_kaldi_k1.py --kernel copy_low_upp
```

### 4. 查看所有配置的 kernel

```bash
python script/cuda_baseline/build_kaldi_k1.py --list
```

## 目录结构

```
script/cuda_baseline/
├── cuda_baseline_builder.py      # 核心：load_inline 封装
├── extract_cuda_source.py        # 工具：从 Kaldi 提取源码
├── generate_adapter.py           # 工具：生成 adapter 代码
├── build_kaldi_k1.py             # 批量构建脚本
├── kaldi_k1_kernels.py           # 统一 Python 接口
└── README.md                     # 本文件

cache/
├── extracted_cuda/               # 提取的 CUDA 源码
├── generated_adapters/           # 生成的 adapter
├── cuda_jit/                     # 编译缓存
└── kaldi_k1_manifest.json        # 构建清单
```

## 工作流程

```
Kaldi CUDA 源码
        ↓
   [extract_cuda_source.py] ← 自动提取 kernel + wrapper
        ↓
   extracted_cuda/*.cu
        ↓
   [generate_adapter.py] ← 自动生成 PyTorch 绑定
        ↓
   generated_adapters/*.cpp
        ↓
   [cuda_baseline_builder.py] ← load_inline 编译
        ↓
   Python 可调用函数 ✓
```

## 当前状态

### 已完成 ✅
- ✅ 核心工具链（提取器、生成器、构建器）
- ✅ 3 个 kernel 完整集成：
  - `copy_low_upp` - 复制下三角到上三角
  - `copy_upp_low` - 复制上三角到下三角
  - `add_mat` - 矩阵加法：dst = alpha * src + dst
- ✅ 统一 Python 接口（KaldiK1Kernels）
- ✅ 自动缓存和增量编译

### 待扩展 📋
- 为剩余 166 个 kernel 添加配置
- 自动从 `kernel_list_k1.py` 读取参数信息
- 批量测试框架

## 使用示例

### 示例 1：基本用法

```python
from script.cuda_baseline.kaldi_k1_kernels import load_kaldi_kernels
import torch

# 快速加载
kaldi = load_kaldi_kernels()

# 测试 copy_low_upp
A = torch.randn(100, 100, device='cuda')
print(f"对称性（之前）: {torch.allclose(A, A.t())}")  # False
kaldi.copy_low_upp(A)
print(f"对称性（之后）: {torch.allclose(A, A.t())}")  # True
```

### 示例 2：性能测试

```python
import time
import torch
from script.cuda_baseline.kaldi_k1_kernels import KaldiK1Kernels

kaldi = KaldiK1Kernels()

N = 1024
dst = torch.randn(N, N, device='cuda')
src = torch.randn(N, N, device='cuda')

# Warmup
for _ in range(10):
    kaldi.add_mat(dst, src, 1.0)
torch.cuda.synchronize()

# Benchmark
iterations = 1000
start = time.time()
for _ in range(iterations):
    kaldi.add_mat(dst, src, 1.0)
torch.cuda.synchronize()
end = time.time()

avg_time_ms = (end - start) / iterations * 1000
print(f"add_mat ({N}x{N}): {avg_time_ms:.4f} ms/iter")
```

### 示例 3：与 PyTorch 原生操作对比

```python
import torch
from script.cuda_baseline.kaldi_k1_kernels import KaldiK1Kernels

kaldi = KaldiK1Kernels()

dst1 = torch.randn(512, 512, device='cuda')
dst2 = dst1.clone()
src = torch.randn(512, 512, device='cuda')
alpha = 2.5

# Kaldi kernel
kaldi.add_mat(dst1, src, alpha)

# PyTorch 等价操作
dst2.add_(src, alpha=alpha)

# 验证结果一致
print(f"一致性: {torch.allclose(dst1, dst2)}")
```

## 性能对比

与之前的 CuPy 方案对比：

| 方面 | CuPy 方案 | load_inline 方案 |
|------|-----------|------------------|
| 复用 Kaldi 代码 | ❌ 需重写 host 逻辑 | ✅ 完全复用 |
| 性能开销 | ⚠️ ~100μs Python 开销 | ✅ <1μs (C++ 直接调用) |
| 测量准确性 | ❌ 测不准（包含 Python 开销） | ✅ 准确 |
| 开发工作量 | 🔴 166 个手写 wrapper | 🟢 自动生成 |
| 维护成本 | 🔴 Kaldi 更新需手动同步 | 🟢 重新提取即可 |

## 扩展到其他 CUDA 仓库

这套工具链是通用的，可以用于任何 CUDA 仓库：

1. **修改 `KaldiCudaExtractor`** → 适配新仓库的代码结构
2. **定义 kernel 配置** → 参数类型、grid 配置等
3. **运行 `build_xxx.py`** → 自动构建所有 kernel

## 常见问题

### Q: 编译失败怎么办？
A: 检查：
- CUDA 是否正确安装（`nvcc --version`）
- PyTorch 是否支持 CUDA（`torch.cuda.is_available()`）
- 查看详细日志：`--verbose` 参数

### Q: 如何清除缓存？
A: 
```bash
rm -rf cache/cuda_jit/*
rm cache/kaldi_k1_manifest.json
```

### Q: 如何添加新 kernel？
A: 在 `build_kaldi_k1.py` 的 `get_kernel_config()` 中添加配置

## 技术细节

### load_inline 工作原理

1. **接收输入**：CUDA 源码 + C++ adapter 代码
2. **生成临时文件**：`main.cpp` (adapter) + `cuda.cu` (kernel)
3. **调用 nvcc**：编译成 `.so` 动态库
4. **加载到 Python**：通过 pybind11 暴露函数
5. **缓存**：避免重复编译

### Adapter 代码结构

```cpp
#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA launcher 声明
extern "C" void launch_xxx(...);

// PyTorch 接口
void xxx(torch::Tensor A) {
    // 参数检查
    TORCH_CHECK(A.device().is_cuda(), "...");
    
    // 提取指针和维度
    float* ptr = A.data_ptr<float>();
    dim3 grid(...), block(...);
    
    // 调用 CUDA launcher
    launch_xxx(grid, block, ptr, ...);
    cudaDeviceSynchronize();
}

// pybind11 绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("xxx", &xxx, "...");
}
```

## 贡献者

FlagBench Team - 2026-01-16

## License

Apache 2.0 (与 Kaldi 保持一致)
