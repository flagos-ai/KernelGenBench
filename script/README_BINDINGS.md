# Kaldi CUDA Bindings 自动构建工具

## 概述

这套工具链用于自动将**任意开源CUDA仓库**的算子暴露为PyTorch扩展，以便：
1. 作为baseline与Triton实现对比
2. 快速验证算子正确性
3. 性能benchmark

**特点：**
- ✅ 自动化：从头文件提取 → 生成binding → 编译 → 测试
- ✅ 通用性：不仅支持Kaldi，可迁移到其他CUDA仓库
- ✅ 可扩展：每个阶段独立，可单独运行或定制

## 工具链架构

```
┌─────────────────────────────────────────────────────────────┐
│                    输入：CUDA仓库                              │
│             (kaldi/src/cudamatrix/*.h, *.cu)                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  阶段1: extract_cuda_kernels.py                              │
│  - 解析cu-kernels-ansi.h提取函数声明                          │
│  - 分析参数类型、输入输出                                      │
│  - 输出: csrc/kaldi_kernels.json                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  阶段2: generate_binding_code.py                             │
│  - 读取JSON，生成C++ wrapper函数                              │
│  - 生成TORCH_LIBRARY注册代码                                  │
│  - 输出: csrc/kaldi_ops.cpp, csrc/setup.py                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  阶段3: compile_bindings.py                                  │
│  - 检查/编译Kaldi库                                           │
│  - 编译PyTorch扩展                                            │
│  - 输出: lib/kaldi_ops.so                                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  阶段4: test_bindings.py                                     │
│  - 加载.so库                                                  │
│  - 验证算子注册                                               │
│  - 输出: test_report.json                                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              输出：torch.ops.kaldi.*                          │
│         可直接在Python中作为baseline使用                       │
└─────────────────────────────────────────────────────────────┘
```

## 快速开始

### 一键构建（推荐）

```bash
# 运行完整构建流程
bash script/build_all.sh
```

### 分步运行

```bash
# 阶段1: 提取CUDA算子
python script/extract_cuda_kernels.py

# 阶段2: 生成binding代码
python script/generate_binding_code.py

# 阶段3: 编译（可选跳过，如遇问题可手动调整）
python script/compile_bindings.py

# 阶段4: 测试
python script/test_bindings.py --lib lib/kaldi_ops.so
```

## 文件说明

### 脚本文件 (script/)

| 文件 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `extract_cuda_kernels.py` | 提取CUDA算子定义 | `cu-kernels-ansi.h` | `kaldi_kernels.json` |
| `generate_binding_code.py` | 生成C++ binding | `kaldi_kernels.json` | `kaldi_ops.cpp`, `setup.py` |
| `compile_bindings.py` | 编译扩展库 | `kaldi_ops.cpp` | `kaldi_ops.so` |
| `test_bindings.py` | 测试bindings | `kaldi_ops.so` | `test_report.json` |
| `build_all.sh` | 串联所有阶段 | - | 完整构建 |

### 生成的文件

```
csrc/
├── kaldi_kernels.json    # 阶段1: 算子定义（JSON格式）
├── kaldi_ops.cpp         # 阶段2: C++ binding代码（6500+行）
└── setup.py              # 阶段2: 编译配置

lib/
└── kaldi_ops.so          # 阶段3: 编译后的扩展库

test_report.json          # 阶段4: 测试报告
```

## 使用示例

### 在Python中使用

```python
import torch

# 加载编译好的库
torch.ops.load_library('lib/kaldi_ops.so')

# 使用Kaldi算子
mat = torch.randn(128, 256, device='cuda')
result = torch.zeros(128, device='cuda')
scratch = torch.empty(256, device='cuda')

# 调用add_row_sum_mat作为baseline
torch.ops.kaldi.add_row_sum_mat(result, mat, scratch, mat, 1.0, 0.0)

# 在测试函数中对比Triton实现
import flagbench
with flagbench.use_gems(REGISTERED_OPS):
    triton_result = torch.ops.kaldi.add_row_sum_mat(...)

# 对比结果
assert torch.allclose(result, triton_result)
```

### 集成到UT生成流程

已提取的129个Kaldi算子可以直接用于：

```bash
# 生成测试用例
python script/generate_ut_sample4k1.py --name add_row_sum_mat

# 运行accuracy测试
python test/test_accuracy_ut.py --name add_row_sum_mat
```

## 迁移到其他CUDA仓库

这套工具设计为通用方案，迁移步骤：

### 1. 准备CUDA仓库

```bash
# 克隆目标仓库
git clone https://github.com/xxx/cuda_repo.git

# 找到kernel定义头文件（类似cu-kernels-ansi.h）
```

### 2. 修改提取脚本

```bash
python script/extract_cuda_kernels.py \
    --input cuda_repo/include/kernels.h \
    --output csrc/custom_kernels.json
```

### 3. 生成并编译

```bash
python script/generate_binding_code.py \
    --input csrc/custom_kernels.json \
    --output csrc/custom_ops.cpp \
    --namespace custom

bash script/build_all.sh
```

## 已知问题与TODO

### 当前状态

✅ **已完成：**
- 成功提取129个Kaldi CUDA算子
- 自动生成6500+行C++ binding代码
- 完整的5阶段构建流程

⚠️ **待优化：**
- 阶段2生成的C++代码有bug（标量参数传递、grid/block处理）
- 需要手动编译Kaldi库
- 类型转换逻辑需要完善

### 优化计划

1. **修复代码生成bug**
   - 修正标量参数传递（`alpha` → 传值而非类型）
   - 完善grid/block自动计算
   - 处理特殊类型（MatrixElement, Int32Pair等）

2. **增强编译流程**
   - 自动检测Kaldi编译状态
   - 提供预编译Kaldi库的下载
   - 支持静态链接

3. **完善测试**
   - 为每个算子生成正确的输入数据
   - 自动验证数值正确性
   - 性能benchmark

## 命令行参数

### build_all.sh

```bash
bash script/build_all.sh [options]

Options:
  --skip-compile    跳过编译阶段（用于只生成代码）
  --skip-test       跳过测试阶段
  --help, -h        显示帮助
```

### 各阶段详细参数

```bash
# 阶段1
python script/extract_cuda_kernels.py \
    --input <header_file> \
    --output <json_file> \
    --cu-impl <cu_file>  # 可选

# 阶段2  
python script/generate_binding_code.py \
    --input <json_file> \
    --output <cpp_file> \
    --cuda-src <cuda_src_dir> \
    --namespace <namespace>

# 阶段3
python script/compile_bindings.py \
    --csrc-dir <csrc_dir> \
    --kaldi-src <kaldi_src> \
    --output-dir <output_dir>

# 阶段4
python script/test_bindings.py \
    --lib <so_file> \
    --namespace <namespace> \
    --test-kernel <kernel_name>  # 可多次使用
```

## 贡献与反馈

这套工具是FlagBench的一部分，用于支持K1和其他CUDA仓库的baseline构建。

如有问题或建议，请联系项目维护者。

---

**Author:** FlagBench Team  
**Date:** 2026-01-16  
**Version:** 1.0.0
