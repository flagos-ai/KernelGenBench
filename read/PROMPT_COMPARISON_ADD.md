# Prompt 对比：PyTorch add vs K1 add_row_sum_mat

## 文件位置

- PyTorch 版本：`flagbench/prompt_example_torch_add.txt`
- K1 版本：`flagbench/prompt_example_add_row_sum_mat.txt`

---

## 结构对比

| 部分 | PyTorch add | K1 add_row_sum_mat | 差异说明 |
|------|-------------|-------------------|----------|
| **1. 任务说明** | "generate a Triton kernel function" | "generate a Triton kernel function" | ✅ 相同 |
| **2. 示例代码** | PyTorch 函数示例：<br>`def add(A, B): return torch.add(A, B)`<br>+ Triton 实现（1D向量） | K1 CUDA wrapper 示例：<br>`inline void cuda_add(...)`<br>+ Triton 实现（2D矩阵） | ⚠️ 示例不同：PyTorch 是向量，K1 是矩阵 |
| **3. 参考实现** | ```python<br># Reference PyTorch implementation<br>def add(*args, **kwargs):<br>    return torch.add(*args, **kwargs)``` | ```python<br># K1 CUDA Wrapper: cuda_add_row_sum_mat<br># Description: 计算矩阵的行和并更新结果向量...``` | ⚠️ 格式不同：PyTorch 是代码，K1 是描述信息 |
| **4. 函数名要求** | `add` | `add_row_sum_mat` | ✅ 相同格式 |
| **5. 输入输出参数** | `Input Args: [InputArg(...), ...]`<br>`Output Args: [OutputArg(...)]`<br>（对象列表格式） | `Input Args: [InputArg(...), ...]`<br>`Output Args: [OutputArg(...)]`<br>（对象列表格式） | ✅ 格式相同，但可读性差 |
| **6. 类型转换说明** | ❌ 无 | ✅ 有：C++ → Python/Triton 类型转换表 | ⚠️ K1 特有 |
| **7. impl_info 处理** | 可能有（但 add 没有） | ❌ 无 | ⚠️ PyTorch 可能有 |
| **8. 格式要求** | "You must use ```python ... ```" | "You must use ```python ... ```" | ✅ 相同 |

---

## 关键差异

### 1. 示例代码
- **PyTorch**: 1D 向量加法示例
- **K1**: 2D 矩阵操作示例（更复杂）

### 2. 参考实现
- **PyTorch**: 简单的 Python wrapper 代码
- **K1**: 描述性信息（wrapper 名称和功能描述）

### 3. 输入输出参数
- **PyTorch**: 通用格式 `*args, **kwargs`
- **K1**: 具体 C++ 类型 `double*`, `const double*`, `MatrixDim` 等

### 4. 类型转换
- **PyTorch**: 不需要（已经是 Python 类型）
- **K1**: 需要（C++ → Python/Triton）

---

## 建议优化

### 输入输出参数格式优化

当前格式（可读性差）：
```
Input Args: [InputArg(arg_name='result', arg_type='double*', ...), ...]
```

建议格式（更易读）：
```
Input Args:
  1. result: double*
  2. mat: const double*
  3. scratch: void*
  4. d: const MatrixDim
  5. alpha: const double
  6. beta: const double
Output Args:
  1. void
```

这样可以：
- 更清晰地展示参数信息
- 便于模型理解参数类型和顺序
- 与示例代码中的参数注释风格一致

