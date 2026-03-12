# Prompt 生成对比：PyTorch vs K1 CUDA

## 调用流程

### PyTorch 版本 (`generate_sample.py`)
```
generate_samples() 
  → TritonKernelGenerator.generate_prompt()
    → generate_prompt_for_new()  # 检测到新生成
      → 使用 PyTorch 版本的 prompt
```

### K1 版本 (`generate_sample4k1.py`)
```
generate_samples() 
  → TritonKernelGenerator.generate_prompt()
    → generate_prompt_for_new()  # 检测到新生成
      → 检测 torch_kernel_code 是否以 "# K1 CUDA Wrapper:" 开头
        → 如果是，调用 generate_prompt_for_k1_cuda()
          → 使用 K1 版本的 prompt
```

---

## Prompt 结构对比

| 部分 | PyTorch 版本 (`generate_prompt_for_new`) | K1 版本 (`generate_prompt_for_k1_cuda`) | 说明 |
|------|------------------------------------------|-----------------------------------------|------|
| **1. 开头说明** | "You are a skilled GPU programmer... Your task is to generate a Triton kernel function." | "You are a skilled GPU programmer... Your task is to implement a Kaldi K1 CUDA wrapper function using Triton." | ✅ 最小修改：只改任务描述 |
| **2. 重要提示** | ❌ 无 | "IMPORTANT: This is a WRAPPER function, not a single kernel..." | ⚠️ K1 特有：强调 wrapper 概念 |
| **3. 示例代码** | PyTorch 函数示例：<br>`def add(A, B): return torch.add(A, B)`<br>+ Triton 实现示例 | CUDA wrapper 示例：<br>`inline void cuda_add(...)`<br>+ Triton 实现示例 | ✅ 最小修改：替换示例来源 |
| **4. 参考实现** | `info.torch_kernel_code`<br>(PyTorch 函数代码) | `info.torch_kernel_code`<br>(K1 CUDA Wrapper 信息，包含 description) | ✅ 相同字段，内容不同 |
| **5. 函数信息** | 函数名：`info.op_name`<br>描述：`info.func_desc` | 函数名：`info.triton_kernel_name`<br>描述：`info.func_desc` | ✅ 最小修改：字段名不同但等价 |
| **6. 输入输出参数** | `Input Args: {info.input_args}`<br>`Output Args: {info.output_args}`<br>(简单列出) | 详细列出每个参数：<br>`1. arg_name: arg_type (desc)`<br>`2. ...` | ⚠️ K1 更详细：逐项列出参数 |
| **7. 多操作符处理** | 有 `impl_info` 处理逻辑<br>(处理多个 ATen 操作符) | ❌ 无 | ⚠️ K1 不需要：wrapper 是单一功能单元 |
| **8. 命名要求** | `info.op_name` | `info.triton_kernel_name` | ✅ 等价：`op_name` 是 `triton_kernel_name` 的属性 |
| **9. 类型转换说明** | ❌ 无 | 详细的 C++ → Python/Triton 类型转换表 | ⚠️ K1 特有：需要类型转换 |
| **10. 实现要求** | 通用要求 | 7 条具体要求（包含 wrapper 多 kernel 处理） | ⚠️ K1 更详细：针对 wrapper 的特殊要求 |
| **11. 格式要求** | "You must use ```python ... ```" | "You must use ```python ... ```" | ✅ 相同 |

---

## 最小化修改建议

### 方案：基于 PyTorch 版本，只替换关键信息

**保持的结构：**
1. ✅ 开头说明（只改任务描述）
2. ✅ 示例代码部分（替换为 K1 示例）
3. ✅ 参考实现部分（使用 `info.torch_kernel_code`）
4. ✅ 输入输出参数部分（保持简单格式，或稍微详细）
5. ✅ 命名要求
6. ✅ 格式要求

**需要添加的 K1 特有内容：**
1. ⚠️ Wrapper 重要提示（可选，因为示例已经说明了）
2. ⚠️ 类型转换说明（重要，因为 C++ 类型需要转换）
3. ⚠️ Wrapper 多 kernel 处理说明（重要）

**可以简化的部分：**
1. 输入输出参数：可以保持 PyTorch 版本的简单格式，不需要逐项列出
2. 示例代码：可以简化，只保留核心转换模式

---

## 推荐的最小修改版本

基于 `generate_prompt_for_new`，只做以下修改：

1. **开头**：改任务描述为 "implement a Kaldi K1 CUDA wrapper function"
2. **示例**：替换为 K1 CUDA wrapper → Triton 的示例
3. **参考代码**：使用 `info.torch_kernel_code`（已经是 K1 格式）
4. **输入输出**：保持简单格式 `Input Args: {info.input_args}`
5. **添加**：类型转换说明（C++ → Python/Triton）
6. **添加**：Wrapper 多 kernel 处理提示（简短）
7. **移除**：`impl_info` 多操作符处理逻辑（K1 不需要）

