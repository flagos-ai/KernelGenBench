# PyTorch K1 CUDA Wrapper Prompt 分析与优化建议

## 📋 当前问题分析

### 1. **参考代码未被使用** ⚠️
- `generate_torch_sample4k1.py` 中定义了 `create_k1_reference_code()` 函数，可以生成详细的参考代码（包括CUDA kernel代码、wrapper代码、usage example等）
- 但是该函数生成的内容**没有被使用**，`create_torch_generate_args()` 没有调用它
- `TorchKernelGenerateArgs` 也没有字段来存储参考代码信息

### 2. **示例过于简单** ⚠️
当前prompt只使用了一个非常基础的 `cuda_add` 示例：
```cpp
inline void cuda_add(dim3 Gr, dim3 Bl, double* mat, double value, MatrixDim d) {
    cudaD_add(Gr, Bl, mat, value, d);
}
```
- 这个示例太简单，无法展示复杂的K1 wrapper模式
- 没有展示涉及多个参数、矩阵维度、stride等复杂情况

### 3. **输入输出参数展示不清晰** ⚠️
当前直接打印对象列表：
```
Input Args: [InputArg(arg_name='result', arg_type='double*', ...), ...]
```
- 对LLM来说不够直观
- 应该格式化为更易读的形式

### 4. **缺少实际K1 wrapper代码** ⚠️
- Prompt中没有包含实际的K1 CUDA wrapper代码作为参考
- 只有函数名和描述，LLM无法看到实际的代码结构

### 5. **缺少算法细节和提示** ⚠️
- `IMPL_INFO_K1` 中可能包含 `algorithm` 和 `hints` 字段，但没有在prompt中使用
- 这些信息对正确实现很重要

### 6. **类型转换说明不够详细** ⚠️
- 当前只有基本的类型映射
- 缺少对复杂情况的说明（如stride处理、维度提取等）

---

## 🎯 优化建议

### 建议1: 使用 `create_k1_reference_code()` 生成的参考代码

**修改点：**
1. 在 `TorchKernelGenerateArgs` 中添加 `reference_code` 字段
2. 在 `create_torch_generate_args()` 中调用 `create_k1_reference_code()` 并传入生成的参考代码
3. 在 `generate_prompt_for_k1_cuda()` 中使用这个参考代码

**示例代码修改：**
```python
# 1. 修改 TorchKernelGenerateArgs
class TorchKernelGenerateArgs(BaseGenerateArgs):
    torch_kernel_name: str
    func_desc: str
    reference_code: Optional[str] = None  # 新增字段
    input_args: List[InputArg] | None = None
    output_args: List[OutputArg] | None = None
    func_type: Optional[str] = None

# 2. 修改 create_torch_generate_args()
def create_torch_generate_args(kernel_name: str, impl_info: Dict[str, Any]) -> TorchKernelGenerateArgs:
    # ... 现有代码 ...
    reference_code = create_k1_reference_code(kernel_name, impl_info)  # 生成参考代码
    
    return TorchKernelGenerateArgs(
        torch_kernel_name=kernel_name,
        func_desc=func_desc,
        reference_code=reference_code,  # 传入参考代码
        input_args=input_args,
        output_args=output_args,
        func_type="k1_cuda",
    )

# 3. 修改 generate_prompt_for_k1_cuda()
def generate_prompt_for_k1_cuda(self, info: TorchKernelGenerateArgs):
    prompt = f"You are a skilled software engineer proficient in PyTorch...\n"
    
    # 如果有参考代码，使用它；否则使用默认示例
    if info.reference_code and len(info.reference_code.strip()) > 0:
        prompt += f"The K1 CUDA wrapper function and implementation details:\n"
        prompt += f"```cpp\n{info.reference_code}\n```\n"
    else:
        # 使用现有的简单示例作为fallback
        prompt += f"K1 CUDA wrapper function:\n..."
```

### 建议2: 改进输入输出参数的展示格式

**当前格式（不友好）：**
```
Input Args: [InputArg(arg_name='result', arg_type='double*', ...), ...]
```

**优化格式（更清晰）：**
```
Input Arguments:
1. result (double*): Output vector that stores the row sum result
2. mat (const double*): Input matrix
3. scratch (void*): Scratch space for computation
4. d (const MatrixDim): Matrix dimension information (num_rows, num_cols, stride)
5. alpha (const double): Scaling factor for row sum
6. beta (const double): Scaling factor for existing result

Output: void (in-place operation, modifies result)
```

**实现代码：**
```python
def format_input_output_args(input_args, output_args) -> str:
    """格式化输入输出参数，使其更易读"""
    lines = ["Input Arguments:"]
    for idx, arg in enumerate(input_args or [], 1):
        desc = f": {arg.arg_desc}" if arg.arg_desc else ""
        lines.append(f"{idx}. {arg.arg_name} ({arg.arg_type}){desc}")
    
    lines.append("")
    lines.append("Output:")
    if output_args and len(output_args) > 0:
        out_arg = output_args[0]
        if out_arg.arg_type == "void":
            lines.append("  void (in-place operation)")
        else:
            desc = f": {out_arg.arg_desc}" if out_arg.arg_desc else ""
            lines.append(f"  {out_arg.arg_type}{desc}")
    return "\n".join(lines)
```

### 建议3: 使用更复杂和真实的示例

**当前示例问题：**
- 只有一个简单的 `add` 操作
- 没有展示复杂情况（如row sum、矩阵乘法等）

**优化建议：**
- 在prompt中提供2-3个不同复杂度的示例
- 包括：
  1. 简单操作（add scalar）
  2. 中等复杂度（row sum）
  3. 复杂操作（如果有）

**示例结构：**
```
Example 1: Simple in-place operation (cuda_add)
Example 2: Reduction operation (cuda_add_row_sum_mat)
Example 3: [根据实际需求添加]
```

### 建议4: 在prompt中包含算法描述和提示

**当前问题：**
- `IMPL_INFO_K1` 中的 `algorithm` 和 `hints` 字段未被使用

**优化建议：**
```python
def generate_prompt_for_k1_cuda(self, info: TorchKernelGenerateArgs):
    # ... 现有代码 ...
    
    # 如果有算法描述，添加到prompt
    if hasattr(info, 'algorithm') and info.algorithm:
        prompt += f"\nAlgorithm Description:\n{info.algorithm}\n"
    
    # 如果有提示，添加到prompt
    if hasattr(info, 'hints') and info.hints:
        prompt += f"\nImplementation Hints:\n{info.hints}\n"
```

### 建议5: 改进类型转换说明

**当前说明不够详细，建议补充：**

```
IMPORTANT: Convert C++ types to PyTorch equivalents:

1. Pointers:
   - double* / float* → torch.Tensor (on CUDA device, mutable)
   - const double* / const float* → torch.Tensor (read-only, use .clone() if modification needed)
   - void* → torch.Tensor or None (depending on context, often scratch space)

2. MatrixDim structure:
   - Extract from tensor: num_rows = tensor.shape[0], num_cols = tensor.shape[1]
   - stride = tensor.stride(0) (row stride)
   - Use tensor.stride(1) for column stride if needed

3. Grid/Block dimensions:
   - dim3 Gr, Bl → Not needed in PyTorch (PyTorch handles this internally)
   - For reduction operations, calculate grid size from tensor dimensions

4. Scalar types:
   - int → Python int
   - double → Python float
   - float → Python float

5. Special considerations:
   - In-place operations: Modify tensor directly using in-place methods (add_, mul_, etc.)
   - Row/column operations: Use torch.sum(..., dim=0 or dim=1) for reductions
   - Stride handling: PyTorch tensors handle strides automatically, but be aware of memory layout
```

### 建议6: 添加实现检查清单

在prompt末尾添加检查清单，帮助LLM生成更准确的代码：

```
Before generating the code, ensure:
- [ ] All input tensors are verified to be on CUDA device
- [ ] Matrix dimensions are correctly extracted from input tensors
- [ ] The operation matches the mathematical description (result = alpha * operation + beta * result)
- [ ] In-place operations modify the correct tensor and return it
- [ ] Edge cases are handled (empty tensors, zero dimensions, etc.)
- [ ] The function signature matches exactly: def {op_name}(...)
```

---

## 📝 完整优化后的Prompt结构建议

```
1. 角色和任务说明
2. 2-3个不同复杂度的示例（包括代码转换）
3. 实际的K1 CUDA wrapper参考代码（如果有）
4. 函数信息（名称、描述）
5. 格式化的输入输出参数列表
6. 算法描述（如果有）
7. 类型转换详细说明
8. 实现指导原则
9. 实现检查清单
10. 格式要求（代码块格式、无解释等）
```

---

## 🔧 推荐的实现优先级

### 高优先级（立即实施）：
1. ✅ **使用 `create_k1_reference_code()` 生成的参考代码** - 这是最重要的改进
2. ✅ **改进输入输出参数展示格式** - 显著提升prompt可读性
3. ✅ **添加参考代码到 `TorchKernelGenerateArgs`** - 必要的架构修改

### 中优先级（后续优化）：
4. ⚠️ 使用更复杂和真实的示例
5. ⚠️ 包含算法描述和提示
6. ⚠️ 改进类型转换说明

### 低优先级（可选）：
7. 📋 添加实现检查清单
8. 📋 根据生成结果质量进一步调整

---

## 💡 额外建议

### 建议7: 添加示例输出验证
在prompt中提供一个示例输入和期望输出，帮助LLM理解期望行为：

```
Example usage:
Input: 
  result = torch.zeros(10, device='cuda')  # 10-element vector
  mat = torch.randn(10, 20, device='cuda')  # 10x20 matrix
  alpha = 1.0, beta = 0.0

Expected behavior:
  After calling add_row_sum_mat(result, mat, scratch, d, alpha, beta):
  result[i] = sum(mat[i, :]) for i in range(10)
```

### 建议8: 根据函数类型定制prompt
不同的K1函数可能有不同的模式：
- Reduction操作（row/column sum）
- Element-wise操作
- Matrix乘法相关操作
- 等等

可以根据 `func_desc` 中的关键词（如"row sum", "column sum"）来定制prompt内容。

---

## 📊 预期改进效果

实施这些优化后，预期：
1. ✅ **代码准确性提升** - 通过提供实际参考代码，LLM能更好地理解K1 wrapper的结构
2. ✅ **参数理解改善** - 格式化的参数列表使LLM更容易理解函数接口
3. ✅ **实现质量提高** - 详细的类型转换说明和实现指导减少常见错误
4. ✅ **一致性增强** - 统一的prompt结构使生成结果更加一致

