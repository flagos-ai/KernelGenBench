## 验证工具

你可以使用以下命令验证你的实现：

```bash
CUDA_VISIBLE_DEVICES={{GPU_ID}} python {{VERIFY_SCRIPT}} --code kernel.py --op {{OPERATOR}} --dataset {{DATASET}} --output-json
```

验证结果会以 JSON 格式输出，包含：
- `passed`: 是否通过验证
- `total_tests`: 总测试数
- `passed_tests`: 通过的测试数
- `error`: 错误信息（如果有）

## 工作流程

请按以下流程工作：

1. **实现初始版本**：根据算子规范编写 Triton kernel 实现，保存到 `kernel.py`
2. **运行验证**：使用上述验证命令检查正确性
3. **分析错误**：如果验证失败，仔细阅读错误信息
4. **修改代码**：根据错误修改 `kernel.py`
5. **重复验证**：重新运行验证，直到通过或你认为已尽最大努力

**重要提示**：
- 每次修改后都要重新验证
- 注意处理边界情况（空 tensor、不同 dtype 等）
- 如果多次尝试后仍有少量测试失败，可以接受并继续

## 输出要求

**完成优化后**，请在回复末尾用以下格式输出你的最终代码：

```python
import torch
import triton
import triton.language as tl

# 你的最终实现代码...
```

确保代码块是完整可运行的，包含所有必要的 import 和函数定义。
