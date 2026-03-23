"""
CupyPromptBuilder - Cupy 框架的 Prompt 构造器

用于从 cuBLAS baseline 函数生成 Triton kernel 的 prompt
"""

from typing import TYPE_CHECKING

from .prompt_builder import PromptBuilder
from flagbench.framework.generate_args import BaseGenerateArgs, CupyGenerateArgs

if TYPE_CHECKING:
    from sandbox.utils.accuracy_utils import VerifyResult


class CupyPromptBuilder(PromptBuilder):
    """Cupy 框架的 Prompt 构造器 - 用于从 cuBLAS baseline 生成 Triton kernel 的 prompt"""

    def build_new(self, gen_args: BaseGenerateArgs) -> str:
        """
        生成新 Triton kernel 的 prompt（从 cuBLAS baseline）

        Args:
            gen_args: 生成参数，实际应为 CupyGenerateArgs 实例
        """
        # 类型转换：确保 gen_args 是 CupyGenerateArgs
        if not isinstance(gen_args, CupyGenerateArgs):
            raise TypeError(f"CupyPromptBuilder requires CupyGenerateArgs, got {type(gen_args)}")

        info: CupyGenerateArgs = gen_args

        prompt = "You are a skilled GPU programmer proficient in Triton. Your task is to generate a Triton kernel function that implements the same functionality as a cuBLAS baseline function.\n"

        # 提供 Triton kernel 示例
        prompt += "\nHere is an example of a cuBLAS baseline function and its corresponding Triton kernel implementation:\n"
        prompt += "cuBLAS baseline function (SAXPY - Level 1 BLAS):\n"
        prompt += """```python
def saxpy_baseline(alpha, x, y):
    '''Single precision a*x + y'''
    return alpha * x + y
```
""".strip() + "\n"

        prompt += "\nTriton kernel implementation:\n"
        prompt += """```python
import triton
import triton.language as tl
import torch

@triton.jit
def saxpy_kernel(
    alpha,
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    output = alpha * x + y

    tl.store(output_ptr + offsets, output, mask=mask)

def saxpy(alpha, x, y):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    saxpy_kernel[grid](alpha, x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
```
""".strip() + "\n"

        prompt += "\n" + "="*60 + "\n"
        prompt += "Now, you must implement a Triton kernel for the following cuBLAS baseline function:\n"
        prompt += "="*60 + "\n\n"

        # 添加 BLAS 操作类型信息
        prompt += f"BLAS Operation Type: {info.blas_operation_type}\n"
        prompt += f"Operation: {info.cupy_kernel_name}\n\n"

        # 添加 baseline 函数代码
        prompt += "cuBLAS Baseline Function:\n"
        prompt += "```python\n"
        prompt += f"{info.baseline_code}\n"
        prompt += "```\n\n"

        # 添加函数描述
        if info.func_desc:
            prompt += f"Function Description:\n{info.func_desc}\n\n"

        # 根据 BLAS Level 提供特定指导
        prompt += self._get_blas_level_guidance(info.blas_operation_type)

        # 添加通用要求
        prompt += "\nGeneral Requirements:\n"
        prompt += "1. The Triton kernel should implement the EXACT same functionality as the baseline function\n"
        prompt += "2. The Python wrapper function name should match the baseline function name\n"
        prompt += "3. Handle edge cases (empty tensors, broadcasting, etc.) appropriately\n"
        prompt += "4. Use appropriate BLOCK_SIZE for optimal performance\n"
        prompt += "5. Include proper masking for boundary conditions\n"
        prompt += "6. The implementation should aim to match or exceed cuBLAS performance\n\n"

        # 添加 wiki reference（如果有）
        if self.mode == "with_wiki" and info.wiki_reference:
            prompt += self._build_wiki_section(info.wiki_reference)

        prompt += "\nPlease provide the complete Triton kernel implementation including:\n"
        prompt += "1. The @triton.jit decorated kernel function\n"
        prompt += "2. The Python wrapper function that calls the kernel\n"
        prompt += "3. Proper imports (triton, triton.language, torch)\n"

        # Add device-specific constraints
        prompt += self._get_device_constraints()

        return prompt

    def _get_blas_level_guidance(self, blas_type: str) -> str:
        """根据 BLAS Level 提供特定的实现指导"""
        guidance = f"\nBLAS {blas_type} Specific Guidance:\n"

        if blas_type == "Level 1":
            guidance += "- Level 1 BLAS operations are vector-vector operations\n"
            guidance += "- Focus on efficient 1D parallelization\n"
            guidance += "- Common operations: axpy, dot, scal, asum, nrm2\n"
            guidance += "- Typically use simple 1D grid with BLOCK_SIZE tuning\n"
        elif blas_type == "Level 2":
            guidance += "- Level 2 BLAS operations are matrix-vector operations\n"
            guidance += "- Consider both row-wise and column-wise parallelization\n"
            guidance += "- Common operations: gemv, ger, trmv, trsv\n"
            guidance += "- May need 2D grid for better performance\n"
        elif blas_type == "Level 3":
            guidance += "- Level 3 BLAS operations are matrix-matrix operations\n"
            guidance += "- Use 2D or 3D grid for parallelization\n"
            guidance += "- Common operations: gemm, syrk, trmm, trsm\n"
            guidance += "- Consider tiling and blocking strategies for cache efficiency\n"
            guidance += "- Pay attention to memory access patterns (coalescing)\n"
        else:  # Extension
            guidance += "- Extension operations may have custom patterns\n"
            guidance += "- Analyze the baseline function carefully\n"
            guidance += "- Choose appropriate parallelization strategy based on operation characteristics\n"

        return guidance + "\n"

    def build_fix(self, gen_args: BaseGenerateArgs) -> str:
        """
        生成修复 Triton kernel 的 prompt

        Args:
            gen_args: 生成参数，实际应为 CupyGenerateArgs 实例
        """
        # 类型转换
        if not isinstance(gen_args, CupyGenerateArgs):
            raise TypeError(f"CupyPromptBuilder requires CupyGenerateArgs, got {type(gen_args)}")

        info: CupyGenerateArgs = gen_args

        # 类型注解和断言：确保 check_result 不为 None
        check_result: 'VerifyResult' = info.check_result  # type: ignore
        assert check_result is not None, "check_result is required for build_fix"

        prompt = "You are a skilled GPU programmer proficient in Triton. Your task is to fix the following Triton kernel function that implements a cuBLAS baseline operation.\n\n"

        prompt += f"BLAS Operation Type: {info.blas_operation_type}\n"
        prompt += f"Operation: {info.cupy_kernel_name}\n\n"

        prompt += "The Triton kernel that needs to be fixed:\n"
        prompt += "```python\n"
        prompt += f"{info.check_result.code}\n"
        prompt += "```\n\n"

        prompt += "cuBLAS Baseline Function (reference):\n"
        prompt += "```python\n"
        prompt += f"{info.baseline_code}\n"
        prompt += "```\n\n"

        prompt += f"Error Information:\n"
        prompt += f"{info.check_result.traceback}\n\n"

        if hasattr(info.check_result, 'params') and info.check_result.params:
            prompt += f"Test Parameters:\n{info.check_result.params}\n\n"

        prompt += "Please analyze the error and provide a fixed version of the Triton kernel.\n"
        prompt += "Focus on:\n"
        prompt += "1. Identifying the root cause of the error\n"
        prompt += "2. Ensuring correctness against the baseline function\n"
        prompt += "3. Maintaining performance characteristics\n"
        prompt += "4. Proper error handling and edge cases\n\n"

        prompt += "Provide the complete fixed implementation.\n"

        return prompt

    def build_optimization(self, gen_args: BaseGenerateArgs) -> str:
        """
        生成优化 Triton kernel 的 prompt

        Args:
            gen_args: 生成参数，实际应为 CupyGenerateArgs 实例
        """
        # 类型转换
        if not isinstance(gen_args, CupyGenerateArgs):
            raise TypeError(f"CupyPromptBuilder requires CupyGenerateArgs, got {type(gen_args)}")

        info: CupyGenerateArgs = gen_args

        # 断言：确保 old_code 不为 None
        assert info.old_code is not None, "old_code is required for build_optimization"

        prompt = "You are a skilled GPU programmer proficient in Triton. Your task is to optimize the following Triton kernel function that implements a cuBLAS baseline operation.\n\n"

        prompt += f"BLAS Operation Type: {info.blas_operation_type}\n"
        prompt += f"Operation: {info.cupy_kernel_name}\n\n"

        prompt += "Current Triton kernel implementation:\n"
        prompt += "```python\n"
        prompt += f"{info.old_code}\n"
        prompt += "```\n\n"

        prompt += "cuBLAS Baseline Function (reference):\n"
        prompt += "```python\n"
        prompt += f"{info.baseline_code}\n"
        prompt += "```\n\n"

        if info.func_desc:
            prompt += f"Function Description:\n{info.func_desc}\n\n"

        prompt += "Optimization Goals:\n"
        prompt += "1. Improve performance to match or exceed cuBLAS\n"
        prompt += "2. Maintain correctness against the baseline function\n"
        prompt += "3. Consider memory access patterns and coalescing\n"
        prompt += "4. Optimize BLOCK_SIZE and grid configuration\n"
        prompt += "5. Reduce memory transactions where possible\n\n"

        # 根据 BLAS Level 提供优化建议
        prompt += self._get_optimization_hints(info.blas_operation_type)

        prompt += "\nProvide an optimized version of the Triton kernel with explanations of the optimizations made.\n"

        return prompt

    def _get_optimization_hints(self, blas_type: str) -> str:
        """根据 BLAS Level 提供优化提示"""
        hints = f"Optimization Hints for {blas_type}:\n"

        if blas_type == "Level 1":
            hints += "- Tune BLOCK_SIZE for vector operations (typically 256-1024)\n"
            hints += "- Minimize memory transactions\n"
            hints += "- Consider vectorized loads/stores\n"
        elif blas_type == "Level 2":
            hints += "- Consider different parallelization strategies (row-wise vs column-wise)\n"
            hints += "- Optimize memory access patterns\n"
            hints += "- Use shared memory for frequently accessed data\n"
        elif blas_type == "Level 3":
            hints += "- Implement tiling/blocking for cache efficiency\n"
            hints += "- Use shared memory for matrix tiles\n"
            hints += "- Optimize grid configuration (2D or 3D)\n"
            hints += "- Consider transpose operations for better coalescing\n"
        else:
            hints += "- Analyze the operation's memory access patterns\n"
            hints += "- Identify opportunities for parallelization\n"
            hints += "- Consider using shared memory for data reuse\n"

        return hints + "\n"

    def _build_wiki_section(self, wiki_reference: any) -> str:
        """构造 wiki reference 部分（如果需要）"""
        section = "\n" + "="*60 + "\n"
        section += "Reference Information:\n"
        section += "="*60 + "\n"
        section += f"{wiki_reference}\n\n"
        return section
