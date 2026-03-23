"""
VllmPromptBuilder - vLLM 框架的 Prompt 构造器

用于从 vLLM baseline 函数生成 Triton kernel 的 prompt
"""

from typing import TYPE_CHECKING

from .prompt_builder import PromptBuilder
from flagbench.framework.generate_args import BaseGenerateArgs, VllmGenerateArgs

if TYPE_CHECKING:
    from sandbox.utils.accuracy_utils import VerifyResult


class VllmPromptBuilder(PromptBuilder):
    """vLLM 框架的 Prompt 构造器 - 用于从 vLLM baseline 生成 Triton kernel 的 prompt"""

    def build_new(self, gen_args: BaseGenerateArgs) -> str:
        """
        生成新 Triton kernel 的 prompt（从 vLLM baseline）

        Args:
            gen_args: 生成参数，实际应为 VllmGenerateArgs 实例
        """
        if not isinstance(gen_args, VllmGenerateArgs):
            raise TypeError(f"VllmPromptBuilder requires VllmGenerateArgs, got {type(gen_args)}")

        info: VllmGenerateArgs = gen_args

        prompt = "You are a skilled GPU programmer proficient in Triton. Your task is to generate a Triton kernel function that implements the same functionality as a vLLM baseline function.\n"

        # 提供 vLLM baseline 示例
        prompt += "\nHere is an example of a vLLM baseline function and its corresponding Triton kernel implementation:\n"
        prompt += "vLLM baseline function (RMS Norm):\n"
        prompt += """```python
def rms_norm_baseline(out, input, weight, epsilon):
    '''RMS normalization'''
    variance = input.pow(2).mean(-1, keepdim=True)
    input_normalized = input * torch.rsqrt(variance + epsilon)
    out.copy_(input_normalized * weight)
```
""".strip() + "\n"

        prompt += "\nTriton kernel implementation (includes both kernel and wrapper function):\n"
        prompt += """```python
import torch
import triton
import triton.language as tl

@triton.jit
def _rms_norm_kernel(
    output_ptr, input_ptr, weight_ptr,
    n_cols, epsilon,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    input_ptrs = input_ptr + row_idx * n_cols + col_offsets
    input_row = tl.load(input_ptrs, mask=mask, other=0.0)

    variance = tl.sum(input_row * input_row, axis=0) / n_cols
    rstd = 1 / tl.sqrt(variance + epsilon)

    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0)
    output = input_row * rstd * weight

    output_ptrs = output_ptr + row_idx * n_cols + col_offsets
    tl.store(output_ptrs, output, mask=mask)

# Wrapper function with EXACT SAME signature as baseline
def rms_norm_baseline(out, input, weight, epsilon):
    n_rows, n_cols = input.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (n_rows,)
    _rms_norm_kernel[grid](out, input, weight, n_cols, epsilon, BLOCK_SIZE=BLOCK_SIZE)
```
""".strip() + "\n"

        # 提供当前任务的 baseline 函数
        prompt += f"\n\nNow, please generate a Triton kernel for the following vLLM baseline function:\n"
        prompt += f"Operation: {info.vllm_kernel_name}\n"
        prompt += f"Type: {info.operation_type}\n"
        prompt += f"Description: {info.func_desc}\n\n"
        prompt += f"Baseline function:\n```python\n{info.baseline_code}\n```\n"

        # 添加 impl_info 信息（如果存在）
        if info.impl_info:
            prompt += f"\nOperator Interface Information:\n"
            if isinstance(info.impl_info, dict):
                if "signature" in info.impl_info:
                    prompt += f"Signature: {info.impl_info['signature']}\n"
                if "input_parameters" in info.impl_info:
                    prompt += f"Input Parameters: {info.impl_info['input_parameters']}\n"
                if "output_parameters" in info.impl_info:
                    prompt += f"Output Parameters: {info.impl_info['output_parameters']}\n"
                if "vllm_api" in info.impl_info:
                    prompt += f"API Call: {info.impl_info['vllm_api']}\n"

        # 从 kernel_name 提取函数名
        func_name = info.vllm_kernel_name.split("::")[-1] if "::" in info.vllm_kernel_name else info.vllm_kernel_name

        # 添加测试环境说明
        prompt += f"\n## Testing Environment\n"
        prompt += f"Your implementation will be tested as follows:\n"
        prompt += f"```python\n"
        prompt += f"# Baseline\n"
        prompt += f"from flagbench.dataset.baseline.vllm.{func_name} import {func_name} as baseline_{func_name}\n"
        prompt += f"ref_out = baseline_{func_name}(...)\n\n"
        prompt += f"# Your Triton implementation\n"
        prompt += f"import flagbench\n"
        prompt += f"act_out = flagbench.triton.{func_name}(...)\n\n"
        prompt += f"# Accuracy verification\n"
        prompt += f"assert_close(act_out, ref_out, dtype)\n"
        prompt += f"```\n"
        prompt += f"**If the function signature doesn't match, the test will fail immediately.**\n"

        # 添加实现要求
        prompt += "\n## Requirements\n"
        prompt += "1. Implement the Triton kernel with proper memory coalescing\n"
        prompt += "2. Use appropriate block sizes for GPU parallelization\n"
        prompt += "3. Handle edge cases and boundary conditions\n"
        prompt += "4. Ensure numerical stability\n"

        # 添加禁止hack说明
        prompt += "\n## IMPORTANT - No Cheating\n"
        prompt += "- You MUST implement the algorithm using Triton kernels (@triton.jit)\n"
        prompt += "- Do NOT call the baseline function or _custom_ops directly\n"
        prompt += "- Do NOT import or use vllm._custom_ops, _custom_ops, or any CUDA C++ extensions\n"
        prompt += "- Your implementation must be a pure Triton kernel solution\n"

        # 添加输出格式要求
        prompt += "\n## Output Format\n"
        prompt += "Generate ONLY the Python code for the Triton kernel implementation:\n"
        prompt += "- Use ```python ... ``` code block format\n"
        prompt += "- Include all necessary imports (torch, triton, etc.)\n"
        prompt += f"- Include the wrapper function with the EXACT SAME signature as baseline (function name: `{func_name}`)\n"
        prompt += "- Include the Triton kernel(s) decorated with @triton.jit\n"
        prompt += "- Do NOT include explanations or test code\n"

        # Add device-specific constraints
        prompt += self._get_device_constraints()

        return prompt

    def build_fix(self, gen_args: BaseGenerateArgs) -> str:
        """
        生成修复 prompt（基于验证失败结果，支持memory history）

        Args:
            gen_args: 生成参数，包含 check_result、old_code 和 history
        """
        if not isinstance(gen_args, VllmGenerateArgs):
            raise TypeError(f"VllmPromptBuilder requires VllmGenerateArgs, got {type(gen_args)}")

        info: VllmGenerateArgs = gen_args
        has_history = info.history is not None and len(info.history) > 0

        prompt = "The previous Triton kernel implementation failed verification. Please analyze the error and generate a corrected version.\n\n"

        # 提供原始 baseline
        prompt += f"vLLM baseline function:\n```python\n{info.baseline_code}\n```\n\n"

        if has_history:
            # 多轮历史：逐轮展示 kernel + 报错
            history = info.history or []
            prompt += f"Below are ALL previous attempts ({len(history)} round(s)) and their error messages. Learn from each failure:\n"
            for entry in history:
                round_num = entry.get("round", "?")
                code = entry.get("code") or ""
                traceback = entry.get("traceback") or "(no error recorded)"
                params = entry.get("params")
                prompt += f"\n--- Attempt Round {round_num} ---\n"
                prompt += f"Kernel code:\n```python\n{code}\n```\n"
                prompt += f"Error message:\n{traceback}\n"
                if params:
                    prompt += f"Test parameters:\n{params}\n"
        else:
            # 单轮兼容逻辑
            if info.old_code:
                prompt += f"Previous failed implementation:\n```python\n{info.old_code}\n```\n\n"
            if info.check_result:
                prompt += f"Error information:\n{info.check_result.traceback}\n\n"
                if info.check_result.params:
                    prompt += f"Test parameters:\n{info.check_result.params}\n\n"

        # 从 kernel_name 提取函数名
        func_name = info.vllm_kernel_name.split("::")[-1] if "::" in info.vllm_kernel_name else info.vllm_kernel_name

        prompt += f"\nPlease fix the issues and provide a corrected implementation.\n"
        prompt += f"\n**CRITICAL**: Your implementation MUST include:\n"
        prompt += f"1. Triton kernel(s) decorated with @triton.jit\n"
        prompt += f"2. A wrapper function named `{func_name}` with the EXACT SAME signature as the baseline\n"
        prompt += f"3. All necessary imports (torch, triton, etc.)\n"
        prompt += f"\n**NO CHEATING**: Do NOT call _custom_ops or baseline functions directly. Implement using pure Triton kernels.\n"

        return prompt

    def build_optimization(self, gen_args: BaseGenerateArgs) -> str:
        """优化场景：暂时复用build_fix逻辑"""
        return self.build_fix(gen_args)
