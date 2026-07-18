# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
CublasPromptBuilder - Prompt builder for the cuBLAS framework
"""

from typing import TYPE_CHECKING
from .prompt_builder import PromptBuilder
from kernelgenbench.framework.generate_args import BaseGenerateArgs, CublasGenerateArgs

if TYPE_CHECKING:
    from sandbox.utils.accuracy_utils import VerifyResult


class CublasPromptBuilder(PromptBuilder):
    """Prompt builder for the cuBLAS framework"""

    def build_new(self, gen_args: BaseGenerateArgs) -> str:
        if not isinstance(gen_args, CublasGenerateArgs):
            raise TypeError(f"CublasPromptBuilder requires CublasGenerateArgs, got {type(gen_args)}")

        info: CublasGenerateArgs = gen_args

        prompt = "You are a skilled GPU programmer proficient in Triton. Your task is to generate a Triton kernel function that implements the same functionality as a cuBLAS baseline function.\\n"

        prompt += "\\nHere is an example of a cuBLAS baseline function and its corresponding Triton kernel implementation:\\n"
        prompt += "cuBLAS baseline function (SAXPY):\\n"
        prompt += """```python
def saxpy(n, alpha, x, incx, y, incy):
    '''SAXPY: y = alpha * x + y'''
    # cuBLAS C API call via ctypes
    cublasSaxpy_v2(handle, n, alpha, x, incx, y, incy)
```
""".strip() + "\\n"

        prompt += "\\nTriton kernel implementation:\\n"
        prompt += """```python
import triton
import triton.language as tl

@triton.jit
def saxpy_kernel(n, alpha, x_ptr, incx, y_ptr, incy, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x_idx = offsets * incx
    y_idx = offsets * incy

    x = tl.load(x_ptr + x_idx, mask=mask)
    y = tl.load(y_ptr + y_idx, mask=mask)

    result = alpha * x + y
    tl.store(y_ptr + y_idx, result, mask=mask)

def saxpy(n, alpha, x, incx, y, incy):
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    saxpy_kernel[grid](n, alpha, x, incx, y, incy, BLOCK_SIZE=1024)
    return y
```
""".strip() + "\\n"

        prompt += f"\\n\\nNow, please generate a Triton kernel for the following cuBLAS baseline function:\\n"
        prompt += f"Operation: {info.cublas_kernel_name}\\n"
        prompt += f"Type: {info.blas_operation_type}\\n"
        prompt += f"Description: {info.func_desc}\\n\\n"
        prompt += f"Baseline function:\\n```python\\n{info.baseline_code}\\n```\\n"

        if info.impl_info:
            prompt += f"\\nOperator Interface Information:\\n"
            if isinstance(info.impl_info, dict):
                if "name" in info.impl_info:
                    prompt += f"Function Name: {info.impl_info['name']}\\n"
                if "operation" in info.impl_info:
                    prompt += f"Operation: {info.impl_info['operation']}\\n"
                if "dtype" in info.impl_info:
                    prompt += f"Data Type: {info.impl_info['dtype']}\\n"
                if "args" in info.impl_info:
                    prompt += f"Arguments: {len(info.impl_info['args'])} parameters\\n"

        # Extract function name from kernel_name
        func_name = info.cublas_kernel_name.split("::")[-1] if "::" in info.cublas_kernel_name else info.cublas_kernel_name

        # Add testing environment description
        prompt += f"\\n## Testing Environment\\n"
        prompt += f"Your implementation will be tested as follows:\\n"
        prompt += f"```python\\n"
        prompt += f"# Baseline (cuBLAS C API wrapper)\\n"
        prompt += f"from kernelgenbench.dataset.baseline.cublas_ctypes.{func_name} import {func_name} as baseline_{func_name}\\n"
        prompt += f"ref_out = baseline_{func_name}(...)\\n\\n"
        prompt += f"# Your Triton implementation\\n"
        prompt += f"import kernelgenbench\\n"
        prompt += f"act_out = kernelgenbench.triton.{func_name}(...)\\n\\n"
        prompt += f"# Accuracy verification\\n"
        prompt += f"assert_close(act_out, ref_out, dtype)\\n"
        prompt += f"```\\n"
        prompt += f"**If the function signature doesn't match, the test will fail immediately.**\\n"

        prompt += "\\n## Requirements\\n"
        prompt += "1. Implement the Triton kernel with proper memory coalescing\\n"
        prompt += "2. Use appropriate block sizes for GPU parallelization\\n"
        prompt += "3. Handle edge cases and boundary conditions\\n"
        prompt += "4. Ensure numerical stability\\n"

        # Add no-cheating notice
        prompt += "\\n## IMPORTANT - No Cheating\\n"
        prompt += "- You MUST implement the algorithm using Triton kernels (@triton.jit)\\n"
        prompt += "- Do NOT call the baseline function or cuBLAS C API directly\\n"
        prompt += "- Do NOT use ctypes to call cuBLAS functions\\n"
        prompt += "- Your implementation must be a pure Triton kernel solution\\n"

        # Add output format requirements
        prompt += "\\n## Output Format\\n"
        prompt += "Generate ONLY the Python code for the Triton kernel implementation:\\n"
        prompt += "- Use ```python ... ``` code block format\\n"
        prompt += "- Include all necessary imports (torch, triton, etc.)\\n"
        prompt += f"- Include the wrapper function with the EXACT SAME signature as baseline (function name: `{func_name}`)\\n"
        prompt += "- Include the Triton kernel(s) decorated with @triton.jit\\n"
        prompt += "- Do NOT include explanations or test code\\n"

        return prompt

    def build_fix(self, gen_args: BaseGenerateArgs) -> str:
        if not isinstance(gen_args, CublasGenerateArgs):
            raise TypeError(f"CublasPromptBuilder requires CublasGenerateArgs, got {type(gen_args)}")

        info: CublasGenerateArgs = gen_args
        has_history = info.history is not None and len(info.history) > 0

        prompt = "The previous Triton kernel implementation failed verification. Please analyze the error and generate a corrected version.\n\n"

        prompt += f"cuBLAS baseline function:\n```python\n{info.baseline_code}\n```\n\n"

        if has_history:
            # Multi-round history: show kernel + error per round
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
            # Single-round compatibility logic
            if info.old_code:
                prompt += f"Previous failed implementation:\n```python\n{info.old_code}\n```\n\n"
            if info.check_result:
                prompt += f"Error information:\n{info.check_result.traceback}\n\n"
                if info.check_result.params:
                    prompt += f"Test parameters:\n{info.check_result.params}\n\n"

        # Extract function name from kernel_name
        func_name = info.cublas_kernel_name.split("::")[-1] if "::" in info.cublas_kernel_name else info.cublas_kernel_name

        prompt += f"\nPlease fix the issues and provide a corrected implementation.\n"
        prompt += f"\n**CRITICAL**: Your implementation MUST include:\n"
        prompt += f"1. Triton kernel(s) decorated with @triton.jit\n"
        prompt += f"2. A wrapper function named `{func_name}` with the EXACT SAME signature as the baseline\n"
        prompt += f"3. All necessary imports (torch, triton, etc.)\n"
        prompt += f"\n**NO CHEATING**: Do NOT call cuBLAS C API or baseline functions directly. Implement using pure Triton kernels.\n"

        return prompt

    def build_optimization(self, gen_args: BaseGenerateArgs) -> str:
        # Optimization scenario: reuse build_fix logic for now
        return self.build_fix(gen_args)
