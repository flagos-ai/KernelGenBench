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
TorchPromptBuilder - Prompt builder for the Torch framework

Migrated prompt generation logic from TritonKernelGenerator
"""

from typing import TYPE_CHECKING

from .prompt_builder import PromptBuilder
from kernelgenbench.framework.generate_args import BaseGenerateArgs, TritonKernelGenerateArgs

if TYPE_CHECKING:
    from sandbox.utils.accuracy_utils import VerifyResult


class TorchPromptBuilder(PromptBuilder):
    """Prompt builder for the Torch framework - generates Triton kernel prompts"""

    def build_new(self, gen_args: BaseGenerateArgs) -> str:
        """
        Generate prompt for a new Triton kernel

        Migrated from TritonKernelGenerator.generate_prompt_for_new

        Args:
            gen_args: Generation arguments, should be a TritonKernelGenerateArgs instance
        """
        # Type check: ensure gen_args is TritonKernelGenerateArgs
        if not isinstance(gen_args, TritonKernelGenerateArgs):
            raise TypeError(f"TorchPromptBuilder requires TritonKernelGenerateArgs, got {type(gen_args)}")

        info: TritonKernelGenerateArgs = gen_args

        prompt = f"You are a skilled GPU programmer proficient in Triton. Your task is to generate a Triton kernel function.\n"
        prompt += f"Here is an example of a PyTorch function and its corresponding Triton kernel implementation:\n"
        prompt += f"PyTorch function:\n"
        prompt += """def add(A, B):
        return torch.add(A, B)
    """.strip() + "\n"
        prompt += f"Triton kernel implementation:\n"
        prompt += """import triton
    import triton.language as tl


    @triton.jit
    def add_kernel(x_ptr,  # *Pointer* to first input vector.
                y_ptr,  # *Pointer* to second input vector.
                output_ptr,  # *Pointer* to output vector.
                n_elements,  # Size of the vector.
                BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
                # NOTE: `constexpr` so it can be used as a shape value.
                ):
        pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)

    def add(x: torch.Tensor, y: torch.Tensor):
        output = torch.empty_like(x)
        assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
        add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
        return output
    """.strip() + "\n"
        prompt += f"You must strictly adhere to the following specifications:\n"
        prompt += f"The Triton kernel should implement the same functionality as the following PyTorch function:\n"
        prompt += "```python\n"
        prompt += f"{info.torch_kernel_code}\n"
        prompt += "```\n"

        # Add impl_info guidance if available and contains multiple operators
        if info.impl_info is not None and len(info.impl_info) > 1:
            prompt += f"\nIMPORTANT: This PyTorch API is internally implemented using multiple ATen C++ operators. "
            prompt += f"You need to provide separate Python wrapper functions for each ATen operator interface, but they can share the same Triton kernel(s).\n"
            prompt += f"The ATen operators involved are:\n"
            for op in info.impl_info:
                prompt += f"  - {op}\n"
            prompt += f"\nYour implementation should include:\n"
            prompt += f"1. One or more @triton.jit decorated kernel function(s) that implement the core computation logic\n"
            prompt += f"2. Multiple Python wrapper functions, each corresponding to one ATen operator interface\n"
            prompt += f"3. For syntax considerations, replace '.' with '_' in the operator names when defining the Python wrapper function names\n"
            prompt += f"\nThe Python wrapper functions should have signatures like:\n"
            for op, _ in info.impl_info:
                py_op_name = op.replace(".", "_")
                prompt += f"  - def {py_op_name}(...): # This wrapper calls the kernel(s)\n"
            prompt += f"\nThese wrapper functions can internally call the same Triton kernel(s) with appropriate parameters.\n"
        else:
            prompt += f"The Python interface function name should be {info.op_name.split('::')[-1]}.\n"

        prompt += f"\nThe input and output args of the function are as follows:\n"
        prompt += f"Input and output Args: \n"

        # Check if input_args exists
        if info.input_args:
            for overload, args in info.input_args.items():
                if overload:
                    prompt += f"  - {info.op_name.split('::')[-1]}_{overload}\n"
                else:
                    prompt += f"  - {info.op_name.split('::')[-1]}\n"
                prompt += f"    Input and Output Args: {args}\n"

        # prompt += f"Output Args: {info.output_args}\n"
        prompt += f"The generated code should include both the Triton kernel definition (with @triton.jit) and the Python wrapper function(s) that launch the kernel.\n"

        # Add critical requirements for broadcast and non-contiguous tensors
        prompt += f"\nCRITICAL REQUIREMENTS:\n"
        prompt += f"1. For pointwise operators, you MUST handle broadcasting correctly. Ensure your kernel supports inputs with different shapes that can be broadcast together according to PyTorch's broadcasting semantics.\n"
        prompt += f"2. You MUST handle non-contiguous tensors correctly. Do not assume input tensors are contiguous in memory. Use proper stride calculations to access elements correctly for tensors with arbitrary memory layouts.\n"

        # Modify this part based on whether multiple operators are involved
        if info.impl_info is not None and len(info.impl_info) > 1:
            prompt += f"You must provide wrapper functions for all the ATen operators mentioned above.\n"
        else:
            prompt += f"The wrapper function name must be exactly the same as the provided function name: {info.op_name.split('::')[-1]}\n"

        if info.user_advice:

            prompt += f"And the following user advice should be considered: {info.user_advice}\n"

        prompt += f"You must generate the valid Triton code directly without any explanations or additional text, and ensure no testing or benchmarking code is included. The code must be complete and ready to run.\n"
        prompt += f"You must use ```python ... ``` to format the code block.\n"

        # Add device constraints
        prompt += self._get_device_constraints()

        return prompt

    def build_fix(self, gen_args: BaseGenerateArgs) -> str:
        """
        Generate prompt for fixing a Triton kernel

        Migrated from TritonKernelGenerator.generate_prompt_for_fix

        Args:
            gen_args: Generation arguments, should be a TritonKernelGenerateArgs instance
        """
        # Type check: ensure gen_args is TritonKernelGenerateArgs
        if not isinstance(gen_args, TritonKernelGenerateArgs):
            raise TypeError(f"TorchPromptBuilder requires TritonKernelGenerateArgs, got {type(gen_args)}")

        info: TritonKernelGenerateArgs = gen_args

        # Type annotation and assertion: ensure check_result is not None
        check_result: 'VerifyResult' = info.check_result  # type: ignore
        assert check_result is not None, "check_result is required for build_fix"

        op_name = check_result.op_name if check_result.op_name else info.op_name

        prompt = f"You are a skilled GPU programmer proficient in Triton. Your task is to fix the following Triton kernel function that implements the same functionality as a given PyTorch function. You must strictly adhere to the following specifications:\n"

        prompt += f"The Triton kernel function that needs to be fixed is:\n"
        prompt += "```python\n"
        prompt += f"{check_result.code}\n"
        prompt += "```\n"

        prompt += f"The Triton kernel should implement the same functionality as the following PyTorch function:\n"
        prompt += "```python\n"
        prompt += f"{info.torch_kernel_code}\n"
        prompt += "```\n"

        prompt += f"The error message from the previous test run is:\n"
        prompt += f"{check_result.traceback}\n"

        if check_result.params:
            prompt += f"\nThe parameters of the test function are as follows:\n"
            prompt += f"{check_result.params}\n"

        # Add impl_info guidance if available and contains multiple operators
        if info.impl_info is not None and len(info.impl_info) > 1:
            prompt += f"\nIMPORTANT: This PyTorch API is internally implemented using multiple ATen C++ operators. "
            prompt += f"You need to provide separate Python wrapper functions for each ATen operator interface, but they can share the same Triton kernel(s).\n"
            prompt += f"The ATen operators involved are:\n"
            for op in info.impl_info:
                prompt += f"  - {op}\n"
            prompt += f"\nYour implementation should include:\n"
            prompt += f"1. One or more @triton.jit decorated kernel function(s) that implement the core computation logic\n"
            prompt += f"2. Multiple Python wrapper functions, each corresponding to one ATen operator interface\n"
            prompt += f"3. For syntax considerations, replace '.' with '_' in the operator names when defining the Python wrapper function names\n"
            prompt += f"\nThe Python wrapper functions should have signatures like:\n"
            for op, _ in info.impl_info:
                py_op_name = op.replace(".", "_")
                prompt += f"  - def {py_op_name}(...): # This wrapper calls the shared kernel(s)\n"
            prompt += f"\nThese wrapper functions can internally call the same Triton kernel(s) with appropriate parameters.\n"
        else:
            prompt += f"\nThe Triton kernel function name should be {op_name}.\n"

        prompt += f"\nThe input and output args of the function are as follows:\n"
        prompt += f"Input Args: {info.input_args}\n"
        prompt += f"Output Args: {info.output_args}\n"
        prompt += f"The generated code should include both the Triton kernel definition (with @triton.jit) and the Python wrapper function(s) that launch the kernel.\n"

        # Add critical requirements for broadcast and non-contiguous tensors
        prompt += f"\nCRITICAL REQUIREMENTS:\n"
        prompt += f"1. For pointwise operators, you MUST handle broadcasting correctly. Ensure your kernel supports inputs with different shapes that can be broadcast together according to PyTorch's broadcasting semantics.\n"
        prompt += f"2. You MUST handle non-contiguous tensors correctly. Do not assume input tensors are contiguous in memory. Use proper stride calculations to access elements correctly for tensors with arbitrary memory layouts.\n"

        # Modify this part based on whether multiple operators are involved
        if info.impl_info is not None and len(info.impl_info) > 1:
            prompt += f"You must provide wrapper functions for all the ATen operators mentioned above.\n"
        else:
            prompt += f"The wrapper function name must be exactly the same as the provided function name: {op_name}\n"

        if info.user_advice:

            prompt += f"And the following user advice should be considered: {info.user_advice}\n"

        prompt += f"You must generate the valid Triton code directly without any explanations or additional text, and ensure no testing or benchmarking code is included. The code must be complete and ready to run.\n"
        prompt += f"You must use ```python ... ``` to format the code block.\n"
        prompt += f"Do not consider the logical correctness of the torch function, only focus on fixing the Triton kernel logic based on the error message provided above.\n"

        # Add device constraints
        prompt += self._get_device_constraints()

        return prompt

    def build_optimization(self, gen_args: BaseGenerateArgs) -> str:
        """
        Generate prompt for optimizing a Triton kernel

        Migrated from TritonKernelGenerator.generate_prompt_for_optimization

        Args:
            gen_args: Generation arguments, should be a TritonKernelGenerateArgs instance
        """
        # Type check: ensure gen_args is TritonKernelGenerateArgs
        if not isinstance(gen_args, TritonKernelGenerateArgs):
            raise TypeError(f"TorchPromptBuilder requires TritonKernelGenerateArgs, got {type(gen_args)}")

        info: TritonKernelGenerateArgs = gen_args

        prompt = f"You are a skilled GPU programmer proficient in Triton. Your task is to optimize the following Triton kernel function that implements the same functionality as a given PyTorch function. You must strictly adhere to the following specifications:\n"
        prompt += f"The Triton kernel function that needs to be optimized is\n"
        prompt += "```python\n"
        prompt += f"{info.old_code}\n"
        prompt += "```\n"
        prompt += f"The function should implement the same functionality as user provided description: {info.func_desc}\n"
        prompt += f"The Triton kernel should implement the same functionality as the following PyTorch function:\n"
        prompt += "```python\n"
        prompt += f"{info.torch_kernel_code}\n"
        prompt += "```\n"
        prompt += f"Make sure to consider the input and output args of the function as follows:\n"
        prompt += f"Input Args: {info.input_args}\n"
        prompt += f"Output Args: {info.output_args}\n"
        prompt += f"The generated code should include both the Triton kernel definition (with @triton.jit) and the Python wrapper function that launches the kernel.\n"
        prompt += f"The wrapper function name must be exactly the same as the provided function name: {info.op_name}\n"

        if info.user_advice:

            prompt += f"And the following user advice should be considered: {info.user_advice}\n"

        prompt += f"You must generate the valid Triton code directly without any explanations or additional text, and ensure no testing or benchmarking code is included. The code must be complete and ready to run.\n"
        prompt += f"You must use ```python ... ``` to format the code block.\n"

        return prompt
