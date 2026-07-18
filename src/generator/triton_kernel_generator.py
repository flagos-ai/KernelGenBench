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

from .generator import BaseGenerator, print_prompt, console
from generator.sampler.utils import extract_first_code
import warnings
from typing import Optional

from .sampler.generate_samples import (
    TritonKernelGenerateArgs,
)
from .prompt_builder import PromptBuilder


class TritonKernelGenerator(BaseGenerator):
    def __init__(self, generation_config, prompt_builder: Optional[PromptBuilder] = None):
        """
        Initialize TritonKernelGenerator

        Args:
            generation_config: Generation configuration
            prompt_builder: Prompt builder (optional). If not provided, TorchPromptBuilder will be created automatically
        """
        super().__init__(generation_config)

        # Backward compatibility: if prompt_builder is not provided, create TorchPromptBuilder automatically
        if prompt_builder is None:
            warnings.warn(
                "TritonKernelGenerator: prompt_builder parameter is not provided. "
                "Automatically creating TorchPromptBuilder for backward compatibility. "
                "This behavior is deprecated and will be removed in a future version. "
                "Please explicitly pass a PromptBuilder instance.",
                DeprecationWarning,
                stacklevel=2
            )
            from .torch_prompt_builder import TorchPromptBuilder
            prompt_builder = TorchPromptBuilder()

        self.prompt_builder = prompt_builder

    # @print_prompt
    def generate_prompt(self, info):
        """
        Generate prompt (delegated to PromptBuilder)

        Args:
            info: Generation arguments (BaseGenerateArgs or subclass)

        Returns:
            Generated prompt string
        """
        # Print appropriate message based on state
        if info.check_result is not None and info.check_result.success is False:
            if info.old_code and info.check_result.code.strip() == info.old_code.strip():
                console.print("Generating prompt for Triton kernel fix...")
            else:
                console.print("Generating prompt for Triton kernel optimization...")
        elif info.old_code is not None and len(info.old_code.strip()) > 0:
            console.print("Generating prompt for Triton kernel optimization...")
        else:
            console.print("Generating prompt for new Triton kernel...")

        # Delegate to PromptBuilder
        return self.prompt_builder.build(info)

    def _init_data(self, kwargs):
        # Save original type
        original_type = type(kwargs)

        config = kwargs.dict()
        if "from_mcp" in config:
            self.from_mcp = config.pop("from_mcp")
        if "check_result" in config and isinstance(config["check_result"], dict):
            from sandbox.utils.accuracy_utils import VerifyResult
            config["check_result"] = VerifyResult(**config["check_result"])

        # ai_advice is only valid for TritonKernelGenerateArgs
        from kernelgenbench.framework.generate_args import TritonKernelGenerateArgs
        if self.generation_config.use_ai_advice is True and original_type == TritonKernelGenerateArgs:
            console.print("AI advice generation is enabled for Triton kernel generator.")
            self.ai_advice = TritonKernelAdviceGenerator(self.generation_config)
            config["user_advice"] = self.ai_advice(TritonKernelGenerateArgs(**config))

        # Restore to original type (do not force cast to TritonKernelGenerateArgs)
        config = original_type(**config)

        # Use the generic op_name property
        self.kernel_name = config.op_name

        return config
    
    def post_process(self, results: list) -> list:
        codes = super().post_process(results)
        names = [r[1].op_name for r in results]
        sample_id = [r[1].sample_id for r in results]
        processed_results = []
        for res in codes:
            extracted_code = extract_first_code(res, ["python", "cpp"])
            if extracted_code is not None:
                processed_results.append(extracted_code.strip())
            else:
                console.print("Code extraction failed, using raw output.")
                if res is None:
                    processed_results.append('')
                else:
                    processed_results.append(res.strip())
        
        # Apply decouple_bench if from_mcp is True
        if self.from_mcp:
            for i in range(len(processed_results)):
                processed_results[i] = self.decouple_bench(processed_results[i], self.kernel_name)
            return processed_results
        else:
            return [[code, name, sample_id] for code, name, sample_id in zip(processed_results, names, sample_id)]

    def decouple_bench(self, code: str, kernel_name: str = None) -> str:
        """
        Convert Triton kernel function name for MCP compatibility.
        - Replace def {kernel_name}( with def {kernel_name}_triton(
        """
        import re

        kernel_name = kernel_name or self.kernel_name
        assert kernel_name is not None, "kernel_name must be provided either in argument or as self.kernel_name"
        
        # Replace the wrapper function name: def {kernel_name}( with def {kernel_name}_triton(
        # Use word boundary to ensure exact match of function name
        pattern = rf'\bdef\s+{re.escape(kernel_name)}\s*\('
        replacement = f'def {kernel_name}_triton('
        code = re.sub(pattern, replacement, code)
        
        return code


class TritonKernelAdviceGenerator(BaseGenerator):
    def __init__(self, generation_config):
        super().__init__(generation_config)

    @print_prompt
    def generate_prompt(self, info: TritonKernelGenerateArgs):
        if info.check_result is not None and info.check_result.success is False:
            if info.check_result.code.strip() == info.old_code.strip():
                console.print("Generating prompt for Triton kernel fix advice...")
                return self.generate_prompt_for_fix(info)
            else:
                console.print("Generating prompt for Triton kernel optimization advice...")
                return self.generate_prompt_for_optimization(info)
        if info.old_code is not None and len(info.old_code.strip()) > 0:
            console.print("Generating prompt for Triton kernel optimization advice...")
            return self.generate_prompt_for_optimization(info)
        console.print("No advice generated for new Triton kernel.")
        return ""
    
    def generate_prompt_for_fix(self, info: TritonKernelGenerateArgs):
        prompt = f"You are a skilled GPU programmer proficient in Triton. Your task is to provide advice on how to fix the following Triton kernel function that implements the same functionality as a given PyTorch function. You must strictly adhere to the following specifications:\n"
        prompt += f"The Triton kernel function that needs to be fixed is\n"
        prompt += "```python\n"
        prompt += f"{info.check_result.code}\n"
        prompt += "```\n"
        prompt += f"The Triton kernel got the following error when running the test function: {info.check_result.traceback}\n"
        prompt += f"The parameters of the test function are as follows:\n"
        prompt += f"{info.check_result.params}\n"
        prompt += f"Please provide concise and specific advice on what needs to be fixed in the Triton kernel function to resolve the error. Focus on the most likely issues based on the error message and the provided parameters.\n"
        prompt += f"Do not consider the logical correctness of the torch function, the test function.\n"
        prompt += f"Make sure keep your advice concise and to the point, do not provide lengthy explanations.\n"
        prompt += f"Do not provide the fixed code, only provide advice on what needs to be changed or improved in the existing code.\n"

        return prompt
    
    def generate_prompt_for_optimization(self, info: TritonKernelGenerateArgs):
        prompt = f"You are a skilled GPU programmer proficient in Triton. Your task is to provide advice on how to optimize the following Triton kernel function that implements the same functionality as a given PyTorch function. You must strictly adhere to the following specifications:\n"
        prompt += f"The Triton kernel function that needs to be optimized is\n"
        prompt += "```python\n"
        prompt += f"{info.old_code}\n"
        prompt += "```\n"
        prompt += f"The function should implement the same functionality as user provided description: {info.func_desc}\n"
        prompt += f"And reference PyTorch function is as follows:\n"
        prompt += "```python\n"
        prompt += f"{info.torch_kernel_code}\n"
        prompt += "```\n"
        prompt += f"Please provide concise and specific advice on what needs to be optimized in the Triton kernel function to fix correctness or improve performance. Focus on the most likely issues based on the provided function description and the reference PyTorch function.\n"
        prompt += f"Do not consider the logical correctness of the torch function, the test function.\n"
        prompt += f"Make sure keep your advice concise and to the point, do not provide lengthy explanations.\n"
        prompt += f"Do not provide the optimized code, only provide advice on what needs to be changed or improved in the existing code.\n"
        
        return prompt
    
    def post_process(self, results: list) -> list:
        results = super().post_process(results)
        print("post process advice results:", results)
         # only return the first result
        return results[0]
