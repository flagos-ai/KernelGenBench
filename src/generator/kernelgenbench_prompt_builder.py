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
KernelGenBench prompt builder

Routes to the corresponding prompt builder based on generate_args type.
"""

from typing import Any


class KernelGenBenchPromptBuilder:
    """KernelGenBench prompt builder - dispatches based on args type"""

    def __init__(self, mode: str = "basic"):
        from .vllm_prompt_builder import VllmPromptBuilder
        from .cublas_prompt_builder import CublasPromptBuilder
        from .torch_prompt_builder import TorchPromptBuilder
        self.vllm_builder = VllmPromptBuilder(mode=mode)
        self.cublas_builder = CublasPromptBuilder(mode=mode)
        self.torch_builder = TorchPromptBuilder(mode=mode)

    def _get_builder(self, gen_args: Any):
        from kernelgenbench.framework.generate_args import VllmGenerateArgs, CublasGenerateArgs, TritonKernelGenerateArgs
        if isinstance(gen_args, VllmGenerateArgs):
            return self.vllm_builder
        elif isinstance(gen_args, CublasGenerateArgs):
            return self.cublas_builder
        elif isinstance(gen_args, TritonKernelGenerateArgs):
            return self.torch_builder
        else:
            raise TypeError(f"Unknown args type: {type(gen_args)}")

    def build(self, gen_args: Any):
        return self._get_builder(gen_args).build(gen_args)

    def build_new(self, gen_args: Any):
        return self._get_builder(gen_args).build_new(gen_args)

    def build_fix(self, gen_args: Any):
        return self._get_builder(gen_args).build_fix(gen_args)

    def build_optimization(self, gen_args: Any):
        return self._get_builder(gen_args).build_optimization(gen_args)
