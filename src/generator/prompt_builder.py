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
PromptBuilder layer

Responsible for constructing prompts used by the LLM to generate Triton kernels.
"""

from abc import ABC, abstractmethod
from kernelgenbench.framework.generate_args import BaseGenerateArgs
from runtime import get_device_constraints


class PromptBuilder(ABC):
    """Base class for prompt builders"""

    def __init__(self, mode: str = "basic"):
        self.mode = mode

    def _get_device_constraints(self) -> str:
        """Get device-specific prompt constraints for the current hardware."""
        return get_device_constraints()

    @abstractmethod
    def build_new(self, gen_args: BaseGenerateArgs) -> str:
        """Build prompt for generating a new kernel"""
        pass

    @abstractmethod
    def build_fix(self, gen_args: BaseGenerateArgs) -> str:
        """Build prompt for fixing a kernel"""
        pass

    @abstractmethod
    def build_optimization(self, gen_args: BaseGenerateArgs) -> str:
        """Build prompt for optimizing a kernel"""
        pass

    def build(self, gen_args: BaseGenerateArgs) -> str:
        """
        Select the appropriate prompt construction method based on gen_args state
        """
        if gen_args.check_result is not None and not gen_args.check_result.success:
            if gen_args.old_code and gen_args.check_result.code and \
               gen_args.check_result.code.strip() == gen_args.old_code.strip():
                return self.build_fix(gen_args)
            else:
                return self.build_optimization(gen_args)
        if gen_args.old_code is not None and len(gen_args.old_code.strip()) > 0:
            return self.build_optimization(gen_args)
        return self.build_new(gen_args)
