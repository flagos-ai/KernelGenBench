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
FrameworkAdapter base class

Defines a unified interface for framework adapters, used to encapsulate framework-specific logic.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
from .generate_args import BaseGenerateArgs


class FrameworkAdapter(ABC):
    """Framework adapter base class, defines a unified interface"""

    @abstractmethod
    def get_operator_function(self, op_name: str) -> Any:
        """
        Get the operator function object.

        Args:
            op_name: operator name, e.g. "aten::add", "cublas::sgemm", "vllm13::rms_norm"

        Returns:
            function object
        """
        pass

    @abstractmethod
    def get_signature_info(self, func: Any, op_name: str) -> Dict:
        """
        Get function signature information.

        Args:
            func: function object
            op_name: operator name

        Returns:
            signature information dictionary
        """
        pass

    @abstractmethod
    def create_generate_args(self, op_name: str, func: Any, impl_info: Any) -> BaseGenerateArgs:
        """
        Create a generate args object.

        Args:
            op_name: operator name
            func: function object
            impl_info: implementation info

        Returns:
            GenerateArgs object
        """
        pass

    @abstractmethod
    def get_reference_code(self, func: Any, op_name: str) -> str:
        """
        Generate reference code.

        Args:
            func: function object
            op_name: operator name

        Returns:
            reference code string
        """
        pass

    @abstractmethod
    def get_impl_info(self, kernel_name: str) -> Any:
        """
        Get implementation info for an operator.

        Args:
            kernel_name: operator name without namespace, e.g. "add", "caxpy"

        Returns:
            implementation info, or None for frameworks that do not need it
        """
        pass

    @property
    @abstractmethod
    def framework_name(self) -> str:
        """Framework name"""
        pass
