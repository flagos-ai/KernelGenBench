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
TorchAdapter - PyTorch framework adapter

Migrated torch-related logic from scripts/utils.py
"""

import torch
from typing import Any, Dict
from .adapter import FrameworkAdapter
from .generate_args import TritonKernelGenerateArgs


class TorchAdapter(FrameworkAdapter):
    """PyTorch framework adapter"""

    @property
    def framework_name(self) -> str:
        """Return the framework name"""
        return "torch"

    def get_operator_function(self, op_name: str) -> Any:
        """
        Get the operator function from torch.ops.

        Args:
            op_name: operator name, e.g. "aten::add"

        Returns:
            function object from torch.ops
        """
        # Parse namespace and kernel_name
        # Format: "aten::add" -> namespace="aten", kernel_name="add"
        if "::" not in op_name:
            raise ValueError(f"Invalid op_name format: {op_name}. Expected format: 'namespace::kernel_name'")

        namespace, kernel_name = op_name.split("::", 1)

        # Get namespace from torch.ops
        if not hasattr(torch.ops, namespace):
            raise AttributeError(f"torch.ops does not have namespace: {namespace}")

        torch_op_namespace = getattr(torch.ops, namespace)

        # Get kernel from namespace
        if not hasattr(torch_op_namespace, kernel_name):
            raise AttributeError(f"torch.ops.{namespace} does not have kernel: {kernel_name}")

        return getattr(torch_op_namespace, kernel_name)

    def get_signature_info(self, func: Any, op_name: str) -> Dict:
        """
        Get the signature info for a torch API.

        Args:
            func: function object from torch.ops
            op_name: operator name

        Returns:
            signature info dict containing input_args, output_args, func_desc
        """
        # Get function description
        func_desc = f"PyTorch operator: {op_name}"
        if hasattr(func, '__doc__') and func.__doc__:
            doc_lines = func.__doc__.strip()
            if doc_lines:
                func_desc = doc_lines

        # Get function signature
        input_args = self._extract_function_signature(func)

        return {
            "input_args": input_args,
            "output_args": None,
            "func_desc": func_desc,
        }

    def _extract_function_signature(self, func: Any) -> Dict[str, str]:
        """
        Extract function signature information.

        For torch.ops operators, use the _schemas attribute.
        """
        if hasattr(func, '_schemas'):
            schema = func._schemas
            # schema is a dict where keys are overload names and values are signature objects
            input_output_info = {k: str(v) for k, v in schema.items()}
            return input_output_info

        # If no _schemas attribute, return empty dict
        return {}

    def create_generate_args(self, op_name: str, func: Any, impl_info: Any) -> TritonKernelGenerateArgs:
        """
        Create a TritonKernelGenerateArgs instance.

        Args:
            op_name: operator name, e.g. "aten::add"
            func: function object from torch.ops
            impl_info: implementation info

        Returns:
            TritonKernelGenerateArgs instance
        """
        # Get signature info
        sig_info = self.get_signature_info(func, op_name)

        # Generate reference code
        torch_kernel_code = self.get_reference_code(func, op_name)

        # Create TritonKernelGenerateArgs
        return TritonKernelGenerateArgs(
            triton_kernel_name=op_name,
            func_desc=sig_info["func_desc"],
            torch_kernel_code=torch_kernel_code,
            input_args=sig_info["input_args"],
            output_args=sig_info["output_args"],
            impl_info=impl_info,
            from_mcp=False,
        )

    def get_reference_code(self, func: Any, op_name: str) -> str:
        """
        Generate torch reference code.

        Args:
            func: function object from torch.ops
            op_name: operator name, e.g. "aten::add"

        Returns:
            reference code string
        """
        # Extract kernel name
        # "aten::add" -> "add"
        kernel_name = op_name.split("::")[-1]

        # Build function name
        # e.g.: "torch.ops.aten.add"
        namespace = op_name.split("::")[0]
        torch_op_func_name = f"torch.ops.{namespace}.{kernel_name}"

        # Generate reference code
        torch_kernel_code = f"""
# Reference PyTorch implementation for {op_name}
import torch

{kernel_name} = {torch_op_func_name}
""".strip()

        return torch_kernel_code

    def get_impl_info(self, kernel_name: str) -> Any:
        """
        Get implementation info for a torch operator.

        Args:
            kernel_name: operator name without namespace, e.g. "add"

        Returns:
            implementation info retrieved from IMPL_INFO
        """
        from kernelgenbench.dataset import IMPL_INFO
        return IMPL_INFO.get(kernel_name)
