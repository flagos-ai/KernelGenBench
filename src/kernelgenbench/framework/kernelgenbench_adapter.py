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
KernelGenBench adapter

Routes to the corresponding framework adapter based on op prefix (aten::/vllm13::/cublas::).
"""

from typing import Any, Dict
from .adapter import FrameworkAdapter
from .generate_args import BaseGenerateArgs


class KernelGenBenchAdapter(FrameworkAdapter):
    """KernelGenBench adapter, dispatches to VllmAdapter, CublasAdapter or TorchAdapter based on op_name prefix."""

    def __init__(self):
        from .vllm_adapter import VllmAdapter
        from .cublas_adapter import CublasAdapter
        from .torch_adapter import TorchAdapter
        self.vllm_adapter = VllmAdapter()
        self.cublas_adapter = CublasAdapter()
        self.torch_adapter = TorchAdapter()

    def _get_adapter(self, op_name: str) -> FrameworkAdapter:
        if op_name.startswith("vllm13::") or op_name.startswith("vllm15::"):
            return self.vllm_adapter
        elif op_name.startswith("cublas::"):
            return self.cublas_adapter
        elif op_name.startswith("aten::"):
            return self.torch_adapter
        else:
            raise ValueError(f"Unknown op_name prefix: {op_name}")

    def get_operator_function(self, op_name: str) -> Any:
        return self._get_adapter(op_name).get_operator_function(op_name)

    def get_signature_info(self, func: Any, op_name: str) -> Dict:
        return self._get_adapter(op_name).get_signature_info(func, op_name)

    def create_generate_args(self, op_name: str, func: Any, impl_info: Any) -> BaseGenerateArgs:
        return self._get_adapter(op_name).create_generate_args(op_name, func, impl_info)

    def get_reference_code(self, func: Any, op_name: str) -> str:
        return self._get_adapter(op_name).get_reference_code(func, op_name)

    def get_impl_info(self, kernel_name: str) -> Any:
        result = self.vllm_adapter.get_impl_info(kernel_name)
        if result is not None:
            return result
        result = self.cublas_adapter.get_impl_info(kernel_name)
        if result is not None:
            return result
        return self.torch_adapter.get_impl_info(kernel_name)

    @property
    def framework_name(self) -> str:
        return "kernelgenbench"
