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
GenerateArgs hierarchy

Defines argument classes used for Triton kernel generation.
"""

from pydantic import BaseModel
from typing import Optional, Any, List
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class InputArg:
    """Input argument definition"""
    arg_name: str
    arg_type: str
    arg_value: Any = None
    arg_default: Any = None
    arg_desc: str = ""


@dataclass
class OutputArg:
    """Output argument definition"""
    arg_type: str
    arg_value: Any = None
    arg_desc: str = ""


class BaseGenerateArgs(BaseModel, ABC):
    """Generate args base class - responsible only for storing operator information"""

    # Common fields (inherited from existing BaseGenerateArgs)
    from_mcp: bool = False
    user_advice: Optional[str] = None
    check_result: Optional[Any] = None  # VerifyResult
    old_code: Optional[str] = None
    sample_id: int = 0

    @property
    @abstractmethod
    def op_name(self):
        """Subclasses must implement this property and return the operator name"""
        pass

    @property
    @abstractmethod
    def framework_name(self) -> str:
        """Subclasses must implement this property and return the framework name"""
        pass

    class Config:
        arbitrary_types_allowed = True  # allow arbitrary types (e.g. function objects)


class TritonKernelGenerateArgs(BaseGenerateArgs):
    """Torch API generate args - used to generate Triton kernels from torch API"""

    # Torch-specific fields (kept consistent with existing TritonKernelGenerateArgs)
    triton_kernel_name: str
    func_desc: str
    torch_kernel_code: str
    input_args: Optional[Any] = None  # List[InputArg] | None | dict
    output_args: Optional[Any] = None  # List[OutputArg] | None
    func_type: Optional[str] = None  # "unary", "binary", "reduction", "other"
    impl_info: Optional[Any] = None  # dict | list

    @property
    def op_name(self):
        return self.triton_kernel_name

    @property
    def framework_name(self) -> str:
        return "torch"



class CublasGenerateArgs(BaseGenerateArgs):
    """cuBLAS baseline generate args"""

    cublas_kernel_name: str
    baseline_func: Any
    baseline_code: str
    func_desc: str
    blas_operation_type: str
    impl_info: Optional[Any] = None

    @property
    def op_name(self):
        return self.cublas_kernel_name

    @property
    def framework_name(self) -> str:
        return "cublas"


class VllmGenerateArgs(BaseGenerateArgs):
    """vLLM baseline generate args"""

    vllm_kernel_name: str
    baseline_func: Any
    baseline_code: str
    func_desc: str
    operation_type: str  # attention, quantization, moe, norm, other
    impl_info: Optional[Any] = None

    @property
    def op_name(self):
        return self.vllm_kernel_name

    @property
    def framework_name(self) -> str:
        return "vllm"
