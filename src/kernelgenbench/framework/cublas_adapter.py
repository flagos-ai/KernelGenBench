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
CublasAdapter - cuBLAS framework adapter
"""

import inspect
import json
from typing import Any, Dict
from .adapter import FrameworkAdapter
from .generate_args import CublasGenerateArgs


class CublasAdapter(FrameworkAdapter):

    def __init__(self):
        self._impl_info_cache = None

    @property
    def framework_name(self) -> str:
        return "cublas"

    def get_operator_function(self, op_name: str) -> Any:
        if "::" not in op_name:
            raise ValueError(f"Invalid op_name format: {op_name}")

        from kernelgenbench.dataset import get_cublas_operators
        cublas_operators = get_cublas_operators()

        if op_name not in cublas_operators:
            raise KeyError(f"Operator {op_name} not found")

        return cublas_operators[op_name]

    def get_signature_info(self, func: Any, op_name: str) -> Dict:
        try:
            sig = inspect.signature(func)
        except (ValueError, TypeError):
            sig = None

        func_desc = f"cuBLAS baseline: {op_name}"
        if func.__doc__:
            func_desc = func.__doc__.strip()

        return {
            "signature": sig,
            "parameters": sig.parameters if sig else {},
            "return_annotation": sig.return_annotation if sig else None,
            "func_desc": func_desc,
        }

    def create_generate_args(self, op_name: str, func: Any, impl_info: Any) -> CublasGenerateArgs:
        sig_info = self.get_signature_info(func, op_name)
        baseline_code = self.get_reference_code(func, op_name)
        blas_type = self._determine_blas_type(op_name)

        return CublasGenerateArgs(
            cublas_kernel_name=op_name,
            baseline_func=func,
            baseline_code=baseline_code,
            func_desc=sig_info["func_desc"],
            blas_operation_type=blas_type,
            impl_info=impl_info,
            from_mcp=False,
        )

    def get_reference_code(self, func: Any, op_name: str) -> str:
        try:
            return inspect.getsource(func).strip()
        except (OSError, TypeError) as e:
            return f"# Unable to get source code for {op_name}: {e}"

    def _determine_blas_type(self, op_name: str) -> str:
        kernel_name = op_name.split("::")[-1].lower()
        op_base = kernel_name[7:] if kernel_name.startswith('cublas') else kernel_name
        op_base = op_base[1:] if len(op_base) > 1 and op_base[0] in 'sdczhz' else op_base

        level3_ops = ['gemm', 'symm', 'hemm', 'syrk', 'herk', 'syr2k', 'her2k', 'trmm', 'trsm']
        if any(op in op_base for op in level3_ops):
            return "Level 3"

        level2_ops = ['gemv', 'gbmv', 'hemv', 'hbmv', 'hpmv', 'symv', 'sbmv', 'spmv',
                      'trmv', 'tbmv', 'tpmv', 'trsv', 'tbsv', 'tpsv',
                      'ger', 'geru', 'gerc', 'her', 'hpr', 'her2', 'hpr2',
                      'syr', 'spr', 'syr2', 'spr2']
        if any(op in op_base for op in level2_ops):
            return "Level 2"

        level1_ops = ['asum', 'axpy', 'copy', 'dot', 'dotc', 'dotu', 'nrm2',
                      'rot', 'rotg', 'rotm', 'rotmg', 'scal', 'swap']
        if any(op in op_base for op in level1_ops):
            return "Level 1"

        extension_ops = ['geam', 'dgmm']
        if any(op in op_base for op in extension_ops):
            return "Extension"

        return "Extension"

    def get_impl_info(self, kernel_name: str) -> Any:
        if self._impl_info_cache is None:
            import os
            json_path = os.path.join(
                os.path.dirname(__file__),
                "../dataset/cublas_IMPL_INFO.json"
            )
            with open(json_path, 'r') as f:
                self._impl_info_cache = json.load(f)

        return self._impl_info_cache.get(kernel_name)
