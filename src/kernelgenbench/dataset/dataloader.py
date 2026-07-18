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

import json
from pathlib import Path
from typing import Dict, List, Optional
import torch
from torch import FunctionSchema
from torch._ops import _OpNamespace
from dataclasses import dataclass

@dataclass
class APIInfo:
    namespace: str
    api: str
    schemas: Dict[str, str | FunctionSchema]
    to_str: bool = True

    def __post_init__(self):
        if self.to_str:
            if self.schemas and isinstance(next(iter(self.schemas.values())), FunctionSchema):
                self.schemas = {k: str(v) for k, v in self.schemas.items()}

class TorchOpsLoader:
    def __init__(self, to_str: bool = True):
        self._cache = {}
        self.to_str = to_str

    def load_namespace(self, namespace: str) -> Dict[str, APIInfo]:
        assert namespace in dir(torch.ops), f"Namespace {namespace} not found in torch.ops"
        assert isinstance(getattr(torch.ops, namespace), _OpNamespace), f"{namespace} is not a valid OpNamespace"
        if namespace in self._cache:
            return self._cache[namespace]
        ns_module = getattr(torch.ops, namespace)
        ops_dict = {}
        for op_name in dir(ns_module):
            op = getattr(ns_module, op_name)
            if callable(op):
                full_name = f"{namespace}::{op_name}"
                ops_dict[full_name] = APIInfo(
                    api=op.__module__ + "." + op.__name__,
                    schemas=op._schemas,
                    namespace=namespace,
                    to_str=self.to_str
                )
        self._cache[namespace] = ops_dict
        return ops_dict
    
    def get_operator(self, namespace: str, op_name: str) -> APIInfo:
        if namespace == "":
            namespace = "aten"
        ns_data = self.load_namespace(namespace)
        full_name = f"{namespace}::{op_name}"
        if full_name in ns_data:
            return ns_data[full_name]
        if hasattr(torch.ops.__getattr__(namespace), op_name):
            op = getattr(torch.ops.__getattr__(namespace), op_name)
            info = APIInfo(
                api=op.__module__ + "." + op.__name__,
                schemas=op._schemas,
                namespace=namespace,
                to_str=self.to_str
            )
            self._cache[namespace][full_name] = info
            return info
        raise KeyError(f"Operator {op_name} not found in namespace {namespace}")
    
    def list_namespaces(self) -> List[str]:
        return [ns for ns in dir(torch.ops) if isinstance(getattr(torch.ops, ns), _OpNamespace)]
    
    def load_all(self) -> Dict[str, APIInfo]:
        all_data = {}
        for namespace in self.list_namespaces():
            namespace_ops = self.load_namespace(namespace)
            all_data.update(namespace_ops)
        return all_data



if __name__ == "__main__":
    loader = TorchOpsLoader()
    aten_ops = loader.load_namespace('aten')
    print(f"Loaded {len(aten_ops)} aten operators.")
    print(aten_ops['abs'])
