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

import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from sandbox.register import Register, register, REGISTERED_OPS
from sandbox.verifier.test_parametrize import Param, parametrize, label

DISPATCH_TORCH_LIB = os.environ.get("DISPATCH_TORCH_LIB", "1") == "1"
aten_lib = torch.library.Library("aten", "IMPL")

current_work_registrar = None

def enable(config, lib=aten_lib, unused=None, registrar=Register):
    global current_work_registrar
    if not DISPATCH_TORCH_LIB:
        current_work_registrar = None
        return
    current_work_registrar = registrar(
        config=config,
        user_unused_ops_list=[] if unused is None else unused,
        lib=lib,
    )


class use_ops:
    def __init__(self, config, unused=None):
        self.lib = torch.library.Library("aten", "IMPL")
        self.unused = [] if unused is None else unused
        self.registrar = Register
        self.config = config

    def __enter__(self):
        enable(config=self.config, lib=self.lib, unused=self.unused, registrar=self.registrar)

    def __exit__(self, exc_type, exc_val, exc_tb):
        global current_work_registrar
        del self.lib
        del self.unused
        del self.registrar
        del current_work_registrar


def all_ops():
    return current_work_registrar.get_all_ops()

accuracy_modules = [
    "kernelgenbench.accuracy.test_ops_with_benchmark",
    "kernelgenbench.accuracy.cublas",
    "kernelgenbench.accuracy.vllm13",
    "kernelgenbench.accuracy.mm_shape_bench",
]

__all__ = [
    "enable",
    "use_ops",
    "register",
    "accuracy_modules",
]
