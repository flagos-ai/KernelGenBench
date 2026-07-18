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

"""Agent methods for kernel generation."""

from .base import BaseMethod, MethodResult
from .naive_cc import NaiveCCMethod
from .normal_cc import NormalCCMethod
from .naive_opencode import NaiveOpenCodeMethod
from .normal_opencode import NormalOpenCodeMethod

_METHODS = {
    "naive_cc": NaiveCCMethod,
    "normal_cc": NormalCCMethod,
    "naive_opencode": NaiveOpenCodeMethod,
    "normal_opencode": NormalOpenCodeMethod,
}


def get_method(name: str) -> BaseMethod:
    if name not in _METHODS:
        available = ", ".join(_METHODS.keys())
        raise ValueError(f"Unknown method: {name}. Available: {available}")
    return _METHODS[name]()


def list_methods() -> list[str]:
    return list(_METHODS.keys())


__all__ = ["BaseMethod", "MethodResult", "get_method", "list_methods",
           "NaiveCCMethod", "NormalCCMethod", "NaiveOpenCodeMethod", "NormalOpenCodeMethod"]
