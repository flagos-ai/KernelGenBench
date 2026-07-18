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

# from . import backend, commom_utils, error
# from .backend.device import DeviceDetector
# import error
# from . import error
from .error import error
from kernelgenbench.dataset import Autograd
from collections import OrderedDict

import logging

from kernelgenbench.dataset import IMPL_INFO, is_pytorch_op
from sandbox.config import DEVICE as device

REGISTERED_OPS = OrderedDict()
REGISTERED_OPS_TORCH = []

def register(api, key, has_backward=Autograd.enable, namespace: str=""):
    """
    Register a function with a key and an optional backward flag.
    """
    if not namespace:
        if api in REGISTERED_OPS:
            if key in REGISTERED_OPS[api]:
                raise ValueError(f"Operation '{key}' is already registered, now REGISTERED_OPS: {REGISTERED_OPS}")
    else:
        if namespace not in REGISTERED_OPS:
            REGISTERED_OPS[namespace] = OrderedDict()
        if key in REGISTERED_OPS[namespace]:
            raise ValueError(f"Operation '{key}' in namespace '{namespace}' is already registered, now REGISTERED_OPS: {REGISTERED_OPS[namespace]}")


    def decorator(func):
        if not namespace:
            if api in REGISTERED_OPS:
                REGISTERED_OPS[api].append((key, func, has_backward))
            else:
                REGISTERED_OPS[api] = [(key, func, has_backward)]
            # REGISTERED_OPS[key] = (key, func, has_backward)
        else:
            REGISTERED_OPS[namespace][key] = (key, func, has_backward)
        # Only check IMPL_INFO for PyTorch operators; skip non-PyTorch operators (e.g. cublas::sgemm)
        if is_pytorch_op(api, namespace=namespace) is False and "::" not in api:
            # api not in IMPL_INFO and no namespace prefix (e.g. cublas::), meaning it is an unregistered aten operator
            logging.warning(f"Operator {key} not found in IMPL_INFO, make sure using bench.{key} directly rather than bench.use_ops")
        import sys
        # package_name = __name__.split('.')[0]
        package_name = "kernelgenbench"
        bench_module = sys.modules[package_name]
        if namespace:
            if not hasattr(bench_module, namespace):
                from types import SimpleNamespace
                setattr(bench_module, namespace, SimpleNamespace())
            setattr(getattr(bench_module, namespace), key, func)
        else:
            setattr(bench_module, key, func)
        return func
    
    return decorator

class Register:
    def __init__(
        self,
        config,
        user_unused_ops_list=None,
        lib=None,
    ):
        # lib is a instance of torch.library.Library
        # self.device = DeviceDetector()
        self.lib = lib
        # reg_key like 'CUDA', reg_bac_key like AutogradCUDA
        # self.reg_key = self.device.name.upper()
        self.reg_key = device.upper()
        self.reg_bac_key = "Autograd" + self.reg_key
        self.all_ops = []
        self.vendor_unused_ops_list = self.get_vendor_unused_op()
        self.unused_ops = user_unused_ops_list + self.vendor_unused_ops_list
        self.config = config
        # self.config_filter()
        self.for_each(last=True)

    # def config_filter(self):
    #     self.config = [
    #         item for item in self.config if item[1].__name__ not in self.unused_ops
    #     ]

    def get_vendor_unused_op(self):
        # if self.device.vendor != commom_utils.vendors.NVIDIA:
        #     return backend.get_curent_device_unused_op(self.device.vendor_name)
        return []

    def register_impl(self, api, key, fn, has_backward):
        try:
            # Check if this is a PyTorch operator
            impl_info = IMPL_INFO.get(api)
            if not impl_info:
                # Non-PyTorch operator: skip torch.library registration
                logging.info(f"Non-PyTorch operator {key} (api={api}), skip torch.library registration")
                return

            # PyTorch operator: register to torch.library
            self.all_ops.append(key)

            if has_backward is Autograd.enable:
                device_key = self.reg_bac_key
            else:
                device_key = self.reg_key

            keys = [impl_key for impl_key, _ in impl_info]
            if key not in keys:
                raise ValueError(f"Operator {key} not found in IMPL_INFO[{api}]")

            self.lib.impl(key, fn, device_key)
        except RuntimeError as e:
            if "already a kernel registered" in str(e):
                pass
            else:
                error.register_error(e)
    def for_each(self, last=False):
        if last:
            if self.config:
                api, info = list(self.config.items())[-1]
                for key, func, has_backward in info:
                    self.register_impl(api, key, func, has_backward)
        else:
            try:
                for api, (key, func, has_backward) in self.config.items():
                    self.register_impl(api, key, func, has_backward)
            except Exception as e:
                error.register_error(e)

    def get_all_ops(self):
        return self.all_ops

    def get_unused_ops(self):
        return self.unused_ops

    # def get_vendor_name(self):
    #     return self.device.vendor_name

    # def get_current_device(self):
    #     return self.device.name
