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

import ast
import importlib.util
import sys
from typing import List, Dict

import os
import types

def scan_registrations(file_path: str) -> List[Dict]:
    """
    Scan a Python file and extract info for all functions decorated with @register.
    Return format: [{"name": registration_name, "func_name": function_name, "args": decorator_args}, ...]
    """
    with open(file_path, 'r') as f:
        tree = ast.parse(f.read())
    
    registrations = []
    
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
            
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
                
            if getattr(decorator.func, 'id', None) == 'register':
                # Parse decorator arguments
                args = []
                for arg in decorator.args:
                    if isinstance(arg, ast.Constant):
                        args.append(arg.value)
                    elif isinstance(arg, ast.Name):
                        args.append(arg.id)
                
                registrations.append({
                    "name": args[0] if args else node.name,
                    "func_name": node.name,
                    "args": args[1:] if len(args) > 1 else []
                })
    
    return registrations

def auto_register_module(module_path_or_source: str):
    # """
    # Auto-register all functions decorated with @register in the module
    # """
    # # Dynamically import module
    # spec = importlib.util.spec_from_file_location("temp_module", module_path)
    # module = importlib.util.module_from_spec(spec)
    # sys.modules["temp_module"] = module
    # spec.loader.exec_module(module)
    """
    Auto-register all functions decorated with @register in the module.
    Supports passing a module file path or source code string.
    """
    module_name = "temp_module"
    if os.path.isfile(module_path_or_source):
        # Import module from file path
        spec = importlib.util.spec_from_file_location(module_name, module_path_or_source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    else:
        # Handle source code string: create a new module object and execute the source
        module = types.ModuleType(module_name)
        sys.modules[module_name] = module
        exec(module_path_or_source, module.__dict__)
    
    # Scan registration info
    # registrations = scan_registrations(module_path)

    # Execute registration
    # for reg in registrations:
    #     func = getattr(module, reg['func_name'])
    #     register(reg['name'], *reg['args'])(func)
    #     print(f"Registered: {reg['name']} -> {reg['func_name']}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Python file path to scan")
    args = parser.parse_args()
    
    auto_register_module(args.file)
