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
import inspect
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass
import torch
from logging import getLogger
from datetime import datetime

from generator.sampler.generate_samples import (
    TritonKernelGenerateArgs,
    GenerationConfig,
    InputArg,
    OutputArg,
)

logger = getLogger(__name__)


def today() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _placeholder(*args, **kwargs):
    raise NotImplementedError("This is a placeholder function.")


def get_torch_api_signature(torch_op_name: str, torch_op_func) -> Dict[str, Any]:
    """
    Extract signature information from a PyTorch API.
    
    Args:
        torch_op_name: Full name of the torch operator (e.g., 'torch.add')
        torch_op_func: The actual torch function object
    
    Returns:
        Dictionary containing input_args, output_args, and func_desc
    """
    # Try to get the docstring for description
    func_desc = f"PyTorch operator: {torch_op_name}"
    if hasattr(torch_op_func, '__doc__') and torch_op_func.__doc__:
        # Extract first line of docstring as description
        doc_lines = torch_op_func.__doc__.strip()
        if doc_lines:
            func_desc = doc_lines
    
    # For now, we'll use generic input/output args since we don't have
    # detailed parameter information without runtime inspection
    # In a real implementation, you might want to use inspect module or
    # parse the docstring to extract parameter information
    
    # TODO : Improve argument extraction logic
    result = get_function_signature(torch_op_func)
    input_args = result["input_args"]
    # output_args = result["output_args"]
    
    # input_args = [InputArg(arg_name="args", arg_type="Any", arg_desc="Input arguments")]
    # output_args = [OutputArg(arg_type="Any", arg_desc="Output result")]
    
    return {
        "input_args": input_args,
        "output_args": None,
        "func_desc": func_desc,
    }


def create_triton_generate_args(torch_op_name: str, torch_op_func_or_namespace: Callable | str, impl_info) -> TritonKernelGenerateArgs:
    """
    Create TritonKernelGenerateArgs for a given PyTorch operator.
    
    Args:
        torch_op_name: Full name of the torch operator (e.g., 'torch.add')
        torch_op_func: The actual torch function object
    
    Returns:
        TritonKernelGenerateArgs instance
    """
    # Extract the function name from the full path
    # e.g., 'torch.add' -> 'add', 'torch.nn.functional.gelu' -> 'gelu'
    kernel_name = torch_op_name.split('.')[-1]
    # Resolve specialized benchmark names (e.g., mm_128x768_768x768_f32 -> mm)
    from kernelgenbench.dataset import resolve_op_func_name
    kernel_name = resolve_op_func_name(kernel_name)
    
    # Get signature information
    if isinstance(torch_op_func_or_namespace, str) and len(torch_op_func_or_namespace) > 0:
        # check torch.ops has the attribute
        if hasattr(torch.ops, torch_op_func_or_namespace):
            # torch_op_func actually is the namespace
            torch_op_name = f"{torch_op_func_or_namespace}::{kernel_name}"
            torch_op_namespace = getattr(torch.ops, torch_op_func_or_namespace)
            torch_op_func = getattr(torch_op_namespace, kernel_name)
            torch_op_func_name = f"{torch_op_namespace.__name__}.{kernel_name}"
    else:
        torch_op_func = torch_op_func_or_namespace
        torch_op_func_name = torch_op_name
    sig_info = get_torch_api_signature(torch_op_name, torch_op_func)
    
    # Create a simple torch kernel code snippet as reference
    torch_kernel_code = f"""
# Reference PyTorch implementation for {torch_op_name}
import torch

{kernel_name} = {torch_op_func_name}
""".strip()
    
    return TritonKernelGenerateArgs(
        triton_kernel_name=torch_op_name,
        func_desc=sig_info["func_desc"],
        torch_kernel_code=torch_kernel_code,
        input_args=sig_info["input_args"],
        # output_args=sig_info["output_args"],
        impl_info=impl_info,
        # func_type="other",  # Could be refined based on operator analysis
        from_mcp=False,
    )


def load_api_to_process_from_test_func_path(test_func_result_path: Path, get_success: bool = True) -> dict[str, str]:
    # check the path is dir or file
    if test_func_result_path.is_dir():
        dirs = [x for x in test_func_result_path.iterdir() if x.is_dir() and x.name.startswith("log_")]
        api_info = {}
        for d in dirs:
            api_info.update(load_api_to_process_from_test_func_result(d, get_success))
        return api_info
    else:
        return load_api_to_process_from_test_func_result(test_func_result_path, get_success)

def load_api_to_process_from_test_func_result(test_func_result_path: Path, get_success: bool = True) -> dict[str, str]:
    """Load API information from a test function result file.

    Args:
        test_func_result_path: Path to the test function result file.
    Returns:
        A list of API information dictionaries.
    """
    test_func_result_path = test_func_result_path / "result.json" if test_func_result_path.name != "result.json" else test_func_result_path
    if not test_func_result_path.exists():
        logger.warning(f"Test function result path {test_func_result_path} does not exist.")
        return {}
    with open(test_func_result_path, "r") as f:
        test_func_results = json.load(f)

    api_info = {}
    for result in test_func_results:
        api_name = result["op_name"]
        ok = result["success"] == get_success
        if ok:
            if "::" in api_name:
                namespace, name = api_name.split("::", 1)
                api_info[name] = namespace
            else:
                logger.warning(f"API name {api_name} does not contain namespace info, defaulting to 'aten'")
                api_info[api_name] = "aten"
    return api_info


def load_right_test_function_from_test_func_dir(path: Path, get_success: bool = True) -> Dict[str, str]:
    # check the path is dir or file
    if path.is_dir():
        dirs = [x for x in path.iterdir() if x.is_dir() and x.name.startswith("log_")]
        test_funcs = {}
        for d in dirs:
            test_funcs.update(load_right_test_function_from_result_path(d / "result.json", get_success))
        return test_funcs
    else:
        return load_right_test_function_from_result_path(path, get_success)


def load_right_test_function_from_result_path(path: Path, get_success: bool = True) -> Dict[str, str]:
    path = path / "result.json" if path.name != "result.json" else path
    if not path.exists():
        logger.warning(f"Test function result path {path} does not exist.")
        return {}
    with open(path, "r") as f:
        eval_result = json.load(f)
    test_funcs = {}
    for item in eval_result:
        if item["success"] == get_success:
            test_funcs[item["op_name"]] = item["test_func"]
    return test_funcs


def convert_accuracy_to_performance_test(test_funcs: Dict[str, str]) -> Dict[str, str]:
    """Convert accuracy test functions to performance test functions.

    This function transforms accuracy test functions into performance test functions
    by applying predefined conversion rules.

    Args:
        test_funcs: Dictionary mapping operator names to accuracy test function code.
                   Format: {op_name: test_func_code}

    Returns:
        Dictionary mapping operator names to performance test function code.
        Format: {op_name: perf_test_func_code}
    """
    result = {}

    for op_name, test_func_code in test_funcs.items():
        try:
            converted_code = _convert_single_test_func(test_func_code)
            result[op_name] = converted_code
        except Exception as e:
            logger.error(f"Failed to convert test function for {op_name}: {e}")
            # Keep original code if conversion fails
            result[op_name] = test_func_code

    return result


def _extract_function_call(full_line: str, var_name: str) -> str:
    """Extract complete function call using parenthesis counting.

    Args:
        full_line: The complete line(s) containing the function call
        var_name: Variable name (e.g., 'ref_out' or 'act_out')

    Returns:
        The extracted function call, or None if not found
    """
    # Find the start of the function call (after '=')
    # Use [^\s(]+ instead of \S+ to stop at the opening parenthesis
    pattern = rf'{var_name}\s*=\s*(torch\.ops\.aten\.[^\s(]+)\('
    match = re.search(pattern, full_line)
    if not match:
        return None

    func_name = match.group(1)
    start_pos = match.end() - 1  # Position of the opening '('

    # Count parentheses to find the end
    paren_count = 0
    i = start_pos
    while i < len(full_line):
        if full_line[i] == '(':
            paren_count += 1
        elif full_line[i] == ')':
            paren_count -= 1
            if paren_count == 0:
                # Found the matching closing parenthesis
                return full_line[match.start(1):i+1]
        i += 1

    return None


def _convert_single_test_func(code: str) -> str:
    """Convert accuracy test to include performance testing.

    Strategy:
    1. Keep all accuracy testing code unchanged
    2. Find assert_close() call
    3. Extract ref and act function calls
    4. Insert performance testing after assert_close()
    5. Add return CustomBenchmarkResult
    """
    lines = code.split('\n')
    new_lines = []

    # Track extracted information
    ref_call = None
    act_call = None
    assert_close_idx = None

    i = 0
    while i < len(lines):
        line = lines[i]

        # Keep all original lines
        new_lines.append(line)

        # Extract reference call: ref_out = torch.ops.aten.xxx(...)
        if 'ref_out' in line and 'torch.ops.aten.' in line and '=' in line:
            # Handle multi-line calls
            full_line = line
            paren_count = line.count('(') - line.count(')')
            j = i + 1
            while paren_count > 0 and j < len(lines):
                full_line += '\n' + lines[j]
                new_lines.append(lines[j])
                paren_count += lines[j].count('(') - lines[j].count(')')
                j += 1
                i = j - 1

            # Extract the function call part using helper
            ref_call = _extract_function_call(full_line, 'ref_out')

        # Extract actual call: act_out = torch.ops.aten.xxx(...) inside with block
        if 'act_out' in line and 'torch.ops.aten.' in line and '=' in line:
            # Handle multi-line calls
            full_line = line
            paren_count = line.count('(') - line.count(')')
            j = i + 1
            while paren_count > 0 and j < len(lines):
                full_line += '\n' + lines[j]
                new_lines.append(lines[j])
                paren_count += lines[j].count('(') - lines[j].count(')')
                j += 1
                i = j - 1

            # Extract the function call part using helper
            act_call = _extract_function_call(full_line, 'act_out')

        # Find assert_close() line
        if 'assert_close' in line and assert_close_idx is None:
            assert_close_idx = len(new_lines) - 1

        i += 1

    # If we found everything, insert performance testing
    if ref_call and act_call and assert_close_idx is not None:
        # Prepare performance testing code
        indent = '    '  # Standard 4-space indent
        perf_code = [
            '',
            indent + '# Performance testing',
            indent + 'import triton',
            indent + 'from sandbox.utils.accuracy_utils import CustomBenchmarkResult',
            '',
            indent + 'quantiles = [0.5, 0.2, 0.8]',
            '',
            indent + '# Benchmark reference implementation',
            indent + 'ms_torch, _, _ = triton.testing.do_bench(',
            indent + f'    lambda: {ref_call},',
            indent + '    rep=100,',
            indent + '    quantiles=quantiles',
            indent + ')',
            '',
            indent + '# Benchmark triton implementation',
            indent + 'with kernelgenbench.use_ops(REGISTERED_OPS):',
            indent + '    ms_triton, _, _ = triton.testing.do_bench(',
            indent + f'        lambda: {act_call},',
            indent + '        rep=100,',
            indent + '        quantiles=quantiles',
            indent + '    )',
            '',
            indent + '# Calculate speedup and return result',
            indent + 'speedup = ms_torch / ms_triton',
            indent + 'result = CustomBenchmarkResult(',
            indent + '    ref_time=ms_torch,',
            indent + '    res_time=ms_triton,',
            indent + '    speedup=speedup,',
            indent + ')',
            indent + 'return result',
        ]

        # Insert after assert_close
        new_lines = new_lines[:assert_close_idx + 1] + perf_code + new_lines[assert_close_idx + 1:]

    return '\n'.join(new_lines)


def _add_clones_to_call(call: str) -> str:
    """Add .clone() to all ref_xxx and act_xxx variables in the call."""
    # Replace ref_xxx with ref_xxx.clone()
    call = re.sub(r'\b(ref_\w+)\b(?!\.clone)', r'\1.clone()', call)
    # Replace act_xxx with act_xxx.clone()
    call = re.sub(r'\b(act_\w+)\b(?!\.clone)', r'\1.clone()', call)
    return call


def load_right_kernel_code_from_acc_verify_dir(path: Path, get_success: bool = True) -> Dict[str, str]:
    # check the path is dir or file
    # breakpoint()
    if path.is_dir():
        dirs = [x for x in path.iterdir() if x.is_dir() and x.name.startswith("log_")]
        kernel_code = {}
        for d in dirs:
            kernel_code.update(load_right_kernel_code_from_acc_verify_result_path(d / "result.json", get_success))
        return kernel_code
    else:
        return load_right_kernel_code_from_acc_verify_result_path(path, get_success)


def load_right_kernel_code_from_acc_verify_result_path(path: Path, get_success: bool = True) -> Dict[str, str]:
    path = path / "result.json" if path.name != "result.json" else path
    if not path.exists():
        logger.warning(f"Test function result path {path} does not exist.")
        return {}
    with open(path, "r") as f:
        eval_result = json.load(f)
    kernel_code = {}
    for item in eval_result:
        if item["success"] == get_success:
            kernel_code[item["op_name"]] = item["code"]
    return kernel_code


def get_function_signature(func: Callable) -> Dict[str, Any]:
    """
    Get the input and output parameter information of a function.

    Supports two types:
    1. Regular Python functions: uses the inspect module
    2. torch.ops operators: uses the _schema attribute

    Args:
        func: Callable object

    Returns:
        Dictionary containing input and output parameter information, format:
        {
            "input_args": [InputArg, ...],
            "output_args": [OutputArg, ...],
            "signature": "complete function signature string"
        }
    """
    result = {
        "input_args": [],
        "output_args": [],
        "signature": ""
    }
    
    # Case 1: torch.ops operator, use _schemas attribute
    if hasattr(func, '_schemas'):
        schema = func._schemas
        result["signature"] = str(schema)
        
        input_output_info = {k: str(v) for k, v in schema.items()}
        return {"input_args": input_output_info}
        # # Parse schema
        # # Format example: "aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"
        # schema_str = str(schema)

        # # Extract input parameters section
        # input_match = re.search(r'\((.*?)\)', schema_str)
        # if input_match:
        #     params_str = input_match.group(1)

        #     # Split parameters (handle * separator)
        #     params_parts = params_str.split('*')
        #     all_params = []

        #     for part in params_parts:
        #         for param in part.split(','):
        #             param = param.strip()
        #             if param and param != '*':
        #                 all_params.append(param)

        #     # Parse each parameter
        #     for param in all_params:
        #         param = param.strip()
        #         if not param:
        #             continue

        #         # Parse parameter: format "Type name" or "Type name=default"
        #         has_default = '=' in param
        #         if has_default:
        #             param_no_default, default_value = param.split('=', 1)
        #         else:
        #             param_no_default = param
        #             default_value = None

        #         parts = param_no_default.strip().split()
        #         if len(parts) >= 2:
        #             param_type = parts[0]
        #             param_name = parts[1]

        #             # Handle type modifiers (e.g. Tensor(a!))
        #             param_type = re.sub(r'\(.*?\)', '', param_type)

        #             # Create InputArg object
        #             input_arg = InputArg(
        #                 arg_name=param_name,
        #                 arg_type=param_type,
        #                 arg_value=None,
        #                 arg_default=default_value.strip() if default_value else None,
        #                 arg_desc=""
        #             )
        #             result["input_args"].append(input_arg)

        # # Extract output parameters
        # output_match = re.search(r'->\s*(.+)$', schema_str)
        # if output_match:
        #     output_type = output_match.group(1).strip()

        #     # Handle multiple return value types (e.g. "(Tensor, Tensor)")
        #     if output_type.startswith('(') and output_type.endswith(')'):
        #         output_types = [t.strip() for t in output_type[1:-1].split(',')]
        #         for out_type in output_types:
        #             # Clean up type modifiers
        #             out_type = re.sub(r'\(.*?\)', '', out_type)
        #             output_arg = OutputArg(
        #                 arg_type=out_type,
        #                 arg_value=None,
        #                 arg_desc=""
        #             )
        #             result["output_args"].append(output_arg)
        #     else:
        #         # Single return value
        #         output_type = re.sub(r'\(.*?\)', '', output_type)
        #         output_arg = OutputArg(
        #             arg_type=output_type,
        #             arg_value=None,
        #             arg_desc=""
        #         )
        #         result["output_args"].append(output_arg)
    
    # Case 2: regular Python function, use inspect
    else:
        try:
            sig = inspect.signature(func)
            result["signature"] = str(sig)
            
            # Parse input parameters
            for param_name, param in sig.parameters.items():
                # Get type annotation
                param_type = (
                    str(param.annotation) 
                    if param.annotation != inspect.Parameter.empty 
                    else "Any"
                )
                
                # Clean up type string (remove <class '...'> wrapper)
                if param_type.startswith("<class '") and param_type.endswith("'>"):
                    param_type = param_type[8:-2]
                
                # Get default value
                param_default = (
                    param.default 
                    if param.default != inspect.Parameter.empty 
                    else None
                )
                
                # Create InputArg object
                input_arg = InputArg(
                    arg_name=param_name,
                    arg_type=param_type,
                    arg_value=None,
                    arg_default=param_default,
                    arg_desc=""
                )
                result["input_args"].append(input_arg)
            
            # Parse return type
            if sig.return_annotation != inspect.Signature.empty:
                return_type = str(sig.return_annotation)
                
                # Clean up type string
                if return_type.startswith("<class '") and return_type.endswith("'>"):
                    return_type = return_type[8:-2]
                
                # Handle multiple return value types (e.g. Tuple[int, str])
                # Simple handling: if it's Tuple/tuple, treat as a single return type
                output_arg = OutputArg(
                    arg_type=return_type,
                    arg_value=None,
                    arg_desc=""
                )
                result["output_args"].append(output_arg)
            # When no type annotation is available
                output_arg = OutputArg(
                    arg_type="Any",
                    arg_value=None,
                    arg_desc=""
                )
                result["output_args"].append(output_arg)
                
        except (ValueError, TypeError) as e:
            # Unable to get signature, return empty result
            result["signature"] = f"<Unable to get signature: {e}>"
    
    return result


def get_function_signature_simple(func: Callable) -> tuple[List[InputArg], List[OutputArg]]:
    """
    Simplified version: directly returns InputArg and OutputArg lists.

    Args:
        func: Callable object

    Returns:
        (InputArg list, OutputArg list)
    """
    sig_info = get_function_signature(func)
    return sig_info["input_args"], sig_info["output_args"]
