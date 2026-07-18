#!/usr/bin/env python3

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
AutoKernel -- Export optimized kernels to HuggingFace Kernels format.

Takes an optimized AutoKernel CUDA or Triton kernel and packages it into the
HuggingFace Kernels project structure for publishing to the HuggingFace Hub.

Usage:
    # Export the current kernel.py (auto-detect backend)
    uv run export_hf.py --name my_matmul

    # Export a specific kernel file
    uv run export_hf.py --name my_matmul --kernel workspace/kernel_matmul_1.py

    # Export with a specific repo ID for upload instructions
    uv run export_hf.py --name my_matmul --repo-id rightnow-ai/matmul-kernel

    # Custom output directory
    uv run export_hf.py --name my_matmul --output workspace/hf_export/

    # After export, upload to HuggingFace Hub:
    #   cd workspace/hf_export/my_matmul
    #   kernels upload . --repo_id rightnow-ai/matmul-kernel

HuggingFace Kernels: https://huggingface.co/docs/kernels/en/index
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
import textwrap
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_KERNEL_PATH = os.path.join(SCRIPT_DIR, "kernel.py")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "workspace", "hf_export")


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def detect_backend(source: str) -> str:
    """
    Detect whether a kernel file uses the CUDA C++ or Triton backend.

    Returns 'cuda' or 'triton'.
    """
    # Explicit BACKEND declaration takes priority
    match = re.search(r'^BACKEND\s*=\s*["\'](\w+)["\']', source, re.MULTILINE)
    if match:
        backend = match.group(1).lower()
        if backend in ("cuda", "triton"):
            return backend

    # Heuristic: look for CUDA indicators
    has_cuda_src = "CUDA_SRC" in source
    has_compile_cuda = "compile_cuda" in source

    if has_cuda_src or has_compile_cuda:
        return "cuda"

    # Heuristic: look for Triton indicators
    has_triton_import = "import triton" in source or "from triton" in source
    has_triton_jit = "@triton.jit" in source or "@triton.autotune" in source

    if has_triton_import or has_triton_jit:
        return "triton"

    # Default to triton if unclear
    return "triton"


# ---------------------------------------------------------------------------
# Kernel type detection
# ---------------------------------------------------------------------------

def detect_kernel_type(source: str) -> Optional[str]:
    """Extract the KERNEL_TYPE from the source file, if declared."""
    match = re.search(r'^KERNEL_TYPE\s*=\s*["\'](\w+)["\']', source, re.MULTILINE)
    if match:
        return match.group(1)
    return None


# ---------------------------------------------------------------------------
# CUDA source extraction
# ---------------------------------------------------------------------------

def extract_cuda_source(source: str) -> Optional[str]:
    """
    Extract the CUDA_SRC string from a Python kernel file.

    Handles:
      - CUDA_SRC = r\"\"\"...\"\"\"
      - CUDA_SRC = \"\"\"...\"\"\"
      - CUDA_SRC = r'''...'''
      - CUDA_SRC = '''...'''

    Returns the raw CUDA C++ source string, or None if not found.
    """
    # Try AST-based extraction first (most robust)
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "CUDA_SRC":
                        if isinstance(node.value, ast.Constant) and isinstance(
                            node.value.value, str
                        ):
                            return node.value.value
                        # Python 3.7 compat: ast.Str
                        if hasattr(ast, "Str") and isinstance(node.value, ast.Str):
                            return node.value.s
    except SyntaxError:
        pass

    # Fallback: regex-based extraction for triple-quoted strings
    # Matches: CUDA_SRC = r"""...""" or CUDA_SRC = """..."""
    for quote in ('"""', "'''"):
        pattern = rf'CUDA_SRC\s*=\s*r?{re.escape(quote)}(.*?){re.escape(quote)}'
        match = re.search(pattern, source, re.DOTALL)
        if match:
            return match.group(1)

    return None


def extract_function_name_from_compile(source: str) -> Optional[str]:
    """
    Extract the function name passed to compile_cuda().

    Looks for patterns like:
      compile_cuda(CUDA_SRC, "matmul_cuda")
      compile_cuda(CUDA_SRC, "softmax_cuda")
    """
    match = re.search(
        r'compile_cuda\s*\(\s*CUDA_SRC\s*,\s*["\'](\w+)["\']', source
    )
    if match:
        return match.group(1)
    return None


# ---------------------------------------------------------------------------
# CUDA function signature parsing
# ---------------------------------------------------------------------------

def extract_function_signatures(cuda_src: str) -> List[Dict[str, str]]:
    """
    Find all torch::Tensor-returning function declarations in the CUDA source.

    Returns a list of dicts with keys:
      - 'return_type': e.g. 'torch::Tensor'
      - 'name': e.g. 'matmul_cuda'
      - 'params': e.g. 'torch::Tensor A, torch::Tensor B'
      - 'full_signature': the complete declaration

    Only extracts non-kernel functions (i.e., the C++ launcher functions that
    PyTorch binds to, not __global__ CUDA kernels).
    """
    # Match: torch::Tensor func_name(params) {
    # Also match at::Tensor, std::vector<torch::Tensor>, void
    pattern = (
        r"^((?:torch::Tensor|at::Tensor|std::vector<torch::Tensor>|void)\s+"
        r"(\w+)\s*\(([^)]*)\))\s*\{"
    )

    results = []
    for match in re.finditer(pattern, cuda_src, re.MULTILINE):
        full_sig = match.group(1).strip()
        func_name = match.group(2)
        params = match.group(3).strip()
        return_type = full_sig.split(func_name)[0].strip()

        # Skip __global__ kernels (they are called from launchers, not from Python)
        # Check if the line before has __global__
        start = match.start()
        preceding = cuda_src[max(0, start - 200) : start]
        if "__global__" in preceding.split("\n")[-1] if preceding else "":
            continue

        # Also skip if the function name suggests it's a device helper
        if func_name.startswith("__"):
            continue

        results.append(
            {
                "return_type": return_type,
                "name": func_name,
                "params": params,
                "full_signature": full_sig,
            }
        )

    return results


def _parse_param_list(params_str: str) -> List[Tuple[str, str]]:
    """
    Parse a C++ parameter list into (type, name) pairs.

    E.g. "torch::Tensor A, torch::Tensor B" -> [("torch::Tensor", "A"), ...]
    """
    if not params_str.strip():
        return []

    results = []
    for param in params_str.split(","):
        param = param.strip()
        if not param:
            continue
        # Remove const and & qualifiers for the binding
        parts = param.split()
        if len(parts) >= 2:
            name = parts[-1].rstrip("&").rstrip("*")
            type_str = " ".join(parts[:-1])
            results.append((type_str, name))

    return results


# ---------------------------------------------------------------------------
# Triton kernel extraction
# ---------------------------------------------------------------------------

def extract_triton_code(source: str) -> str:
    """
    Extract the Triton kernel code from a Python file.

    Returns everything from the first import statement onward, skipping
    the module docstring and KERNEL_TYPE/BACKEND declarations.
    """
    lines = source.split("\n")

    # Find the first import line
    import_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            import_idx = i
            break

    if import_idx is not None:
        return "\n".join(lines[import_idx:])

    # Fallback: return everything after KERNEL_TYPE line
    for i, line in enumerate(lines):
        if line.strip().startswith("KERNEL_TYPE"):
            return "\n".join(lines[i + 1 :])

    return source


# ---------------------------------------------------------------------------
# File generation: build.toml
# ---------------------------------------------------------------------------

def generate_build_toml(
    name: str,
    functions: List[Dict[str, str]],
    backend: str = "cuda",
    repo_id: str = "",
) -> str:
    """
    Generate the build.toml file for HF Kernels.

    Parameters
    ----------
    name : str
        Kernel project name.
    functions : list
        List of function signature dicts from extract_function_signatures.
    backend : str
        'cuda' or 'triton'.
    repo_id : str
        HuggingFace repo ID for the hub section.
    """
    if backend == "cuda":
        return textwrap.dedent(f"""\
            [general]
            name = "{name}"
            license = "Apache-2.0"
            version = 1
            backends = ["cuda"]

            [general.hub]
            repo-id = "{repo_id}"

            [torch]
            src = [
              "torch-ext/torch_binding.cpp",
              "torch-ext/torch_binding.h",
            ]

            [kernel.{name}]
            backend = "cuda"
            depends = ["torch"]
            src = ["{name}_cuda/kernel.cu"]
        """)
    else:
        # Triton kernels are pure Python -- no CUDA compilation needed
        return textwrap.dedent(f"""\
            [general]
            name = "{name}"
            license = "Apache-2.0"
            version = 1

            [general.hub]
            repo-id = "{repo_id}"
        """)


# ---------------------------------------------------------------------------
# File generation: torch_binding.cpp / .h
# ---------------------------------------------------------------------------

def _param_to_torch_schema(param_type: str, param_name: str) -> str:
    """Convert a C++ parameter type to a torch.library schema type."""
    t = param_type.strip()
    # Remove const, &, * qualifiers for schema
    for q in ("const ", "&", "*", "__restrict__"):
        t = t.replace(q, "").strip()

    type_map = {
        "torch::Tensor": "Tensor",
        "at::Tensor": "Tensor",
        "int": "int",
        "int64_t": "int",
        "int32_t": "int",
        "float": "float",
        "double": "float",
        "bool": "bool",
    }
    schema_type = type_map.get(t, "Tensor")
    return f"{schema_type} {param_name}"


def _build_ops_schema(func: Dict[str, str]) -> str:
    """Build a torch.library ops.def() schema string for a function."""
    params = _parse_param_list(func["params"])
    schema_params = ", ".join(
        _param_to_torch_schema(ptype, pname) for ptype, pname in params
    )

    ret = func["return_type"].strip()
    if "vector" in ret:
        schema_ret = "Tensor[]"
    elif "void" in ret:
        schema_ret = "()"
    else:
        schema_ret = "Tensor"

    return f"{func['name']}({schema_params}) -> {schema_ret}"


def generate_torch_binding_cpp(
    name: str,
    functions: List[Dict[str, str]],
) -> str:
    """
    Generate torch_binding.cpp with TORCH_LIBRARY_EXPAND registration.

    Uses the HuggingFace Kernels convention: torch/library.h + registration.h
    + REGISTER_EXTENSION macro for compatibility with the kernel-builder
    Nix build pipeline.
    """
    # Build ops.def() and ops.impl() lines
    ops_lines = []
    for func in functions:
        schema = _build_ops_schema(func)
        ops_lines.append(f'  ops.def("{schema}");')
        ops_lines.append(f'  ops.impl("{func["name"]}", torch::kCUDA, &{func["name"]});')

    ops_str = "\n".join(ops_lines)

    return (
        "#include <torch/library.h>\n"
        "\n"
        '#include "registration.h"\n'
        '#include "torch_binding.h"\n'
        "\n"
        "TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {\n"
        f"{ops_str}\n"
        "}\n"
        "\n"
        "REGISTER_EXTENSION(TORCH_EXTENSION_NAME)\n"
    )


def generate_torch_binding_h(
    functions: List[Dict[str, str]],
) -> str:
    """Generate torch_binding.h with forward declarations."""
    forward_decls = []
    for func in functions:
        forward_decls.append(f"{func['full_signature']};")

    forward_decls_str = "\n".join(forward_decls)

    return textwrap.dedent(f"""\
        #pragma once
        #include <torch/torch.h>

        {forward_decls_str}
    """)


# ---------------------------------------------------------------------------
# File generation: flake.nix (Nix build system for kernel-builder)
# ---------------------------------------------------------------------------

def generate_flake_nix(name: str) -> str:
    """Generate the flake.nix for the HF kernel-builder Nix pipeline."""
    return textwrap.dedent(f"""\
        {{
          description = "Flake for {name} kernel";

          inputs = {{
            kernel-builder.url = "github:huggingface/kernels";
          }};

          outputs =
            {{
              self,
              kernel-builder,
            }}:
            kernel-builder.lib.genKernelFlakeOutputs {{
              inherit self;
              path = ./.;
            }};
        }}
    """)


# ---------------------------------------------------------------------------
# File generation: __init__.py
# ---------------------------------------------------------------------------

def generate_init_py(
    name: str,
    functions: List[Dict[str, str]],
    repo_id: str,
    backend: str = "cuda",
) -> str:
    """Generate the Python __init__.py for the HF Kernels module."""
    first_func = functions[0]["name"] if functions else "kernel_fn"

    if backend == "cuda":
        return textwrap.dedent(f'''\
            """
            {name} - Optimized GPU kernel exported from AutoKernel
            https://github.com/RightNow-AI/autokernel

            Usage:
                from kernels import get_kernel
                module = get_kernel("{repo_id}")
                result = module.{first_func}(input)
            """
            from ._C import *  # noqa: F401,F403
        ''')
    else:
        # Triton kernel: import the Python module directly
        return textwrap.dedent(f'''\
            """
            {name} - Optimized Triton GPU kernel exported from AutoKernel
            https://github.com/RightNow-AI/autokernel

            Usage:
                from kernels import get_kernel
                module = get_kernel("{repo_id}")
                result = module.kernel_fn(input)
            """
            from .kernel import kernel_fn  # noqa: F401
        ''')


# ---------------------------------------------------------------------------
# Main export pipeline: CUDA
# ---------------------------------------------------------------------------

def _export_cuda_kernel(
    source: str,
    name: str,
    output_dir: str,
    repo_id: str,
) -> None:
    """Export a CUDA C++ kernel to HF Kernels format."""

    # Extract the CUDA source string
    cuda_src = extract_cuda_source(source)
    if cuda_src is None:
        print("ERROR: Could not extract CUDA_SRC from kernel file.")
        print("       Expected a CUDA_SRC = r\"\"\"...\"\"\" string assignment.")
        sys.exit(1)

    # Parse function signatures from the CUDA source
    functions = extract_function_signatures(cuda_src)
    if not functions:
        # Try to detect from compile_cuda call
        func_name = extract_function_name_from_compile(source)
        if func_name:
            print(
                f"WARNING: Could not parse function signatures from CUDA source. "
                f"Using function name from compile_cuda call: {func_name}"
            )
            # Create a placeholder signature -- the user may need to adjust
            functions = [
                {
                    "return_type": "torch::Tensor",
                    "name": func_name,
                    "params": "torch::Tensor input",
                    "full_signature": f"torch::Tensor {func_name}(torch::Tensor input)",
                }
            ]
        else:
            print("ERROR: Could not find any torch::Tensor-returning functions in CUDA source.")
            print("       The CUDA source should contain launcher functions like:")
            print("         torch::Tensor my_kernel_cuda(torch::Tensor A, torch::Tensor B) { ... }")
            sys.exit(1)

    # Create directory structure (matches kernels-community convention)
    project_dir = os.path.join(output_dir, name)
    kernel_cuda_dir = os.path.join(project_dir, f"{name}_cuda")
    torch_ext_dir = os.path.join(project_dir, "torch-ext")

    os.makedirs(kernel_cuda_dir, exist_ok=True)
    os.makedirs(torch_ext_dir, exist_ok=True)

    # 1. Write kernel.cu
    kernel_cu_path = os.path.join(kernel_cuda_dir, "kernel.cu")
    cuda_src_clean = cuda_src.strip()
    with open(kernel_cu_path, "w", encoding="utf-8") as f:
        f.write(cuda_src_clean)
        f.write("\n")
    print(f"  Created {os.path.relpath(kernel_cu_path, output_dir)}")

    # 2. Write build.toml
    build_toml_path = os.path.join(project_dir, "build.toml")
    build_toml_content = generate_build_toml(name, functions, backend="cuda", repo_id=repo_id)
    with open(build_toml_path, "w", encoding="utf-8") as f:
        f.write(build_toml_content)
    print(f"  Created {os.path.relpath(build_toml_path, output_dir)}")

    # 3. Write torch_binding.cpp (TORCH_LIBRARY_EXPAND style)
    binding_cpp_path = os.path.join(torch_ext_dir, "torch_binding.cpp")
    binding_cpp_content = generate_torch_binding_cpp(name, functions)
    with open(binding_cpp_path, "w", encoding="utf-8") as f:
        f.write(binding_cpp_content)
    print(f"  Created {os.path.relpath(binding_cpp_path, output_dir)}")

    # 4. Write torch_binding.h
    binding_h_path = os.path.join(torch_ext_dir, "torch_binding.h")
    binding_h_content = generate_torch_binding_h(functions)
    with open(binding_h_path, "w", encoding="utf-8") as f:
        f.write(binding_h_content)
    print(f"  Created {os.path.relpath(binding_h_path, output_dir)}")

    # 5. Write flake.nix (Nix build for kernel-builder pipeline)
    flake_nix_path = os.path.join(project_dir, "flake.nix")
    flake_nix_content = generate_flake_nix(name)
    with open(flake_nix_path, "w", encoding="utf-8") as f:
        f.write(flake_nix_content)
    print(f"  Created {os.path.relpath(flake_nix_path, output_dir)}")

    # Print function summary
    print()
    print(f"  Exported {len(functions)} function(s):")
    for func in functions:
        print(f"    - {func['full_signature']}")


# ---------------------------------------------------------------------------
# Main export pipeline: Triton
# ---------------------------------------------------------------------------

def _export_triton_kernel(
    source: str,
    name: str,
    output_dir: str,
    repo_id: str,
) -> None:
    """
    Export a Triton kernel to HF Kernels format.

    Triton kernels are already Python, so the export is simpler: package
    the Triton code as a Python module. No CUDA compilation needed -- the
    Triton JIT compiler handles everything at runtime.
    """
    # Create directory structure
    project_dir = os.path.join(output_dir, name)
    module_dir = os.path.join(project_dir, name)

    os.makedirs(module_dir, exist_ok=True)

    # 1. Write the Triton kernel as kernel.py inside the module
    triton_code = extract_triton_code(source)
    kernel_py_path = os.path.join(module_dir, "kernel.py")
    with open(kernel_py_path, "w", encoding="utf-8") as f:
        f.write(triton_code.strip())
        f.write("\n")
    print(f"  Created {os.path.relpath(kernel_py_path, output_dir)}")

    # 2. Write __init__.py
    init_py_path = os.path.join(module_dir, "__init__.py")
    # For Triton, functions are the Python entry points (kernel_fn)
    functions = [{"name": "kernel_fn"}]
    init_py_content = generate_init_py(name, functions, repo_id, backend="triton")
    with open(init_py_path, "w", encoding="utf-8") as f:
        f.write(init_py_content)
    print(f"  Created {os.path.relpath(init_py_path, output_dir)}")

    # 3. Write a minimal pyproject.toml for the Triton package
    pyproject_path = os.path.join(project_dir, "pyproject.toml")
    pyproject_content = textwrap.dedent(f"""\
        [project]
        name = "{name}"
        version = "0.1.0"
        description = "Optimized Triton GPU kernel exported from AutoKernel"
        requires-python = ">=3.10"
        dependencies = [
            "torch>=2.4.0",
            "triton>=3.3.0",
        ]
    """)
    with open(pyproject_path, "w", encoding="utf-8") as f:
        f.write(pyproject_content)
    print(f"  Created {os.path.relpath(pyproject_path, output_dir)}")

    # 4. Write a README for the Hub repo
    readme_path = os.path.join(project_dir, "README.md")
    readme_content = textwrap.dedent(f"""\
        # {name}

        Optimized Triton GPU kernel exported from [AutoKernel](https://github.com/RightNow-AI/autokernel).

        ## Usage

        ```python
        from {name}.kernel import kernel_fn

        result = kernel_fn(input_tensor)
        ```

        ## Requirements

        - PyTorch >= 2.4.0
        - Triton >= 3.3.0
        - NVIDIA GPU
    """)
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"  Created {os.path.relpath(readme_path, output_dir)}")

    print()
    print("  Exported Triton kernel as a Python package.")
    print("  Entry point: kernel_fn()")


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------

def export_kernel(
    kernel_path: str,
    name: str,
    output_dir: str,
    repo_id: Optional[str] = None,
) -> str:
    """
    Main export pipeline.

    Parameters
    ----------
    kernel_path : str
        Path to the AutoKernel kernel file (kernel.py or similar).
    name : str
        Name for the exported kernel project (used in build.toml, module name).
        Must be a valid Python identifier.
    output_dir : str
        Directory where the HF Kernels project will be created.
    repo_id : str, optional
        HuggingFace repo ID (e.g., "rightnow-ai/matmul-kernel").
        Used in documentation and __init__.py usage examples.

    Returns
    -------
    str
        Path to the exported project directory.
    """
    # Validate name is a valid Python identifier
    if not name.isidentifier():
        print(f"ERROR: '{name}' is not a valid Python identifier.")
        print("       Use a name like 'my_matmul' or 'fused_attention'.")
        sys.exit(1)

    # Default repo_id
    if repo_id is None:
        repo_id = f"your-username/{name}"

    # Read the kernel file
    if not os.path.exists(kernel_path):
        print(f"ERROR: Kernel file not found: {kernel_path}")
        sys.exit(1)

    with open(kernel_path, "r", encoding="utf-8") as f:
        source = f.read()

    if not source.strip():
        print(f"ERROR: Kernel file is empty: {kernel_path}")
        sys.exit(1)

    # Detect backend
    backend = detect_backend(source)
    kernel_type = detect_kernel_type(source)

    print(f"=== AutoKernel HuggingFace Kernels Export ===")
    print()
    print(f"  Kernel file:  {kernel_path}")
    print(f"  Backend:      {backend}")
    if kernel_type:
        print(f"  Kernel type:  {kernel_type}")
    print(f"  Project name: {name}")
    print(f"  Repo ID:      {repo_id}")
    print(f"  Output:       {output_dir}")
    print()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    project_dir = os.path.join(output_dir, name)

    # Check if project directory already exists
    if os.path.exists(project_dir):
        print(f"WARNING: Output directory already exists: {project_dir}")
        print("         Files will be overwritten.")
        print()

    # Export based on backend
    if backend == "cuda":
        _export_cuda_kernel(source, name, output_dir, repo_id)
    else:
        _export_triton_kernel(source, name, output_dir, repo_id)

    print()
    print("=" * 60)
    print("  Export complete!")
    print("=" * 60)
    print()
    print("  Next steps:")
    print()
    if backend == "cuda":
        print("  1. Review the exported files:")
        print(f"     ls {project_dir}/")
        print()
        print("  2. Build with Nix (cross-compiles for all PyTorch/CUDA combos):")
        print(f"     cd {project_dir}")
        print("     nix flake lock   # generates flake.lock")
        print("     nix run -L .#build-and-copy")
        print()
        print("     Or use HF's kernel-builder terraform setup:")
        print("     https://github.com/huggingface/kernels/tree/main/terraform")
        print()
        print("  3. Upload to HuggingFace Hub:")
        print(f"     nix run .#kernels -- upload --repo-id {repo_id}")
        print()
        print("  4. Use from anywhere:")
        print("     from kernels import get_kernel")
        print(f'     module = get_kernel("{repo_id}")')
        functions_in_src = extract_cuda_source(source)
        if functions_in_src:
            funcs = extract_function_signatures(functions_in_src)
            if funcs:
                print(f"     result = module.{funcs[0]['name']}(input)")
    else:
        print("  1. Review the exported files:")
        print(f"     ls {project_dir}/")
        print()
        print("  2. Upload to HuggingFace Hub:")
        print("     # First: pip install huggingface-hub && huggingface-cli login")
        print(f"     cd {project_dir}")
        print(f"     huggingface-cli upload {repo_id} . .")
        print()
        print("  3. Use from anywhere:")
        print(f"     from {name}.kernel import kernel_fn")
        print("     result = kernel_fn(input)")

    print()
    return project_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export an optimized AutoKernel kernel to HuggingFace Kernels format. "
            "Supports both CUDA C++ and Triton backends."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              # Export the default kernel.py
              uv run export_hf.py --name my_matmul

              # Export a specific kernel file with repo ID
              uv run export_hf.py --name my_matmul --kernel workspace/kernel_matmul_1.py \\
                                  --repo-id rightnow-ai/matmul-kernel

              # Custom output directory
              uv run export_hf.py --name my_matmul --output /tmp/hf_export/
        """),
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help=(
            "Name for the exported kernel project. Must be a valid Python identifier "
            "(e.g., 'my_matmul', 'fused_attention')."
        ),
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default=DEFAULT_KERNEL_PATH,
        help=f"Path to the kernel file to export (default: kernel.py)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for the HF Kernels project (default: workspace/hf_export/)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help=(
            "HuggingFace repo ID (e.g., 'rightnow-ai/matmul-kernel'). "
            "Used in documentation and usage examples."
        ),
    )

    args = parser.parse_args()

    export_kernel(
        kernel_path=args.kernel,
        name=args.name,
        output_dir=args.output,
        repo_id=args.repo_id,
    )


if __name__ == "__main__":
    main()
