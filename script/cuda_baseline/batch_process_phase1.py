#!/usr/bin/env python3
"""
Phase 1: Batch process standard pattern kernels (127 kernels)
- standard_dim3_matrixdim (90 kernels)
- vector_int_grid (26 kernels)
- sizet_grid (4 kernels)
- standard_dim3_other (7 kernels)
"""

import sys
import os
import json
import re

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from flagbench.dataset.kernel_list_k1 import IMPL_INFO_K1

REPO_TOP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
KALDI_REPO = "/share/project/zpy/k1_repo"
CACHE_DIR = os.path.join(REPO_TOP_DIR, "cache")


def load_classification():
    """Load kernel classification from JSON."""
    json_path = os.path.join(CACHE_DIR, "kernel_classification.json")
    with open(json_path, 'r') as f:
        return json.load(f)


def get_phase1_kernels(classification):
    """Get all Phase 1 kernel names."""
    phase1_kernels = []
    for category in classification:
        if 'Phase 1' in category['description']:
            phase1_kernels.extend(category['kernels'])
    return phase1_kernels


def extract_cuda_kernel_implementations(kernel_names):
    """
    Extract CUDA kernel implementations from Kaldi repository.
    Returns a dictionary: {kernel_name: cuda_source}
    """
    cu_kernels_path = os.path.join(KALDI_REPO, "src/cudamatrix/cu-kernels.cu")
    
    with open(cu_kernels_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    results = {}
    
    for kernel_name in kernel_names:
        # Try to find the kernel implementation
        # Pattern 1: __global__ void _kernel_name(...)
        pattern1 = rf'(__global__\s+void\s+_{kernel_name}\s*\([^)]*\)\s*\{{(?:[^{{}}]|\{{[^{{}}]*\}})*\}})'
        
        # Pattern 2: For multiple overloads (float/double)
        # We need both _cuda_kernel_name_F and _cuda_kernel_name_D
        pattern_f = rf'(__global__\s+void\s+_cuda_{kernel_name}_[FD]\s*\([^)]*\)\s*\{{(?:[^{{}}]|\{{[^{{}}]*\}})*\}})'
        
        matches = re.findall(pattern1, content, re.DOTALL | re.MULTILINE)
        if not matches:
            matches = re.findall(pattern_f, content, re.DOTALL | re.MULTILINE)
        
        if matches:
            results[kernel_name] = '\n\n'.join(matches)
        else:
            print(f"⚠ Warning: Could not find CUDA implementation for {kernel_name}")
    
    return results


def extract_extern_c_wrappers(kernel_names):
    """
    Extract extern "C" wrapper functions from cu-kernels-ansi.h
    Returns a dictionary: {kernel_name: wrapper_source}
    """
    cu_kernels_ansi_path = os.path.join(KALDI_REPO, "src/cudamatrix/cu-kernels-ansi.h")
    
    with open(cu_kernels_ansi_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    results = {}
    
    for kernel_name in kernel_names:
        # Pattern: find cudaF_kernel_name and cudaD_kernel_name functions
        pattern = rf'(inline\s+void\s+cuda[FD]_{kernel_name}\s*\([^)]*\)\s*\{{[^}}]*\}})'
        
        matches = re.findall(pattern, content, re.DOTALL | re.MULTILINE)
        
        if matches:
            results[kernel_name] = '\n\n'.join(matches)
        else:
            print(f"⚠ Warning: Could not find extern C wrapper for {kernel_name}")
    
    return results


def extract_matrixdim_structure():
    """Extract MatrixDim structure definition."""
    cu_matrixdim_path = os.path.join(KALDI_REPO, "src/cudamatrix/cu-matrixdim.h")
    
    with open(cu_matrixdim_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract typedef
    pattern = r'(typedef\s+struct\s+MatrixDim_\s*\{[^}]+\}\s*MatrixDim;)'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        return match.group(1)
    else:
        # Fallback: use known definition
        return """typedef struct MatrixDim_ {
    int32_cuda rows;
    int32_cuda cols;
    int32_cuda stride;
} MatrixDim;"""


def extract_helper_functions():
    """Extract helper functions like n_blocks."""
    helpers = """
// Helper function for grid size calculation
inline int n_blocks(int size, int block_size) {
    return size / block_size + ((size % block_size == 0) ? 0 : 1);
}

#define CU1DBLOCK 256
#define CU2DBLOCK 16
"""
    return helpers


def generate_unified_cuda_source(kernel_infos, cuda_implementations, extern_wrappers):
    """
    Generate a unified CUDA source file containing all Phase 1 kernels.
    """
    lines = []
    
    # Header
    lines.append("// Auto-generated unified CUDA source for Phase 1 kernels")
    lines.append("// Total kernels: " + str(len(kernel_infos)))
    lines.append("")
    
    # Includes and typedefs
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append("typedef int32_t int32_cuda;")
    lines.append("typedef uint32_t uint32_cuda;")
    lines.append("typedef int32_t MatrixIndexT_cuda;")
    lines.append("")
    
    # MatrixDim structure
    lines.append(extract_matrixdim_structure())
    lines.append("")
    
    # Helper functions
    lines.append(extract_helper_functions())
    lines.append("")
    
    # Additional type definitions
    lines.append("// Additional type definitions")
    lines.append("template<typename Real>")
    lines.append("struct MatrixElement {")
    lines.append("    int32_cuda row;")
    lines.append("    int32_cuda column;")
    lines.append("    Real weight;")
    lines.append("};")
    lines.append("")
    lines.append("struct Int32Pair {")
    lines.append("    int32_cuda first;")
    lines.append("    int32_cuda second;")
    lines.append("};")
    lines.append("")
    lines.append("struct CuBlockMatrixData {")
    lines.append("    void* data;")
    lines.append("    int32_cuda row_offset;")
    lines.append("    int32_cuda col_offset;")
    lines.append("    MatrixDim matrix_dim;")
    lines.append("};")
    lines.append("")
    
    # CUDA kernel implementations
    lines.append("// ============================================================")
    lines.append("// CUDA Kernel Implementations")
    lines.append("// ============================================================")
    lines.append("")
    
    for kernel_name in sorted(kernel_infos.keys()):
        if kernel_name in cuda_implementations:
            lines.append(f"// Kernel: {kernel_name}")
            lines.append(cuda_implementations[kernel_name])
            lines.append("")
    
    # Extern C wrappers
    lines.append("// ============================================================")
    lines.append("// Extern C Launcher Functions")
    lines.append("// ============================================================")
    lines.append("")
    lines.append('extern "C" {')
    lines.append("")
    
    for kernel_name in sorted(kernel_infos.keys()):
        if kernel_name in extern_wrappers:
            lines.append(f"// Wrapper: {kernel_name}")
            lines.append(extern_wrappers[kernel_name])
            lines.append("")
    
    lines.append("} // extern C")
    lines.append("")
    
    return '\n'.join(lines)


def generate_grid_calculation_code(param):
    """Generate grid/block calculation code based on parameter type."""
    param_type = param.get('type', '')
    param_name = param.get('name', '')
    
    if param_type == 'dim3':
        if param_name == 'Gr':
            return None  # Will be calculated based on next parameters
        elif param_name == 'Bl':
            return None
    elif param_type == 'int' and param_name in ['Gr', 'Bl']:
        return None  # 1D grid
    elif param_type == 'size_t' and param_name in ['Gr', 'Bl']:
        return None  # size_t grid
    
    return None


def determine_grid_pattern(input_args):
    """Determine the grid/block pattern from input arguments."""
    if len(input_args) < 2:
        return 'none'
    
    first = input_args[0]
    second = input_args[1]
    
    if first.get('type') == 'dim3' and second.get('type') == 'dim3':
        return 'dim3'
    elif first.get('type') == 'int' and second.get('type') == 'int':
        if first.get('name') == 'Gr' and second.get('name') == 'Bl':
            return 'int_1d'
    elif first.get('type') == 'size_t' and second.get('type') == 'size_t':
        return 'sizet'
    
    return 'unknown'


def generate_cpp_adapter_function(kernel_name, kernel_info):
    """Generate C++ adapter function for a single kernel."""
    input_args = kernel_info.get('input_args', [])
    grid_pattern = determine_grid_pattern(input_args)
    
    lines = []
    lines.append(f"torch::Tensor {kernel_name}(")
    
    # Build parameter list (skip Gr and Bl, we'll calculate them)
    params = []
    data_params = []
    
    skip_next = 0
    for i, arg in enumerate(input_args):
        if skip_next > 0:
            skip_next -= 1
            continue
        
        arg_name = arg.get('name', '')
        arg_type = arg.get('type', '')
        
        # Skip grid/block parameters
        if arg_name in ['Gr', 'Bl']:
            continue
        
        # Convert types to torch
        if arg_type in ['double*', 'float*']:
            params.append(f"torch::Tensor {arg_name}")
            data_params.append(arg_name)
        elif arg_type in ['const double*', 'const float*']:
            params.append(f"const torch::Tensor& {arg_name}")
            data_params.append(arg_name)
        elif arg_type == 'MatrixDim':
            # Will be computed from tensor
            pass
        elif arg_type == 'const MatrixDim':
            pass
        elif 'double' in arg_type or 'float' in arg_type:
            params.append(f"double {arg_name}")
        elif 'int' in arg_type:
            params.append(f"int {arg_name}")
        else:
            params.append(f"void* {arg_name}")  # Generic pointer
    
    lines.append("    " + ",\n    ".join(params))
    lines.append(") {")
    
    # Function body
    lines.append("    // TODO: Implement adapter logic")
    lines.append(f"    // Call cuda[F|D]_{kernel_name}(...)")
    lines.append("    return torch::Tensor();")
    lines.append("}")
    
    return '\n'.join(lines)


def generate_unified_cpp_adapter(kernel_infos):
    """Generate unified C++ adapter file."""
    lines = []
    
    # Header
    lines.append("// Auto-generated C++ adapter for Phase 1 kernels")
    lines.append("")
    lines.append("#include <torch/extension.h>")
    lines.append("#include <cuda_runtime.h>")
    lines.append("")
    
    # MatrixDim helper
    lines.append("// Helper to get MatrixDim from tensor")
    lines.append("struct MatrixDim {")
    lines.append("    int rows;")
    lines.append("    int cols;")
    lines.append("    int stride;")
    lines.append("};")
    lines.append("")
    lines.append("MatrixDim get_matrix_dim(const torch::Tensor& tensor) {")
    lines.append("    MatrixDim dim;")
    lines.append("    dim.rows = tensor.size(0);")
    lines.append("    dim.cols = tensor.size(1);")
    lines.append("    dim.stride = tensor.stride(0);")
    lines.append("    return dim;")
    lines.append("}")
    lines.append("")
    
    # Extern C declarations
    lines.append("// Extern C function declarations")
    lines.append('extern "C" {')
    for kernel_name in sorted(kernel_infos.keys()):
        lines.append(f"    void cudaF_{kernel_name}(...);")
        lines.append(f"    void cudaD_{kernel_name}(...);")
    lines.append("}")
    lines.append("")
    
    # Adapter functions
    lines.append("// Adapter functions")
    for kernel_name, kernel_info in sorted(kernel_infos.items()):
        lines.append("")
        adapter_code = generate_cpp_adapter_function(kernel_name, kernel_info)
        lines.append(adapter_code)
    
    # Pybind11 module
    lines.append("")
    lines.append("PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {")
    for kernel_name in sorted(kernel_infos.keys()):
        lines.append(f'    m.def("{kernel_name}", &{kernel_name}, "Kaldi {kernel_name}");')
    lines.append("}")
    lines.append("")
    
    return '\n'.join(lines)


def main():
    """Main entry point."""
    print("=" * 80)
    print("PHASE 1: BATCH PROCESSING STANDARD PATTERN KERNELS")
    print("=" * 80)
    
    # Load classification
    classification = load_classification()
    phase1_kernels = get_phase1_kernels(classification)
    
    print(f"\nTotal Phase 1 kernels: {len(phase1_kernels)}")
    print(f"Kernels: {', '.join(phase1_kernels[:10])}...")
    
    # Get kernel info
    kernel_infos = {name: IMPL_INFO_K1[name] for name in phase1_kernels if name in IMPL_INFO_K1}
    
    print(f"\nStep 1: Extracting CUDA kernel implementations...")
    cuda_implementations = extract_cuda_kernel_implementations(list(kernel_infos.keys()))
    print(f"✓ Extracted {len(cuda_implementations)} CUDA implementations")
    
    print(f"\nStep 2: Extracting extern C wrappers...")
    extern_wrappers = extract_extern_c_wrappers(list(kernel_infos.keys()))
    print(f"✓ Extracted {len(extern_wrappers)} extern C wrappers")
    
    print(f"\nStep 3: Generating unified CUDA source...")
    cuda_source = generate_unified_cuda_source(kernel_infos, cuda_implementations, extern_wrappers)
    
    # Save CUDA source
    cuda_output_path = os.path.join(CACHE_DIR, "extracted_cuda", "phase1_unified.cu")
    os.makedirs(os.path.dirname(cuda_output_path), exist_ok=True)
    with open(cuda_output_path, 'w') as f:
        f.write(cuda_source)
    print(f"✓ Saved CUDA source to: {cuda_output_path}")
    print(f"  Size: {len(cuda_source)} bytes")
    
    print(f"\nStep 4: Generating C++ adapter...")
    cpp_adapter = generate_unified_cpp_adapter(kernel_infos)
    
    # Save adapter
    adapter_output_path = os.path.join(CACHE_DIR, "generated_adapters", "phase1_adapter.cpp")
    os.makedirs(os.path.dirname(adapter_output_path), exist_ok=True)
    with open(adapter_output_path, 'w') as f:
        f.write(cpp_adapter)
    print(f"✓ Saved C++ adapter to: {adapter_output_path}")
    print(f"  Size: {len(cpp_adapter)} bytes")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Phase 1 kernels to process:  {len(phase1_kernels)}")
    print(f"CUDA implementations found:  {len(cuda_implementations)}")
    print(f"Extern C wrappers found:     {len(extern_wrappers)}")
    print(f"Missing implementations:     {len(phase1_kernels) - len(cuda_implementations)}")
    
    if len(cuda_implementations) < len(phase1_kernels):
        missing = set(phase1_kernels) - set(cuda_implementations.keys())
        print(f"\nMissing kernels: {', '.join(sorted(missing)[:10])}...")
    
    print("\n✓ Phase 1 extraction complete!")
    print("  Next step: Fix extraction patterns for missing kernels")


if __name__ == '__main__':
    main()
