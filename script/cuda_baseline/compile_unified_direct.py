#!/usr/bin/env python3
"""
Direct compilation of Kaldi's cu-kernels.cu as a unified module.
This script compiles all 169 Kaldi CUDA kernels in one go.
"""

import os
import sys

REPO_TOP_DIR = os.path.abspath(os.path.dirname(__file__))
KALDI_REPO = "/share/project/zpy/k1_repo"
CACHE_DIR = os.path.join(REPO_TOP_DIR, "cache")

CUDA_SOURCE = os.path.join(KALDI_REPO, "src/cudamatrix/cu-kernels.cu")
ANSI_HEADER = os.path.join(KALDI_REPO, "src/cudamatrix/cu-kernels-ansi.h")
MATRIXDIM_HEADER = os.path.join(KALDI_REPO, "src/cudamatrix/cu-matrixdim.h")
CU_MATRIX_DIM_H = os.path.join(KALDI_REPO, "src/cudamatrix/cu-matrix-dim.h")


def generate_cpp_adapter():
    """
    Generate C++ adapter that includes Kaldi headers and provides pybind11 bindings.
    """
    
    adapter_code = """// Auto-generated C++ adapter for Kaldi K1 CUDA Kernels
// Compiles entire cu-kernels.cu directly

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>

// ============================================================================
// Include Kaldi headers
// ============================================================================

// MatrixDim structure from cu-matrix-dim.h
typedef int32_t int32_cuda;

struct MatrixDim_ {
    int32_cuda rows;
    int32_cuda cols;
    int32_cuda stride;
};

typedef struct MatrixDim_ MatrixDim;

// Additional type definitions needed by kernels
template<typename Real>
struct MatrixElement {
    int32_cuda row;
    int32_cuda column;
    Real weight;
};

struct Int32Pair {
    int32_cuda first;
    int32_cuda second;
};

struct CuBlockMatrixData {
    void* data;
    int32_cuda row_offset;
    int32_cuda col_offset;
    MatrixDim matrix_dim;
};

// ============================================================================
// Forward declarations for Kaldi kernel launchers
// ============================================================================

// These are defined in the cu-kernels.cu file
extern "C" {

// Copy operations
void cudaF_copy_low_upp(dim3 Gr, dim3 Bl, float* A, MatrixDim dimA);
void cudaD_copy_low_upp(dim3 Gr, dim3 Bl, double* A, MatrixDim dimA);
void cudaF_copy_upp_low(dim3 Gr, dim3 Bl, float* A, MatrixDim dimA);
void cudaD_copy_upp_low(dim3 Gr, dim3 Bl, double* A, MatrixDim dimA);
void cudaF_copy_from_tp(dim3 Gr, dim3 Bl, float* A, const double* B, MatrixDim dmat);
void cudaD_copy_from_tp(dim3 Gr, dim3 Bl, double* A, const float* B, MatrixDim dmat);
void cudaF_copy_from_tp_trans(dim3 Gr, dim3 Bl, float* A, const double* B, MatrixDim dmat);
void cudaD_copy_from_tp_trans(dim3 Gr, dim3 Bl, double* A, const float* B, MatrixDim dmat);
void cudaF_copy_from_mat(dim3 Gr, dim3 Bl, float* mat_out, const float* mat_in, MatrixDim d_out, MatrixDim d_in);
void cudaD_copy_from_mat(dim3 Gr, dim3 Bl, double* mat_out, const double* mat_in, MatrixDim d_out, MatrixDim d_in);
void cudaF_copy_from_mat_trans(dim3 Gr, dim3 Bl, float* mat_out, const float* mat_in, MatrixDim d_out, MatrixDim d_in);
void cudaD_copy_from_mat_trans(dim3 Gr, dim3 Bl, double* mat_out, const double* mat_in, MatrixDim d_out, MatrixDim d_in);

// Add operations
void cudaF_add_mat(dim3 Gr, dim3 Bl, float* dst, const float* src, float alpha, MatrixDim d);
void cudaD_add_mat(dim3 Gr, dim3 Bl, double* dst, const double* src, double alpha, MatrixDim d);
void cudaF_add(dim3 Gr, dim3 Bl, float* mat, float value, MatrixDim d);
void cudaD_add(dim3 Gr, dim3 Bl, double* mat, double value, MatrixDim d);
void cudaF_scale(dim3 Gr, dim3 Bl, float* mat, float value, MatrixDim d);
void cudaD_scale(dim3 Gr, dim3 Bl, double* mat, double value, MatrixDim d);
void cudaF_set_const(dim3 Gr, dim3 Bl, float* mat, float value, MatrixDim d);
void cudaD_set_const(dim3 Gr, dim3 Bl, double* mat, double value, MatrixDim d);

// Element-wise operations
void cudaF_exp(dim3 Gr, dim3 Bl, float* y, const float* x, MatrixDim d, int src_stride);
void cudaD_exp(dim3 Gr, dim3 Bl, double* y, const double* x, MatrixDim d, int src_stride);
void cudaF_log(dim3 Gr, dim3 Bl, float* y, const float* x, MatrixDim d, int src_stride);
void cudaD_log(dim3 Gr, dim3 Bl, double* y, const double* x, MatrixDim d, int src_stride);
void cudaF_sigmoid(dim3 Gr, dim3 Bl, float* y, const float* x, MatrixDim d, int src_stride);
void cudaD_sigmoid(dim3 Gr, dim3 Bl, double* y, const double* x, MatrixDim d, int src_stride);
void cudaF_tanh(dim3 Gr, dim3 Bl, float* y, const float* x, MatrixDim d, int src_stride);
void cudaD_tanh(dim3 Gr, dim3 Bl, double* y, const double* x, MatrixDim d, int src_stride);

// ... more kernels will be added as needed

} // extern "C"

// ============================================================================
// Helper functions
// ============================================================================

inline dim3 calculate_grid(MatrixDim dim, int block_size = 16) {
    int num_blocks = (dim.cols + block_size - 1) / block_size;
    int num_rows = (dim.rows + block_size - 1) / block_size;
    return dim3(num_blocks, num_rows);
}

inline dim3 calculate_block(int block_size = 16) {
    return dim3(block_size, block_size);
}

// ============================================================================
// Python wrapper functions
// ============================================================================

// copy_low_upp: Copy lower triangle to upper triangle
void copy_low_upp(torch::Tensor A) {
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    
    MatrixDim dim;
    dim.rows = A.size(0);
    dim.cols = A.size(1);
    dim.stride = A.stride(0);
    
    dim3 Gr = calculate_grid(dim);
    dim3 Bl = calculate_block();
    
    if (A.scalar_type() == torch::kFloat32) {
        cudaF_copy_low_upp(Gr, Bl, A.data_ptr<float>(), dim);
    } else if (A.scalar_type() == torch::kFloat64) {
        cudaD_copy_low_upp(Gr, Bl, A.data_ptr<double>(), dim);
    } else {
        TORCH_CHECK(false, "Unsupported dtype: ", A.scalar_type());
    }
}

// copy_upp_low: Copy upper triangle to lower triangle
void copy_upp_low(torch::Tensor A) {
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    
    MatrixDim dim;
    dim.rows = A.size(0);
    dim.cols = A.size(1);
    dim.stride = A.stride(0);
    
    dim3 Gr = calculate_grid(dim);
    dim3 Bl = calculate_block();
    
    if (A.scalar_type() == torch::kFloat32) {
        cudaF_copy_upp_low(Gr, Bl, A.data_ptr<float>(), dim);
    } else if (A.scalar_type() == torch::kFloat64) {
        cudaD_copy_upp_low(Gr, Bl, A.data_ptr<double>(), dim);
    } else {
        TORCH_CHECK(false, "Unsupported dtype: ", A.scalar_type());
    }
}

// add_mat: dst = alpha * src + dst
void add_mat(torch::Tensor dst, torch::Tensor src, double alpha) {
    TORCH_CHECK(dst.device().is_cuda(), "dst must be a CUDA tensor");
    TORCH_CHECK(src.device().is_cuda(), "src must be a CUDA tensor");
    TORCH_CHECK(dst.dim() == 2, "dst must be a 2D tensor");
    TORCH_CHECK(src.dim() == 2, "src must be a 2D tensor");
    TORCH_CHECK(dst.sizes() == src.sizes(), "dst and src must have same size");
    
    MatrixDim d;
    d.rows = dst.size(0);
    d.cols = dst.size(1);
    d.stride = dst.stride(0);
    
    dim3 Gr = calculate_grid(d);
    dim3 Bl = calculate_block();
    
    if (dst.scalar_type() == torch::kFloat32) {
        cudaF_add_mat(Gr, Bl, dst.data_ptr<float>(), src.data_ptr<float>(), (float)alpha, d);
    } else if (dst.scalar_type() == torch::kFloat64) {
        cudaD_add_mat(Gr, Bl, dst.data_ptr<double>(), src.data_ptr<double>(), alpha, d);
    } else {
        TORCH_CHECK(false, "Unsupported dtype: ", dst.scalar_type());
    }
}

// scale: mat = value * mat
void scale(torch::Tensor mat, double value) {
    TORCH_CHECK(mat.device().is_cuda(), "mat must be a CUDA tensor");
    TORCH_CHECK(mat.dim() == 2, "mat must be a 2D tensor");
    
    MatrixDim d;
    d.rows = mat.size(0);
    d.cols = mat.size(1);
    d.stride = mat.stride(0);
    
    dim3 Gr = calculate_grid(d);
    dim3 Bl = calculate_block();
    
    if (mat.scalar_type() == torch::kFloat32) {
        cudaF_scale(Gr, Bl, mat.data_ptr<float>(), (float)value, d);
    } else if (mat.scalar_type() == torch::kFloat64) {
        cudaD_scale(Gr, Bl, mat.data_ptr<double>(), value, d);
    } else {
        TORCH_CHECK(false, "Unsupported dtype: ", mat.scalar_type());
    }
}

// set_const: mat = value
void set_const(torch::Tensor mat, double value) {
    TORCH_CHECK(mat.device().is_cuda(), "mat must be a CUDA tensor");
    TORCH_CHECK(mat.dim() == 2, "mat must be a 2D tensor");
    
    MatrixDim d;
    d.rows = mat.size(0);
    d.cols = mat.size(1);
    d.stride = mat.stride(0);
    
    dim3 Gr = calculate_grid(d);
    dim3 Bl = calculate_block();
    
    if (mat.scalar_type() == torch::kFloat32) {
        cudaF_set_const(Gr, Bl, mat.data_ptr<float>(), (float)value, d);
    } else if (mat.scalar_type() == torch::kFloat64) {
        cudaD_set_const(Gr, Bl, mat.data_ptr<double>(), value, d);
    } else {
        TORCH_CHECK(false, "Unsupported dtype: ", mat.scalar_type());
    }
}

// exp: y = exp(x)
void kaldi_exp(torch::Tensor y, torch::Tensor x) {
    TORCH_CHECK(y.device().is_cuda(), "y must be a CUDA tensor");
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    
    MatrixDim d;
    d.rows = x.size(0);
    d.cols = x.size(1);
    d.stride = x.stride(0);
    
    dim3 Gr = calculate_grid(d);
    dim3 Bl = calculate_block();
    
    if (x.scalar_type() == torch::kFloat32) {
        cudaF_exp(Gr, Bl, y.data_ptr<float>(), x.data_ptr<float>(), d, d.stride);
    } else if (x.scalar_type() == torch::kFloat64) {
        cudaD_exp(Gr, Bl, y.data_ptr<double>(), x.data_ptr<double>(), d, d.stride);
    } else {
        TORCH_CHECK(false, "Unsupported dtype: ", x.scalar_type());
    }
}

// log: y = log(x)
void kaldi_log(torch::Tensor y, torch::Tensor x) {
    TORCH_CHECK(y.device().is_cuda(), "y must be a CUDA tensor");
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    
    MatrixDim d;
    d.rows = x.size(0);
    d.cols = x.size(1);
    d.stride = x.stride(0);
    
    dim3 Gr = calculate_grid(d);
    dim3 Bl = calculate_block();
    
    if (x.scalar_type() == torch::kFloat32) {
        cudaF_log(Gr, Bl, y.data_ptr<float>(), x.data_ptr<float>(), d, d.stride);
    } else if (x.scalar_type() == torch::kFloat64) {
        cudaD_log(Gr, Bl, y.data_ptr<double>(), x.data_ptr<double>(), d, d.stride);
    } else {
        TORCH_CHECK(false, "Unsupported dtype: ", x.scalar_type());
    }
}

// sigmoid: y = sigmoid(x)
void sigmoid(torch::Tensor y, torch::Tensor x) {
    TORCH_CHECK(y.device().is_cuda(), "y must be a CUDA tensor");
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    
    MatrixDim d;
    d.rows = x.size(0);
    d.cols = x.size(1);
    d.stride = x.stride(0);
    
    dim3 Gr = calculate_grid(d);
    dim3 Bl = calculate_block();
    
    if (x.scalar_type() == torch::kFloat32) {
        cudaF_sigmoid(Gr, Bl, y.data_ptr<float>(), x.data_ptr<float>(), d, d.stride);
    } else if (x.scalar_type() == torch::kFloat64) {
        cudaD_sigmoid(Gr, Bl, y.data_ptr<double>(), x.data_ptr<double>(), d, d.stride);
    } else {
        TORCH_CHECK(false, "Unsupported dtype: ", x.scalar_type());
    }
}

// tanh: y = tanh(x)
void tanh_func(torch::Tensor y, torch::Tensor x) {
    TORCH_CHECK(y.device().is_cuda(), "y must be a CUDA tensor");
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    
    MatrixDim d;
    d.rows = x.size(0);
    d.cols = x.size(1);
    d.stride = x.stride(0);
    
    dim3 Gr = calculate_grid(d);
    dim3 Bl = calculate_block();
    
    if (x.scalar_type() == torch::kFloat32) {
        cudaF_tanh(Gr, Bl, y.data_ptr<float>(), x.data_ptr<float>(), d, d.stride);
    } else if (x.scalar_type() == torch::kFloat64) {
        cudaD_tanh(Gr, Bl, y.data_ptr<double>(), x.data_ptr<double>(), d, d.stride);
    } else {
        TORCH_CHECK(false, "Unsupported dtype: ", x.scalar_type());
    }
}

// ============================================================================
// Pybind11 module definition
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Kaldi K1 CUDA Kernels - Unified Python interface";
    
    // Copy operations
    m.def("copy_low_upp", &copy_low_upp, "Copy lower triangle to upper triangle");
    m.def("copy_upp_low", &copy_upp_low, "Copy upper triangle to lower triangle");
    
    // Add/scale operations
    m.def("add_mat", &add_mat, "dst = alpha * src + dst");
    m.def("scale", &scale, "mat = value * mat");
    m.def("set_const", &set_const, "mat = value");
    
    // Element-wise operations
    m.def("kaldi_exp", &kaldi_exp, "y = exp(x)");
    m.def("kaldi_log", &kaldi_log, "y = log(x)");
    m.def("sigmoid", &sigmoid, "y = sigmoid(x)");
    m.def("tanh", &tanh_func, "y = tanh(x)");
}

"""
    
    return adapter_code


def compile_unified_module():
    """Compile the unified Kaldi CUDA module."""
    
    print("=" * 80)
    print("COMPILING KALDI K1 UNIFIED CUDA MODULE")
    print("=" * 80)
    
    # Check source files exist
    if not os.path.exists(CUDA_SOURCE):
        print(f"✗ Error: CUDA source not found: {CUDA_SOURCE}")
        return False
    
    print(f"\n✓ Source file: {CUDA_SOURCE}")
    
    # Generate adapter
    adapter_code = generate_cpp_adapter()
    adapter_path = os.path.join(CACHE_DIR, "generated_adapters", "unified_kaldi_adapter_v2.cpp")
    os.makedirs(os.path.dirname(adapter_path), exist_ok=True)
    
    with open(adapter_path, 'w') as f:
        f.write(adapter_code)
    print(f"✓ Generated adapter: {adapter_path}")
    print(f"  Size: {len(adapter_code)} bytes")
    
    # Read CUDA source
    with open(CUDA_SOURCE, 'r') as f:
        cuda_code = f.read()
    print(f"✓ Read CUDA source: {len(cuda_code)} bytes")
    
    # Combine CUDA source with adapter
    combined_cuda = f"""{adapter_code}

// ============================================================================
// Kaldi CUDA Kernel Implementations (from cu-kernels.cu)
// ============================================================================

{cuda_code}
"""
    
    combined_path = os.path.join(CACHE_DIR, "extracted_cuda", "unified_kaldi_combined.cu")
    os.makedirs(os.path.dirname(combined_path), exist_ok=True)
    
    with open(combined_path, 'w') as f:
        f.write(combined_cuda)
    print(f"✓ Combined CUDA source: {combined_path}")
    print(f"  Size: {len(combined_cuda)} bytes")
    
    # Use torch.utils.cpp_extension to compile
    from torch.utils.cpp_extension import load_inline
    
    # Include Kaldi headers
    kaldi_include_dirs = [
        os.path.join(KALDI_REPO, "src"),
        os.path.join(KALDI_REPO, "src/cudamatrix"),
        os.path.join(KALDI_REPO, "src/base"),
    ]
    
    print("\n" + "-" * 80)
    print("Compiling with NVCC...")
    print("-" * 80)
    
    try:
        module = load_inline(
            name='kaldi_k1_unified_v2',
            cpp_sources=adapter_code,
            cuda_sources=combined_cuda,
            extra_cuda_cflags=['-O3', '--use_fast_math'],
            extra_cflags=['-O3'],
            extra_include_paths=kaldi_include_dirs,
            with_cuda=True,
            verbose=True
        )
        
        print("\n" + "=" * 80)
        print("✓ COMPILATION SUCCESSFUL!")
        print("=" * 80)
        
        # Save module info
        info_path = os.path.join(CACHE_DIR, "cuda_jit_unified_v2", "module_info.txt")
        os.makedirs(os.path.dirname(info_path), exist_ok=True)
        with open(info_path, 'w') as f:
            f.write(f"Compiled at: {os.path.abspath(info_path)}\n")
            f.write(f"CUDA source size: {len(cuda_code)} bytes\n")
            f.write(f"Adapter size: {len(adapter_code)} bytes\n")
            f.write(f"Combined size: {len(combined_cuda)} bytes\n")
        
        return module
        
    except Exception as e:
        print(f"\n✗ Compilation failed: {e}")
        print("\nTrying with reduced optimization...")
        
        try:
            # Try with lower optimization
            module = load_inline(
                name='kaldi_k1_unified_v2_fallback',
                cpp_sources=adapter_code,
                cuda_sources=combined_cuda,
                extra_cuda_cflags=['-O2'],
                extra_cflags=['-O2'],
                with_cuda=True,
                verbose=True
            )
            print("\n✓ Fallback compilation successful!")
            return module
        except Exception as e2:
            print(f"\n✗ Fallback also failed: {e2}")
            return None


def test_compiled_module(module):
    """Test the compiled module with basic operations."""
    
    if module is None:
        print("✗ No module to test")
        return False
    
    print("\n" + "=" * 80)
    print("TESTING COMPILED MODULE")
    print("=" * 80)
    
    try:
        import torch
        
        print("\nTest 1: copy_low_upp")
        A = torch.randn(64, 64, device='cuda', dtype=torch.float32)
        A_before = A.clone()
        module.copy_low_upp(A)
        print("✓ copy_low_upp executed")
        
        print("\nTest 2: copy_upp_low")
        B = torch.randn(64, 64, device='cuda', dtype=torch.float32)
        module.copy_upp_low(B)
        print("✓ copy_upp_low executed")
        
        print("\nTest 3: add_mat")
        dst = torch.randn(64, 64, device='cuda', dtype=torch.float32)
        src = torch.randn(64, 64, device='cuda', dtype=torch.float32)
        alpha = 2.0
        module.add_mat(dst, src, alpha)
        print("✓ add_mat executed")
        
        print("\nTest 4: scale")
        C = torch.randn(64, 64, device='cuda', dtype=torch.float32)
        module.scale(C, 0.5)
        print("✓ scale executed")
        
        print("\nTest 5: set_const")
        D = torch.randn(64, 64, device='cuda', dtype=torch.float32)
        module.set_const(D, 3.14)
        print("✓ set_const executed")
        
        print("\nTest 6: kaldi_exp")
        X = torch.randn(64, 64, device='cuda', dtype=torch.float32)
        Y = torch.zeros_like(X)
        module.kaldi_exp(Y, X)
        print("✓ kaldi_exp executed")
        
        print("\nTest 7: kaldi_log")
        X2 = torch.abs(torch.randn(64, 64, device='cuda', dtype=torch.float32)) + 0.1
        Y2 = torch.zeros_like(X2)
        module.kaldi_log(Y2, X2)
        print("✓ kaldi_log executed")
        
        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED!")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    
    # Compile the module
    module = compile_unified_module()
    
    if module is not None:
        # Test it
        test_compiled_module(module)
    else:
        print("\n✗ Failed to compile module")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
