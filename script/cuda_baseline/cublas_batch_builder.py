#!/usr/bin/env python3
"""
CuBLAS Batch Builder - 自动生成cuBLAS Python绑定
================================================

使用Registry模式，动态生成C++ wrapper代码并编译为Python模块。
"""

import os
import sys
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

REPO_TOP_DIR = os.path.abspath(os.path.dirname(__file__))
CACHE_DIR = os.path.join(REPO_TOP_DIR, "cache")


@dataclass
class CuBLASOperator:
    """cuBLAS算子定义"""
    name: str                      # 例如: cublasSgemm
    template: str                  # 模板类型: GEMM_FLOAT, AXPY_FLOAT等
    description: str = ""          # 功能描述
    dtype: str = "float"           # 数据类型: float, double, half
    cblas_func: str = ""           # 对应的cuBLAS函数名

@dataclass
class Template:
    """代码模板"""
    cpp_wrapper: str               # C++ wrapper代码
    pybind11_binding: str = ""     # pybind11绑定
    test_code: str = ""            # 测试代码


# 预定义模板库
TEMPLATES = {
    'GEMM_FLOAT': Template(
        cpp_wrapper="""
// ============================================================================
// GEMM_FLOAT: Single-precision matrix multiplication (C = alpha * A * B + beta * C)
// ============================================================================
void {{func_name}}(
    torch::Tensor C,
    double alpha,
    torch::Tensor A,
    torch::Tensor B,
    double beta
) {
    TORCH_CHECK(C.device().is_cuda(), "C must be CUDA tensor");
    TORCH_CHECK(A.device().is_cuda(), "A must be CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be CUDA tensor");
    
    TORCH_CHECK(C.scalar_type() == torch::kFloat32, "C must be float32");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.scalar_type() == torch::kFloat32, "B must be float32");
    
    // PyTorch is Row-Major, cuBLAS is Column-Major
    // For C = A * B (Row-Major), we call cuBLAS with:
    //   C^T = B^T * A^T (Column-Major)
    // So we swap A and B, and swap M and N
    
    int M = A.size(0);  // rows of A
    int N = B.size(1);  // cols of B
    int K = A.size(1);  // cols of A = rows of B
    
    TORCH_CHECK(C.size(0) == M, "C.rows must match A.rows");
    TORCH_CHECK(C.size(1) == N, "C.cols must match B.cols");
    TORCH_CHECK(A.size(1) == K, "A.cols must match B.rows");
    
    // Use cuBLAS with Column-Major: op(B) * op(A)
    float f_alpha = static_cast<float>(alpha);
    float f_beta = static_cast<float>(beta);
    
    // cuBLAS expects Column-Major, so we pass B first, then A
    // Dimensions: B is (K, N), A is (M, K) in Column-Major
    cublasSgemm(
        get_handle(),
        CUBLAS_OP_N, CUBLAS_OP_N,  // No transpose for both (since we swap)
        N, M, K,                   // Note: swapped dimensions
        &f_beta,                   // beta
        B.data_ptr<float>(), N,    // B (stride N in Column-Major)
        A.data_ptr<float>(), K,    // A (stride K in Column-Major)
        &f_alpha,                  // alpha
        C.data_ptr<float>(), N     // C (stride N in Column-Major)
    );
}
""",
        pybind11_binding="""
    m.def("{{func_name}}", &{{func_name}},
          "{{desc}}\\n\\n"
          "Args:\\n"
          "    C: Output matrix (M x N), modified in-place\\n"
          "    alpha: Scalar multiplier for A * B\\n"
          "    A: Input matrix (M x K)\\n"
          "    B: Input matrix (K x N)\\n"
          "    beta: Scalar multiplier for C");
""",
        test_code="""
def test_gemm():
    M, K, N = 64, 32, 64
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)
    C_torch = torch.randn(M, N, device='cuda', dtype=torch.float32)
    C_ref = C_torch.clone()
    
    alpha, beta = 2.0, 0.5
    torch.addmm(C_ref, A, B, beta=beta, alpha=alpha)
    
    C_cublas = C_torch.clone()
    module.sgemm(C_cublas, alpha, A, B, beta)
    
    assert torch.allclose(C_cublas, C_ref, rtol=1e-4), "GEMM test failed"
    print("✓ cublasSgemm test passed")
"""
    ),
    
    'AXPY_FLOAT': Template(
        cpp_wrapper="""
// ============================================================================
// AXPY_FLOAT: Vector operation (Y = alpha * X + Y)
// ============================================================================
void {{func_name}}(
    torch::Tensor Y,
    double alpha,
    torch::Tensor X
) {
    TORCH_CHECK(Y.device().is_cuda(), "Y must be CUDA tensor");
    TORCH_CHECK(X.device().is_cuda(), "X must be CUDA tensor");
    
    TORCH_CHECK(Y.scalar_type() == torch::kFloat32, "Y must be float32");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32, "X must be float32");
    
    TORCH_CHECK(Y.dim() == 1, "Y must be 1D vector");
    TORCH_CHECK(X.dim() == 1, "X must be 1D vector");
    
    int n = Y.size(0);
    TORCH_CHECK(X.size(0) == n, "X and Y must have same size");
    
    float f_alpha = static_cast<float>(alpha);
    
    cublasSaxpy(
        get_handle(),
        n,
        &f_alpha,
        X.data_ptr<float>(),
        1,
        Y.data_ptr<float>(),
        1
    );
}
""",
        pybind11_binding="""
    m.def("{{func_name}}", &{{func_name}},
          "{{desc}}\\n\\n"
          "Args:\\n"
          "    Y: Input/Output vector (n), modified in-place\\n"
          "    alpha: Scalar multiplier for X\\n"
          "    X: Input vector (n)");
""",
        test_code="""
def test_axpy():
    n = 1024
    X = torch.randn(n, device='cuda', dtype=torch.float32)
    Y = torch.randn(n, device='cuda', dtype=torch.float32)
    Y_ref = Y.clone()
    
    alpha = 2.0
    Y_ref.add_(X, alpha=alpha)
    
    module.saxpy(Y, alpha, X)
    
    assert torch.allclose(Y, Y_ref, rtol=1e-4), "AXPY test failed"
    print("✓ cublasSaxpy test passed")
"""
    ),
    
    'GEMM_DOUBLE': Template(
        cpp_wrapper="""
// ============================================================================
// GEMM_DOUBLE: Double-precision matrix multiplication
// ============================================================================
void {{func_name}}(
    torch::Tensor C,
    double alpha,
    torch::Tensor A,
    torch::Tensor B,
    double beta
) {
    TORCH_CHECK(C.device().is_cuda(), "C must be CUDA tensor");
    TORCH_CHECK(A.device().is_cuda(), "A must be CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be CUDA tensor");
    
    TORCH_CHECK(C.scalar_type() == torch::kFloat64, "C must be float64");
    TORCH_CHECK(A.scalar_type() == torch::kFloat64, "A must be float64");
    TORCH_CHECK(B.scalar_type() == torch::kFloat64, "B must be float64");
    
    int M = A.size(0);
    int N = B.size(1);
    int K = A.size(1);
    
    TORCH_CHECK(C.size(0) == M, "C.rows must match A.rows");
    TORCH_CHECK(C.size(1) == N, "C.cols must match B.cols");
    
    cublasDgemm(
        get_handle(),
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &beta,
        B.data_ptr<double>(), N,
        A.data_ptr<double>(), K,
        &alpha,
        C.data_ptr<double>(), N
    );
}
""",
        pybind11_binding="""
    m.def("{{func_name}}", &{{func_name}},
          "{{desc}}\\n\\n"
          "Double-precision matrix multiplication.");
""",
        test_code="""
def test_gemm_double():
    M, K, N = 64, 32, 64
    A = torch.randn(M, K, device='cuda', dtype=torch.float64)
    B = torch.randn(K, N, device='cuda', dtype=torch.float64)
    C_torch = torch.randn(M, N, device='cuda', dtype=torch.float64)
    C_ref = C_torch.clone()
    
    alpha, beta = 2.0, 0.5
    torch.addmm(C_ref, A, B, beta=beta, alpha=alpha)
    
    C_cublas = C_torch.clone()
    module.dgemm(C_cublas, alpha, A, B, beta)
    
    assert torch.allclose(C_cublas, C_ref, rtol=1e-8), "DGEMM test failed"
    print("✓ cublasDgemm test passed")
"""
    ),
    
    'SCAL_FLOAT': Template(
        cpp_wrapper="""
// ============================================================================
// SCAL_FLOAT: Vector scaling (X = alpha * X)
// ============================================================================
void {{func_name}}(
    torch::Tensor X,
    double alpha
) {
    TORCH_CHECK(X.device().is_cuda(), "X must be CUDA tensor");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32, "X must be float32");
    TORCH_CHECK(X.dim() == 1, "X must be 1D vector");
    
    int n = X.size(0);
    float f_alpha = static_cast<float>(alpha);
    
    cublasSscal(
        get_handle(),
        n,
        &f_alpha,
        X.data_ptr<float>(),
        1
    );
}
""",
        pybind11_binding="""
    m.def("{{func_name}}", &{{func_name}},
          "{{desc}}\\n\\n"
          "Args:\\n"
          "    X: Input/Output vector (n), scaled in-place\\n"
          "    alpha: Scalar multiplier");
""",
        test_code="""
def test_scal():
    n = 1024
    X = torch.randn(n, device='cuda', dtype=torch.float32)
    X_ref = X.clone()
    
    alpha = 2.0
    X_ref.mul_(alpha)
    
    module.sscal(X, alpha)
    
    assert torch.allclose(X, X_ref, rtol=1e-4), "SCAL test failed"
    print("✓ cublasSscal test passed")
"""
    ),
    
    'DOT_FLOAT': Template(
        cpp_wrapper="""
// ============================================================================
// DOT_FLOAT: Dot product of two vectors
// ============================================================================
torch::Tensor {{func_name}}(
    torch::Tensor X,
    torch::Tensor Y
) {
    TORCH_CHECK(X.device().is_cuda(), "X must be CUDA tensor");
    TORCH_CHECK(Y.device().is_cuda(), "Y must be CUDA tensor");
    
    TORCH_CHECK(X.scalar_type() == torch::kFloat32, "X must be float32");
    TORCH_CHECK(Y.scalar_type() == torch::kFloat32, "Y must be float32");
    
    TORCH_CHECK(X.dim() == 1, "X must be 1D vector");
    TORCH_CHECK(Y.dim() == 1, "Y must be 1D vector");
    
    int n = X.size(0);
    TORCH_CHECK(Y.size(0) == n, "X and Y must have same size");
    
    float result;
    cublasSdot(
        get_handle(),
        n,
        X.data_ptr<float>(),
        1,
        Y.data_ptr<float>(),
        1,
        &result
    );
    
    return torch::tensor(result, torch::kCUDA);
}
""",
        pybind11_binding="""
    m.def("{{func_name}}", &{{func_name}},
          py::return_value_policy::move,
          "{{desc}}\\n\\n"
          "Returns:\\n"
          "    Scalar tensor with dot product");
""",
        test_code="""
def test_dot():
    n = 1024
    X = torch.randn(n, device='cuda', dtype=torch.float32)
    Y = torch.randn(n, device='cuda', dtype=torch.float32)
    
    ref = torch.dot(X, Y)
    result = module.sdot(X, Y)
    
    assert torch.allclose(result, ref, rtol=1e-4), "DOT test failed"
    print("✓ cublasSdot test passed")
"""
    ),
    
    'GEMM_HALF': Template(
        cpp_wrapper="""
// ============================================================================
// GEMM_HALF: Half-precision (FP16) matrix multiplication
// ============================================================================
void {{func_name}}(
    torch::Tensor C,
    double alpha,
    torch::Tensor A,
    torch::Tensor B,
    double beta
) {
    TORCH_CHECK(C.device().is_cuda(), "C must be CUDA tensor");
    TORCH_CHECK(A.device().is_cuda(), "A must be CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be CUDA tensor");
    
    TORCH_CHECK(C.scalar_type() == torch::kHalf, "C must be float16");
    TORCH_CHECK(A.scalar_type() == torch::kHalf, "A must be float16");
    TORCH_CHECK(B.scalar_type() == torch::kHalf, "B must be float16");
    
    int M = A.size(0);
    int N = B.size(1);
    int K = A.size(1);
    
    TORCH_CHECK(C.size(0) == M, "C.rows must match A.rows");
    TORCH_CHECK(C.size(1) == N, "C.cols must match B.cols");
    
    __half h_alpha = __float2half_rn(static_cast<float>(alpha));
    __half h_beta = __float2half_rn(static_cast<float>(beta));
    
    cublasHgemm(
        get_handle(),
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &h_beta,
        reinterpret_cast<const __half*>(B.data_ptr<at::Half>()), N,
        reinterpret_cast<const __half*>(A.data_ptr<at::Half>()), K,
        &h_alpha,
        reinterpret_cast<__half*>(C.data_ptr<at::Half>()), N
    );
}
""",
        pybind11_binding="""
    m.def("{{func_name}}", &{{func_name}},
          "{{desc}}\\n\\n"
          "Half-precision (FP16) matrix multiplication.");
""",
        test_code="""
def test_gemm_half():
    M, K, N = 64, 32, 64
    A = torch.randn(M, K, device='cuda', dtype=torch.float16)
    B = torch.randn(K, N, device='cuda', dtype=torch.float16)
    C_torch = torch.randn(M, N, device='cuda', dtype=torch.float16)
    C_ref = C_torch.clone()
    
    alpha, beta = 2.0, 0.5
    # Use float32 for reference computation
    A_f = A.float()
    B_f = B.float()
    C_ref_f = C_ref.float()
    torch.addmm(C_ref_f, A_f, B_f, beta=beta, alpha=alpha)
    C_ref = C_ref_f.half()
    
    C_cublas = C_torch.clone()
    module.hgemm(C_cublas, alpha, A, B, beta)
    
    assert torch.allclose(C_cublas, C_ref, rtol=1e-2, atol=1e-2), "HGEMM test failed"
    print("✓ cublasHgemm test passed")
"""
    ),
}


class CuBLASBatchBuilder:
    """
    cuBLAS批量构建器
    
    使用方法:
        builder = CuBLASBatchBuilder()
        builder.register_operator('cublasSgemm', 'GEMM_FLOAT')
        builder.register_operator('cublasSaxpy', 'AXPY_FLOAT')
        module = builder.build()
    """
    
    def __init__(self, module_name: str = "cublas_baseline"):
        self.module_name = module_name
        self.operators: List[CuBLASOperator] = []
        self.include_dirs = [
            '/usr/local/cuda-12.4/targets/x86_64-linux/include',
        ]
        self.extra_cflags = []
        self.extra_cuda_cflags = []
        self.extra_ldflags = ['-lcublas']
        
    def register_operator(
        self,
        name: str,
        template: str,
        description: str = "",
        dtype: str = "float",
        cblas_func: str = ""
    ) -> 'CuBLASBatchBuilder':
        """注册一个cuBLAS算子"""
        if template not in TEMPLATES:
            raise ValueError(f"Unknown template: {template}. Available: {list(TEMPLATES.keys())}")
        
        op = CuBLASOperator(
            name=name,
            template=template,
            description=description or f"cuBLAS {name} operation",
            dtype=dtype,
            cblas_func=cblas_func or name
        )
        self.operators.append(op)
        return self
    
    def generate_cpp_source(self) -> str:
        """生成完整的C++源代码"""
        lines = []
        
        # 1. 头文件
        lines.append("// Auto-generated cuBLAS Python bindings")
        lines.append(f"// Module: {self.module_name}")
        lines.append(f"// Operators: {len(self.operators)}")
        lines.append("")
        lines.append("#include <torch/extension.h>")
        lines.append("#include <cublas_v2.h>")
        lines.append("#include <iostream>")
        lines.append("")
        
        # 2. 单例Handle
        lines.append("// ============================================================================")
        lines.append("// Singleton cuBLAS Handle")
        lines.append("// ============================================================================")
        lines.append("static cublasHandle_t get_handle() {")
        lines.append("    static cublasHandle_t handle = nullptr;")
        lines.append("    if (handle == nullptr) {")
        lines.append("        cublasCreate(&handle);")
        lines.append("    }")
        lines.append("    return handle;")
        lines.append("}")
        lines.append("")
        
        # 3. 操作符实现
        lines.append("// ============================================================================")
        lines.append("// Operator Implementations")
        lines.append("// ============================================================================")
        lines.append("")
        
        for op in self.operators:
            template = TEMPLATES[op.template]
            # 使用唯一函数名避免与cuBLAS原生函数冲突
            func_name = f"kaldi_{op.name.replace('cublas', '').lower()}"
            op._python_name = op.name  # 保存原始名称用于pybind
            op._func_name = func_name  # 保存C++函数名
            cpp_code = template.cpp_wrapper.replace("{{func_name}}", func_name)
            cpp_code = cpp_code.replace("{{desc}}", op.description)
            lines.append(cpp_code)
            lines.append("")
        
        # 4. pybind11模块定义
        lines.append("// ============================================================================")
        lines.append("// Pybind11 Module Definition")
        lines.append("// ============================================================================")
        lines.append("")
        lines.append("PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {")
        lines.append(f'    m.doc() = "cuBLAS Baseline - {self.module_name}";')
        lines.append("")
        lines.append("    // cuBLAS version info")
        lines.append("    int version;")
        lines.append("    cublasGetVersion(get_handle(), &version);")
        lines.append(f'    m.attr("CUBLAS_VERSION") = std::to_string(version);')
        lines.append("")
        lines.append("    // Operator bindings")
        lines.append("")
        
        for op in self.operators:
            template = TEMPLATES[op.template]
            # 使用唯一的C++函数名
            func_name = op._func_name
            binding = template.pybind11_binding.replace("{{func_name}}", func_name)
            binding = binding.replace("{{desc}}", op.description)
            # 使用原始名称作为Python接口名（如sgemm, saxpy）
            python_name = op.name.replace('cublas', '').lower()
            binding = binding.replace('m.def("{{func_name}}"', f'm.def("{python_name}"')
            # 缩进pybind11代码
            lines.append("    " + "\n    ".join(binding.split("\n")))
        
        lines.append("}")
        lines.append("")
        
        return "\n".join(lines)
    
    def generate_test_script(self) -> str:
        """生成测试脚本"""
        lines = []
        lines.append("#!/usr/bin/env python3")
        lines.append('"""Auto-generated test script for cuBLAS baseline"""')
        lines.append("")
        lines.append("import torch")
        lines.append("import sys")
        lines.append("")
        lines.append("def run_all_tests(module):")
        lines.append('    """Run all operator tests"""')
        lines.append('    print("=" * 80)')
        lines.append('    print("CuBLAS Baseline - Running Tests")')
        lines.append('    print("=" * 80)')
        lines.append("")
        
        for op in self.operators:
            template = TEMPLATES[op.template]
            if template.test_code:
                test = template.test_code
                # 缩进
                lines.append("    " + "\n    ".join(test.split("\n")))
                lines.append("")
        
        lines.append('    print("=" * 80)')
        lines.append('    print("All tests passed!")')
        lines.append('    print("=" * 80)')
        lines.append("")
        lines.append("if __name__ == '__main__':")
        lines.append("    from cublas_baseline import *")
        lines.append("    run_all_tests(None)")
        lines.append("")
        
        return "\n".join(lines)
    
    def build(self, verbose: bool = False):
        """构建并编译模块"""
        from torch.utils.cpp_extension import load_inline
        
        cpp_source = self.generate_cpp_source()
        
        # 保存生成的代码
        cache_path = os.path.join(CACHE_DIR, "cublas_baseline")
        os.makedirs(cache_path, exist_ok=True)
        
        cpp_path = os.path.join(cache_path, "generated_wrapper.cpp")
        with open(cpp_path, 'w') as f:
            f.write(cpp_source)
        
        print(f"Generated C++ source: {cpp_path}")
        print(f"Source size: {len(cpp_source)} bytes")
        print(f"Operators: {len(self.operators)}")
        
        # 编译
        print("\nCompiling with load_inline...")
        
        try:
            module = load_inline(
                name=self.module_name,
                cpp_sources=cpp_source,
                extra_cflags=['-O3', '-std=c++17'],
                extra_ldflags=self.extra_ldflags,
                extra_include_paths=self.include_dirs,
                verbose=verbose
            )
            
            print(f"\n✓ Compilation successful!")
            print(f"Module: {module}")
            
            # 保存模块信息
            info_path = os.path.join(cache_path, "module_info.json")
            with open(info_path, 'w') as f:
                json.dump({
                    "module_name": self.module_name,
                    "operators": [op.name for op in self.operators],
                    "cpp_source": cpp_path,
                }, f, indent=2)
            
            return module
            
        except Exception as e:
            print(f"\n✗ Compilation failed: {e}")
            raise


# 预定义Registry - 所有cuBLAS算子
CUBLAS_OPERATORS_REGISTRY = [
    # GEMM系列 - 矩阵乘法
    {'name': 'cublasSgemm', 'template': 'GEMM_FLOAT', 'description': 'Single-precision matrix multiplication (C = alpha * A * B + beta * C)'},
    {'name': 'cublasDgemm', 'template': 'GEMM_DOUBLE', 'description': 'Double-precision matrix multiplication'},
    {'name': 'cublasHgemm', 'template': 'GEMM_HALF', 'description': 'Half-precision (FP16) matrix multiplication'},
    
    # AXPY系列 - 向量加法
    {'name': 'cublasSaxpy', 'template': 'AXPY_FLOAT', 'description': 'Single-precision vector addition (Y = alpha * X + Y)'},
    
    # SCAL系列 - 向量缩放
    {'name': 'cublasSscal', 'template': 'SCAL_FLOAT', 'description': 'Single-precision vector scaling (X = alpha * X)'},
    
    # DOT系列 - 点积
    {'name': 'cublasSdot', 'template': 'DOT_FLOAT', 'description': 'Single-precision dot product'},
]


def build_default_baseline():
    """构建默认的cuBLAS baseline"""
    print("=" * 80)
    print("BUILDING CUBLAS BASELINE")
    print("=" * 80)
    
    builder = CuBLASBatchBuilder(module_name="cublas_baseline")
    
    # 注册所有算子
    for op_def in CUBLAS_OPERATORS_REGISTRY:
        builder.register_operator(
            name=op_def['name'],
            template=op_def['template'],
            description=op_def['description']
        )
    
    # 构建
    module = builder.build()
    
    return module


if __name__ == '__main__':
    module = build_default_baseline()
