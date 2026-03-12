#!/usr/bin/env python3
"""
阶段2: 生成C++ PyTorch binding代码

功能：
1. 读取阶段1生成的JSON文件
2. 为每个kernel生成对应的C++ wrapper函数
3. 生成TORCH_LIBRARY注册代码
4. 生成完整的kaldi_ops.cpp文件
5. 生成编译配置文件setup.py

类型转换策略：
- Real* (output) → at::Tensor&  (in-place modification)
- const Real* → const at::Tensor&
- void* → at::Tensor& (scratch space)
- MatrixDim → 从Tensor提取: {rows, cols, stride}
- dim3 Gr, Bl → 忽略（PyTorch自动管理）
- scalar (Real) → double/float (根据输入tensor dtype自动dispatch)

使用方法：
python script/generate_binding_code.py \
    --input csrc/kaldi_kernels.json \
    --output csrc/kaldi_ops.cpp \
    --cuda-src kaldi/src/cudamatrix \
    --namespace kaldi
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any


class BindingCodeGenerator:
    """生成PyTorch C++ binding代码"""
    
    def __init__(self, kernels: Dict[str, Any], cuda_src_dir: Path, namespace: str = "kaldi"):
        self.kernels = kernels
        self.cuda_src_dir = cuda_src_dir
        self.namespace = namespace
        self.generated_functions = []
        
    def generate_matrixdim_struct(self) -> str:
        """生成MatrixDim结构体定义（从tensor提取）"""
        return """
// MatrixDim structure (from Kaldi)
struct MatrixDim {
  int32_t rows;
  int32_t cols;
  int32_t stride;
};

// Helper: Extract MatrixDim from Tensor
inline MatrixDim get_matrix_dim(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.dim() == 2, "Expected 2D tensor for MatrixDim");
  MatrixDim dim;
  dim.rows = tensor.size(0);
  dim.cols = tensor.size(1);
  dim.stride = tensor.stride(0);
  return dim;
}
"""
    
    def generate_type_aliases(self) -> str:
        """生成类型别名"""
        return """
// Type aliases for compatibility
using MatrixIndexT_cuda = int32_t;
using int32_cuda = int32_t;

// Helper structures
template<typename Real>
struct MatrixElement {
  int32_t row;
  int32_t column;
  Real weight;
};

struct Int32Pair {
  int32_t first;
  int32_t second;
};
"""
    
    def cpp_type_from_param(self, param: Dict[str, Any], for_signature: bool = True) -> str:
        """将参数转换为C++类型
        
        Args:
            param: 参数信息字典
            for_signature: 是否用于函数签名（vs 函数调用）
        """
        ptype = param["type"]
        is_const = param.get("is_const", False)
        is_output = param.get("is_output", False)
        
        if ptype == "pointer":
            base_type = param.get("base_type", "Real")
            if base_type == "Real":
                # Tensor类型
                if for_signature:
                    if is_output or not is_const:
                        return "at::Tensor&"
                    else:
                        return "const at::Tensor&"
                else:
                    # 调用时需要data_ptr
                    return f"tensor_{param['name']}.data_ptr<scalar_t>()"
            else:
                # 其他指针类型 (int32_cuda*, MatrixIndexT_cuda*, etc.)
                if for_signature:
                    if is_output or not is_const:
                        return "at::Tensor&"
                    else:
                        return "const at::Tensor&"
                else:
                    return f"tensor_{param['name']}.data_ptr<{base_type}>()"
                    
        elif ptype == "void_pointer":
            if for_signature:
                return "at::Tensor&"
            else:
                return f"tensor_{param['name']}.data_ptr<void>()"
                
        elif ptype == "MatrixDim":
            if for_signature:
                return "const at::Tensor&"
            else:
                return f"get_matrix_dim(tensor_{param['name']})"
                
        elif ptype == "dim3":
            # dim3参数通常是grid/block，PyTorch自动管理，忽略
            return None
            
        elif ptype == "scalar":
            base_type = param.get("base_type", "Real")
            if base_type == "Real":
                return "scalar_t"
            elif base_type == "int":
                return "int"
            elif base_type == "bool":
                return "bool"
            else:
                return base_type
        
        return "void*"  # fallback
    
    def generate_wrapper_function(self, kernel_name: str, kernel_info: Dict[str, Any]) -> str:
        """为单个kernel生成wrapper函数"""
        params = kernel_info["parameters"]
        has_grid_block = kernel_info.get("has_grid_block", False)
        
        # 过滤掉grid/block参数
        filtered_params = [p for p in params if p["type"] not in ["dim3", "int", "size_t"] or p["name"] not in ["Gr", "Bl"]]
        
        # 生成函数签名
        sig_params = []
        for param in filtered_params:
            cpp_type = self.cpp_type_from_param(param, for_signature=True)
            if cpp_type:
                # 为了避免命名冲突，tensor参数用tensor_前缀
                if cpp_type in ["at::Tensor&", "const at::Tensor&"]:
                    sig_params.append(f"{cpp_type} tensor_{param['name']}")
                else:
                    sig_params.append(f"{cpp_type} {param['name']}")
        
        sig_str = ", ".join(sig_params)
        
        # 生成函数体
        func_body = f"""
// Wrapper for {kernel_name}
// {kernel_info.get('description', '')}
void {kernel_name}({sig_str}) {{
  // Dispatch based on tensor dtype
  AT_DISPATCH_FLOATING_TYPES(tensor_{filtered_params[0]['name']}.scalar_type(), "{kernel_name}", [&] {{
"""
        
        # 生成kernel调用参数
        call_params = []
        for param in params:
            if param["type"] in ["dim3"] or param["name"] in ["Gr", "Bl"]:
                # 跳过grid/block参数，或者根据需要自动计算
                if param["name"] == "Gr":
                    # 简单策略：grid = (num_elements + 255) / 256
                    call_params.append("/* grid calculated automatically */")
                    continue
                elif param["name"] == "Bl":
                    call_params.append("/* block calculated automatically */")
                    continue
            
            cpp_expr = self.cpp_type_from_param(param, for_signature=False)
            if cpp_expr and "/*" not in cpp_expr:
                call_params.append(cpp_expr)
        
        # 调用实际的CUDA kernel
        # 需要根据scalar_t选择cudaF_或cudaD_
        func_body += f"""    
    // Call the actual CUDA kernel
    if constexpr (std::is_same<scalar_t, float>::value) {{
      cudaF_{kernel_name}({", ".join(call_params)});
    }} else if constexpr (std::is_same<scalar_t, double>::value) {{
      cudaD_{kernel_name}({", ".join(call_params)});
    }}
  }});
  
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}}
"""
        
        return func_body
    
    def generate_torch_library_registration(self) -> str:
        """生成TORCH_LIBRARY注册代码"""
        code = f"""
TORCH_LIBRARY({self.namespace}, m) {{
"""
        
        for kernel_name, kernel_info in self.kernels.items():
            params = kernel_info["parameters"]
            filtered_params = [p for p in params if p["type"] not in ["dim3", "int", "size_t"] or p["name"] not in ["Gr", "Bl"]]
            
            # 生成schema字符串
            schema_params = []
            for param in filtered_params:
                cpp_type = self.cpp_type_from_param(param, for_signature=True)
                if cpp_type:
                    # PyTorch schema类型
                    if "Tensor&" in cpp_type and "const" not in cpp_type:
                        schema_params.append(f"Tensor(a!) {param['name']}")
                    elif "Tensor&" in cpp_type:
                        schema_params.append(f"Tensor {param['name']}")
                    elif cpp_type == "scalar_t":
                        schema_params.append(f"Scalar {param['name']}")
                    elif cpp_type == "int":
                        schema_params.append(f"int {param['name']}")
                    elif cpp_type == "bool":
                        schema_params.append(f"bool {param['name']}")
            
            schema_str = ", ".join(schema_params)
            
            code += f'  m.def("{kernel_name}({schema_str}) -> ()", {kernel_name});\n'
        
        code += "}\n"
        return code
    
    def generate_full_cpp_file(self) -> str:
        """生成完整的C++文件"""
        code = f"""// Auto-generated PyTorch C++ extension for Kaldi CUDA kernels
// Generated by FlagBench binding generator
// DO NOT EDIT MANUALLY

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// Include Kaldi CUDA headers
extern "C" {{
#include "{self.cuda_src_dir / 'cu-matrixdim.h'}"
#include "{self.cuda_src_dir / 'cu-kernels-ansi.h'}"
}}

{self.generate_matrixdim_struct()}

{self.generate_type_aliases()}

// ==================== Wrapper Functions ====================
"""
        
        # 生成所有wrapper函数
        for kernel_name, kernel_info in self.kernels.items():
            code += self.generate_wrapper_function(kernel_name, kernel_info)
            code += "\n"
        
        code += "\n// ==================== PyTorch Library Registration ====================\n"
        code += self.generate_torch_library_registration()
        
        return code
    
    def save_cpp_file(self, output_path: Path):
        """保存C++文件"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cpp_code = self.generate_full_cpp_file()
        output_path.write_text(cpp_code, encoding='utf-8')
        
        print(f"Generated C++ binding code: {output_path}")
        print(f"  Total kernels: {len(self.kernels)}")
        print(f"  Lines of code: {len(cpp_code.splitlines())}")
    
    def generate_setup_py(self, output_path: Path, cpp_file: Path):
        """生成setup.py编译配置"""
        setup_code = f'''# Auto-generated setup.py for Kaldi CUDA bindings
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Kaldi source directory
KALDI_SRC = os.path.join(os.path.dirname(__file__), "../kaldi/src")

setup(
    name='{self.namespace}_ops',
    ext_modules=[
        CUDAExtension(
            name='{self.namespace}_ops',
            sources=['{cpp_file.name}'],
            include_dirs=[
                KALDI_SRC,
                os.path.join(KALDI_SRC, "cudamatrix"),
            ],
            library_dirs=[
                os.path.join(KALDI_SRC, "cudamatrix"),
            ],
            libraries=['kaldi-cudamatrix'],  # Link against libkaldi-cudamatrix
            extra_compile_args={{
                'cxx': ['-O3', '-std=c++14'],
                'nvcc': ['-O3', '--expt-relaxed-constexpr'],
            }},
        ),
    ],
    cmdclass={{
        'build_ext': BuildExtension
    }}
)
'''
        output_path.write_text(setup_code, encoding='utf-8')
        print(f"Generated setup.py: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate PyTorch C++ binding code from kernel JSON"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("csrc/kaldi_kernels.json"),
        help="Input JSON file from stage 1"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("csrc/kaldi_ops.cpp"),
        help="Output C++ file"
    )
    parser.add_argument(
        "--cuda-src",
        type=Path,
        default=Path("kaldi/src/cudamatrix"),
        help="Kaldi CUDA source directory"
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="kaldi",
        help="PyTorch ops namespace (torch.ops.<namespace>)"
    )
    parser.add_argument(
        "--setup-output",
        type=Path,
        default=Path("csrc/setup.py"),
        help="Output setup.py file"
    )
    
    args = parser.parse_args()
    
    # 转换为绝对路径
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    input_file = project_root / args.input
    output_file = project_root / args.output
    cuda_src_dir = project_root / args.cuda_src
    setup_output = project_root / args.setup_output
    
    # 读取JSON
    print(f"Reading kernel definitions from: {input_file}")
    with input_file.open('r', encoding='utf-8') as f:
        kernels = json.load(f)
    
    print(f"Loaded {len(kernels)} kernels")
    
    # 生成binding代码
    generator = BindingCodeGenerator(kernels, cuda_src_dir, args.namespace)
    generator.save_cpp_file(output_file)
    generator.generate_setup_py(setup_output, output_file)
    
    print("\n✓ Stage 2 completed successfully!")


if __name__ == "__main__":
    main()
