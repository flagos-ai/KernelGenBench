#!/usr/bin/env python3
"""
阶段1: 从CUDA仓库提取所有算子信息

功能：
1. 解析 cu-kernels-ansi.h 文件，提取所有 cudaF_* 和 cudaD_* 函数声明
2. 分析函数签名，提取参数信息
3. 将 float/double 版本的函数合并为统一的算子定义
4. 输出 JSON 格式的算子列表

输出格式：
{
  "add_row_sum_mat": {
    "description": "从cu-kernels.cu中提取的注释或默认描述",
    "float_signature": "void cudaF_add_row_sum_mat(...)",
    "double_signature": "void cudaD_add_row_sum_mat(...)",
    "parameters": [
      {"name": "result", "type": "pointer", "base_type": "Real", "is_const": false, "is_output": true},
      {"name": "mat", "type": "pointer", "base_type": "Real", "is_const": true, "is_output": false},
      {"name": "scratch", "type": "void_pointer", "is_const": false, "is_output": false},
      {"name": "d", "type": "MatrixDim", "is_const": true, "is_output": false},
      {"name": "alpha", "type": "scalar", "base_type": "Real", "is_const": true, "is_output": false},
      {"name": "beta", "type": "scalar", "base_type": "Real", "is_const": true, "is_output": false}
    ],
    "has_grid_block": false
  }
}

使用方法：
python script/extract_cuda_kernels.py \
    --input kaldi/src/cudamatrix/cu-kernels-ansi.h \
    --output csrc/kaldi_kernels.json \
    --cu-impl kaldi/src/cudamatrix/cu-kernels.cu  # 可选，用于提取注释
"""

import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict


class CUDAKernelExtractor:
    """从CUDA头文件中提取kernel信息"""
    
    # 类型映射
    TYPE_PATTERNS = {
        'pointer': r'(float|double|int32_cuda|MatrixIndexT_cuda)\s*\*',
        'void_pointer': r'void\s*\*',
        'MatrixDim': r'MatrixDim',
        'dim3': r'dim3',
        'scalar': r'(float|double|int|size_t|bool)',
        'int': r'int',
        'size_t': r'size_t',
        'bool': r'bool',
    }
    
    def __init__(self, header_file: Path, cu_impl_file: Optional[Path] = None):
        self.header_file = header_file
        self.cu_impl_file = cu_impl_file
        self.kernels = {}
        self.descriptions = {}
        
    def extract_descriptions(self):
        """从.cu文件中提取函数描述（可选）"""
        if not self.cu_impl_file or not self.cu_impl_file.exists():
            return
        
        try:
            content = self.cu_impl_file.read_text(encoding='utf-8', errors='ignore')
            # 简单的启发式：查找函数定义前的注释
            # 这部分可以根据实际代码风格调整
            pattern = r'/\*\*(.*?)\*/\s*void\s+(cudaF_|cudaD_)(\w+)'
            matches = re.finditer(pattern, content, re.DOTALL)
            for match in matches:
                comment = match.group(1).strip()
                kernel_name = match.group(3)
                # 清理注释
                comment = re.sub(r'^\s*\*\s*', '', comment, flags=re.MULTILINE)
                self.descriptions[kernel_name] = comment.strip()
        except Exception as e:
            print(f"Warning: Failed to extract descriptions from {self.cu_impl_file}: {e}")
    
    def parse_parameter(self, param_str: str) -> Dict[str, Any]:
        """解析单个参数"""
        param_str = param_str.strip()
        
        param_info = {
            "name": "",
            "type": "unknown",
            "base_type": None,
            "is_const": "const" in param_str,
            "is_output": False,
            "raw": param_str
        }
        
        # 去掉const关键字来分析类型
        type_str = param_str.replace('const', '').strip()
        
        # void*
        if re.search(r'void\s*\*', type_str):
            param_info["type"] = "void_pointer"
            # 提取参数名
            name_match = re.search(r'\*\s*(\w+)$', param_str)
            if name_match:
                param_info["name"] = name_match.group(1)
                
        # 指针类型
        elif '*' in type_str:
            param_info["type"] = "pointer"
            # 提取基础类型
            base_match = re.search(r'(float|double|int32_cuda|MatrixIndexT_cuda|MatrixElement|Int32Pair)', type_str)
            if base_match:
                base_type = base_match.group(1)
                if base_type in ['float', 'double']:
                    param_info["base_type"] = "Real"
                else:
                    param_info["base_type"] = base_type
            # 提取参数名
            name_match = re.search(r'\*\s*(\w+)$', param_str)
            if name_match:
                param_info["name"] = name_match.group(1)
            # 判断是否是输出参数（非const指针通常是输出）
            if not param_info["is_const"] and base_match:
                param_info["is_output"] = True
                
        # MatrixDim
        elif 'MatrixDim' in type_str:
            param_info["type"] = "MatrixDim"
            name_match = re.search(r'MatrixDim\s+(\w+)$', param_str)
            if name_match:
                param_info["name"] = name_match.group(1)
                
        # dim3
        elif 'dim3' in type_str:
            param_info["type"] = "dim3"
            name_match = re.search(r'dim3\s+(\w+)$', param_str)
            if name_match:
                param_info["name"] = name_match.group(1)
                
        # 标量类型
        else:
            scalar_match = re.search(r'(float|double|int|size_t|bool)\s+(\w+)$', param_str)
            if scalar_match:
                base_type = scalar_match.group(1)
                param_info["type"] = "scalar"
                if base_type in ['float', 'double']:
                    param_info["base_type"] = "Real"
                else:
                    param_info["base_type"] = base_type
                param_info["name"] = scalar_match.group(2)
        
        return param_info
    
    def parse_function_signature(self, signature: str) -> Tuple[str, List[Dict[str, Any]], bool]:
        """解析函数签名
        
        Returns:
            (kernel_name, parameters, has_grid_block)
        """
        # 提取函数名
        func_match = re.search(r'cuda[FD]_(\w+)\s*\(', signature)
        if not func_match:
            return None, [], False
        
        kernel_name = func_match.group(1)
        
        # 提取参数列表
        params_match = re.search(r'\((.*?)\);?$', signature, re.DOTALL)
        if not params_match:
            return kernel_name, [], False
        
        params_str = params_match.group(1)
        
        # 分割参数（注意处理嵌套的<>）
        params = []
        depth = 0
        current_param = ""
        for char in params_str:
            if char == '<':
                depth += 1
            elif char == '>':
                depth -= 1
            elif char == ',' and depth == 0:
                if current_param.strip():
                    params.append(current_param.strip())
                current_param = ""
                continue
            current_param += char
        
        if current_param.strip():
            params.append(current_param.strip())
        
        # 解析每个参数
        parsed_params = []
        has_grid_block = False
        
        for param in params:
            param_info = self.parse_parameter(param)
            parsed_params.append(param_info)
            
            # 检查是否有grid/block参数
            if param_info["type"] in ["dim3", "int", "size_t"] and \
               param_info["name"] in ["Gr", "Bl"]:
                has_grid_block = True
        
        return kernel_name, parsed_params, has_grid_block
    
    def extract_kernels(self):
        """提取所有kernel定义"""
        print(f"Reading CUDA header file: {self.header_file}")
        
        if not self.header_file.exists():
            raise FileNotFoundError(f"Header file not found: {self.header_file}")
        
        # 首先提取描述
        self.extract_descriptions()
        
        content = self.header_file.read_text(encoding='utf-8', errors='ignore')
        
        # 提取所有函数声明（支持多行）
        # 匹配 void cudaF_xxx(...) 或 void cudaD_xxx(...)
        pattern = r'void\s+(cudaF_|cudaD_)(\w+)\s*\([^)]*(?:\([^)]*\)[^)]*)*\)\s*;'
        matches = re.finditer(pattern, content, re.DOTALL | re.MULTILINE)
        
        # 按kernel名称分组
        kernel_groups = defaultdict(lambda: {"float": None, "double": None})
        
        for match in matches:
            full_signature = match.group(0)
            prefix = match.group(1)
            kernel_name = match.group(2)
            
            # 解析签名
            parsed_name, params, has_grid_block = self.parse_function_signature(full_signature)
            
            if parsed_name:
                variant = "float" if prefix == "cudaF_" else "double"
                kernel_groups[kernel_name][variant] = {
                    "signature": full_signature.strip(),
                    "parameters": params,
                    "has_grid_block": has_grid_block
                }
        
        # 合并float/double版本
        for kernel_name, variants in kernel_groups.items():
            # 优先使用float版本的参数（通常两者一致）
            params = None
            has_grid_block = False
            
            if variants["float"]:
                params = variants["float"]["parameters"]
                has_grid_block = variants["float"]["has_grid_block"]
            elif variants["double"]:
                params = variants["double"]["parameters"]
                has_grid_block = variants["double"]["has_grid_block"]
            
            if params is not None:
                self.kernels[kernel_name] = {
                    "description": self.descriptions.get(kernel_name, f"K1 CUDA kernel: {kernel_name}"),
                    "float_signature": variants["float"]["signature"] if variants["float"] else None,
                    "double_signature": variants["double"]["signature"] if variants["double"] else None,
                    "parameters": params,
                    "has_grid_block": has_grid_block
                }
        
        print(f"Extracted {len(self.kernels)} unique kernels")
        return self.kernels
    
    def save_to_json(self, output_file: Path):
        """保存到JSON文件"""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with output_file.open('w', encoding='utf-8') as f:
            json.dump(self.kernels, f, indent=2, ensure_ascii=False)
        
        print(f"Saved kernel definitions to: {output_file}")
        
        # 打印统计信息
        total = len(self.kernels)
        with_float = sum(1 for k in self.kernels.values() if k["float_signature"])
        with_double = sum(1 for k in self.kernels.values() if k["double_signature"])
        with_grid_block = sum(1 for k in self.kernels.values() if k["has_grid_block"])
        
        print(f"\nStatistics:")
        print(f"  Total kernels: {total}")
        print(f"  With float variant: {with_float}")
        print(f"  With double variant: {with_double}")
        print(f"  With grid/block params: {with_grid_block}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract CUDA kernel definitions from header file"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("kaldi/src/cudamatrix/cu-kernels-ansi.h"),
        help="Input CUDA header file (cu-kernels-ansi.h)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("csrc/kaldi_kernels.json"),
        help="Output JSON file"
    )
    parser.add_argument(
        "--cu-impl",
        type=Path,
        default=Path("kaldi/src/cudamatrix/cu-kernels.cu"),
        help="Optional: CUDA implementation file for extracting descriptions"
    )
    
    args = parser.parse_args()
    
    # 转换为绝对路径（基于项目根目录）
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    input_file = project_root / args.input
    output_file = project_root / args.output
    cu_impl_file = project_root / args.cu_impl if args.cu_impl else None
    
    # 执行提取
    extractor = CUDAKernelExtractor(input_file, cu_impl_file)
    extractor.extract_kernels()
    extractor.save_to_json(output_file)
    
    print("\n✓ Stage 1 completed successfully!")


if __name__ == "__main__":
    main()
