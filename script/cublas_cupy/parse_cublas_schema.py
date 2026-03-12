#!/usr/bin/env python3
"""
cuBLAS Schema Parser
从 cublas_v2.h 头文件中提取所有函数签名，生成结构化的 JSON schema
"""

import re
import json
from typing import List, Dict, Any


# ============================================================================
# 配置：需要提取的函数类别
# ============================================================================
WANTED_CATEGORIES = [
    # Level 1 BLAS (向量操作)
    'axpy', 'scal', 'copy', 'swap', 'dot', 'nrm2', 'asum', 'iamax', 'iamin',
    'rot', 'rotg', 'rotm', 'rotmg',
    
    # Level 2 BLAS (矩阵-向量操作)
    'gemv', 'gbmv', 'symv', 'sbmv', 'spmv', 'trmv', 'tbmv', 'tpmv',
    'trsv', 'tbsv', 'tpsv', 'ger', 'syr', 'spr', 'syr2', 'spr2',
    'hemv', 'hbmv', 'hpmv', 'her', 'hpr', 'her2', 'hpr2',
    
    # Level 3 BLAS (矩阵-矩阵操作)
    'gemm', 'symm', 'syrk', 'syr2k', 'trmm', 'trsm',
    'hemm', 'herk', 'her2k',
    
    # Batched operations
    'gemmbatched', 'gemmstridedbatched',
    'geam', 'dgmm', 'getrfbatched', 'getribatched', 'getrsbatched',
]


def infer_dtype(func_name: str) -> str:
    """
    从函数名推断数据类型
    
    cuBLAS 命名规则：cublas<T><operation>
    - S: float (single precision)
    - D: double (double precision) 
    - C: complex float
    - Z: complex double
    - H: half (float16)
    """
    # 提取类型前缀（第7个字符，索引6）
    # 例如：cublasSgemm -> S, cublasDaxpy -> D
    if len(func_name) < 7:
        return 'unknown'
    
    type_char = func_name[6].lower()
    
    dtype_map = {
        's': 'float32',
        'd': 'float64',
        'h': 'float16',
        'c': 'complex64',
        'z': 'complex128',
    }
    
    return dtype_map.get(type_char, 'unknown')


def classify_argument_role(arg_name: str, arg_type: str, func_name: str) -> str:
    """
    判断参数的角色
    
    角色类型：
    - context: cuBLAS handle
    - scalar: 标量参数（alpha, beta 等）
    - input: 只读输入
    - output: 只写输出
    - inout: 输入输出（in-place）
    - value: 普通值参数（维度等）
    """
    arg_name_lower = arg_name.lower()
    
    # Handle
    if 'handle' in arg_name_lower:
        return 'context'
    
    # 维度参数（int, enum 等非指针类型）
    if '*' not in arg_type:
        # 特殊处理：result 参数可能是输出
        if 'result' in arg_name_lower:
            return 'output'
        return 'value'
    
    # 标量参数（通常是 const scalar* 形式）
    # 必须满足：(1) 是指针 (2) 名字是 alpha/beta/c/s (3) 是 const
    # 注意：这里的 'c' 和 's' 是指 rotation 函数里的 c 和 s，不是 gemm 的 C
    is_const = 'const' in arg_type
    if is_const and arg_name_lower in ['alpha', 'beta']:
        return 'scalar'
    
    # rot 函数的 c 和 s 参数（const float* c, const float* s）
    if is_const and arg_name_lower in ['c', 's'] and len(arg_name) == 1:
        # 单字符参数名，且是 const，很可能是 rot 的标量参数
        return 'scalar'
    
    # 指针类型参数
    if is_const:
        return 'input'
    else:
        # 非 const 指针
        # 特殊情况：某些函数的输出参数
        if any(kw in arg_name_lower for kw in ['result', 'info', 'work']):
            return 'output'
        
        # 对于 BLAS 操作，通常非 const 的矩阵/向量是 inout
        # 例如：saxpy 的 y, gemm 的 C
        return 'inout'


def is_wanted_function(func_name: str) -> bool:
    """检查函数是否在需要提取的类别中"""
    func_lower = func_name.lower()
    
    # 排除 64 位版本（我们只需要标准版本）
    if '_64' in func_lower:
        return False
    
    # 检查是否匹配任何需要的类别
    return any(cat in func_lower for cat in WANTED_CATEGORIES)


def parse_cublas_header(header_path: str) -> List[Dict[str, Any]]:
    """
    解析 cuBLAS 头文件，提取函数签名
    
    Args:
        header_path: cublas_v2.h 文件路径
    
    Returns:
        函数签名列表，每个元素包含：
        {
            "name": "cublasSgemm",
            "operation": "gemm",
            "dtype": "float32",
            "args": [
                {"name": "handle", "type": "cublasHandle_t", "role": "context"},
                {"name": "transa", "type": "cublasOperation_t", "role": "value"},
                ...
            ]
        }
    """
    with open(header_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 正则表达式：匹配 cublasStatus_t 开头的函数声明
    # 例如: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemm_v2(...);
    # 需要匹配 CUBLASAPI 前缀
    pattern = re.compile(
        r'(?:CUBLASAPI\s+)?cublasStatus_t\s+(?:CUBLASWINAPI\s+)?(cublas[A-Z][a-zA-Z0-9_]+)\s*\(([^;]+?)\)\s*;',
        re.MULTILINE | re.DOTALL
    )
    
    functions = []
    
    for match in pattern.finditer(content):
        func_name = match.group(1)
        raw_args = match.group(2)
        
        # 过滤：只提取需要的函数
        if not is_wanted_function(func_name):
            continue
        
        # 推断数据类型和操作类型
        dtype = infer_dtype(func_name)
        operation = extract_operation_name(func_name)
        
        # 解析参数列表
        args = parse_arguments(raw_args, func_name)
        
        functions.append({
            "name": func_name,
            "operation": operation,
            "dtype": dtype,
            "args": args
        })
    
    return functions


def extract_operation_name(func_name: str) -> str:
    """
    从函数名提取操作名称
    
    例如：
    - cublasSgemm_v2 -> gemm
    - cublasDaxpy_v2 -> axpy
    - cublasHgemm_v2 -> gemm
    """
    # 移除前缀 "cublas"
    without_prefix = func_name[6:]
    
    # 移除类型字符（S/D/C/Z/H）
    if without_prefix and without_prefix[0] in 'SDCZHsdczh':
        without_prefix = without_prefix[1:]
    
    # 移除 _v2 后缀
    if without_prefix.endswith('_v2'):
        without_prefix = without_prefix[:-3]
    
    # 转为小写
    return without_prefix.lower()


def parse_arguments(raw_args: str, func_name: str) -> List[Dict[str, str]]:
    """
    解析函数参数列表
    
    Args:
        raw_args: 原始参数字符串，例如 "cublasHandle_t handle, int n, const float *x, ..."
        func_name: 函数名（用于角色推断）
    
    Returns:
        参数列表，每个元素包含 name, type, role
    """
    # 预处理：移除换行符和多余空格
    clean_args = ' '.join(raw_args.split())
    
    # 按逗号分割参数
    arg_tokens = split_arguments(clean_args)
    
    args = []
    for token in arg_tokens:
        token = token.strip()
        if not token:
            continue
        
        # 解析类型和名称
        # 例如: "const float *alpha" -> type="const float *", name="alpha"
        arg_match = re.search(r'^(.*?)\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\[[^\]]*\])?)$', token)
        
        if arg_match:
            arg_type = arg_match.group(1).strip()
            arg_name = arg_match.group(2).strip()
            
            # 移除数组后缀（如果有）
            # 例如: "x[10]" -> "x"
            arg_name = re.sub(r'\[.*\]', '', arg_name)
            
            # 判断参数角色
            role = classify_argument_role(arg_name, arg_type, func_name)
            
            args.append({
                "name": arg_name,
                "type": arg_type,
                "role": role
            })
    
    return args


def split_arguments(args_str: str) -> List[str]:
    """
    智能分割参数列表（处理嵌套的逗号）
    
    例如: "int n, void (*func)(int, int), float *x"
    不能简单用 split(',')，因为函数指针里也有逗号
    """
    result = []
    current = []
    depth = 0  # 括号深度
    
    for char in args_str:
        if char == '(':
            depth += 1
            current.append(char)
        elif char == ')':
            depth -= 1
            current.append(char)
        elif char == ',' and depth == 0:
            # 顶层逗号，分割
            result.append(''.join(current))
            current = []
        else:
            current.append(char)
    
    # 添加最后一个参数
    if current:
        result.append(''.join(current))
    
    return result


def save_schema(functions: List[Dict[str, Any]], output_path: str):
    """保存 schema 到 JSON 文件"""
    schema = {
        "version": "1.0",
        "source": "cublas_v2.h",
        "total_functions": len(functions),
        "functions": functions
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Schema saved to: {output_path}")
    print(f"📊 Total functions extracted: {len(functions)}")


def print_summary(functions: List[Dict[str, Any]]):
    """打印统计摘要"""
    print("\n" + "="*60)
    print("📋 cuBLAS Schema Parser - Summary")
    print("="*60)
    
    # 按操作类型分组
    operations = {}
    for func in functions:
        op = func['operation']
        if op not in operations:
            operations[op] = []
        operations[op].append(func['name'])
    
    print(f"\n🔢 Total functions: {len(functions)}")
    print(f"🔧 Operation types: {len(operations)}")
    
    print("\n📊 Functions by operation:")
    for op, funcs in sorted(operations.items()):
        print(f"  {op:20s}: {len(funcs):3d} functions")
    
    # 按数据类型分组
    dtypes = {}
    for func in functions:
        dt = func['dtype']
        dtypes[dt] = dtypes.get(dt, 0) + 1
    
    print("\n📊 Functions by dtype:")
    for dt, count in sorted(dtypes.items()):
        print(f"  {dt:15s}: {count:3d} functions")
    
    print("\n" + "="*60)


def main():
    """主函数"""
    import os
    
    # 路径配置
    header_path = "/usr/local/cuda/include/cublas_api.h"  # 真正的函数声明在这里
    output_dir = "/share/project/zpy/flagbench/script/cublas_cupy"
    output_path = os.path.join(output_dir, "cublas_ops.json")
    
    print("🚀 Starting cuBLAS schema parser...")
    print(f"📂 Header file: {header_path}")
    print(f"💾 Output file: {output_path}")
    
    # 解析头文件
    try:
        functions = parse_cublas_header(header_path)
    except FileNotFoundError:
        print(f"❌ Error: Header file not found: {header_path}")
        return
    except Exception as e:
        print(f"❌ Error parsing header file: {e}")
        raise
    
    # 保存结果
    save_schema(functions, output_path)
    
    # 打印摘要
    print_summary(functions)
    
    # 打印前3个函数的详细信息（用于检查）
    print("\n🔍 Sample functions (first 3):")
    print("="*60)
    for func in functions[:3]:
        print(f"\nFunction: {func['name']}")
        print(f"  Operation: {func['operation']}")
        print(f"  Dtype: {func['dtype']}")
        print(f"  Arguments ({len(func['args'])}):")
        for arg in func['args']:
            print(f"    - {arg['name']:15s} ({arg['role']:8s}): {arg['type']}")
    
    print("\n✅ Done! Please check the output file for correctness.")


if __name__ == "__main__":
    main()
