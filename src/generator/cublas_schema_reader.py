#!/usr/bin/env python3
"""
cuBLAS Schema 读取工具

用于从 cublas_ops.json 中读取 cuBLAS 函数签名信息
"""

import json
import os

# Schema 文件路径（使用绝对路径）
SCHEMA_PATH = "/share/project/zpy/flagbench/script/cublas_cupy/cublas_ops.json"


def load_cublas_schema():
    """加载 cuBLAS schema"""
    try:
        with open(SCHEMA_PATH, 'r') as f:
            schema = json.load(f)
        print(f"✅ 成功加载 cuBLAS schema: {schema['total_functions']} 个函数")
        return schema
    except Exception as e:
        print(f"❌ 加载 cuBLAS schema 失败: {e}")
        print(f"📂 Schema 路径: {SCHEMA_PATH}")
        return None


def find_cublas_function(operation_name):
    """
    查找 cuBLAS 函数
    
    Args:
        operation_name: 操作名称（如 'gemm', 'axpy'）
    
    Returns:
        匹配的函数列表（可能有多个 dtype 变体）
    """
    schema = load_cublas_schema()
    if schema is None:
        return []
    
    matching_funcs = [f for f in schema['functions'] if f['operation'] == operation_name.lower()]
    
    if matching_funcs:
        print(f"✅ 找到 {len(matching_funcs)} 个 {operation_name} 函数")
        for func in matching_funcs:
            print(f"   - {func['name']} ({func['dtype']})")
    else:
        print(f"❌ 未找到操作: {operation_name}")
    
    return matching_funcs


def print_function_signature(func_info):
    """打印函数签名"""
    print(f"\n{'='*60}")
    print(f"函数名: {func_info['name']}")
    print(f"操作:   {func_info['operation']}")
    print(f"类型:   {func_info['dtype']}")
    print(f"{'='*60}")
    print(f"参数 ({len(func_info['args'])} 个):")
    
    for arg in func_info['args']:
        role_symbol = {
            'context': '🔧',
            'input': '📥',
            'output': '📤',
            'inout': '🔄',
            'scalar': '⚙️',
            'value': '🔢'
        }.get(arg['role'], '❓')
        
        print(f"  {role_symbol} {arg['name']:15s} ({arg['role']:8s}): {arg['type']}")


if __name__ == "__main__":
    # 测试：查找 sgemm 函数
    print("测试：查找 cuBLAS SGEMM 函数\n")
    functions = find_cublas_function('gemm')
    
    if functions:
        for func in functions:
            if func['dtype'] == 'float32':
                print_function_signature(func)
