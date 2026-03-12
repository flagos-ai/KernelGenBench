#!/usr/bin/env python3
"""
示例：使用 generate_prompt_for_cublas 生成 cuBLAS baseline 函数

这个脚本展示如何：
1. 加载 cuBLAS schema
2. 调用 generate_prompt_for_cublas 生成 prompt
3. 使用 LLM 生成 CuPy wrapper 函数
"""

import json
import sys
import os

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from generator.test_func_generator import TestFuncGenerator
from generator.sampler.utils import TestFuncGenerateArgs


def load_cublas_schema(schema_path: str, function_name: str) -> dict:
    """
    从 cublas_ops.json 加载指定函数的 schema
    
    Args:
        schema_path: cublas_ops.json 的路径
        function_name: cuBLAS 函数名，如 "cublasSgemm_v2"
    
    Returns:
        函数的 schema dict
    """
    with open(schema_path, 'r') as f:
        data = json.load(f)
    
    # 查找函数
    for func in data['functions']:
        if func['name'] == function_name:
            return func
    
    raise ValueError(f"Function {function_name} not found in schema")


def generate_cublas_baseline_prompt(
    cublas_function_name: str,
    triton_kernel_name: str,
    schema_path: str = None
):
    """
    生成 cuBLAS baseline 的 prompt
    
    Args:
        cublas_function_name: cuBLAS 函数名，如 "cublasSgemm_v2"
        triton_kernel_name: 要对比的 Triton kernel 名称
        schema_path: cublas_ops.json 路径（默认使用项目中的路径）
    
    Returns:
        生成的 prompt 字符串
    """
    # 默认 schema 路径
    if schema_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        schema_path = os.path.join(script_dir, 'cublas_ops.json')
    
    # 加载 schema
    cublas_schema = load_cublas_schema(schema_path, cublas_function_name)
    
    # 创建 TestFuncGenerateArgs
    info = TestFuncGenerateArgs(
        kernel_name=triton_kernel_name,
        ops_namespace="aten",  # 默认命名空间
        op_name=f"test_{triton_kernel_name}_vs_cublas",
        sample_id=0,
        user_advice=None
    )
    
    # 创建生成器（需要一个配置，这里用空配置）
    class DummyConfig:
        def __init__(self):
            pass
    
    generator = TestFuncGenerator(DummyConfig())
    
    # 生成 prompt
    prompt = generator.generate_prompt_for_cublas(info, cublas_schema)
    
    return prompt, cublas_schema


def main():
    """主函数：演示用法"""
    
    print("="*70)
    print("cuBLAS Baseline Prompt 生成器")
    print("="*70)
    
    # 示例 1: SGEMM
    print("\n示例 1: 为 cublasSgemm_v2 生成 prompt\n")
    
    prompt, schema = generate_cublas_baseline_prompt(
        cublas_function_name="cublasSgemm_v2",
        triton_kernel_name="matmul_triton"
    )
    
    print(f"✅ 成功加载 schema:")
    print(f"  - 函数名: {schema['name']}")
    print(f"  - 操作: {schema['operation']}")
    print(f"  - 数据类型: {schema['dtype']}")
    print(f"  - 参数数量: {len(schema['args'])}")
    
    print(f"\n📝 生成的 Prompt (前 500 字符):")
    print("-" * 70)
    print(prompt[:500] + "...")
    print("-" * 70)
    
    # 保存完整 prompt 到文件
    output_file = "/tmp/cublas_sgemm_prompt.txt"
    with open(output_file, 'w') as f:
        f.write(prompt)
    print(f"\n💾 完整 prompt 已保存到: {output_file}")
    
    # 示例 2: SAXPY
    print("\n" + "="*70)
    print("示例 2: 为 cublasSaxpy_v2 生成 prompt\n")
    
    prompt2, schema2 = generate_cublas_baseline_prompt(
        cublas_function_name="cublasSaxpy_v2",
        triton_kernel_name="axpy_triton"
    )
    
    print(f"✅ 成功加载 schema:")
    print(f"  - 函数名: {schema2['name']}")
    print(f"  - 操作: {schema2['operation']}")
    print(f"  - 数据类型: {schema2['dtype']}")
    print(f"  - 参数数量: {len(schema2['args'])}")
    
    output_file2 = "/tmp/cublas_saxpy_prompt.txt"
    with open(output_file2, 'w') as f:
        f.write(prompt2)
    print(f"\n💾 完整 prompt 已保存到: {output_file2}")
    
    print("\n" + "="*70)
    print("✅ 完成！现在你可以将这些 prompt 发送给 LLM 生成 CuPy baseline 函数")
    print("="*70)
    
    # 打印使用说明
    print("\n📖 使用说明:")
    print("1. 将生成的 prompt 发送给 LLM (如 GPT-4, Claude)")
    print("2. LLM 会生成一个 CuPy wrapper 函数")
    print("3. 将生成的函数保存到 script/cublas_cupy/baselines/{operation}_baseline.py")
    print("4. 在测试中导入并使用这个 baseline 函数")
    print("\n示例:")
    print("  from script.cublas_cupy.baselines.gemm_baseline import gemm_cublas_baseline")
    print("  result = gemm_cublas_baseline(A, B, alpha=1.0, beta=0.0)")


if __name__ == "__main__":
    main()
