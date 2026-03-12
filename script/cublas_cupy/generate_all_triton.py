#!/usr/bin/env python3
"""
批量生成所有47个cuBLAS操作的Triton kernel实现

这个脚本会为每个cuBLAS操作生成Triton实现
"""

import os
import sys
import json
import subprocess
from pathlib import Path

# 添加src路径
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def load_coverage_data():
    """加载覆盖率数据，获取所有操作名称"""
    coverage_path = SCRIPT_DIR / 'cupy_cublas_coverage.json'
    with open(coverage_path, 'r') as f:
        data = json.load(f)
    return data

def extract_operations(functions):
    """从函数列表中提取操作名称（去除dtype前缀）"""
    operations = set()
    for func_name in functions:
        # 例如: cublasSaxpy_v2 -> axpy, cublasSgemm_v2 -> gemm
        # 去除 cublas 前缀
        name = func_name.replace('cublas', '')
        # 去除 _v2 后缀
        name = name.replace('_v2', '')
        # 去除dtype前缀 (S/D/C/Z/H)
        if name and name[0] in ['S', 'D', 'C', 'Z', 'H']:
            operation = name[1:].lower()
            operations.add(operation)
    return sorted(operations)

def main():
    print("="*80)
    print("批量生成所有cuBLAS Triton Kernel实现")
    print("="*80)
    print()
    
    # 加载覆盖率数据
    coverage_data = load_coverage_data()
    with_cupy = coverage_data['functions']['with_cupy']
    
    print(f"总共有 {len(with_cupy)} 个cuBLAS函数需要生成Triton实现")
    
    # 提取操作名称
    operations = extract_operations(with_cupy)
    print(f"涉及 {len(operations)} 个操作:")
    print(f"  {', '.join(operations)}")
    print()
    
    # 生成参数
    output_dir = PROJECT_ROOT / "output_triton_cublas"
    schema_path = SCRIPT_DIR / "cublas_ops.json"
    
    print("生成配置:")
    print(f"  输出目录: {output_dir}")
    print(f"  Schema文件: {schema_path}")
    print(f"  模型: gpt-5")
    print(f"  温度: 0.0")
    print(f"  并发workers: 10")
    print()
    
    # 为每个操作生成Triton kernel
    print("="*80)
    print("开始生成...")
    print("="*80)
    print()
    
    failed_ops = []
    successful_ops = []
    
    for i, operation in enumerate(operations, 1):
        print(f"[{i}/{len(operations)}] 生成 {operation} 的Triton kernel...")
        
        cmd = [
            "python",
            str(SCRIPT_DIR / "generate_triton_cublas.py"),
            "--name", operation,
            "--output-dir", str(output_dir),
            "--schema-path", str(schema_path),
            "--server-type", "panda",
            "--model-name", "gpt-5",
            "--temperature", "0.0",
            "--max-tokens", "16384",
            "--num-samples", "1",
            "--num-workers", "10",
        ]
        
        try:
            # 设置环境变量
            env = os.environ.copy()
            env['PANDA_API_KEY'] = os.getenv('PANDA_API_KEY', '')
            
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            
            if result.returncode == 0:
                print(f"  ✓ {operation} 生成成功")
                successful_ops.append(operation)
            else:
                print(f"  ✗ {operation} 生成失败")
                print(f"    错误: {result.stderr[:200]}")
                failed_ops.append(operation)
        except subprocess.TimeoutExpired:
            print(f"  ✗ {operation} 超时")
            failed_ops.append(operation)
        except Exception as e:
            print(f"  ✗ {operation} 出错: {e}")
            failed_ops.append(operation)
        
        print()
    
    # 统计结果
    print("="*80)
    print("生成完成!")
    print("="*80)
    print()
    print(f"统计:")
    print(f"  总数: {len(operations)}")
    print(f"  成功: {len(successful_ops)}")
    print(f"  失败: {len(failed_ops)}")
    print(f"  成功率: {len(successful_ops)/len(operations)*100:.1f}%")
    
    if failed_ops:
        print()
        print(f"失败的操作: {', '.join(failed_ops)}")
    
    print()
    print(f"输出目录: {output_dir}")

if __name__ == '__main__':
    main()
