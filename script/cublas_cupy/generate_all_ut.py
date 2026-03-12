#!/usr/bin/env python3
"""
批量生成所有47个cuBLAS baseline的单元测试

这个脚本会为每个cuBLAS操作（axpy, gemm等）生成UT，
每个操作包含多个dtype变体（例如saxpy, daxpy, caxpy, zaxpy）
"""

import os
import sys
import json
import argparse
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

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="批量生成所有cuBLAS baseline的单元测试"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "output_ut_cublas"),
        help="输出目录路径 (默认: PROJECT_ROOT/output_ut_cublas)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt-5",
        help="使用的模型名称 (默认: gpt-5)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="生成温度 (默认: 0.0)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=10,
        help="并发worker数量 (默认: 10)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="最大token数 (默认: 16384)"
    )
    parser.add_argument(
        "--server-type",
        type=str,
        default="panda",
        help="服务器类型 (默认: panda)"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    print("="*80)
    print("批量生成所有cuBLAS UT测试")
    print("="*80)
    print()

    # 加载覆盖率数据
    coverage_data = load_coverage_data()
    with_cupy = coverage_data['functions']['with_cupy']

    print(f"总共有 {len(with_cupy)} 个cuBLAS函数需要生成UT")

    # 提取操作名称
    operations = extract_operations(with_cupy)
    print(f"涉及 {len(operations)} 个操作:")
    print(f"  {', '.join(operations)}")
    print()

    # 生成参数
    output_dir = Path(args.output_dir)
    schema_path = SCRIPT_DIR / "cublas_ops.json"

    print("生成配置:")
    print(f"  输出目录: {output_dir}")
    print(f"  Schema文件: {schema_path}")
    print(f"  模型: {args.model_name}")
    print(f"  温度: {args.temperature}")
    print(f"  并发workers: {args.num_workers}")
    print(f"  最大tokens: {args.max_tokens}")
    print(f"  服务器类型: {args.server_type}")
    print()
    
    # 为每个操作生成UT
    print("="*80)
    print("开始生成...")
    print("="*80)
    print()
    
    failed_ops = []
    successful_ops = []
    
    for i, operation in enumerate(operations, 1):
        print(f"[{i}/{len(operations)}] 生成 {operation} 的UT...")
        
        cmd = [
            "python",
            str(SCRIPT_DIR / "generate_ut_cublas.py"),
            "--name", operation,
            "--output-dir", str(output_dir),
            "--schema-path", str(schema_path),
            "--server-type", args.server_type,
            "--model-name", args.model_name,
            "--temperature", str(args.temperature),
            "--max-tokens", str(args.max_tokens),
            "--num-samples", "1",
            "--num-workers", str(args.num_workers),
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
