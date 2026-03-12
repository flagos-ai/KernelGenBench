#!/usr/bin/env python3
"""
测试脚本：验证新增的 101 个算子测试
基于 test_accuracy_ut.py，使用 Verifier 来测试算子

使用方法:
    python test_accuracy_all.py --mode collect  # 收集所有新增的算子
    python test_accuracy_all.py --mode run      # 运行所有新增算子的测试
    python test_accuracy_all.py --mode run --quick  # 快速测试（只运行前10个）
    python test_accuracy_all.py --mode run --name "addmm"  # 测试指定算子
"""

import os
os.environ["DISPATCH_TORCH_LIB"] = "0"
os.environ["FLAGBENCH_UPCAST"] = "0"

import sys
import argparse
import subprocess
import re
from pathlib import Path
from sandbox.verifier import Verifier, VerifyConfig, VerifyRequest, Source
from flagbench import PYTORCH_OPERATORS

# Mock triton code（用于 Verifier）
mock_triton_code = "mock triton code"

# 9 个修改的测试文件（在 src/flagbench/accuracy/ 目录下）
TEST_FILES = [
    'test_unary_pointwise_ops.py',
    'test_binary_pointwise_ops.py',
    'test_reduction_ops.py',
    'test_blas_ops.py',
    'test_attention_ops.py',
    'test_special_ops.py',
    'test_norm_ops.py',
    'test_general_reduction_ops.py',
    'test_tensor_constructor_ops.py',
]


def get_test_dir():
    """获取测试文件目录"""
    # 测试文件在 src/flagbench/accuracy/
    current_dir = Path(__file__).parent
    test_dir = current_dir.parent / "src" / "flagbench" / "accuracy"
    if not test_dir.exists():
        print(f"❌ 测试目录不存在: {test_dir}")
        sys.exit(1)
    return test_dir


def extract_labels_from_file(filepath):
    """
    从测试文件中提取所有 @label("operator_name") 装饰器
    返回：(算子名称, 测试函数名) 的列表
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    matches = []
    current_labels = []
    
    for i, line in enumerate(lines):
        # 查找 @label("operator_name")
        label_match = re.match(r'@label\(["\']([^"\']+)["\']\)', line.strip())
        if label_match:
            # 跳过 "inplace" 标签
            if label_match.group(1) != "inplace":
                current_labels.append(label_match.group(1))
        
        # 查找 def test_xxx
        func_match = re.match(r'def\s+(test_\w+)\s*\(', line.strip())
        if func_match and current_labels:
            test_func = func_match.group(1)
            # 取最后一个非 "inplace" 的标签
            for label in reversed(current_labels):
                if label != "inplace":
                    matches.append((label, test_func))
                    break
            current_labels = []
    
    return matches  # [(operator_name, test_function_name), ...]


def collect_new_operators(test_dir):
    """
    收集新增的算子（通过扫描测试文件中的 @label 装饰器和 git diff）
    返回：{operator_name: [(filename, test_func), ...]} 字典
    """
    print("收集 9 个修改文件中新增的算子:")
    print("=" * 80)
    
    all_operators = {}  # {op_name: [(filename, test_func), ...]}
    total_new = 0
    
    for filename in TEST_FILES:
        filepath = test_dir / filename
        
        if not filepath.exists():
            print(f"\n❌ {filename}: 文件不存在")
            continue
        
        # 提取当前文件的所有 @label
        current_labels = extract_labels_from_file(filepath)
        
        # 使用 git diff 找出新增的测试函数
        try:
            result = subprocess.run(
                ['git', 'diff', 'HEAD~1', str(filepath)],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=test_dir
            )
            
            if result.returncode == 0:
                diff_output = result.stdout
                # 查找新增的 def test_xxx 函数
                new_test_funcs = re.findall(r'^\+def (test_\w+)\(', diff_output, re.MULTILINE)
                new_test_funcs_set = set(new_test_funcs)
            else:
                new_test_funcs_set = set()
        except Exception as e:
            print(f"  ⚠️ git diff 失败: {e}")
            new_test_funcs_set = set()
        
        # 过滤：只保留新增的测试函数对应的算子
        new_operators_in_file = []
        for op_name, test_func in current_labels:
            if test_func in new_test_funcs_set:
                new_operators_in_file.append((op_name, test_func))
                
                # 添加到总字典
                if op_name not in all_operators:
                    all_operators[op_name] = []
                all_operators[op_name].append((filename, test_func))
        
        print(f"\n✅ {filename}:")
        print(f"   总 @label: {len(current_labels)} 个")
        print(f"   新增测试: {len(new_test_funcs_set)} 个")
        print(f"   新增算子: {len(new_operators_in_file)} 个")
        
        if new_operators_in_file:
            # 显示前 5 个新增算子
            op_names = [op for op, _ in new_operators_in_file]
            print(f"   算子示例: {', '.join(op_names[:5])}{'...' if len(op_names) > 5 else ''}")
        
        total_new += len(new_operators_in_file)
    
    print("\n" + "=" * 80)
    print(f"汇总:")
    print(f"  扫描文件: {len(TEST_FILES)} 个")
    print(f"  新增测试: {total_new} 个")
    print(f"  涉及算子: {len(all_operators)} 个（去重后）")
    
    return all_operators


def test_operator_with_verifier(op_name, test_sources):
    """
    使用 Verifier 测试单个算子
    
    Args:
        op_name: 算子名称，例如 "addmm"
        test_sources: [(filename, test_func), ...] 列表
    
    Returns:
        tuple: (bool, str) - (测试是否通过, 状态消息)
    """
    # 检查算子是否在 PYTORCH_OPERATORS 中
    # PYTORCH_OPERATORS 的键格式是 "torch.xxx" 或 "torch.Tensor.xxx_" 等
    possible_keys = [
        f"torch.{op_name}",
        f"torch.Tensor.{op_name}",
        f"torch.nn.functional.{op_name}",
        f"torch.nn.{op_name}",
        f"torch.ops.aten.{op_name}",
        f"torch._C._nn.{op_name}",
        f"torch.linalg.{op_name}",
    ]
    
    op_key = None
    for key in possible_keys:
        if key in PYTORCH_OPERATORS:
            op_key = key
            break
    
    if op_key is None:
        return False, "自定义算子"
    
    # 使用找到的键
    
    # 创建 Verifier 配置
    config = VerifyConfig(
        run_name=f"test_{op_name}",
        test_type="accuracy",
        run_dir=f"/tmp/flagbench_test_runs/{op_name}",
        store_type="local",
        strict_check=True,
        seed=42,
        sample_id=0,
        save_log=True,
    )
    
    verifier = Verifier(config)
    
    # 构建 VerifyRequest（使用完整的 torch.xxx 格式）
    source = Source(
        source=mock_triton_code,
        function_name=op_key  # 使用 torch.xxx 格式
    )
    
    verify_request = VerifyRequest(source=[source])
    
    try:
        # 运行验证
        print(f"  🔍 测试算子: {op_name} ({op_key})")
        print(f"     测试来源: {', '.join([f'{fn}::{tf}' for fn, tf in test_sources])}")
        
        verifier.only_verify(
            name_source_map=[verify_request],
            test_type="accuracy"
        )
        
        print(f"  ✅ 通过")
        return True, "通过"
        
    except Exception as e:
        error_msg = str(e)
        print(f"  ❌ 失败: {error_msg}")
        return False, error_msg


def run_tests(operators_dict, quick_mode=False, single_op=None):
    """
    运行所有算子测试
    
    Args:
        operators_dict: {op_name: [(filename, test_func), ...]}
        quick_mode: 是否快速模式（只测试前 10 个）
        single_op: 只测试指定算子
    """
    print("\n" + "=" * 80)
    print("运行 Verifier 测试:")
    print("=" * 80)
    
    if single_op:
        if single_op not in operators_dict:
            print(f"❌ 算子 '{single_op}' 不在新增列表中")
            return {}
        operators_to_test = {single_op: operators_dict[single_op]}
    else:
        operators_to_test = operators_dict
    
    if quick_mode:
        print("\n⚡ 快速模式：只测试前 10 个算子")
        operators_to_test = dict(list(operators_to_test.items())[:10])
    
    results = {}
    passed = 0
    failed = 0
    custom = 0
    
    for i, (op_name, test_sources) in enumerate(operators_to_test.items(), 1):
        print(f"\n[{i}/{len(operators_to_test)}] 测试: {op_name}")
        print("-" * 80)
        
        success, msg = test_operator_with_verifier(op_name, test_sources)
        results[op_name] = (success, msg)
        
        if msg == "自定义算子":
            custom += 1
            print(f"  ⚠️ 跳过：这是自定义算子（不在 PYTORCH_OPERATORS）")
        elif success:
            passed += 1
        else:
            failed += 1
    
    # 打印总结
    print("\n" + "=" * 80)
    print("测试总结:")
    print("=" * 80)
    print(f"✅ 通过: {passed} 个")
    print(f"❌ 失败: {failed} 个")
    print(f"⚠️  自定义算子（跳过）: {custom} 个")
    print(f"📊 总计: {len(results)} 个")
    
    if failed > 0:
        print("\n❌ 失败算子列表:")
        for op_name, (success, msg) in results.items():
            if not success and msg != "自定义算子":
                print(f"  - {op_name}: {msg}")
    
    if custom > 0:
        print("\n⚠️  自定义算子列表（无法用 ut.py 测试）:")
        for op_name, (success, msg) in results.items():
            if msg == "自定义算子":
                print(f"  - {op_name}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="测试新增的算子（使用 Verifier）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --mode collect                    # 收集所有新增算子
  %(prog)s --mode run                        # 运行所有测试
  %(prog)s --mode run --quick                # 快速测试前 10 个
  %(prog)s --mode run --name "addmm"         # 只测试 addmm 算子
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['collect', 'run'],
        default='collect',
        help='运行模式：collect（收集算子）或 run（运行测试）'
    )
    
    parser.add_argument(
        '--name',
        type=str,
        help='只测试指定的算子（例如：addmm）'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='快速模式：只运行前 10 个测试'
    )
    
    args = parser.parse_args()
    
    # 获取测试目录
    test_dir = get_test_dir()
    print(f"测试目录: {test_dir}\n")
    
    # 收集新增算子
    operators_dict = collect_new_operators(test_dir)
    
    if args.mode == 'collect':
        # 只收集，打印列表
        print("\n" + "=" * 80)
        print("新增算子列表:")
        print("=" * 80)
        for i, (op_name, sources) in enumerate(sorted(operators_dict.items()), 1):
            print(f"{i:3d}. {op_name:30s} ({len(sources)} 个测试)")
        return 0
    
    elif args.mode == 'run':
        if not operators_dict:
            print("\n❌ 没有找到可测试的算子")
            return 1
        
        # 运行测试
        results = run_tests(operators_dict, args.quick, args.name)
        
        # 返回适当的退出码
        failed_count = sum(1 for success in results.values() if not success)
        return 1 if failed_count > 0 else 0
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
