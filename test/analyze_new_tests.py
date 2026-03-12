#!/usr/bin/env python3
"""
统计9个测试文件中新增的测试函数和对应的算子

使用方法:
    python test/analyze_new_tests.py
"""

import os
import sys
import subprocess
import re
from pathlib import Path
from collections import defaultdict

# 9 个修改的测试文件
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
    current_dir = Path(__file__).parent
    test_dir = current_dir.parent / "src" / "flagbench" / "accuracy"
    if not test_dir.exists():
        print(f"❌ 测试目录不存在: {test_dir}")
        sys.exit(1)
    return test_dir


def extract_labels_from_file(filepath):
    """
    从测试文件中提取所有 @label("operator_name") 和对应的测试函数
    返回：{test_func_name: [label1, label2, ...]} 字典
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')
    
    test_func_labels = {}  # {test_func: [labels]}
    current_labels = []
    current_func = None
    
    for i, line in enumerate(lines):
        # 查找 @label("operator_name")
        label_match = re.search(r'@label\(["\']([^"\']+)["\']\)', line)
        if label_match:
            label_name = label_match.group(1)
            # 跳过 "inplace" 标签
            if label_name != "inplace":
                current_labels.append(label_name)
        
        # 查找 def test_xxx
        func_match = re.match(r'^\s*def\s+(test_\w+)\s*\(', line)
        if func_match:
            test_func = func_match.group(1)
            # 保存上一个函数的标签
            if current_func and current_labels:
                test_func_labels[current_func] = current_labels.copy()
            
            # 开始新的函数
            current_func = test_func
            current_labels = []
    
    # 保存最后一个函数的标签
    if current_func and current_labels:
        test_func_labels[current_func] = current_labels.copy()
    
    return test_func_labels


def get_new_test_functions(filepath, base_ref="HEAD~1"):
    """
    使用 git diff 找出新增的测试函数
    返回：新增的测试函数名列表
    """
    try:
        # 尝试多个可能的基准点
        refs_to_try = [base_ref, "origin/master", "upstream/master"]
        
        for ref in refs_to_try:
            try:
                result = subprocess.run(
                    ['git', 'diff', ref, '--', str(filepath)],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=filepath.parent.parent.parent.parent  # 回到仓库根目录
                )
                
                if result.returncode == 0 and result.stdout:
                    # 查找新增的 def test_xxx 函数
                    new_test_funcs = re.findall(r'^\+\s*def\s+(test_\w+)\s*\(', result.stdout, re.MULTILINE)
                    if new_test_funcs:
                        return list(set(new_test_funcs))  # 去重
            except:
                continue
        
        return []
    except Exception as e:
        print(f"  ⚠️ git diff 失败: {e}")
        return []


def analyze_new_tests():
    """分析新增的测试"""
    test_dir = get_test_dir()
    print(f"测试目录: {test_dir}\n")
    print("=" * 80)
    print("分析新增测试...")
    print("=" * 80)
    
    all_new_tests = defaultdict(list)  # {operator_name: [(filename, test_func), ...]}
    file_stats = {}
    
    for filename in TEST_FILES:
        filepath = test_dir / filename
        
        if not filepath.exists():
            print(f"\n❌ {filename}: 文件不存在")
            continue
        
        # 提取当前文件的所有标签
        all_labels = extract_labels_from_file(filepath)
        
        # 获取新增的测试函数
        new_test_funcs = get_new_test_functions(filepath)
        new_test_funcs_set = set(new_test_funcs)
        
        # 找出新增测试对应的算子
        new_operators = defaultdict(list)  # {operator: [test_funcs]}
        for test_func, labels in all_labels.items():
            if test_func in new_test_funcs_set:
                for label in labels:
                    if label != "inplace":
                        new_operators[label].append(test_func)
                        all_new_tests[label].append((filename, test_func))
        
        # 统计信息
        file_stats[filename] = {
            'total_labels': len(all_labels),
            'new_tests': len(new_test_funcs_set),
            'new_operators': len(new_operators),
            'operators': new_operators
        }
        
        # 打印文件统计
        print(f"\n📄 {filename}:")
        print(f"   总测试函数: {len(all_labels)} 个")
        print(f"   新增测试: {len(new_test_funcs_set)} 个")
        print(f"   涉及算子: {len(new_operators)} 个")
        
        if new_operators:
            # 显示前10个算子
            op_names = list(new_operators.keys())[:10]
            print(f"   算子示例: {', '.join(op_names)}{'...' if len(new_operators) > 10 else ''}")
    
    # 打印总结
    print("\n" + "=" * 80)
    print("汇总统计:")
    print("=" * 80)
    print(f"📊 扫描文件: {len(TEST_FILES)} 个")
    
    total_new_tests = sum(stats['new_tests'] for stats in file_stats.values())
    total_new_operators = len(all_new_tests)
    
    print(f"📝 新增测试函数: {total_new_tests} 个")
    print(f"🔧 涉及算子: {total_new_operators} 个（去重后）")
    
    # 按文件统计
    print("\n" + "=" * 80)
    print("按文件统计:")
    print("=" * 80)
    for filename, stats in sorted(file_stats.items()):
        print(f"\n{filename}:")
        print(f"  新增测试: {stats['new_tests']} 个")
        print(f"  涉及算子: {stats['new_operators']} 个")
        if stats['operators']:
            for op, funcs in sorted(stats['operators'].items()):
                print(f"    - {op}: {len(funcs)} 个测试 ({', '.join(funcs[:3])}{'...' if len(funcs) > 3 else ''})")
    
    # 按算子统计
    print("\n" + "=" * 80)
    print("按算子统计（所有新增的算子）:")
    print("=" * 80)
    for i, (op_name, sources) in enumerate(sorted(all_new_tests.items()), 1):
        # 去重测试函数
        unique_funcs = list(set([tf for _, tf in sources]))
        files = list(set([fn for fn, _ in sources]))
        print(f"{i:3d}. {op_name:30s} - {len(unique_funcs):2d} 个测试, 来自 {len(files)} 个文件")
        if len(files) == 1:
            print(f"     {files[0]}::{', '.join(unique_funcs[:5])}{'...' if len(unique_funcs) > 5 else ''}")
    
    # 保存到文件
    output_file = test_dir.parent.parent / "new_tests_summary.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("新增测试统计\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"总计: {total_new_tests} 个测试函数, {total_new_operators} 个算子\n\n")
        
        f.write("按算子详细列表:\n")
        f.write("-" * 80 + "\n")
        for op_name, sources in sorted(all_new_tests.items()):
            unique_funcs = list(set([tf for _, tf in sources]))
            files = list(set([fn for fn, _ in sources]))
            f.write(f"\n{op_name}:\n")
            f.write(f"  测试函数 ({len(unique_funcs)} 个): {', '.join(unique_funcs)}\n")
            f.write(f"  来源文件: {', '.join(files)}\n")
    
    print(f"\n✅ 详细统计已保存到: {output_file}")
    
    return all_new_tests, file_stats


if __name__ == "__main__":
    try:
        analyze_new_tests()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

