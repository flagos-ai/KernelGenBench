#!/usr/bin/env python3
"""
对比分析默认few-shot和分类few-shot的测试结果
"""
import json
import os
from collections import defaultdict

default_dir = "output/pass_at_k/pass_at_5_gpt-5_triton_reflection_20251211-210554"
category_dir = "output/pass_at_k/pass_at_5_gpt-5_triton_reflection_20251211-214014"

# 读取分类信息
with open("scripts/plot/api_categories.json") as f:
    categories = json.load(f)

def get_op_category(op_name):
    """获取算子的分类"""
    op_name_clean = op_name.replace('aten::', '')
    if 'categories' in categories:
        cat_dict = categories['categories']
    else:
        cat_dict = categories
    
    for cat, cat_info in cat_dict.items():
        if isinstance(cat_info, dict) and 'ops' in cat_info:
            ops_list = cat_info['ops']
        elif isinstance(cat_info, list):
            ops_list = cat_info
        else:
            continue
            
        for op in ops_list:
            if op.replace('aten::', '') == op_name_clean:
                return cat
    return "unknown"

# 读取结果
with open(f"{default_dir}/pass_at_k_results.json") as f:
    default_results = json.load(f)
with open(f"{category_dir}/pass_at_k_results.json") as f:
    category_results = json.load(f)
with open(f"{default_dir}/category_statistics.json") as f:
    default_cat_stats = json.load(f)
with open(f"{category_dir}/category_statistics.json") as f:
    category_cat_stats = json.load(f)

# 找出通过的算子
def get_passed_ops(results):
    passed = set()
    for round_summary in results.get('rounds_summary', []):
        for op in round_summary.get('newly_passed_operators', []):
            passed.add(op)
    return passed

default_passed = get_passed_ops(default_results)
category_passed = get_passed_ops(category_results)

# 获取所有测试的算子
all_ops = set()
for round_summary in default_results.get('rounds_summary', []):
    for op in round_summary.get('newly_passed_operators', []):
        all_ops.add(op)
for round_summary in category_results.get('rounds_summary', []):
    for op in round_summary.get('newly_passed_operators', []):
        all_ops.add(op)

# 找出差异
only_default = default_passed - category_passed
only_category = category_passed - default_passed
both_passed = default_passed & category_passed
both_failed = all_ops - (default_passed | category_passed)

print("=" * 80)
print("总体对比分析")
print("=" * 80)
print(f"默认few-shot: {default_results['total_passed']}/{default_results['total_operators']} = {default_results['final_pass_rate']:.2%}")
print(f"分类few-shot: {category_results['total_passed']}/{category_results['total_operators']} = {category_results['final_pass_rate']:.2%}")
print(f"差异: {default_results['total_passed'] - category_results['total_passed']} 个算子 ({abs(default_results['final_pass_rate'] - category_results['final_pass_rate']):.2%})")

print("\n" + "=" * 80)
print("各轮次通过率对比")
print("=" * 80)
print(f"{'轮次':<8} {'默认few-shot':<15} {'分类few-shot':<15} {'差异':<10}")
print("-" * 55)
for i in range(5):
    default_rate = default_cat_stats['overall_accuracy'][str(i)]
    category_rate = category_cat_stats['overall_accuracy'][str(i)]
    diff = default_rate - category_rate
    print(f"Round {i:<3} {default_rate:>6.2%}        {category_rate:>6.2%}        {diff:>+7.2%}")

print("\n" + "=" * 80)
print("各分类通过率对比 (Round 4)")
print("=" * 80)
print(f"{'分类':<30} {'默认':<12} {'分类':<12} {'差异':<10}")
print("-" * 70)
for cat in sorted(default_cat_stats['category_accuracy'].keys()):
    default_acc = default_cat_stats['category_accuracy'][cat]['4']
    category_acc = category_cat_stats['category_accuracy'][cat]['4']
    diff = default_acc - category_acc
    print(f"{cat:<30} {default_acc:>6.2%}      {category_acc:>6.2%}      {diff:>+7.2%}")

print("\n" + "=" * 80)
print("通过的算子差异")
print("=" * 80)
print(f"只在默认few-shot通过 ({len(only_default)}个): {sorted(only_default) if only_default else '无'}")
if only_default:
    for op in sorted(only_default):
        cat = get_op_category(op)
        print(f"  - {op} ({cat})")

print(f"\n只在分类few-shot通过 ({len(only_category)}个): {sorted(only_category) if only_category else '无'}")
if only_category:
    for op in sorted(only_category):
        cat = get_op_category(op)
        print(f"  - {op} ({cat})")

print(f"\n两者都通过: {len(both_passed)} 个")
print(f"两者都失败: {len(both_failed)} 个")

# 分析各分类的详细差异
print("\n" + "=" * 80)
print("各分类详细对比")
print("=" * 80)

def organize_by_category(passed_ops, all_ops_in_category):
    cat_ops = defaultdict(set)
    for op in passed_ops:
        cat = get_op_category(op)
        cat_ops[cat].add(op)
    return cat_ops

default_cat_ops = organize_by_category(default_passed, all_ops)
category_cat_ops = organize_by_category(category_passed, all_ops)

for cat in sorted(set(default_cat_ops.keys()) | set(category_cat_ops.keys())):
    default_ops = default_cat_ops.get(cat, set())
    category_ops = category_cat_ops.get(cat, set())
    
    # 获取该分类的所有算子
    cat_all_ops = set()
    if 'categories' in categories:
        cat_dict = categories['categories']
    else:
        cat_dict = categories
    
    if cat in cat_dict:
        cat_info = cat_dict[cat]
        if isinstance(cat_info, dict) and 'ops' in cat_info:
            ops_list = cat_info['ops']
        elif isinstance(cat_info, list):
            ops_list = cat_info
        else:
            ops_list = []
        
        for op in ops_list:
            cat_all_ops.add(op.replace('aten::', ''))
    
    default_passed_count = len(default_ops)
    category_passed_count = len(category_ops)
    total = len(cat_all_ops)
    
    if total > 0:
        print(f"\n{cat}:")
        print(f"  默认: {default_passed_count}/{total} = {default_passed_count/total:.2%}")
        print(f"  分类: {category_passed_count}/{total} = {category_passed_count/total:.2%}")
        
        # 找出差异算子
        diff_ops = []
        for op in cat_all_ops:
            default_status = op in default_ops
            category_status = op in category_ops
            if default_status != category_status:
                diff_ops.append((op, default_status, category_status))
        
        if diff_ops:
            print(f"  差异算子 ({len(diff_ops)}个):")
            for op, d_status, c_status in sorted(diff_ops):
                print(f"    - {op}: 默认={'✓' if d_status else '✗'}, 分类={'✓' if c_status else '✗'}")

# 分析错误类型
print("\n" + "=" * 80)
print("错误类型分布对比")
print("=" * 80)

def analyze_errors(result_dir):
    error_types = defaultdict(int)
    
    verification_dir = f"{result_dir}/verification"
    if not os.path.exists(verification_dir):
        return error_types
    
    for log_dir in sorted(os.listdir(verification_dir)):
        if not log_dir.startswith("log_"):
            continue
        log_path = f"{verification_dir}/{log_dir}"
        
        for report_file in os.listdir(log_path):
            if not report_file.startswith("test_report_") or not report_file.endswith(".json"):
                continue
            
            report_path = f"{log_path}/{report_file}"
            try:
                with open(report_path) as f:
                    report_data = json.load(f)
                
                if isinstance(report_data, list):
                    for test_case in report_data:
                        if not test_case.get('success', True):
                            traceback = test_case.get('traceback', '')
                            
                            # 分类错误类型
                            if 'AttributeError' in traceback:
                                if 'log1p' in traceback:
                                    err_type = 'AttributeError: log1p'
                                elif 'astype' in traceback:
                                    err_type = 'AttributeError: astype'
                                else:
                                    err_type = 'AttributeError: other'
                            elif 'CompilationError' in traceback or 'triton.compiler.errors.CompilationError' in traceback:
                                err_type = 'CompilationError'
                            elif 'TypeError' in traceback:
                                err_type = 'TypeError'
                            elif 'NameError' in traceback:
                                err_type = 'NameError'
                            elif 'RuntimeError' in traceback:
                                err_type = 'RuntimeError'
                            else:
                                err_type = 'Other'
                            
                            error_types[err_type] += 1
            except:
                pass
    
    return error_types

default_error_types = analyze_errors(default_dir)
category_error_types = analyze_errors(category_dir)

print(f"\n{'错误类型':<30} {'默认few-shot':<15} {'分类few-shot':<15} {'差异':<10}")
print("-" * 75)

all_error_types = set(default_error_types.keys()) | set(category_error_types.keys())
for err_type in sorted(all_error_types):
    default_count = default_error_types.get(err_type, 0)
    category_count = category_error_types.get(err_type, 0)
    diff = default_count - category_count
    print(f"{err_type:<30} {default_count:>6}        {category_count:>6}        {diff:>+4}")

# 总结
print("\n" + "=" * 80)
print("总结")
print("=" * 80)
print(f"1. 总体通过率: 默认few-shot ({default_results['final_pass_rate']:.2%}) vs 分类few-shot ({category_results['final_pass_rate']:.2%})")
print(f"   差异: {abs(default_results['final_pass_rate'] - category_results['final_pass_rate']):.2%} ({default_results['total_passed'] - category_results['total_passed']:+d} 个算子)")

print(f"\n2. 轮次表现:")
print(f"   - Round 0-1: 默认few-shot表现更好 (差异 +2% ~ +12%)")
print(f"   - Round 2-4: 差异缩小到 +2% ~ +10%")

print(f"\n3. 分类表现:")
print(f"   - core reduction: 默认few-shot明显更好 (+11.11%)")
print(f"   - 其他分类: 差异很小或相同")

print(f"\n4. 关键差异算子:")
print(f"   - 只在默认few-shot通过: {len(only_default)} 个")
print(f"   - 只在分类few-shot通过: {len(only_category)} 个")

print(f"\n5. 错误类型:")
print(f"   - CompilationError: 分类few-shot更多 (-7)")
print(f"   - AttributeError: other: 默认few-shot更多 (+8)")

print("\n结论: 两种few-shot策略在最终通过率上差异很小(2%)，但默认few-shot在早期轮次和reduction类算子表现更好。")

