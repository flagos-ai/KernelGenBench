"""
批量生成有CuPy包装的47个cuBLAS baseline函数
"""
import os
import sys
import json
import argparse
from datetime import datetime

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from generator.baseline_func_generator import BaselineFuncGenerator
from generator.sampler.generate_samples import GenerationConfig

def load_cublas_schema():
    """加载cuBLAS schema"""
    schema_path = os.path.join(os.path.dirname(__file__), 'cublas_ops.json')
    with open(schema_path, 'r') as f:
        data = json.load(f)
    return data['functions']

def load_coverage_data():
    """加载覆盖率数据，获取有CuPy包装的函数列表"""
    coverage_path = os.path.join(os.path.dirname(__file__), 'cupy_cublas_coverage.json')
    with open(coverage_path, 'r') as f:
        data = json.load(f)
    return set(data['functions']['with_cupy'])

def main():
    parser = argparse.ArgumentParser(description='生成有CuPy包装的cuBLAS baseline函数')
    parser.add_argument('--num-workers', type=int, default=10, help='并发worker数量')
    parser.add_argument('--model', type=str, default='deepseek-v3-0324', help='LLM模型')
    parser.add_argument('--temperature', type=float, default=0.0, help='采样温度')
    parser.add_argument('--max-tokens', type=int, default=4096, help='最大token数')
    parser.add_argument('--dry-run', action='store_true', help='仅生成列表，不实际调用LLM')
    args = parser.parse_args()
    
    print("="*80)
    print("批量生成有CuPy包装的cuBLAS baseline函数")
    print("="*80)
    
    # 加载数据
    all_functions = load_cublas_schema()
    with_cupy_names = load_coverage_data()
    
    print(f"\n总cuBLAS函数: {len(all_functions)}")
    print(f"有CuPy包装: {len(with_cupy_names)}")
    
    # 过滤出有CuPy包装的函数
    functions_to_generate = [f for f in all_functions if f['name'] in with_cupy_names]
    
    print(f"\n将生成 {len(functions_to_generate)} 个baseline函数:")
    
    # 按操作分组显示
    by_operation = {}
    for func in functions_to_generate:
        op = func['operation']
        if op not in by_operation:
            by_operation[op] = []
        by_operation[op].append(func['name'])
    
    for op, funcs in sorted(by_operation.items()):
        print(f"  {op:15s}: {len(funcs)} 个变体")
    
    if args.dry_run:
        print("\n[DRY RUN] 函数列表:")
        for func in sorted(functions_to_generate, key=lambda x: x['name']):
            print(f"  - {func['name']:30s} ({func['operation']:10s}, {func['dtype']})")
        print(f"\n总计: {len(functions_to_generate)} 个函数")
        return
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"baseline_cublas_with_cupy_{args.model}_temp_{args.temperature}_{timestamp}"
    output_dir = os.path.join(os.path.dirname(__file__), '../../output_baseline_cublas', run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 配置生成器
    config = GenerationConfig(
        run_name=run_name,
        sample_id=0,
        num_samples=1,
        test_type='accuracy',
        name_list=None,
        num_workers=args.num_workers,
        api_query_interval=0.0,
        server_type='panda',
        model_name=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        run_dir=output_dir,
        save_log=True,
        verbose=False,
        store_type='local',
        log_prompt=True,
        backend='triton',
        greedy_sample=False,
        base_url='http://localhost:8000/v1',
        strict_check=False,
        seed=42,
        use_ai_advice=False
    )
    
    print(f"\n配置:")
    print(f"  模型: {args.model}")
    print(f"  温度: {args.temperature}")
    print(f"  最大tokens: {args.max_tokens}")
    print(f"  并发workers: {args.num_workers}")
    print(f"  输出目录: {output_dir}")
    
    # 创建生成器
    generator = BaselineFuncGenerator(config)
    
    print("\n" + "="*80)
    print("开始生成...")
    print("="*80)
    
    # 批量生成 (使用 __call__ 方法)
    results = generator(functions_to_generate)
    
    # 统计结果
    successful = 0
    failed = 0
    
    baseline_dir = os.path.join(output_dir, 'baseline_0')
    os.makedirs(baseline_dir, exist_ok=True)
    
    # 从生成结果中提取函数名，建立result到func的映射
    import re
    result_to_func = {}
    for result in results:
        if isinstance(result, str) and len(result) > 10:
            # Extract function name from @register decorator or def statement
            match = re.search(r'def\s+(\w+)\s*\(', result)
            if match:
                result_func_name = match.group(1)
                result_to_func[result_func_name] = result
    
    # 按原始函数列表顺序保存
    for i, func in enumerate(functions_to_generate):
        func_name = func['name']
        kernel_name = func_name.replace('cublas', '').replace('_v2', '').lower()
        
        # 从映射中找到对应的生成结果
        result = result_to_func.get(kernel_name)
        
        # 检查结果
        if result is not None:
            successful += 1
            # 保存文件
            output_file = os.path.join(baseline_dir, f"{kernel_name}.py")
            with open(output_file, 'w') as f:
                f.write(result)
            print(f"✓ [{i+1}/{len(functions_to_generate)}] {func_name:30s} -> {kernel_name}.py")
        else:
            failed += 1
            print(f"✗ [{i+1}/{len(functions_to_generate)}] {func_name:30s} - 生成失败")
    
    # 保存摘要
    summary = {
        'timestamp': timestamp,
        'model': args.model,
        'temperature': args.temperature,
        'total_functions': len(functions_to_generate),
        'successful': successful,
        'failed': failed,
        'success_rate': successful / len(functions_to_generate) * 100 if functions_to_generate else 0,
        'functions': [
            {
                'name': f['name'],
                'operation': f['operation'],
                'dtype': f['dtype'],
                'status': 'success' if isinstance(results[i], str) and len(results[i]) > 10 else 'failed'
            }
            for i, f in enumerate(functions_to_generate)
        ]
    }
    
    summary_file = os.path.join(output_dir, 'generation_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*80)
    print("生成完成!")
    print("="*80)
    print(f"\n统计:")
    print(f"  总数: {len(functions_to_generate)}")
    print(f"  成功: {successful}")
    print(f"  失败: {failed}")
    print(f"  成功率: {successful / len(functions_to_generate) * 100:.1f}%")
    print(f"\n输出目录: {output_dir}")
    print(f"摘要文件: {summary_file}")
    
if __name__ == '__main__':
    main()
