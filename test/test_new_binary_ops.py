#!/usr/bin/env python3
"""
逐个测试 test_binary_pointwise_ops.py 中新增的算子
"""

import os
import sys
import subprocess
import re
from pathlib import Path

# test_binary_pointwise_ops.py 中新增的算子及其在 PYTORCH_OPERATORS 中的格式
BINARY_OPS = [
    ('addcdiv', 'torch.addcdiv'),
    ('addcmul', 'torch.addcmul'),
    ('clamp_min_', 'torch.Tensor.clamp_min_'),
    ('fill_', 'torch.fill_'),
    ('floor_divide', 'torch.floor_divide'),
    ('floor_divide_', 'torch.Tensor.floor_divide_'),
    ('gelu_and_mul', 'gelu_and_mul'),  # 自定义算子，不在 PYTORCH_OPERATORS 中，直接用算子名
    ('lerp', 'torch.lerp'),
    ('lerp_', 'torch.Tensor.lerp_'),
    ('masked_fill', 'torch.masked_fill'),
    ('masked_fill_', 'torch.Tensor.masked_fill_'),
    ('nan_to_num', 'torch.nan_to_num'),
    ('polar', 'torch.polar'),
    ('silu_and_mul', 'silu_and_mul'),  # 自定义算子，不在 PYTORCH_OPERATORS 中，直接用算子名
    ('threshold', 'torch.nn.functional.threshold'),
]


def test_operator(op_name, pytorch_key):
    """测试单个算子"""
    if pytorch_key is None:
        print(f"\n⚠️  {op_name}: 未指定 PYTORCH_OPERATORS 键，跳过")
        return None, "未指定键"
    
    print(f"\n{'='*80}")
    print(f"测试算子: {op_name} ({pytorch_key})")
    print(f"{'='*80}")
    
    script_path = Path(__file__).parent / "test_accuracy_ut.py"
    
    try:
        # 使用正确的环境
        env = os.environ.copy()
        env['LD_PRELOAD'] = '/usr/local/cuda/compat/lib/libcuda.so'
        python_path = '/share/project/tj/miniconda3/envs/test/bin/python'
        
        result = subprocess.run(
            [python_path, str(script_path), '--name', pytorch_key],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=script_path.parent,
            env=env
        )
        
        output = result.stdout + result.stderr
        
        if result.returncode == 0:
            # 检查测试是否真正通过
            # 通过标志：有 "✅ 通过" 或者 "Passed=XX, Failed=0" 且没有 "❌ 失败"
            has_success = "✅ 通过" in output or ("Passed=" in output and "Failed=0" in output)
            has_failure = "❌ 失败" in output or ("Failed=" in output and "Failed=0" not in output)
            
            # 如果有明确的失败标志，返回失败
            if has_failure:
                return False, output
            # 如果有明确的成功标志，返回成功
            elif has_success:
                return True, output
            # 否则检查测试统计信息
            else:
                # 尝试从输出中提取测试统计
                # 查找 "Total=XX, Passed=XX, Failed=XX" 模式
                stats_match = re.search(r'Total=(\d+),\s*Passed=(\d+),\s*Failed=(\d+)', output)
                if stats_match:
                    total, passed, failed = map(int, stats_match.groups())
                    if failed == 0 and passed > 0:
                        return True, output
                    else:
                        return False, output
                # 如果找不到统计信息，默认认为失败（需要人工检查）
                return False, output
        else:
            return False, output
            
    except subprocess.TimeoutExpired:
        return False, "测试超时（>60秒）"
    except Exception as e:
        return False, f"执行错误: {str(e)}"


def main():
    print("开始测试 test_binary_pointwise_ops.py 中新增的算子")
    print("=" * 80)
    
    results = {}
    passed = 0
    failed = 0
    skipped = 0
    
    for op_name, pytorch_key in BINARY_OPS:
        success, msg = test_operator(op_name, pytorch_key)
        results[op_name] = (success, msg)
        
        if success is None:
            skipped += 1
        elif success:
            passed += 1
            print(f"✅ {op_name}: 通过")
        else:
            failed += 1
            print(f"❌ {op_name}: 失败")
            # 打印错误信息的前几行
            if msg:
                lines = msg.split('\n')
                error_lines = [l for l in lines if any(keyword in l.lower() for keyword in ['error', 'exception', 'traceback', 'failed', '失败', 'keyerror'])]
                if error_lines:
                    for line in error_lines[:10]:
                        if line.strip():
                            print(f"   {line}")
                else:
                    for line in lines[:10]:
                        if line.strip() and not line.startswith('['):
                            print(f"   {line}")
    
    # 打印总结
    print("\n" + "=" * 80)
    print("测试总结:")
    print(f"  ✅ 通过: {passed} 个")
    print(f"  ❌ 失败: {failed} 个")
    print(f"  ⚠️  跳过: {skipped} 个")
    print(f"  📊 总计: {len(BINARY_OPS)} 个")
    
    # 打印失败详情
    if failed > 0:
        print("\n" + "=" * 80)
        print("失败算子详情:")
        print("=" * 80)
        for op_name, (success, msg) in results.items():
            if success is False:
                print(f"\n❌ {op_name}:")
                print("-" * 80)
                if msg:
                    # 只显示关键错误信息
                    lines = msg.split('\n')
                    error_lines = [l for l in lines if any(keyword in l.lower() for keyword in ['error', 'exception', 'traceback', 'failed', '失败', 'keyerror'])]
                    if error_lines:
                        for line in error_lines[:20]:
                            if line.strip():
                                print(f"  {line}")
                    else:
                        # 显示测试统计信息
                        stats_match = re.search(r'Total=(\d+),\s*Passed=(\d+),\s*Failed=(\d+)', msg)
                        if stats_match:
                            total, passed, failed = map(int, stats_match.groups())
                            print(f"  测试统计: Total={total}, Passed={passed}, Failed={failed}")
                        else:
                            for line in lines[:15]:
                                if line.strip() and not line.startswith('['):
                                    print(f"  {line}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

