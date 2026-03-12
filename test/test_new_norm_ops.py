#!/usr/bin/env python3
"""
逐个测试 test_norm_ops.py 中新增的算子
"""

import os
import sys
import subprocess
import re
from pathlib import Path

# test_norm_ops.py 中新增的算子及其在 PYTORCH_OPERATORS 中的格式
# 排除自定义算子：fused_add_rms_norm, skip_layer_norm, weight_norm
NORM_OPS = [
    ('batch_norm', 'torch.batch_norm'),
    ('layer_norm', 'torch.nn.functional.layer_norm'),
    ('rms_norm', 'torch.rms_norm'),
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
            has_success = "✅ 通过" in output or ("Passed=" in output and "Failed=0" in output)
            has_failure = "❌ 失败" in output or ("Failed=" in output and "Failed=0" not in output)
            
            if has_failure:
                return False, output
            elif has_success:
                return True, output
            else:
                stats_match = re.search(r'Total=(\d+),\s*Passed=(\d+),\s*Failed=(\d+)', output)
                if stats_match:
                    total, passed, failed = map(int, stats_match.groups())
                    if failed == 0 and passed > 0:
                        return True, output
                    else:
                        return False, output
                return False, output
        else:
            return False, output
            
    except subprocess.TimeoutExpired:
        return False, "测试超时（>60秒）"
    except Exception as e:
        return False, f"执行错误: {str(e)}"


def main():
    print("开始测试 test_norm_ops.py 中新增的算子")
    print("=" * 80)
    
    results = {}
    passed = 0
    failed = 0
    skipped = 0
    
    for op_name, pytorch_key in NORM_OPS:
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
            if msg:
                lines = msg.split('\n')
                error_lines = [l for l in lines if any(keyword in l.lower() for keyword in ['error', 'exception', 'traceback', 'failed', '失败', 'keyerror', 'nameerror', 'attributeerror'])]
                if error_lines:
                    for line in error_lines[:15]:
                        if line.strip():
                            print(f"   {line}")
                else:
                    for line in lines[:15]:
                        if line.strip() and not line.startswith('['):
                            print(f"   {line}")
    
    print("\n" + "=" * 80)
    print("测试总结:")
    print(f"  ✅ 通过: {passed} 个")
    print(f"  ❌ 失败: {failed} 个")
    print(f"  ⚠️  跳过: {skipped} 个")
    print(f"  📊 总计: {len(NORM_OPS)} 个")
    
    if failed > 0:
        print("\n" + "=" * 80)
        print("失败算子详情:")
        print("=" * 80)
        for op_name, (success, msg) in results.items():
            if success is False:
                print(f"\n❌ {op_name}:")
                print("-" * 80)
                if msg:
                    lines = msg.split('\n')
                    error_lines = [l for l in lines if any(keyword in l.lower() for keyword in ['error', 'exception', 'traceback', 'failed', '失败'])]
                    if error_lines:
                        for line in error_lines[:25]:
                            if line.strip():
                                print(f"  {line}")
                    else:
                        stats_match = re.search(r'Total=(\d+),\s*Passed=(\d+),\s*Failed=(\d+)', msg)
                        if stats_match:
                            total, passed, failed = map(int, stats_match.groups())
                            print(f"  测试统计: Total={total}, Passed={passed}, Failed={failed}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

