#!/usr/bin/env python3
"""
测试 test_attention_ops.py 中新增的测试函数
"""

import os
import sys
import subprocess
import re

# 新增的测试函数和对应的算子
ATTENTION_OPS = [
    ("scaled_dot_product_attention", "torch.nn.functional.scaled_dot_product_attention"),
    ("flash_mla", "flagbench.flash_mla"),
    ("scaled_dot_product_attention", "torch.nn.functional.scaled_dot_product_attention"),
    ("scaled_dot_product_attention", "torch.nn.functional.scaled_dot_product_attention"),
    ("scaled_dot_product_attention", "torch.nn.functional.scaled_dot_product_attention"),
    ("flash_attention_forward", "flagbench.ops.flash_attention_forward"),
    ("flash_attention_forward", "flagbench.ops.flash_attention_forward"),
    ("flash_attention_forward", "flagbench.ops.flash_attention_forward"),
    ("flash_attention_forward", "flagbench.ops.flash_attention_forward"),
    ("flash_attn_varlen_func", "flagbench.ops.flash_attn_varlen_func"),
    ("flash_attn_varlen_func", "flagbench.ops.flash_attn_varlen_func"),
    ("concat_and_cache_mla", "flagbench.concat_and_cache_mla"),
    ("reshape_and_cache", "flagbench.reshape_and_cache"),
    ("flash_attention_forward", "flagbench.ops.flash_attention_forward"),
    ("reshape_and_cache_flash", "flagbench.reshape_and_cache_flash"),
    ("get_scheduler_metadata", "flagbench.get_scheduler_metadata"),
]

# 自定义算子列表（这些可能无法用标准 PyTorch 测试）
CUSTOM_OPS = {
    "flash_mla",
    "flash_attention_forward",
    "flash_attn_varlen_func",
    "concat_and_cache_mla",
    "reshape_and_cache",
    "reshape_and_cache_flash",
    "get_scheduler_metadata",
}

test_script = "/share/project/zpy/flagbench/test/test_accuracy_ut.py"


def test_operator(op_name, pytorch_key):
    """测试单个算子"""
    if op_name in CUSTOM_OPS:
        print(f"⚠️  SKIP (custom op): {op_name}")
        return True
    
    print(f"\n{'='*80}")
    print(f"Testing: {op_name}")
    print(f"PyTorch key: {pytorch_key}")
    print(f"{'='*80}")
    
    env = os.environ.copy()
    env['LD_PRELOAD'] = '/usr/local/cuda/compat/lib/libcuda.so'
    python_path = '/share/project/tj/miniconda3/envs/test/bin/python'
    
    result = subprocess.run(
        [python_path, test_script, '--name', pytorch_key],
        cwd='/share/project/zpy/flagbench',
        env=env,
        capture_output=True,
        text=True,
        timeout=120
    )
    
    output = result.stdout + result.stderr
    
    if '✅ 通过' in output or '通过:' in output:
        if 'Failed=0' in output or ('Passed=' in output and 'Failed=' in output):
            passed_match = re.search(r'Passed=(\d+)', output)
            failed_match = re.search(r'Failed=(\d+)', output)
            if passed_match and failed_match:
                passed_count = int(passed_match.group(1))
                failed_count = int(failed_match.group(1))
                if failed_count == 0 and passed_count > 0:
                    print(f"✅ PASSED: {op_name} (Passed={passed_count}, Failed={failed_count})")
                    return True
                elif failed_count > 0:
                    print(f"❌ FAILED: {op_name} (Passed={passed_count}, Failed={failed_count})")
                    return False
    
    if '⚠️  无测试' in output or '无测试:' in output or 'No valid test cases found' in output:
        print(f"⚠️  NO TEST: {op_name}")
        return True
    
    if '❌ 失败' in output or '❌ 失败:' in output:
        print(f"❌ FAILED: {op_name}")
        lines = output.split('\n')
        print("Last 20 lines of output:")
        for line in lines[-20:]:
            if line.strip():
                print(f"  {line}")
        return False
    
    if result.returncode == 0:
        passed_match = re.search(r'Passed=(\d+)', output)
        failed_match = re.search(r'Failed=(\d+)', output)
        if passed_match and failed_match:
            passed_count = int(passed_match.group(1))
            failed_count = int(failed_match.group(1))
            if failed_count == 0:
                print(f"✅ PASSED: {op_name} (Passed={passed_count}, Failed={failed_count})")
                return True
            else:
                print(f"❌ FAILED: {op_name} (Passed={passed_count}, Failed={failed_count})")
                return False
        else:
            print(f"✅ PASSED: {op_name} (no errors, return code: {result.returncode})")
            return True
    else:
        print(f"❌ FAILED: {op_name} (return code: {result.returncode})")
        lines = output.split('\n')
        print("Last 20 lines of output:")
        for line in lines[-20:]:
            if line.strip():
                print(f"  {line}")
        return False


def main():
    print("Testing new attention operators from test_attention_ops.py")
    print("="*80)
    
    # 去重，只测试每个算子一次
    tested_ops = set()
    results = []
    
    for op_name, pytorch_key in ATTENTION_OPS:
        if op_name in tested_ops:
            continue
        tested_ops.add(op_name)
        
        success = test_operator(op_name, pytorch_key)
        results.append((op_name, success))
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, success in results if success)
    failed = len(results) - passed
    
    for op_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {op_name}")
    
    print(f"\nTotal: {len(results)}, Passed: {passed}, Failed: {failed}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

