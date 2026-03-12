#!/usr/bin/env python3
"""
Test script for newly added operators in test_reduction_ops.py
"""
import subprocess
import sys
import os

# Custom operators to skip (not in PYTORCH_OPERATORS)
CUSTOM_OPS = [
    'topk_softmax',
    'conv_depthwise2d',
]

# Map of label names to PyTorch operator keys
# Note: verifier extracts the last part of the key (e.g., 'cross_entropy' from 'torch.nn.functional.cross_entropy')
# and uses it to find the label. So we need to use the actual label name or the last part of the key.
REDUCTION_OPS = [
    ('amax', 'torch.amax'),
    ('argmax', 'torch.argmax'),
    ('argmin', 'torch.argmin'),
    ('CrossEntropyLoss', 'torch.nn.functional.cross_entropy'),  # Label is CrossEntropyLoss, but key maps to cross_entropy
    ('nll_loss', 'torch.nn.functional.nll_loss'),
    ('cumsum', 'torch.cumsum'),
    ('cummin', 'torch.cummin'),
    ('cummax', 'torch.cummax'),
    ('nonzero', 'torch.nonzero'),
    ('count_nonzero', 'torch.count_nonzero'),
    ('log_softmax', 'torch.log_softmax'),
    ('softmax', 'torch.softmax'),
    ('var_mean', 'torch.var_mean'),
    ('scatter', 'torch.scatter'),
    ('select_scatter', 'torch.select_scatter'),
    ('slice_scatter', 'torch.slice_scatter'),
    ('index_add', 'torch.index_add'),
    ('index_fill', 'torch.index_fill'),
    ('index_select', 'torch.index_select'),
    ('masked_select', 'torch.masked_select'),
    ('conv1d', 'torch.conv1d'),
    ('index_put', 'torch.index_put'),
    ('mse_loss', 'torch.nn.functional.mse_loss'),
    ('scatter_', 'torch.Tensor.scatter_'),
    ('trace', 'torch.trace'),
    ('gather', 'torch.gather'),
    ('index_add_', 'torch.Tensor.index_add_'),
    ('max_pool2d', 'torch.max_pool2d'),
    ('max_pool2d_backward', 'torch.max_pool2d_backward'),
    ('conv2d', 'torch.nn.functional.conv2d'),
    ('conv3d', 'torch.nn.functional.conv3d'),
    ('index_put_', 'torch.index_put_'),
    ('index', 'torch.ops.aten.index'),
    ('std', 'torch.std'),
]

def test_operator(op_name, pytorch_key):
    """Test a single operator"""
    if op_name in CUSTOM_OPS:
        print(f"⏭️  Skipping custom operator: {op_name}")
        return True
    
    print(f"\n{'='*60}")
    print(f"Testing: {op_name} ({pytorch_key})")
    print(f"{'='*60}")
    
    python_path = "/share/project/tj/miniconda3/envs/test/bin/python"
    test_script = "/share/project/zpy/flagbench/test/test_accuracy_ut.py"
    
    env = os.environ.copy()
    env['LD_PRELOAD'] = '/usr/local/cuda/compat/lib/libcuda.so'
    
    try:
        result = subprocess.run(
            [python_path, test_script, '--name', pytorch_key],
            cwd='/share/project/zpy/flagbench',
            env=env,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )
        
        output = result.stdout + result.stderr
        
        # Check for success indicators - look for "✅ 通过" or "Passed=X, Failed=0" pattern
        if '✅ 通过' in output or '通过:' in output:
            # Check if there are any failures
            if 'Failed=0' in output or ('Passed=' in output and 'Failed=' in output):
                # Extract numbers
                import re
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
        
        # Check for "无测试" (no test) indicator
        if '⚠️  无测试' in output or '无测试:' in output or 'No valid test cases found' in output:
            print(f"⚠️  NO TEST: {op_name}")
            return True  # Not a failure, just no test
        
        # Check for failure indicators
        if '❌ 失败' in output or '❌ 失败:' in output:
            print(f"❌ FAILED: {op_name}")
            # Print last 20 lines of output
            lines = output.split('\n')
            print("Last 20 lines of output:")
            for line in lines[-20:]:
                if line.strip():
                    print(f"  {line}")
            return False
        
        # If no clear indicator, check return code
        if result.returncode == 0:
            # Try to parse the output for test statistics
            import re
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
            # Print last 20 lines of output
            lines = output.split('\n')
            print("Last 20 lines of output:")
            for line in lines[-20:]:
                if line.strip():
                    print(f"  {line}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏱️  TIMEOUT: {op_name} (exceeded 5 minutes)")
        return False
    except Exception as e:
        print(f"❌ ERROR testing {op_name}: {e}")
        return False

def main():
    print("="*60)
    print("Testing newly added operators in test_reduction_ops.py")
    print("="*60)
    
    passed = 0
    failed = 0
    skipped = 0
    failed_ops = []
    
    for op_name, pytorch_key in REDUCTION_OPS:
        if op_name in CUSTOM_OPS:
            skipped += 1
            continue
        
        success = test_operator(op_name, pytorch_key)
        if success:
            passed += 1
        else:
            failed += 1
            failed_ops.append((op_name, pytorch_key))
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total operators: {len(REDUCTION_OPS)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    
    if failed_ops:
        print(f"\nFailed operators:")
        for op_name, pytorch_key in failed_ops:
            print(f"  - {op_name} ({pytorch_key})")
    
    return 0 if failed == 0 else 1

if __name__ == '__main__':
    sys.exit(main())

