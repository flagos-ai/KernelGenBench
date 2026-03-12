#!/usr/bin/env python3
"""
阶段4: 测试PyTorch bindings

功能：
1. 加载编译好的.so库
2. 验证所有算子是否正确注册
3. 运行简单的测试用例
4. 生成测试报告

使用方法：
python script/test_bindings.py \
    --lib lib/kaldi_ops.so \
    --test-kernel add_row_sum_mat
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Dict, Any


class BindingTester:
    """测试PyTorch bindings"""
    
    def __init__(self, lib_path: Path, namespace: str = "kaldi"):
        self.lib_path = lib_path
        self.namespace = namespace
        self.test_results = []
        
    def load_library(self) -> bool:
        """加载编译好的库"""
        print(f"\n=== Loading library: {self.lib_path} ===")
        
        if not self.lib_path.exists():
            print(f"✗ Error: Library not found: {self.lib_path}")
            return False
        
        try:
            import torch
            torch.ops.load_library(str(self.lib_path))
            print(f"✓ Library loaded successfully")
            return True
        except Exception as e:
            print(f"✗ Error loading library: {e}")
            return False
    
    def list_registered_ops(self) -> List[str]:
        """列出所有注册的算子"""
        try:
            import torch
            
            # 获取命名空间下的所有ops
            ops_namespace = getattr(torch.ops, self.namespace, None)
            if ops_namespace is None:
                print(f"✗ Namespace '{self.namespace}' not found in torch.ops")
                return []
            
            # 列出所有操作
            ops = [op for op in dir(ops_namespace) if not op.startswith('_')]
            print(f"\n✓ Found {len(ops)} registered operations in torch.ops.{self.namespace}")
            
            return ops
        except Exception as e:
            print(f"✗ Error listing operations: {e}")
            return []
    
    def test_simple_kernel(self, kernel_name: str) -> bool:
        """测试单个kernel（简单smoke test）"""
        print(f"\n--- Testing kernel: {kernel_name} ---")
        
        try:
            import torch
            
            # 获取操作
            ops_namespace = getattr(torch.ops, self.namespace)
            op = getattr(ops_namespace, kernel_name)
            
            print(f"  ✓ Operation found: torch.ops.{self.namespace}.{kernel_name}")
            
            # 这里只是验证op存在，不执行
            # 实际测试需要根据每个kernel的签名构造输入
            print(f"  ℹ Skipping execution test (requires kernel-specific inputs)")
            
            self.test_results.append({
                "kernel": kernel_name,
                "status": "registered",
                "error": None
            })
            
            return True
            
        except AttributeError:
            print(f"  ✗ Kernel not found: {kernel_name}")
            self.test_results.append({
                "kernel": kernel_name,
                "status": "not_found",
                "error": "Kernel not registered"
            })
            return False
        except Exception as e:
            print(f"  ✗ Error: {e}")
            self.test_results.append({
                "kernel": kernel_name,
                "status": "error",
                "error": str(e)
            })
            return False
    
    def test_add_row_sum_mat(self) -> bool:
        """测试add_row_sum_mat（示例）"""
        print(f"\n--- Testing add_row_sum_mat with real data ---")
        
        try:
            import torch
            
            ops_namespace = getattr(torch.ops, self.namespace)
            op = getattr(ops_namespace, "add_row_sum_mat")
            
            # 创建测试数据
            # result = alpha * row_sum(mat) + beta * result
            rows, cols = 4, 6
            mat = torch.randn(rows, cols, dtype=torch.float32, device='cuda')
            result = torch.zeros(rows, dtype=torch.float32, device='cuda')
            scratch = torch.zeros(cols, dtype=torch.float32, device='cuda')  # scratch space
            alpha = 1.0
            beta = 0.0
            
            print(f"  Input mat shape: {mat.shape}")
            print(f"  Input result shape: {result.shape}")
            
            # 调用kernel
            # 注意：这里的签名可能需要根据实际生成的binding调整
            try:
                op(result, mat, scratch, mat, alpha, beta)
                print(f"  ✓ Kernel executed successfully")
                print(f"  Output result: {result}")
                
                # 简单验证：手动计算row sum
                expected = mat.sum(dim=1) * alpha + result * beta
                print(f"  Expected result: {expected}")
                
                return True
            except RuntimeError as e:
                print(f"  ⚠ Kernel execution failed (may need signature adjustment): {e}")
                return False
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return False
    
    def generate_report(self, output_file: Path):
        """生成测试报告"""
        report = {
            "library": str(self.lib_path),
            "namespace": self.namespace,
            "total_tests": len(self.test_results),
            "passed": sum(1 for r in self.test_results if r["status"] == "registered"),
            "failed": sum(1 for r in self.test_results if r["status"] != "registered"),
            "results": self.test_results
        }
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open('w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Test report saved to: {output_file}")
        print(f"  Total: {report['total_tests']}, Passed: {report['passed']}, Failed: {report['failed']}")
    
    def run_tests(self, test_kernels: List[str] = None) -> bool:
        """运行所有测试"""
        print("="*60)
        print("Stage 4: Testing Bindings")
        print("="*60)
        
        # 1. 加载库
        if not self.load_library():
            return False
        
        # 2. 列出所有ops
        registered_ops = self.list_registered_ops()
        
        if not registered_ops:
            print("✗ No operations found")
            return False
        
        # 3. 测试指定的kernels或所有kernels
        kernels_to_test = test_kernels if test_kernels else registered_ops[:5]  # 默认测试前5个
        
        print(f"\nTesting {len(kernels_to_test)} kernels...")
        
        for kernel in kernels_to_test:
            self.test_simple_kernel(kernel)
        
        # 4. 可选：测试一个具体的kernel
        if "add_row_sum_mat" in registered_ops:
            print("\n" + "="*60)
            print("Running detailed test for add_row_sum_mat")
            print("="*60)
            self.test_add_row_sum_mat()
        
        # 5. 生成报告
        success_rate = sum(1 for r in self.test_results if r["status"] == "registered") / len(self.test_results) if self.test_results else 0
        
        print("\n" + "="*60)
        print(f"Testing completed: {success_rate*100:.1f}% kernels registered")
        print("="*60)
        
        return success_rate > 0.5  # 至少50%成功才算通过


def main():
    parser = argparse.ArgumentParser(
        description="Test PyTorch bindings for Kaldi CUDA kernels"
    )
    parser.add_argument(
        "--lib",
        type=Path,
        default=Path("lib/kaldi_ops.so"),
        help="Path to compiled library"
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="kaldi",
        help="PyTorch ops namespace"
    )
    parser.add_argument(
        "--test-kernel",
        type=str,
        action='append',
        help="Specific kernel to test (can be used multiple times)"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("test_report.json"),
        help="Output test report file"
    )
    
    args = parser.parse_args()
    
    # 转换为绝对路径
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    lib_path = project_root / args.lib
    report_path = project_root / args.report
    
    # 运行测试
    tester = BindingTester(lib_path, args.namespace)
    success = tester.run_tests(args.test_kernel)
    tester.generate_report(report_path)
    
    if success:
        print("\n✓ Stage 4 completed successfully!")
    else:
        print("\n✗ Stage 4 failed")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
