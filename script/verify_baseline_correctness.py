#!/usr/bin/env python3
"""
Verification script for baseline cuBLAS functions.

This script validates that all 47 baseline functions:
1. Have correct imports (cupy, cublas, DLPack)
2. Use proper CuPy cuBLAS API calls
3. Can execute successfully with test data
4. Return correct torch.Tensor outputs
5. Produce mathematically correct results for simple operations

Usage:
    python script/verify_baseline_correctness.py
    python script/verify_baseline_correctness.py --verbose
    python script/verify_baseline_correctness.py --name saxpy,sgemm
"""

import os
import sys
import json
import re
import argparse
import importlib
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict

# Add src to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
import cupy as cp
from rich.console import Console
from rich.table import Table
from rich import print as rprint

# Disable Torch library dispatch
os.environ['DISPATCH_TORCH_LIB'] = '0'
os.environ['FLAGBENCH_UPCAST'] = '0'

console = Console()


@dataclass
class ValidationResult:
    """Result of validation for a single baseline function."""
    function_name: str
    file_path: str
    
    # Static checks
    has_cupy_import: bool = False
    has_cublas_import: bool = False
    has_dlpack_import: bool = False
    has_register_decorator: bool = False
    has_correct_namespace: bool = False
    cublas_api_call: str = ""
    has_invalid_api_usage: bool = False  # e.g., cublas.create()
    
    # Dynamic checks
    can_import: bool = False
    import_error: str = ""
    can_execute: bool = False
    execution_error: str = ""
    returns_tensor: bool = False
    
    # Numerical correctness (for simple ops)
    numerical_check: str = "N/A"  # "PASS", "FAIL", "SKIP", "N/A"
    numerical_error: str = ""
    
    @property
    def is_valid(self) -> bool:
        """Overall validation status."""
        static_ok = (
            self.has_cupy_import and
            self.has_cublas_import and
            self.has_dlpack_import and
            self.has_register_decorator and
            self.has_correct_namespace and
            not self.has_invalid_api_usage and
            bool(self.cublas_api_call)
        )
        dynamic_ok = (
            self.can_import and
            self.can_execute and
            self.returns_tensor
        )
        return static_ok and dynamic_ok


class BaselineVerifier:
    """Verifies correctness of baseline cuBLAS functions."""
    
    def __init__(self, baseline_dir: Path, coverage_file: Path):
        self.baseline_dir = baseline_dir
        self.coverage_file = coverage_file
        self.results: List[ValidationResult] = []
        
        # Load coverage data
        with open(coverage_file) as f:
            self.coverage_data = json.load(f)
        
        # Expected cuBLAS API functions
        self.expected_apis = {
            'nrm2': 'cublas.nrm2',
            'dot': 'cublas.dot',
            'dotu': 'cublas.dotu',
            'dotc': 'cublas.dotc',
            'scal': 'cublas.scal',
            'axpy': 'cublas.axpy',
            'asum': 'cublas.asum',
            'gemv': 'cublas.gemv',
            'sbmv': 'cublas.sbmv',
            'ger': 'cublas.ger',
            'geru': 'cublas.geru',
            'gerc': 'cublas.gerc',
            'gemm': 'cublas.gemm',
            'syrk': 'cublas.syrk',
            'geam': 'cublas.geam',
            'dgmm': 'cublas.dgmm',
            # hgemm uses CuPy array operations (cuBLAS doesn't support float16 well in CuPy)
            'hgemm': None,
        }
        
        # Operations that use CuPy array operations instead of cuBLAS directly
        # hgemm: cuBLAS doesn't support float16 well in CuPy, so use CuPy @ operator
        self.non_cublas_ops = {'gemm'}
    
    def get_operation_from_function_name(self, func_name: str) -> str:
        """Extract operation name from function (e.g., 'saxpy' -> 'axpy')."""
        # Remove dtype prefix (s, d, c, z, h)
        for prefix in ['s', 'd', 'c', 'z', 'h']:
            if func_name.startswith(prefix):
                return func_name[1:]
        return func_name
    
    def static_analysis(self, file_path: Path) -> ValidationResult:
        """Perform static code analysis on a baseline file."""
        func_name = file_path.stem
        result = ValidationResult(
            function_name=func_name,
            file_path=str(file_path)
        )
        
        # Read file content
        code = file_path.read_text()
        
        # Check imports
        result.has_cupy_import = 'import cupy' in code
        result.has_cublas_import = 'from cupy import cublas' in code
        result.has_dlpack_import = 'from torch.utils.dlpack import' in code
        
        # Check register decorator
        result.has_register_decorator = '@register' in code
        namespace_match = re.search(r'namespace\s*=\s*["\'](\w+)["\']', code)
        result.has_correct_namespace = (
            namespace_match and namespace_match.group(1) == 'baseline'
        )
        
        # Check for invalid API usage
        result.has_invalid_api_usage = 'cublas.create()' in code
        
        # Find cuBLAS API call
        operation = self.get_operation_from_function_name(func_name)
        expected_api = self.expected_apis.get(operation, '')
        
        # Skip API check for non-cuBLAS operations (like hgemm)
        if operation in self.non_cublas_ops:
            result.cublas_api_call = 'cupy_array_ops'
            return result
        
        # Search for cublas.xxx() calls
        api_pattern = r'cublas\.(\w+)\s*\('
        api_matches = re.findall(api_pattern, code)
        if api_matches:
            result.cublas_api_call = f"cublas.{api_matches[0]}"
        
        return result
    
    def dynamic_test(self, result: ValidationResult) -> None:
        """Perform dynamic testing by importing and executing the function."""
        func_name = result.function_name
        
        # Try to import
        try:
            import flagbench.baseline
            from sandbox.register import REGISTERED_OPS
            
            result.can_import = True
            
            # Check if registered
            if 'baseline' not in REGISTERED_OPS:
                result.import_error = "Namespace 'baseline' not found in REGISTERED_OPS"
                result.can_import = False
                return
            
            if func_name not in REGISTERED_OPS['baseline']:
                result.import_error = f"Function '{func_name}' not registered in baseline namespace"
                result.can_import = False
                return
                
        except Exception as e:
            result.can_import = False
            result.import_error = str(e)
            return
        
        # Try to execute with test data
        try:
            # REGISTERED_OPS stores (name, func, autograd_info) tuples
            registered_entry = REGISTERED_OPS['baseline'][func_name]
            if isinstance(registered_entry, tuple):
                func = registered_entry[1]  # Extract function from tuple
            else:
                func = registered_entry
            
            test_data = self.generate_test_data(func_name)
            
            if test_data is None:
                result.execution_error = "No test data generator available"
                return
            
            # Execute
            output = func(**test_data)
            result.can_execute = True
            
            # Check return type
            result.returns_tensor = isinstance(output, torch.Tensor)
            
            # Numerical correctness check
            self.check_numerical_correctness(func_name, test_data, output, result)
            
        except Exception as e:
            result.can_execute = False
            result.execution_error = str(e)
    
    def generate_test_data(self, func_name: str) -> Dict[str, Any]:
        """Generate simple test data for a function."""
        device = 'cuda'
        
        # Determine dtype
        if func_name.startswith('s'):
            dtype = torch.float32
        elif func_name.startswith('d'):
            dtype = torch.float64
        elif func_name.startswith('c'):
            dtype = torch.complex64
        elif func_name.startswith('z'):
            dtype = torch.complex128
        elif func_name.startswith('h'):
            dtype = torch.float16
        else:
            dtype = torch.float32
        
        operation = self.get_operation_from_function_name(func_name)
        
        # BLAS Level 1
        if operation == 'nrm2':
            return {'n': 10, 'x': torch.randn(10, dtype=dtype, device=device), 'incx': 1, 'result': torch.randn(1, dtype=dtype, device=device)}
        
        elif operation == 'dot':
            return {
                'n': 10,
                'x': torch.randn(10, dtype=dtype, device=device),
                'incx': 1,
                'y': torch.randn(10, dtype=dtype, device=device),
                'incy': 1,
                'result': torch.randn(1, dtype=dtype, device=device)
            }
        
        elif operation in ['dotu', 'dotc']:
            return {
                'n': 10,
                'x': torch.randn(10, dtype=dtype, device=device),
                'incx': 1,
                'y': torch.randn(10, dtype=dtype, device=device),
                'incy': 1,
                'result': torch.randn(1, dtype=dtype, device=device)
            }
        
        elif operation == 'scal':
            return {
                'n': 10,
                'alpha': 2.0,
                'x': torch.randn(10, dtype=dtype, device=device),
                'incx': 1
            }
        
        elif operation == 'axpy':
            return {
                'n': 10,
                'alpha': 2.0,
                'x': torch.randn(10, dtype=dtype, device=device),
                'incx': 1,
                'y': torch.randn(10, dtype=dtype, device=device),
                'incy': 1
            }
        
        elif operation == 'asum':
            return {'n': 10, 'x': torch.randn(10, dtype=dtype, device=device), 'incx': 1, 'result': torch.randn(1, dtype=dtype, device=device)}
        
        # BLAS Level 2
        elif operation == 'gemv':
            return {
                'trans': 'N',
                'm': 4,
                'n': 4,
                'alpha': 1.0,
                'A': torch.randn(4, 4, dtype=dtype, device=device),
                'lda': 4,
                'x': torch.randn(4, dtype=dtype, device=device),
                'incx': 1,
                'beta': 0.0,
                'y': torch.randn(4, dtype=dtype, device=device),
                'incy': 1
            }
        
        elif operation == 'sbmv':
            return {
                'uplo': 'U',
                'n': 4,
                'k': 1,
                'alpha': 1.0,
                'A': torch.randn(4, 4, dtype=dtype, device=device),
                'lda': 4,
                'x': torch.randn(4, dtype=dtype, device=device),
                'incx': 1,
                'beta': 0.0,
                'y': torch.randn(4, dtype=dtype, device=device),
                'incy': 1
            }
        
        elif operation == 'ger':
            return {
                'm': 4,
                'n': 4,
                'alpha': 1.0,
                'x': torch.randn(4, dtype=dtype, device=device),
                'incx': 1,
                'y': torch.randn(4, dtype=dtype, device=device),
                'incy': 1,
                'A': torch.randn(4, 4, dtype=dtype, device=device),
                'lda': 4
            }
        
        elif operation in ['geru', 'gerc']:
            return {
                'm': 4,
                'n': 4,
                'alpha': 1.0,
                'x': torch.randn(4, dtype=dtype, device=device),
                'incx': 1,
                'y': torch.randn(4, dtype=dtype, device=device),
                'incy': 1,
                'A': torch.randn(4, 4, dtype=dtype, device=device),
                'lda': 4
            }
        
        # BLAS Level 3
        elif operation == 'gemm':
            return {
                'transa': 'N',
                'transb': 'N',
                'm': 4,
                'n': 4,
                'k': 4,
                'alpha': 1.0,
                'A': torch.randn(4, 4, dtype=dtype, device=device),
                'lda': 4,
                'B': torch.randn(4, 4, dtype=dtype, device=device),
                'ldb': 4,
                'beta': 0.0,
                'C': torch.randn(4, 4, dtype=dtype, device=device),
                'ldc': 4
            }
        
        elif operation == 'hgemm':
            # hgemm uses float16
            hgemm_dtype = torch.float16
            return {
                'transa': 'N',
                'transb': 'N',
                'm': 4,
                'n': 4,
                'k': 4,
                'alpha': 1.0,
                'A': torch.randn(4, 4, dtype=hgemm_dtype, device=device),
                'lda': 4,
                'B': torch.randn(4, 4, dtype=hgemm_dtype, device=device),
                'ldb': 4,
                'beta': 0.0,
                'C': torch.randn(4, 4, dtype=hgemm_dtype, device=device),
                'ldc': 4
            }
        
        elif operation == 'syrk':
            return {
                'uplo': 'U',
                'trans': 'N',
                'n': 4,
                'k': 4,
                'alpha': 1.0,
                'A': torch.randn(4, 4, dtype=dtype, device=device),
                'lda': 4,
                'beta': 0.0,
                'C': torch.randn(4, 4, dtype=dtype, device=device),
                'ldc': 4
            }
        
        # Extensions
        elif operation == 'geam':
            return {
                'transa': 'N',
                'transb': 'N',
                'm': 4,
                'n': 4,
                'alpha': 1.0,
                'A': torch.randn(4, 4, dtype=dtype, device=device),
                'lda': 4,
                'beta': 1.0,
                'B': torch.randn(4, 4, dtype=dtype, device=device),
                'ldb': 4,
                'C': torch.randn(4, 4, dtype=dtype, device=device),
                'ldc': 4
            }
        
        elif operation == 'dgmm':
            return {
                'mode': 'L',
                'm': 4,
                'n': 4,
                'A': torch.randn(4, 4, dtype=dtype, device=device),
                'lda': 4,
                'x': torch.randn(4, dtype=dtype, device=device),
                'incx': 1,
                'C': torch.randn(4, 4, dtype=dtype, device=device),
                'ldc': 4
            }
        
        return None
    
    def check_numerical_correctness(self, func_name: str, test_data: Dict, output: torch.Tensor, result: ValidationResult) -> None:
        """Check numerical correctness for simple operations."""
        operation = self.get_operation_from_function_name(func_name)
        
        try:
            # Only check simple operations where we can easily verify
            if operation == 'axpy':
                # y = alpha * x + y
                alpha = test_data['alpha']
                x = test_data['x']
                y_original = test_data['y'].clone()
                expected = alpha * x + y_original
                
                if torch.allclose(output, expected, rtol=1e-4, atol=1e-5):
                    result.numerical_check = "PASS"
                else:
                    result.numerical_check = "FAIL"
                    max_diff = (output - expected).abs().max().item()
                    result.numerical_error = f"Max difference: {max_diff:.2e}"
            
            elif operation == 'scal':
                # x = alpha * x
                alpha = test_data['alpha']
                x_original = test_data['x'].clone()
                expected = alpha * x_original
                
                if torch.allclose(output, expected, rtol=1e-4, atol=1e-5):
                    result.numerical_check = "PASS"
                else:
                    result.numerical_check = "FAIL"
                    max_diff = (output - expected).abs().max().item()
                    result.numerical_error = f"Max difference: {max_diff:.2e}"
            
            else:
                result.numerical_check = "SKIP"
                
        except Exception as e:
            result.numerical_check = "ERROR"
            result.numerical_error = str(e)
    
    def verify_all(self, filter_names: List[str] = None) -> None:
        """Verify all baseline functions."""
        py_files = sorted([f for f in self.baseline_dir.glob("*.py") if f.name != "__init__.py"])
        
        if filter_names:
            py_files = [f for f in py_files if f.stem in filter_names]
        
        console.print(f"\n[bold cyan]Verifying {len(py_files)} baseline functions...[/bold cyan]\n")
        
        for file_path in py_files:
            # Static analysis
            result = self.static_analysis(file_path)
            
            # Dynamic testing
            if result.has_cupy_import and result.has_cublas_import:
                self.dynamic_test(result)
            
            self.results.append(result)
            
            # Live feedback
            status = "✓" if result.is_valid else "✗"
            color = "green" if result.is_valid else "red"
            console.print(f"[{color}]{status}[/{color}] {result.function_name}")
    
    def generate_report(self, verbose: bool = False) -> None:
        """Generate validation report."""
        total = len(self.results)
        valid = sum(1 for r in self.results if r.is_valid)
        
        # Summary table
        table = Table(title=f"\n📊 Baseline Verification Summary ({valid}/{total} valid)")
        table.add_column("Check", style="cyan")
        table.add_column("Pass", style="green")
        table.add_column("Fail", style="red")
        table.add_column("Rate", style="yellow")
        
        checks = [
            ("CuPy Import", "has_cupy_import"),
            ("cuBLAS Import", "has_cublas_import"),
            ("DLPack Import", "has_dlpack_import"),
            ("Register Decorator", "has_register_decorator"),
            ("Correct Namespace", "has_correct_namespace"),
            ("No Invalid API", lambda r: not r.has_invalid_api_usage),
            ("Has cuBLAS Call", lambda r: bool(r.cublas_api_call)),
            ("Can Import", "can_import"),
            ("Can Execute", "can_execute"),
            ("Returns Tensor", "returns_tensor"),
        ]
        
        for check_name, attr in checks:
            if callable(attr):
                passed = sum(1 for r in self.results if attr(r))
            else:
                passed = sum(1 for r in self.results if getattr(r, attr))
            failed = total - passed
            rate = f"{passed/total*100:.1f}%"
            table.add_row(check_name, str(passed), str(failed), rate)
        
        console.print(table)
        
        # Numerical correctness
        num_pass = sum(1 for r in self.results if r.numerical_check == "PASS")
        num_fail = sum(1 for r in self.results if r.numerical_check == "FAIL")
        num_skip = sum(1 for r in self.results if r.numerical_check == "SKIP")
        console.print(f"\n[bold]Numerical Correctness:[/bold] {num_pass} pass, {num_fail} fail, {num_skip} skip\n")
        
        # Failed functions
        failed = [r for r in self.results if not r.is_valid]
        if failed:
            console.print(f"\n[bold red]❌ {len(failed)} Failed Functions:[/bold red]\n")
            for r in failed:
                console.print(f"  • {r.function_name}")
                if not r.has_cupy_import:
                    console.print("    - Missing CuPy import")
                if not r.has_cublas_import:
                    console.print("    - Missing cuBLAS import")
                if not r.has_dlpack_import:
                    console.print("    - Missing DLPack import")
                if r.has_invalid_api_usage:
                    console.print("    - Invalid API usage (e.g., cublas.create())")
                if not r.cublas_api_call:
                    console.print("    - No cuBLAS API call found")
                if not r.can_import:
                    console.print(f"    - Import error: {r.import_error}")
                if not r.can_execute:
                    console.print(f"    - Execution error: {r.execution_error}")
                if not r.returns_tensor:
                    console.print("    - Does not return torch.Tensor")
                console.print()
        
        # Verbose output
        if verbose:
            console.print("\n[bold]Detailed Results:[/bold]\n")
            for r in self.results:
                status_icon = "✓" if r.is_valid else "✗"
                status_color = "green" if r.is_valid else "red"
                console.print(f"[{status_color}]{status_icon} {r.function_name}[/{status_color}]")
                console.print(f"  API Call: {r.cublas_api_call or 'NOT FOUND'}")
                console.print(f"  Numerical: {r.numerical_check}")
                if r.numerical_error:
                    console.print(f"  Error: {r.numerical_error}")
                console.print()
    
    def save_json_report(self, output_file: Path) -> None:
        """Save detailed results to JSON."""
        data = {
            "summary": {
                "total": len(self.results),
                "valid": sum(1 for r in self.results if r.is_valid),
                "invalid": sum(1 for r in self.results if not r.is_valid)
            },
            "results": [asdict(r) for r in self.results]
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        console.print(f"\n[green]✓[/green] Detailed report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Verify baseline cuBLAS functions")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--name", help="Comma-separated list of function names to verify")
    parser.add_argument("--output", "-o", default="baseline_verification_report.json", help="Output JSON file")
    args = parser.parse_args()
    
    # Paths
    baseline_dir = REPO_ROOT / "src" / "flagbench" / "baseline"
    coverage_file = REPO_ROOT / "script" / "cublas_cupy" / "cupy_cublas_coverage.json"
    
    # Verify
    verifier = BaselineVerifier(baseline_dir, coverage_file)
    
    filter_names = args.name.split(',') if args.name else None
    verifier.verify_all(filter_names)
    
    # Report
    verifier.generate_report(verbose=args.verbose)
    verifier.save_json_report(REPO_ROOT / "script" / args.output)
    
    # Exit code
    valid_count = sum(1 for r in verifier.results if r.is_valid)
    if valid_count == len(verifier.results):
        console.print("\n[bold green]🎉 All baseline functions are valid![/bold green]\n")
        sys.exit(0)
    else:
        console.print(f"\n[bold red]⚠️  {len(verifier.results) - valid_count} functions need attention[/bold red]\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
