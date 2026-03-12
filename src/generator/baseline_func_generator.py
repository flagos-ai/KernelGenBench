import json
from typing import Dict, Any

from .generator import BaseGenerator, console
from generator.sampler.utils import extract_first_code

class BaselineFuncGenerator(BaseGenerator):
    """
    Generator for cuBLAS baseline functions using CuPy.
    
    This generator creates wrapper functions that call cuBLAS operations via CuPy.
    It intelligently uses cupy.cublas.xxx() when available, or falls back to
    CuPy array operations when direct cuBLAS wrappers are not available.
    """
    
    def __init__(self, generation_config):
        super().__init__(generation_config)
    
    def generate_prompt(self, info) -> str:
        """
        Generate prompt for creating a cuBLAS baseline function.
        
        Args:
            info: BaselineGenerateArgs object containing cublas_func
            
        Returns:
            Prompt string for LLM
        """
        # Extract cublas_func from info object
        cublas_func = info.cublas_func
        func_name = cublas_func['name']
        operation = cublas_func['operation']
        dtype = cublas_func['dtype']
        
        # Extract kernel name (e.g., "cublasSgemm_v2" -> "sgemm")
        kernel_name = func_name.replace('cublas', '').replace('_v2', '').lower()
        
        prompt = "You are an expert in CUDA, cuBLAS, and CuPy. Your task is to create a baseline wrapper function that calls cuBLAS operations via CuPy. You must strictly adhere to the following specifications:\\n\\n"
        
        # 基本信息
        prompt += "## Task Overview\\n"
        prompt += f"- **Function name**: `{kernel_name}`\\n"
        prompt += f"- **cuBLAS operation**: `{operation}`\\n"
        prompt += f"- **cuBLAS function**: `{func_name}`\\n"
        prompt += f"- **Data type**: `{dtype}`\\n\\n"
        
        # cuBLAS 函数签名
        prompt += "## cuBLAS Function Signature\\n"
        prompt += f"The cuBLAS function has the following signature:\\n"
        prompt += "```c\\n"
        prompt += f"{func_name}(\\n"
        for arg in cublas_func['args']:
            role_marker = {
                'context': '// cuBLAS handle',
                'input': '// input tensor',
                'output': '// output tensor',
                'inout': '// input/output tensor',
                'scalar': '// scalar parameter',
                'value': '// dimension/stride parameter'
            }.get(arg['role'], '')
            prompt += f"    {arg['type']} {arg['name']},  {role_marker}\\n"
        prompt += ");\\n"
        prompt += "```\\n\\n"
        
        # Python 函数签名（排除 handle）
        prompt += "## Required Python Function Signature\\n"
        prompt += "Your Python wrapper function MUST have this exact signature (excluding the cuBLAS handle):\\n"
        prompt += "```python\\n"
        prompt += f"def {kernel_name}(\\n"
        params = []
        for arg in cublas_func['args']:
            if arg['role'] == 'context':  # Skip handle
                continue
            params.append(arg['name'])
        prompt += ", ".join(params)
        prompt += "\\n):\\n"
        prompt += "    pass\\n"
        prompt += "```\\n\\n"
        
        # Available CuPy cuBLAS functions
        prompt += "## Available CuPy cuBLAS Functions\\n"
        prompt += "CuPy provides the following direct cuBLAS wrappers in the `cupy.cublas` module:\\n"
        prompt += "- **BLAS Level 1**: `axpy`, `dot`, `dotu`, `dotc`, `nrm2`, `scal`, `asum`, `iamax`, `iamin`\\n"
        prompt += "- **BLAS Level 2**: `gemv`, `ger`, `geru`, `gerc`\\n"
        prompt += "- **BLAS Level 3**: `gemm`, `syrk`\\n"
        prompt += "- **Other**: `geam`, `dgmm`, `sbmv`\\n\\n"
        prompt += "**IMPORTANT**: If a matching function exists in `cupy.cublas`, you MUST use it directly.\\n"
        prompt += "If no direct wrapper exists, implement the operation using CuPy array operations.\\n\\n"
        
        # Implementation guidelines
        prompt += "## Implementation Guidelines\\n\\n"
        
        prompt += "### 1. Required Imports\\n"
        prompt += "Your code MUST start with these imports:\\n"
        prompt += "```python\\n"
        prompt += "from sandbox.register import register\\n"
        prompt += "from flagbench.dataset import Autograd\\n"
        prompt += "import torch\\n"
        prompt += "import cupy as cp\\n"
        prompt += "from cupy import cublas  # For direct cuBLAS calls\\n"
        prompt += "from torch.utils.dlpack import to_dlpack, from_dlpack\\n"
        prompt += "```\\n\\n"
        
        prompt += "### 2. Registration Decorator\\n"
        prompt += "The function MUST be decorated with:\\n"
        prompt += "```python\\n"
        prompt += f'@register("CUDA", "{kernel_name}", has_backward=Autograd.disable, namespace="baseline")\\n'
        prompt += f"def {kernel_name}(...):\\n"
        prompt += "```\\n\\n"
        
        prompt += "### 3. Implementation Strategy\\n"
        prompt += f"For `{func_name}`, CuPy provides `cupy.cublas.{operation}()` - use it directly!\\n\\n"
        prompt += f"**CRITICAL**: Use `cublas.{operation}(...)` directly - DO NOT use `cublas.create()`!\\n"
        prompt += f"- ✅ CORRECT: `result = cublas.{operation}(args...)`\\n"
        prompt += f"- ❌ WRONG: `cublas.{operation}(cublas.create(), args...)`\\n"
        prompt += f"- CuPy automatically manages the cuBLAS handle internally\\n\\n"
        
        prompt += "**Step 2**: Convert PyTorch tensors to CuPy arrays\\n"
        prompt += "```python\\n"
        prompt += "# Zero-copy conversion using DLPack\\n"
        prompt += "x_cp = cp.from_dlpack(to_dlpack(x))\\n"
        prompt += "```\\n\\n"
        
        prompt += "**Step 3**: Perform the cuBLAS operation\\n"
        prompt += "```python\\n"
        prompt += f"# Option A: Direct cuBLAS call (if available)\\n"
        prompt += f"result_cp = cublas.{operation}(...)\\n\\n"
        prompt += f"# Option B: CuPy array operations (fallback)\\n"
        prompt += f"result_cp = cp.some_operation(...)\\n"
        prompt += "```\\n\\n"
        
        prompt += "**Step 4**: Convert result back to PyTorch\\n"
        prompt += "```python\\n"
        prompt += "ref_out = from_dlpack(result_cp.toDlpack())\\n"
        prompt += "return ref_out\\n"
        prompt += "```\\n\\n"
        
        prompt += "### 4. CRITICAL: Return Value\\n"
        prompt += "**The last two lines of your function MUST be:**\\n"
        prompt += "```python\\n"
        prompt += "ref_out = from_dlpack(...toDlpack())  # or torch.tensor(...) for scalars\\n"
        prompt += "return ref_out\\n"
        prompt += "```\\n"
        prompt += "**Do NOT return the expression directly. Always assign to `ref_out` first.**\\n\\n"
        
        # Operation-specific guidance with CORRECT CuPy API
        prompt += "## Operation-Specific CuPy API Usage\\n"
        if operation == 'gemm':
            prompt += "**GEMM** - `cublas.gemm(transa, transb, a, b, out=None, alpha=1.0, beta=0.0)`:\\n"
            prompt += "```python\\n"
            prompt += "# Correct usage - NO cublas.create()!\\n"
            prompt += "C_cp = cublas.gemm('N', 'N', A_cp, B_cp, alpha=alpha, beta=beta)  # beta=0\\n"
            prompt += "# OR for in-place update (beta != 0):\\n"
            prompt += "C_cp = cublas.gemm('N', 'N', A_cp, B_cp, out=C_cp, alpha=alpha, beta=beta)\\n"
            prompt += "```\\n"
        elif operation == 'axpy':
            prompt += "**AXPY** - `cublas.axpy(a, x, y)` modifies y in-place and returns None:\\n"
            prompt += "```python\\n"
            prompt += "# Correct usage - axpy returns None, y is modified in-place!\\n"
            prompt += "cublas.axpy(alpha, x_cp, y_cp)  # y = alpha*x + y, returns None\\n"
            prompt += "ref_out = from_dlpack(y_cp.toDlpack())  # Return the modified y_cp\\n"
            prompt += "```\\n"
        elif operation == 'dot':
            prompt += "**DOT** - `cublas.dot(x, y, out)` returns nothing, result goes to out:\\n"
            prompt += "```python\\n"
            prompt += "# Correct usage - must pre-allocate output buffer\\n"
            prompt += "result_cp = cp.empty(1, dtype=x_cp.dtype)  # Pre-allocate output\\n"
            prompt += "cublas.dot(x_cp, y_cp, result_cp)  # Result goes to result_cp\\n"
            prompt += "ref_out = from_dlpack(result_cp.toDlpack())\\n"
            prompt += "```\\n"
        elif operation in ['dotu', 'dotc']:
            prompt += f"**{operation.upper()}** - `cublas.{operation}(x, y, out)` for complex dot:\\n"
            prompt += "```python\\n"
            prompt += f"# Correct usage - must pre-allocate output buffer\\n"
            prompt += f"result_cp = cp.empty(1, dtype=x_cp.dtype)\\n"
            prompt += f"cublas.{operation}(x_cp, y_cp, result_cp)\\n"
            prompt += "ref_out = from_dlpack(result_cp.toDlpack())\\n"
            prompt += "```\\n"
        elif operation == 'nrm2':
            prompt += "**NRM2** - `cublas.nrm2(x, out)` returns Euclidean norm to out buffer:\\n"
            prompt += "```python\\n"
            prompt += "# Correct usage - must pre-allocate output buffer\\n"
            prompt += "result_cp = cp.empty(1, dtype=x_cp.dtype)\\n"
            prompt += "cublas.nrm2(x_cp, result_cp)\\n"
            prompt += "ref_out = from_dlpack(result_cp.toDlpack())\\n"
            prompt += "```\\n"
        elif operation == 'asum':
            prompt += "**ASUM** - `cublas.asum(x, out)` returns sum of abs values to out buffer:\\n"
            prompt += "```python\\n"
            prompt += "# Correct usage - must pre-allocate output buffer\\n"
            prompt += "result_cp = cp.empty(1, dtype=x_cp.dtype)\\n"
            prompt += "cublas.asum(x_cp, result_cp)\\n"
            prompt += "ref_out = from_dlpack(result_cp.toDlpack())\\n"
            prompt += "```\\n"
        elif operation == 'scal':
            prompt += "**SCAL** - `cublas.scal(a, x)` modifies x in-place:\\n"
            prompt += "```python\\n"
            prompt += "cublas.scal(alpha, x_cp)  # x = alpha * x (in-place)\\n"
            prompt += "ref_out = from_dlpack(x_cp.toDlpack())\\n"
            prompt += "```\\n"
        elif operation == 'gemv':
            prompt += "**GEMV** - `cublas.gemv(transa, alpha, a, x, beta, y)` modifies y in-place and returns None:\\n"
            prompt += "```python\\n"
            prompt += "# Correct usage - gemv returns None, y is modified in-place!\\n"
            prompt += "cublas.gemv(transa, alpha, A_cp, x_cp, beta, y_cp)\\n"
            prompt += "ref_out = from_dlpack(y_cp.toDlpack())  # Return the modified y_cp\\n"
            prompt += "```\\n"
        elif operation in ['ger', 'geru', 'gerc']:
            prompt += f"**{operation.upper()}** - `cublas.{operation}(alpha, x, y)` returns rank-1 matrix:\\n"
            prompt += "```python\\n"
            prompt += f"# Correct usage - {operation} returns new matrix, no 'out' parameter\\n"
            prompt += f"result_cp = cublas.{operation}(alpha, x_cp, y_cp)\\n"
            prompt += "ref_out = from_dlpack(result_cp.toDlpack())\\n"
            prompt += "```\\n"
        elif operation == 'syrk':
            prompt += "**SYRK** - `cublas.syrk(trans, a, out=None, alpha=1.0, beta=0.0, lower=False)`:\\n"
            prompt += "```python\\n"
            prompt += "C_cp = cublas.syrk('N', A_cp, alpha=alpha, beta=beta, lower=False)\\n"
            prompt += "```\\n"
        elif operation == 'geam':
            prompt += "**GEAM** - `cublas.geam(transa, transb, alpha, a, beta, b, out=None)`:\\n"
            prompt += "```python\\n"
            prompt += "C_cp = cublas.geam('N', 'N', alpha, A_cp, beta, B_cp)\\n"
            prompt += "```\\n"
        elif operation == 'dgmm':
            prompt += "**DGMM** - `cublas.dgmm(side, a, x, out=None, incx=1)`:\\n"
            prompt += "```python\\n"
            prompt += "result_cp = cublas.dgmm('L', A_cp, x_cp)  # side: 'L' or 'R'\\n"
            prompt += "```\\n"
        elif operation == 'sbmv':
            prompt += "**SBMV** - `cublas.sbmv(uplo, alpha, a, x, beta, y)` modifies y in-place:\\n"
            prompt += "```python\\n"
            prompt += "# CuPy sbmv signature: sbmv(uplo, alpha, A, x, beta, y)\\n"
            prompt += "cublas.sbmv(uplo, alpha, A_cp, x_cp, beta, y_cp)  # y = alpha*A*x + beta*y\\n"
            prompt += "ref_out = from_dlpack(y_cp.toDlpack())\\n"
            prompt += "```\\n"
        elif operation in ['iamax', 'iamin']:
            prompt += f"**{operation.upper()}** - `cublas.{operation}(x)` returns index:\\n"
            prompt += "```python\\n"
            prompt += f"idx = cublas.{operation}(x_cp)  # Returns index (1-based)\\n"
            prompt += "ref_out = torch.tensor(idx, device='cuda')\\n"
            prompt += "```\\n"
        else:
            prompt += f"**{operation.upper()}** - `cublas.{operation}(...)`:\\n"
            prompt += "```python\\n"
            prompt += f"result_cp = cublas.{operation}(...)  # Use appropriate args\\n"
            prompt += "```\\n"
        prompt += "\\n"
        
        # Example
        prompt += "## Example: GEMM Implementation\\n"
        prompt += "```python\\n"
        prompt += "from sandbox.register import register\\n"
        prompt += "from flagbench.dataset import Autograd\\n"
        prompt += "import torch\\n"
        prompt += "import cupy as cp\\n"
        prompt += "from cupy import cublas\\n"
        prompt += "from torch.utils.dlpack import to_dlpack, from_dlpack\\n\\n"
        prompt += '@register("CUDA", "sgemm", has_backward=Autograd.disable, namespace="baseline")\\n'
        prompt += "def sgemm(A, B, C, alpha=1.0, beta=0.0):\\n"
        prompt += '    """CuPy cuBLAS baseline for sgemm: C = alpha*A@B + beta*C"""\\n'
        prompt += "    # Convert to CuPy\\n"
        prompt += "    A_cp = cp.from_dlpack(to_dlpack(A))\\n"
        prompt += "    B_cp = cp.from_dlpack(to_dlpack(B))\\n"
        prompt += "    \\n"
        prompt += "    # Call cuBLAS via CuPy\\n"
        prompt += "    if beta == 0.0:\\n"
        prompt += "        C_cp = cublas.gemm('N', 'N', A_cp, B_cp, alpha=alpha, beta=0.0)\\n"
        prompt += "    else:\\n"
        prompt += "        C_cp = cp.from_dlpack(to_dlpack(C))\\n"
        prompt += "        C_cp = cublas.gemm('N', 'N', A_cp, B_cp, out=C_cp, alpha=alpha, beta=beta)\\n"
        prompt += "    \\n"
        prompt += "    # Convert back and return\\n"
        prompt += "    ref_out = from_dlpack(C_cp.toDlpack())\\n"
        prompt += "    return ref_out\\n"
        prompt += "```\\n\\n"
        
        # Constraints
        prompt += "## Important Constraints\\n"
        prompt += "1. DO NOT include any explanations, markdown headers, or text outside the code block\\n"
        prompt += "2. Only output the complete Python function code\\n"
        prompt += "3. Wrap your code in ```python ... ``` code block\\n"
        prompt += "4. Include a concise docstring describing the operation\\n"
        prompt += "5. Always use `ref_out` variable before return (never `return expression` directly)\\n"
        prompt += "6. For scalar results, use `torch.tensor(result, device='cuda')` or `torch.tensor(result.item(), device='cuda')`\\n"
        prompt += "7. Handle in-place operations correctly (e.g., AXPY, SCAL modify input)\\n"
        prompt += f"8. Function name MUST be `{kernel_name}` (lowercase)\\n"
        prompt += f"9. Use `cublas.{operation}()` if available in CuPy, otherwise implement manually\\n\\n"
        
        prompt += f"Now generate the complete baseline function for `{kernel_name}`.\\n"
        
        return prompt
    
    def _init_data(self, cublas_func: Dict[str, Any]):
        """
        Initialize data for generation.
        
        Args:
            cublas_func: cuBLAS function schema
            
        Returns:
            Object with op_name attribute (used by parent class workflow)
        """
        # Create a simple wrapper object with required attributes
        class BaselineGenerateArgs:
            def __init__(self, cublas_func):
                self.cublas_func = cublas_func
                # Extract kernel name for op_name
                func_name = cublas_func['name']
                self.op_name = func_name.replace('cublas', '').replace('_v2', '').lower()
                self.operation = cublas_func['operation']
        
        return BaselineGenerateArgs(cublas_func)
    
    def post_process(self, results: list) -> list:
        """
        Post-process generated code.
        
        Args:
            results: List of generation results (can be tuples if failed)
            
        Returns:
            List of processed code strings
        """
        processed = []
        for result in results:
            # Handle failed generations (returns tuple: (False, work_obj, ''))
            if isinstance(result, tuple):
                success, work_obj, code = result
                if not success:
                    console.print(f"[red]✗ Generation failed for {work_obj.op_name}")
                    processed.append("")  # Empty string for failed generation
                    continue
            else:
                code = result
            
            console.rule("[bold blue]Raw Output from LLM")
            console.print(code, markup=False)
            console.rule("[bold blue]End of Raw Output")
            
            # Extract code from markdown block
            extracted = extract_first_code(code, ["python"])
            
            console.rule("[bold blue]Extracted Code Block")
            console.print(extracted if extracted else "⚠️  No code block found", markup=False)
            console.rule("[bold blue]End of Extracted Code Block")
            
            if extracted:
                processed.append(extracted)
            else:
                console.print("[yellow]Warning: Code extraction failed, using raw output")
                processed.append(code if isinstance(code, str) else "")
        
        return processed
