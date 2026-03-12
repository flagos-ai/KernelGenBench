# cuBLAS Triton Kernel 生成改进记录

## 改进概述

本文档记录了对 `script/cublas_cupy/generate_triton_cublas.py` 脚本的改进。

## 改进内容

### 1. 添加 `get_baseline_operators()` 函数

**功能**：读取 `src/flagbench/baseline/` 目录，自动获取所有有baseline文件的算子名。

```python
def get_baseline_operators() -> set:
    """
    Get the set of operator names from baseline directory.
    
    Returns:
        Set of operator names (e.g., {'sgemm', 'dgemm', 'caxpy', ...})
    """
    baseline_dir = PROJECT_ROOT / "src" / "flagbench" / "baseline"
    operators = set()
    
    if baseline_dir.exists():
        for f in baseline_dir.glob('*.py'):
            if f.name != '__init__.py':
                operators.add(f.stem)
    
    logger.info(f"Found {len(operators)} baseline operators")
    return operators
```

**效果**：自动识别47个有baseline文件的算子。

### 2. 添加 `filter_by_baseline()` 函数

**功能**：从cuBLAS函数列表中过滤出有baseline文件的函数。

```python
def filter_by_baseline(functions: List[Dict], baseline_ops: set) -> List[Dict]:
    """
    Filter functions to only include those that have baseline files.
    
    Args:
        functions: List of cuBLAS function dictionaries
        baseline_ops: Set of baseline operator names
    
    Returns:
        Filtered list of functions that have baseline files
    """
```

**效果**：支持只生成有baseline的算子，避免生成无用的算子。

### 3. 修改 `generate_kernels()` 函数

**功能**：支持 `name=baseline` 参数，与 `name=all` 并列。

**修改前**：
- `name=all` - 生成所有219个cuBLAS函数

**修改后**：
- `name=all` - 生成所有219个cuBLAS函数
- `name=baseline` - 只生成47个有baseline的算子
- `name=<operation>` - 生成特定operation（如 `gemm`, `axpy` 等）

```python
# Get the list of functions to process
if name.lower() == "all":
    functions_to_process = all_functions
    logger.info(f"Processing all {len(functions_to_process)} cuBLAS functions.")
elif name.lower() == "baseline":
    # Only generate for operators with baseline files
    baseline_ops = get_baseline_operators()
    functions_to_process = filter_by_baseline(all_functions, baseline_ops)
    logger.info(f"Processing {len(functions_to_process)} cuBLAS functions (baseline operators only).")
else:
    # Find functions matching the operation name
    matching_funcs = [f for f in all_functions if f['operation'].lower() == name.lower()]
```

### 4. Prompt中使用Baseline签名

**功能**：生成prompt时，从baseline文件提取函数签名和参数，强制模型使用相同的签名。

**实现位置**：`src/generator/triton_kernel_generator.py`

```python
def get_baseline_signature(self, kernel_name: str) -> str:
    """从baseline文件提取函数签名"""
    baseline_path = Path(f"src/flagbench/baseline/{kernel_name}.py")
    if not baseline_path.exists():
        return ""
    
    code = baseline_path.read_text()
    
    # 提取函数定义行
    for line in code.split('\n'):
        if f'def {kernel_name}(' in line:
            return line.strip()
    
    return ""

def get_baseline_params(self, kernel_name: str) -> list:
    """提取baseline参数名列表"""
    signature = self.get_baseline_signature(kernel_name)
    if not signature:
        return []
    
    # 提取括号内的参数
    match = re.search(r'def\s+\w+\s*\((.*?)\):', signature)
    if match:
        params_str = match.group(1)
        params = [p.strip().split(':')[0].strip() for p in params_str.split(',') if p.strip()]
        return params
    return []
```

**效果**：Prompt中包含Baseline函数签名，强制模型使用相同的参数名和顺序。

## 使用方法

### 生成所有cuBLAS函数
```bash
python script/cublas_cupy/generate_triton_cublas.py \
  --name all \
  --output-dir ./output_all \
  --model-name gpt-5 \
  --num-workers 10
```

### 只生成baseline中的47个算子
```bash
python script/cublas_cupy/generate_triton_cublas.py \
  --name baseline \
  --output-dir ./output_baseline \
  --model-name gpt-5 \
  --num-workers 10
```

### 生成特定operation
```bash
python script/cublas_cupy/generate_triton_cublas.py \
  --name gemm \
  --output-dir ./output_gemm \
  --model-name gpt-5 \
  --num-workers 10
```

## Baseline目录

当前 `src/flagbench/baseline/` 包含47个算子：

| 分类 | 算子 |
|------|------|
| axpy | caxpy, daxpy, saxpy, zaxpy |
| dot | cdotc, cdotu, ddot, sdot, zdotc, zdotu |
| scal | cscal, csscal, dscal, sscal, zscal, zdscal |
| gemm | cgemm, dgemm, hgemm, sgemm, zgemm |
| gemv | cgemv, dgemv, sgemv, zgemv |
| geam | cgeam, dgeam, sgeam, zgeam |
| syrk | csyrk, dsyrk, ssyrk, zsyrk |
| ger | cgerc, cgeru, dger, sger, zgerc, zgeru |
| dgmm | cdgmm, ddgmm, sdgmm, zdgmm |
| asum | dasum, sasum |
| nrm2 | dnrm2, snrm2 |
| sbmv | dsbmv, ssbmv |

## 总结

改进后的脚本：
1. **更灵活** - 支持 `all`, `baseline`, 或具体operation名称
2. **更精准** - 只生成有baseline的47个算子，避免无用工作
3. **更规范** - Prompt中使用baseline签名，确保模型生成代码的一致性
