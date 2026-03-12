#!/usr/bin/env python3
"""使用金山云 API 为 cuBLAS baseline 生成 Triton 实现"""

import os
import sys
from pathlib import Path
import requests
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# 金山云 API 配置
KSYUN_API_URL = "https://kspmas.ksyun.com/v1/chat/completions"
KSYUN_API_KEY = "8407460c-9a3d-4a32-bb0d-43e91a74304f"

def call_ksyun_api(prompt, model="mog-1", temperature=0.0, max_tokens=8192):
    """调用金山云 API"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {KSYUN_API_KEY}"
    }

    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    print(f"Calling KSYun API (model={model})...")
    response = requests.post(KSYUN_API_URL, headers=headers, json=data, timeout=180)
    response.raise_for_status()
    result = response.json()
    return result['choices'][0]['message']['content']

def generate_triton_for_cublas(cublas_func_name):
    """为指定的 cuBLAS 函数生成 Triton 实现"""

    # 读取 baseline 代码
    baseline_path = PROJECT_ROOT / "src/flagbench/dataset/baseline/cublas_ctypes" / f"{cublas_func_name}.py"

    if not baseline_path.exists():
        print(f"✗ Baseline file not found: {baseline_path}")
        return None

    baseline_code = baseline_path.read_text()

    # 提取函数签名
    baseline_signature = ""
    for line in baseline_code.split('\n'):
        if f'def {cublas_func_name}(' in line:
            baseline_signature = line.strip()
            break

    print(f"✓ Loaded baseline: {cublas_func_name}")
    print(f"  Signature: {baseline_signature}")

    return baseline_code, baseline_signature

def build_prompt(cublas_func_name, baseline_code, baseline_signature):
    """构建 prompt"""
    # 提取 kernel 名称（去掉 cublas 前缀和 _v2 后缀）
    kernel_name = cublas_func_name.replace('cublas', '').replace('_v2', '').lower()

    prompt = f"""You are an expert GPU programmer proficient in Triton. Your task is to implement a Triton kernel that replicates the functionality of a cuBLAS operation.

## Task Overview
Implement a Triton kernel for the cuBLAS operation: **{cublas_func_name}**
- Kernel name: `{kernel_name}`

## Baseline Reference (cuBLAS C API Wrapper)
The following baseline implementation uses ctypes to call the cuBLAS C API.
**CRITICAL**: Your Triton kernel MUST have the EXACT SAME function signature as the baseline.

### Function Signature (MUST MATCH EXACTLY)
```python
{baseline_signature}
```

### Complete Baseline Implementation
```python
{baseline_code}
```

## Implementation Requirements

### 1. Function Signature (CRITICAL)
Your Triton kernel MUST use the EXACT SAME function signature as the baseline:
- Same function name: `{cublas_func_name}`
- Same parameter names and order
- Same return type

### 2. Testing Environment
Your implementation will be tested as follows:
```python
# Baseline (cuBLAS C API wrapper)
from flagbench.dataset.baseline.cublas_ctypes.{cublas_func_name} import {cublas_func_name} as baseline_{cublas_func_name}
ref_out = baseline_{cublas_func_name}(...)

# Your Triton implementation
import flagbench
act_out = flagbench.triton.{cublas_func_name}(...)

# Accuracy verification
assert_close(act_out, ref_out, dtype)
```
**If the function signature doesn't match, the test will fail immediately.**

### 3. Implementation Guidelines
- Use Triton's `@triton.jit` decorator for GPU kernels
- Implement proper grid/block configuration for parallel execution
- Handle tensor shapes, strides, and data types correctly
- Ensure numerical accuracy matches cuBLAS (within reasonable tolerance)
- Return the same output tensor(s) as the baseline

### 4. Performance Guidelines
- Choose BLOCK_SIZE as power-of-2 (e.g. 128, 256, 1024) for optimal GPU utilization
- Use `tl.constexpr` for compile-time constants (BLOCK_SIZE, etc.)
- Avoid unnecessary tensor copies — modify output tensor in-place when possible
- For matrix operations: use tiled computation with proper shared memory usage
- For vector operations: use simple 1D grid with each program handling BLOCK_SIZE elements
- Minimize Python-level overhead in the wrapper function (no runtime checks, no unnecessary allocations)

## Output Format
Generate ONLY the Python code for the Triton kernel implementation.
- Use ```python ... ``` code block format
- Include all necessary imports (torch, triton, etc.)
- Include the wrapper function with the exact signature from baseline
- Include the Triton kernel(s) decorated with @triton.jit
- Do NOT include explanations or test code

Now, generate the Triton kernel implementation:
"""
    return prompt

def generate_single(cublas_func_name, model="mog-1", output_dir=None):
    """为单个cuBLAS函数生成Triton实现"""
    print("=" * 60)
    print(f"Generating Triton kernel for: {cublas_func_name}")
    print("=" * 60)

    # 1. 加载 baseline
    result = generate_triton_for_cublas(cublas_func_name)
    if result is None:
        return False

    baseline_code, baseline_signature = result

    # 2. 构建 prompt
    prompt = build_prompt(cublas_func_name, baseline_code, baseline_signature)
    print(f"\n✓ Prompt built ({len(prompt)} chars)")

    # 3. 调用 API
    try:
        triton_code_raw = call_ksyun_api(prompt, model=model)
        print(f"\n✓ API call successful")
        print(f"  Raw response length: {len(triton_code_raw)} chars")

        # 检查返回内容是否为空
        if not triton_code_raw or len(triton_code_raw.strip()) == 0:
            print(f"\n✗ API returned empty content")
            return False

        # 清理代码：去掉 markdown 代码块标记
        triton_code = triton_code_raw.strip()
        if triton_code.startswith("```python"):
            triton_code = triton_code[len("```python"):].strip()
        elif triton_code.startswith("```"):
            triton_code = triton_code[3:].strip()

        if triton_code.endswith("```"):
            triton_code = triton_code[:-3].strip()

        # 再次检查清理后的内容
        if not triton_code or len(triton_code.strip()) == 0:
            print(f"\n✗ API returned empty content after cleaning")
            print(f"  Raw response preview: {triton_code_raw[:200]}")
            return False

        print(f"  Cleaned code length: {len(triton_code)} chars")

    except Exception as e:
        print(f"\n✗ API call failed: {e}")
        return False

    # 4. 创建输出目录结构：每个算子一个文件夹
    if output_dir:
        base_dir = Path(output_dir)
    else:
        base_dir = Path(__file__).parent / "output"

    # 为每个算子创建单独的文件夹
    func_dir = base_dir / cublas_func_name
    func_dir.mkdir(parents=True, exist_ok=True)

    # 保存 prompt.txt
    prompt_path = func_dir / "prompt.txt"
    prompt_path.write_text(prompt)

    # 保存 triton 实现
    triton_path = func_dir / f"{cublas_func_name}_triton.py"
    triton_path.write_text(triton_code)

    print(f"\n✓ Prompt saved to: {prompt_path}")
    print(f"✓ Triton code saved to: {triton_path}")
    return True

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate Triton kernel for cuBLAS baseline")
    parser.add_argument("--name", type=str, help="cuBLAS function name (e.g., cublasSaxpy_v2)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory path")
    parser.add_argument("--model", type=str, default="mog-1", help="Model name")
    parser.add_argument("--batch", action="store_true", help="Generate for all 9 baselines")
    parser.add_argument("--workers", type=int, default=10, help="Number of concurrent workers")
    args = parser.parse_args()

    # 批量生成模式
    if args.batch:
        BATCH_FUNCTIONS = [
            "cublasCgemmStridedBatched",
            "cublasCgemvStridedBatched",
            "cublasDgemmStridedBatched",
            "cublasDgemvStridedBatched",
            "cublasHgemmStridedBatched",
            "cublasSgemmStridedBatched",
            "cublasSgemvStridedBatched",
            "cublasZgemmStridedBatched",
            "cublasZgemvStridedBatched"
        ]

        # 创建带时间戳的输出目录
        if args.output_dir:
            output_base = Path(args.output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output_base = Path(__file__).parent / "output" / f"triton_cublas_{args.model}_temp_0.0_{timestamp}"

        output_base.mkdir(parents=True, exist_ok=True)

        print("=" * 80)
        print("批量生成模式：为9个cuBLAS baseline生成Triton实现（并发）")
        print(f"输出目录: {output_base}")
        print(f"并发数: {args.workers}")
        print("=" * 80)
        print()

        # 使用线程池并发生成
        results = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            # 提交所有任务
            future_to_func = {
                executor.submit(generate_single, func_name, args.model, str(output_base)): func_name
                for func_name in BATCH_FUNCTIONS
            }

            # 收集结果
            for i, future in enumerate(as_completed(future_to_func), 1):
                func_name = future_to_func[future]
                try:
                    success = future.result()
                    results.append((func_name, success))
                    status = "✓" if success else "✗"
                    print(f"[{i}/{len(BATCH_FUNCTIONS)}] {status} {func_name}")
                except Exception as e:
                    results.append((func_name, False))
                    print(f"[{i}/{len(BATCH_FUNCTIONS)}] ✗ {func_name}: {e}")

        # 汇总结果
        print("\n" + "=" * 80)
        print("生成结果汇总")
        print("=" * 80)
        success_count = sum(1 for _, success in results if success)
        print(f"\n成功: {success_count}/{len(BATCH_FUNCTIONS)}")
        print(f"失败: {len(BATCH_FUNCTIONS) - success_count}/{len(BATCH_FUNCTIONS)}")
        print("\n详细结果:")
        for func_name, success in sorted(results, key=lambda x: x[0]):
            status = "✓" if success else "✗"
            print(f"  {status} {func_name}")
        return

    # 单个生成模式
    if not args.name:
        print("错误: 请指定 --name 或使用 --batch 进行批量生成")
        return

    success = generate_single(args.name, model=args.model, output_dir=args.output_dir)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()

