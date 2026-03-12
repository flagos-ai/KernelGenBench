#!/usr/bin/env python3
"""
生成 cuBLAS test_func（parametrize 格式的准确性+性能测试）
使用金山云 API
"""

import os
import sys
import json
import requests
from pathlib import Path
from datetime import datetime

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# 动态加载 config
import importlib.util
config_path = Path(__file__).parent.parent / "cublas_c_api_config.py"
spec = importlib.util.spec_from_file_location("cublas_config", config_path)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
CUBLAS_C_API_CONFIG = config_module.CUBLAS_C_API_CONFIG

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
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    print(f"Calling KSYun API (model={model})...")
    response = requests.post(KSYUN_API_URL, headers=headers, json=data, timeout=180)
    response.raise_for_status()
    result = response.json()
    return result['choices'][0]['message']['content']


# ============================================================
# 参数模板：根据 base_op 类型提供推荐的参数组合
# ============================================================
PARAM_TEMPLATES = {
    'axpy': {
        'n': [1, 32, 71, 160, 497, 1024, 4113, 4096, 5333],
        'alpha': [1.0, 0.0, 0.001, -0.999, 100.001, -111.999, 0.5, -0.5],
        'incx': [1, 2, 3],
        'incy': [1, 2, 3],
        'perf_threshold': 'n >= 1024',
    },
    'scal': {
        'n': [1, 32, 71, 160, 497, 1024, 4113, 4096, 5333],
        'alpha': [1.0, 0.0, 0.001, -0.999, 100.001, -111.999, 0.5, -0.5],
        'incx': [1, 2, 3],
        'perf_threshold': 'n >= 1024',
    },
    'gemm': {
        'M_N_K': [(1, 1, 1), (16, 16, 16), (32, 64, 16), (64, 32, 48),
                  (128, 256, 64), (256, 128, 128), (17, 33, 65),
                  (128, 128, 128), (256, 256, 256)],
        'alpha_beta': [(1.0, 0.0), (0.0, 1.0), (0.5, 0.5),
                       (-1.0, 1.0), (2.0, -0.5), (0.001, 0.999)],
        'trans': [('N', 'N'), ('N', 'T'), ('T', 'N'), ('T', 'T')],
        'batchCount': [1, 2, 4, 8],
        'perf_threshold': 'M >= 128 and N >= 128 and K >= 64',
    },
    'gemv': {
        'M_N': [(16, 16), (32, 64), (64, 128), (128, 64),
                (128, 128), (128, 256), (256, 128), (256, 256), (17, 33)],
        'alpha_beta': [(1.0, 0.0), (0.0, 1.0), (0.5, 0.5),
                       (-1.0, 1.0), (1.0, 0.5), (0.001, 0.999)],
        'trans': ['N', 'T'],
        'batchCount': [1, 2, 4, 8],
        'perf_threshold': 'M >= 128 and N >= 64',
    },
    'dot': {
        'n': [1, 32, 71, 160, 497, 1024, 4113, 4096, 5333],
        'incx': [1, 2, 3],
        'incy': [1, 2, 3],
        'perf_threshold': 'n >= 1024',
    },
}


# ============================================================
# Shot 示例：cublasSscal_v2 的完整 test_func（作为 few-shot）
# ============================================================
SHOT_SCAL = '''import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
import torch

@label("cublasSscal_v2")
@parametrize("n", [1, 32, 71, 160, 497, 1024, 4113, 4096, 5333])
@parametrize("alpha", [1.0, 0.0, 0.001, -0.999, 100.001, -111.999, 0.5, -0.5])
@parametrize("incx", [1, 2, 3])
@parametrize("dtype", [torch.float32])
def test_accuracy_cublasSscal_v2(n, alpha, incx, dtype):
    x = torch.randn(n * incx, dtype=dtype, device='cuda')
    x_ref = x.clone()
    x_act = x.clone()

    ref_out = flagbench.baseline.cublasSscal_v2(n, alpha, x_ref, incx)
    act_out = flagbench.triton.cublasSscal_v2(n, alpha, x_act, incx)

    assert_close(act_out, ref_out, dtype)

    # Performance Test
    from sandbox.utils.accuracy_utils import CustomBenchmarkResult
    if n < 1024:
        return None

    x_bench = torch.randn(n * incx, dtype=dtype, device='cuda')
    for _ in range(10):
        _ = flagbench.baseline.cublasSscal_v2(n, alpha, x_bench.clone(), incx)
        _ = flagbench.triton.cublasSscal_v2(n, alpha, x_bench.clone(), incx)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = flagbench.baseline.cublasSscal_v2(n, alpha, x_bench.clone(), incx)
    end_event.record()
    torch.cuda.synchronize()
    ms_baseline = start_event.elapsed_time(end_event) / 100

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = flagbench.triton.cublasSscal_v2(n, alpha, x_bench.clone(), incx)
    end_event.record()
    torch.cuda.synchronize()
    ms_triton = start_event.elapsed_time(end_event) / 100

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
'''

SHOT_GEMM = '''import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
import torch

@label("cublasSgemmStridedBatched")
@parametrize("M, N, K", [
    (1, 1, 1), (16, 16, 16), (32, 64, 16), (64, 32, 48),
    (128, 256, 64), (256, 128, 128), (17, 33, 65),
    (128, 128, 128), (256, 256, 256),
])
@parametrize("alpha, beta", [
    (1.0, 0.0), (0.0, 1.0), (0.5, 0.5),
    (-1.0, 1.0), (2.0, -0.5), (0.001, 0.999),
])
@parametrize("transa, transb", [("N", "N"), ("N", "T"), ("T", "N"), ("T", "T")])
@parametrize("batchCount", [1, 2, 4, 8])
@parametrize("dtype", [torch.float32])
def test_accuracy_cublasSgemmStridedBatched(M, N, K, alpha, beta, transa, transb, batchCount, dtype):
    A_shape = (batchCount, K, M) if transa == 'T' else (batchCount, M, K)
    B_shape = (batchCount, N, K) if transb == 'T' else (batchCount, K, N)
    A = torch.randn(A_shape, dtype=dtype, device='cuda')
    B = torch.randn(B_shape, dtype=dtype, device='cuda')
    C = torch.randn(batchCount, M, N, dtype=dtype, device='cuda')
    strideA = A.shape[1] * A.shape[2]
    strideB = B.shape[1] * B.shape[2]
    strideC = M * N
    lda = A.shape[1]
    ldb = B.shape[1]
    ldc = M
    C_ref = C.clone()
    C_act = C.clone()

    ref_out = flagbench.baseline.cublasSgemmStridedBatched(
        transa, transb, M, N, K, alpha, A, lda, strideA,
        B, ldb, strideB, beta, C_ref, ldc, strideC, batchCount)
    act_out = flagbench.triton.cublasSgemmStridedBatched(
        transa, transb, M, N, K, alpha, A, lda, strideA,
        B, ldb, strideB, beta, C_act, ldc, strideC, batchCount)
    assert_close(act_out, ref_out, dtype, reduce_dim=K)

    from sandbox.utils.accuracy_utils import CustomBenchmarkResult
    if M < 128 or N < 128 or K < 64:
        return None

    A_bench = torch.randn(A_shape, dtype=dtype, device='cuda')
    B_bench = torch.randn(B_shape, dtype=dtype, device='cuda')
    for _ in range(10):
        C_w = torch.randn(batchCount, M, N, dtype=dtype, device='cuda')
        _ = flagbench.baseline.cublasSgemmStridedBatched(
            transa, transb, M, N, K, alpha, A_bench, lda, strideA,
            B_bench, ldb, strideB, beta, C_w.clone(), ldc, strideC, batchCount)
        _ = flagbench.triton.cublasSgemmStridedBatched(
            transa, transb, M, N, K, alpha, A_bench, lda, strideA,
            B_bench, ldb, strideB, beta, C_w.clone(), ldc, strideC, batchCount)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    C_b = torch.randn(batchCount, M, N, dtype=dtype, device='cuda')
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = flagbench.baseline.cublasSgemmStridedBatched(
            transa, transb, M, N, K, alpha, A_bench, lda, strideA,
            B_bench, ldb, strideB, beta, C_b.clone(), ldc, strideC, batchCount)
    end_event.record()
    torch.cuda.synchronize()
    ms_baseline = start_event.elapsed_time(end_event) / 100

    C_t = torch.randn(batchCount, M, N, dtype=dtype, device='cuda')
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = flagbench.triton.cublasSgemmStridedBatched(
            transa, transb, M, N, K, alpha, A_bench, lda, strideA,
            B_bench, ldb, strideB, beta, C_t.clone(), ldc, strideC, batchCount)
    end_event.record()
    torch.cuda.synchronize()
    ms_triton = start_event.elapsed_time(end_event) / 100

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
'''


# ============================================================
# Helper functions
# ============================================================
def _get_shot_for_op(base_op):
    """根据 base_op 类型选择合适的 shot 示例"""
    if base_op in ('gemm',):
        return SHOT_GEMM
    else:
        return SHOT_SCAL


def _get_param_template(base_op):
    """获取参数模板"""
    base_op = base_op.lower()
    if base_op in PARAM_TEMPLATES:
        return PARAM_TEMPLATES[base_op]
    # 默认使用 scal 模板
    return PARAM_TEMPLATES['scal']


# ============================================================
# Prompt builder
# ============================================================
def build_prompt(func_name, config, baseline_code, baseline_signature):
    """构建生成 test_func 的 prompt"""
    base_op = config['base_op'].lower()
    dtype = config['dtype']
    shot = _get_shot_for_op(base_op)
    param_template = _get_param_template(base_op)

    # 构建参数模板描述
    param_desc = json.dumps(param_template, indent=2, default=str)

    prompt = f"""You are an expert in CUDA testing. Generate a parametrized accuracy + performance test function for the cuBLAS operation: **{func_name}**

## Function Info
- Function Name: {func_name}
- Base Operation: {config['base_op']}
- Data Type: {dtype}
- BLAS Level: {config['level']}
- Description: {config['description']}

## Baseline Function Signature
```python
{baseline_signature}
```

## Baseline Implementation (for reference)
```python
{baseline_code}
```

## Recommended Parameter Combinations
{param_desc}

## RULES (MUST follow exactly)

### 1. Imports and Structure
- Use EXACTLY these imports:
```python
import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
import torch
```
- Do NOT import baseline directly. Use `flagbench.baseline.{func_name}` and `flagbench.triton.{func_name}`.

### 2. Parametrize Rules
- Use `@label("{func_name}")` decorator
- Use `@parametrize` for EACH parameter dimension
- Include rich parameter values: edge cases (1, 32), non-aligned (71, 497, 17, 33), power-of-2 (1024, 4096), large (5333)
- For alpha/beta: include 1.0, 0.0, fractional (0.5, 0.001), negative (-0.999, -1.0), large (100.001)
- For complex types: use complex values like (1+0j), (0.5+0.3j), (0.001+0.001j)
- For trans: include both 'N' and 'T' (and 'C' for complex conjugate if applicable)
- For batchCount (if batched): [1, 2, 4, 8]
- For incx/incy (if applicable): [1, 2, 3]

### 3. Test Function Body
- Function name: `test_accuracy_{func_name}`
- Create input tensors with correct dtype and device='cuda'
- Clone output tensors for ref and act
- Call `flagbench.baseline.{func_name}(...)` for reference
- Call `flagbench.triton.{func_name}(...)` for actual
- Use `assert_close(act_out, ref_out, dtype)` for comparison
- For GEMM/GEMV with reduce dimension, use `assert_close(act_out, ref_out, dtype, reduce_dim=K)` or similar

### 4. Performance Test (MUST include)
- After accuracy test, add performance benchmark
- Import `CustomBenchmarkResult` from `sandbox.utils.accuracy_utils`
- Skip small sizes (return None if below threshold)
- Warmup: 10 iterations of both baseline and triton
- Benchmark: use `torch.cuda.Event(enable_timing=True)`, 100 iterations each
- Return `CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)`

### 5. CRITICAL Performance Test Pattern
```python
    from sandbox.utils.accuracy_utils import CustomBenchmarkResult
    if <size_too_small>:
        return None

    # warmup
    for _ in range(10):
        _ = flagbench.baseline.{func_name}(...)
        _ = flagbench.triton.{func_name}(...)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = flagbench.baseline.{func_name}(...)
    end_event.record()
    torch.cuda.synchronize()
    ms_baseline = start_event.elapsed_time(end_event) / 100

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = flagbench.triton.{func_name}(...)
    end_event.record()
    torch.cuda.synchronize()
    ms_triton = start_event.elapsed_time(end_event) / 100

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
```

## Complete Example (follow this pattern exactly)
```python
{shot}
```

## Output
Generate ONLY the Python code. Use ```python ... ``` code block. No explanations.
"""
    return prompt


# ============================================================
# Generate single / batch
# ============================================================
def generate_single(func_name, model="mog-1", output_dir=None):
    """为单个 cuBLAS 函数生成 test_func"""
    print("=" * 60)
    print(f"Generating test_func for: {func_name}")
    print("=" * 60)

    # 1. 获取 config
    if func_name not in CUBLAS_C_API_CONFIG:
        print(f"Error: {func_name} not in config")
        return False
    config = CUBLAS_C_API_CONFIG[func_name]

    # 2. 读取 baseline 代码
    baseline_path = (PROJECT_ROOT / "src" / "flagbench" /
                     "dataset" / "baseline" / "cublas_ctypes" /
                     f"{func_name}.py")
    if not baseline_path.exists():
        print(f"Error: baseline not found: {baseline_path}")
        return False

    baseline_code = baseline_path.read_text()

    # 提取函数签名
    baseline_signature = ""
    for line in baseline_code.split('\n'):
        if f'def {func_name}(' in line:
            baseline_signature = line.strip()
            break

    print(f"  Signature: {baseline_signature}")

    # 3. 构建 prompt
    prompt = build_prompt(
        func_name, config, baseline_code, baseline_signature)
    print(f"  Prompt length: {len(prompt)} chars")

    # 4. 调用 API
    try:
        raw = call_ksyun_api(prompt, model=model)
        if not raw or not raw.strip():
            print("Error: API returned empty")
            return False

        # 清理 markdown 代码块
        code = raw.strip()
        if code.startswith("```python"):
            code = code[len("```python"):].strip()
        elif code.startswith("```"):
            code = code[3:].strip()
        if code.endswith("```"):
            code = code[:-3].strip()

        if not code.strip():
            print("Error: empty after cleaning")
            return False

    except Exception as e:
        print(f"Error: API call failed: {e}")
        return False

    # 5. 保存
    if output_dir:
        out_path = Path(output_dir)
    else:
        out_path = Path(__file__).parent

    out_path.mkdir(parents=True, exist_ok=True)
    code_file = out_path / f"test_{func_name}.py"
    code_file.write_text(code)

    prompt_file = out_path / f"test_{func_name}_prompt.txt"
    prompt_file.write_text(prompt)

    print(f"  Saved: {code_file}")
    print(f"  Prompt: {prompt_file}")
    return True


def main():
    """主函数"""
    import argparse
    from concurrent.futures import ThreadPoolExecutor, as_completed

    parser = argparse.ArgumentParser(
        description="Generate test_func for cuBLAS operators")
    parser.add_argument("--name", type=str,
                        help="cuBLAS function name (e.g., cublasSscal_v2)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory path")
    parser.add_argument("--model", type=str, default="mog-1",
                        help="Model name")
    parser.add_argument("--batch", action="store_true",
                        help="Generate for all configured functions")
    parser.add_argument("--workers", type=int, default=5,
                        help="Number of concurrent workers for batch mode")
    args = parser.parse_args()

    if args.batch:
        func_names = list(CUBLAS_C_API_CONFIG.keys())

        if args.output_dir:
            output_base = Path(args.output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output_base = (Path(__file__).parent /
                           f"output_test_func_{args.model}_{timestamp}")

        output_base.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print(f"Batch mode: generating test_func for {len(func_names)} functions")
        print(f"Output: {output_base}")
        print(f"Workers: {args.workers}")
        print("=" * 60)

        results = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_func = {
                executor.submit(
                    generate_single, fn, args.model, str(output_base)
                ): fn
                for fn in func_names
            }
            for i, future in enumerate(as_completed(future_to_func), 1):
                fn = future_to_func[future]
                try:
                    success = future.result()
                    results.append((fn, success))
                    status = "OK" if success else "FAIL"
                    print(f"[{i}/{len(func_names)}] {status} {fn}")
                except Exception as e:
                    results.append((fn, False))
                    print(f"[{i}/{len(func_names)}] FAIL {fn}: {e}")

        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        ok = sum(1 for _, s in results if s)
        print(f"Success: {ok}/{len(func_names)}")
        for fn, s in sorted(results):
            print(f"  {'OK' if s else 'FAIL'} {fn}")
        return

    if not args.name:
        print("Error: specify --name or use --batch")
        sys.exit(1)

    success = generate_single(
        args.name, model=args.model, output_dir=args.output_dir)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
