#!/usr/bin/env python3
"""
简单示例：直接打印 cuBLAS GEMM 的 prompt

运行方法：
    cd /share/project/zpy/flagbench
    python script/cublas_cupy/simple_example.py
"""

import json
import os

# 获取 schema 路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEMA_PATH = os.path.join(SCRIPT_DIR, 'cublas_ops.json')


def load_function_schema(function_name: str):
    """加载指定 cuBLAS 函数的 schema"""
    with open(SCHEMA_PATH, 'r') as f:
        data = json.load(f)
    
    for func in data['functions']:
        if func['name'] == function_name:
            return func
    
    raise ValueError(f"Function {function_name} not found")


def generate_simple_prompt(cublas_func_name: str):
    """生成简化版的 prompt"""
    schema = load_function_schema(cublas_func_name)
    
    prompt = f"""# 任务：为 {cublas_func_name} 生成 CuPy baseline 函数

## cuBLAS 函数信息
- 函数名: {schema['name']}
- 操作类型: {schema['operation']}
- 数据类型: {schema['dtype']}

## 参数列表:
"""
    
    for arg in schema['args']:
        prompt += f"- {arg['name']} ({arg['role']}): {arg['type']}\n"
    
    prompt += f"""
## 要求
生成一个 Python 函数 `{schema['operation']}_cublas_baseline`，实现以下功能：
1. 接收 PyTorch tensor 作为输入
2. 使用 DLPack 转换为 CuPy array（零拷贝）
3. 调用 CuPy 的高层 API（如 cp.dot()）
4. 将结果转回 PyTorch tensor

## 示例代码框架：
```python
def {schema['operation']}_cublas_baseline(...):
    import cupy as cp
    from torch.utils.dlpack import to_dlpack, from_dlpack
    
    # 转换输入
    # TODO: 转换 PyTorch tensors 为 CuPy arrays
    
    # 调用 CuPy
    # TODO: 调用合适的 CuPy 函数
    
    # 转回 PyTorch
    return from_dlpack(result.toDlpack())
```

请生成完整的函数代码。
"""
    
    return prompt


def main():
    print("="*70)
    print("cuBLAS Baseline Prompt 生成器 - 简化版")
    print("="*70)
    
    # 测试几个常用函数
    functions_to_test = [
        "cublasSgemm_v2",
        "cublasSaxpy_v2", 
        "cublasSdot_v2",
        "cublasSscal_v2"
    ]
    
    for func_name in functions_to_test:
        print(f"\n{'='*70}")
        print(f"函数: {func_name}")
        print('='*70)
        
        try:
            prompt = generate_simple_prompt(func_name)
            print(prompt)
            
            # 保存到文件
            output_file = f"/tmp/{func_name}_prompt.txt"
            with open(output_file, 'w') as f:
                f.write(prompt)
            print(f"\n💾 已保存到: {output_file}")
            
        except ValueError as e:
            print(f"❌ 错误: {e}")
        
        # 只打印第一个，避免输出太长
        break
    
    print("\n" + "="*70)
    print("✅ 完成！")
    print("="*70)


if __name__ == "__main__":
    main()
