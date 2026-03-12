#!/usr/bin/env python3
"""
测试脚本：使用新的 C API prompt 生成3个 baseline 函数
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# 直接读取配置文件，避免导入依赖
config_file = PROJECT_ROOT / "src/flagbench/dataset/cublas_c_api_config.py"
with open(config_file, 'r') as f:
    content = f.read()
exec(content)

# 直接读取 BaselinePromptBuilder 类代码
prompt_builder_file = PROJECT_ROOT / "src/generator/baseline/baseline_prompt_builder.py"
with open(prompt_builder_file, 'r') as f:
    builder_code = f.read()
exec(builder_code)

def main():
    # 选择3个测试函数
    test_functions = [
        'cublasSaxpy_v2',   # Level 1: y = alpha * x + y
        'cublasSgemm_v2',   # Level 3: C = alpha * A @ B + beta * C
        'cublasSdot_v2',    # Level 1: result = x^T * y
    ]

    builder = BaselinePromptBuilder()

    print("=" * 80)
    print("生成 3 个测试 baseline 函数的 Prompt")
    print("=" * 80)

    for func_name in test_functions:
        if func_name not in CUBLAS_C_API_CONFIG:
            print(f"\n❌ 函数 {func_name} 不在配置中")
            continue

        config = CUBLAS_C_API_CONFIG[func_name]
        kernel_name = func_name.replace('cublas', '').replace('_v2', '').lower()

        print(f"\n{'=' * 80}")
        print(f"函数: {func_name} -> {kernel_name}")
        print(f"操作: {config['base_op']}, 类型: {config['dtype']}, Level: {config['level']}")
        print(f"{'=' * 80}\n")

        # 生成 prompt
        prompt = builder.build_prompt(func_name, config)

        # 保存 prompt 到文件
        prompt_file = PROJECT_ROOT / f"test_prompt_{kernel_name}.txt"
        with open(prompt_file, 'w') as f:
            f.write(prompt)

        print(f"✓ Prompt 已保存到: {prompt_file}")

        # 打印 prompt 的前500个字符
        print(f"\nPrompt 预览 (前500字符):")
        print("-" * 80)
        print(prompt[:500])
        print("...")
        print("-" * 80)

if __name__ == "__main__":
    main()
