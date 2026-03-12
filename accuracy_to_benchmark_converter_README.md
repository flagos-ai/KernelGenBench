# Accuracy to Benchmark Converter

## 功能描述

`accuracy_to_benchmark_converter.py` 是一个将正确性测试函数转换为加速比测试函数的工具脚本。

## 主要接口

### 1. `convert_accuracy_to_benchmark_content(content: str) -> str`

**入参：**
- `content: str` - 正确性测试文件的内容字符串

**出参：**
- `str` - 转换后的加速比测试文件内容字符串（包含完整的导入和测试函数）

**使用场景：** 适合集成到其他工具中，直接处理字符串内容，无需文件IO。

### 2. `convert_accuracy_to_benchmark_file(input_file: str, output_file: str) -> None`

**入参：**
- `input_file: str` - 输入的正确性测试文件路径
- `output_file: str` - 输出的加速比测试文件路径

**出参：**
- `None` - 直接写入文件，无返回值

**使用场景：** 适合命令行使用或批量文件转换。

## 集成到其他工具

### 方式一：内容转换（推荐）

```python
from accuracy_to_benchmark_converter import convert_accuracy_to_benchmark_content

# 读取正确性测试文件内容
with open('test_accuracy.py', 'r') as f:
    accuracy_content = f.read()

# 转换为加速比测试内容
benchmark_content = convert_accuracy_to_benchmark_content(accuracy_content)

# 写入新文件或进一步处理
with open('test_benchmark.py', 'w') as f:
    f.write(benchmark_content)
```

### 方式二：直接文件转换

```python
from accuracy_to_benchmark_converter import convert_accuracy_to_benchmark_file

# 直接转换文件
convert_accuracy_to_benchmark_file('test_accuracy.py', 'test_benchmark.py')
```

### 方式三：批量处理

```python
from accuracy_to_benchmark_converter import convert_accuracy_to_benchmark_content
import os

def batch_convert(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith('_accuracy.py'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename.replace('_accuracy.py', '_benchmark.py'))
            
            with open(input_path, 'r') as f:
                content = f.read()
            
            benchmark_content = convert_accuracy_to_benchmark_content(content)
            
            with open(output_path, 'w') as f:
                f.write(benchmark_content)
```

## 转换规则

1. **函数名转换：** `def test_xxx()` → `def xxx_benchmark()`
2. **标签转换：** `@label("op_name")` → `@label("op_name_benchmark")`
3. **保留装饰器：** `@parametrize` 装饰器保持不变
4. **核心逻辑替换：** 
   - 移除 `gems_assert_close` 等断言
   - 添加 `triton.testing.do_bench` 性能测试代码
   - 添加 `CustomBenchmarkResult` 返回结果

## 命令行使用

```bash
# 测试模式（使用默认测试文件）
python accuracy_to_benchmark_converter.py

# 指定输入输出文件
python accuracy_to_benchmark_converter.py input_accuracy.py output_benchmark.py
```

