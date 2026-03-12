# 测试脚本使用说明

## 概述

`test_accuracy_all.py` 脚本用于验证从 FlagGems 迁移过来的 pytest 测试函数是否正确集成到 flagbench 中。

## 关键区别

### test_accuracy_ut.py vs test_accuracy_all.py

| 特性 | test_accuracy_ut.py | test_accuracy_all.py (新) |
|-----|-------------------|------------------------|
| **测试对象** | PYTORCH_OPERATORS 中的算子 | pytest 测试函数 |
| **测试方式** | 使用 Verifier + mock code | 直接运行 pytest |
| **测试数量** | 201 个算子 | 101 个新增测试函数 |
| **测试级别** | 算子级别 | 测试函数级别 |
| **验证内容** | 算子注册是否正确 | 测试逻辑、参数化是否正确 |
| **适用场景** | 验证 kernel 实现 | 验证测试迁移是否成功 |

### 为什么不能用 test_accuracy_ut.py？

❌ **test_accuracy_ut.py 的局限性**：
```python
# test_accuracy_ut.py 测试的是这个：
PYTORCH_OPERATORS = {
    'torch.addmm': torch.addmm,  # 只是一个算子
    'torch.addmv': torch.addmv,
}

# 但我们新增的是这些：
def test_accuracy_addmm():      # 测试函数 1
def test_accuracy_addmm_out():  # 测试函数 2（同一个算子的不同用法）
def test_accuracy_addmv():      # 测试函数 3
def test_accuracy_addmv_out():  # 测试函数 4
```

✅ **test_accuracy_all.py 的优势**：
- 直接运行 pytest 测试函数
- 验证参数化（@parametrize）是否正确
- 验证测试数据生成逻辑
- 验证 gems_assert_close 等断言
- 检测测试函数是否能正常发现和执行

## 使用方法

### 1. 收集模式（查看有哪些测试）

```bash
cd /share/project/zpy/flagbench/test
python test_accuracy_all.py --mode collect
```

**输出示例**：
```
收集新增的测试函数:
======================================================================

✅ test_unary_pointwise_ops.py:
   预期 API 数量: 24
   找到测试函数: 15
   示例: test_accuracy_atan, test_accuracy_atan_, test_accuracy_celu...
   ⚠️  缺失测试: contiguous, diagonal, eye, ...

总结:
  预期总数: 66 个 API
  找到测试: 36 个
  缺失测试: 30 个
```

### 2. 运行模式（执行测试）

#### 运行所有测试
```bash
python test_accuracy_all.py --mode run
```

#### 快速测试（只运行部分）
```bash
python test_accuracy_all.py --mode run --quick
```

#### 测试特定文件
```bash
python test_accuracy_all.py --mode run --file test_blas_ops.py
```

### 3. 组合使用

```bash
# 先收集看看某个文件有什么测试
python test_accuracy_all.py --mode collect --file test_blas_ops.py

# 然后运行该文件的测试
python test_accuracy_all.py --mode run --file test_blas_ops.py
```

## 当前状态

### 已找到的测试（36 个）

| 文件 | 找到 | 预期 | 比例 |
|-----|-----|-----|------|
| test_unary_pointwise_ops.py | 15 | 24 | 62% |
| test_binary_pointwise_ops.py | 9 | 11 | 82% |
| test_reduction_ops.py | 9 | 11 | 82% |
| test_blas_ops.py | 3 | 4 | 75% |
| test_attention_ops.py | 0 | 7 | 0% |
| test_special_ops.py | 0 | 1 | 0% |
| **总计** | **36** | **58** | **62%** |

### 缺失的文件（3 个）
- test_indexing_ops.py
- test_pointwise_ops.py  
- test_rwkv_ops.py

### 缺失的测试（30 个）

可能的原因：
1. **测试函数名不匹配**：API 名和测试函数名不一致
2. **测试在其他文件中**：可能被归类到其他测试文件
3. **测试未迁移**：确实没有迁移过来

## 建议的测试流程

### 步骤 1: 收集信息
```bash
python test_accuracy_all.py --mode collect
```
查看哪些测试已经存在，哪些缺失。

### 步骤 2: 快速验证
```bash
python test_accuracy_all.py --mode run --quick --file test_blas_ops.py
```
先测试一个小文件，确保脚本正常工作。

### 步骤 3: 逐个文件测试
```bash
# 测试 BLAS 操作
python test_accuracy_all.py --mode run --file test_blas_ops.py

# 测试一元操作
python test_accuracy_all.py --mode run --file test_unary_pointwise_ops.py

# 测试二元操作
python test_accuracy_all.py --mode run --file test_binary_pointwise_ops.py
```

### 步骤 4: 全量测试
```bash
python test_accuracy_all.py --mode run
```

## 预期结果

### 成功的标志
- ✅ pytest 能发现所有测试函数
- ✅ 参数化正确生成测试用例
- ✅ 测试能正常运行（可能有失败，但不应该崩溃）
- ✅ gems_assert_close 断言正常工作

### 可能的问题

#### 1. 导入错误
```python
ImportError: cannot import name 'xxx'
```
**解决**: 检查 flagbench 的导入路径

#### 2. 参数化错误
```python
ValueError: invalid parametrize value
```
**解决**: 检查 @parametrize 装饰器的参数

#### 3. 测试未发现
```python
collected 0 items
```
**解决**: 检查测试函数命名是否以 `test_` 开头

#### 4. 断言失败
```python
AssertionError: Tensor comparison failed
```
**解决**: 这是正常的测试失败，说明精度不匹配（这是 OK 的，因为我们还没实现 Triton kernel）

## 与 pytest 直接运行的区别

### 直接使用 pytest
```bash
cd /share/project/zpy/flagbench/src/flagbench/accuracy
pytest test_blas_ops.py -v
```

### 使用 test_accuracy_all.py
```bash
cd /share/project/zpy/flagbench/test
python test_accuracy_all.py --mode run --file test_blas_ops.py
```

**优势**：
- 自动统计测试数量
- 批量运行多个文件
- 生成汇总报告
- 更好的错误处理

## 下一步

1. **验证注册**：确认 IMPL_INFO 和 PYTORCH_OPERATORS 的注册是否正确
2. **运行测试**：使用 `test_accuracy_all.py --mode run` 运行所有测试
3. **分析失败**：查看哪些测试失败，是否是预期的（因为缺少 Triton kernel 实现）
4. **补充缺失**：根据 collect 模式的结果，补充缺失的测试函数或文件

---
创建时间: 2025-11-17
版本: 1.0
