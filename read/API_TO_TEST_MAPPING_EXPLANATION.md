# API 到测试的映射机制说明

## 概述

系统通过以下步骤将注册的 API 与测试文件中的 `@label("")` 进行匹配：

1. **收集注册的算子**：从 `IMPL_INFO` 中获取所有已注册的算子名称
2. **收集测试 label**：从测试文件中提取所有 `@label("operator_name")` 装饰器
3. **建立映射关系**：通过映射表将测试 label 映射到注册的算子名称
4. **匹配检查**：找出哪些注册的算子没有对应的测试

---

## 详细流程

### 步骤 1: 收集注册的算子

**代码位置**: `check_missing_tests.py` 的 `get_registered_operators()`

```python
def get_registered_operators():
    # 从 IMPL_INFO 获取所有算子名称
    registered = set(IMPL_INFO.keys())
    
    # 添加别名（从 IMPL_INFO 中已有的别名）
    aliases = {
        "dropout": "native_dropout",
        "group_norm": "native_group_norm",
        "layer_norm": "native_layer_norm",
        "weight_norm": "_weight_norm_interface",
    }
    
    # 添加别名对应的算子
    for alias, real_name in aliases.items():
        if real_name in registered:
            registered.add(alias)
    
    return registered
```

**说明**:
- 从 `IMPL_INFO` 字典中提取所有的 key（算子名称）
- 添加别名，例如 `dropout` 是 `native_dropout` 的别名
- 返回所有注册的算子名称集合

**示例**:
- `IMPL_INFO` 中有: `"abs"`, `"add"`, `"native_dropout"`, `"native_layer_norm"` 等
- 添加别名后: `"dropout"`, `"layer_norm"` 等也被加入

---

### 步骤 2: 收集测试文件中的 label

**代码位置**: `check_missing_tests.py` 的 `extract_labels_from_file()` 和 `get_all_test_labels()`

```python
def extract_labels_from_file(filepath):
    """从测试文件中提取所有 @label 装饰器中的算子名称"""
    labels = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 匹配 @label("operator_name") 或 @label('operator_name')
    pattern = r'@label\(["\']([^"\']+)["\']\)'
    matches = re.findall(pattern, content)
    
    for match in matches:
        # 跳过 "inplace" 标签
        if match != "inplace":
            labels.add(match)
    
    return labels

def get_all_test_labels():
    """收集所有测试文件中的 label"""
    all_labels = set()
    label_to_files = defaultdict(list)
    
    for filename in TEST_FILES:
        filepath = TEST_DIR / filename
        labels = extract_labels_from_file(filepath)
        all_labels.update(labels)
        
        for label in labels:
            label_to_files[label].append(filename)
    
    return all_labels, label_to_files
```

**说明**:
- 遍历所有测试文件（`test_*.py`）
- 使用正则表达式 `r'@label\(["\']([^"\']+)["\']\)'` 匹配 `@label("xxx")` 或 `@label('xxx')`
- 跳过 `"inplace"` 标签（这是特殊标签，不是算子名称）
- 返回所有 label 的集合，以及每个 label 出现在哪些文件中

**测试文件示例**:
```python
# test_unary_pointwise_ops.py
@label("abs")
def test_accuracy_abs(...):
    ...

@label("relu")
def test_accuracy_relu(...):
    ...
```

**提取结果**:
- `all_labels = {"abs", "relu", ...}`
- `label_to_files = {"abs": ["test_unary_pointwise_ops.py"], ...}`

---

### 步骤 3: 建立映射关系

**代码位置**: `check_missing_tests.py` 的 `get_label_to_registered_mapping()`

```python
def get_label_to_registered_mapping():
    """获取测试 label 到注册算子名称的映射"""
    label_to_registered = {
        # 别名映射（测试 label -> 注册算子名称）
        "dropout": "native_dropout",  # 测试中的 dropout label 对应注册的 native_dropout
        "native_dropout": "native_dropout",
        "group_norm": "native_group_norm",
        "native_group_norm": "native_group_norm",
        "layer_norm": "native_layer_norm",
        "native_layer_norm": "native_layer_norm",
        "weight_norm": "_weight_norm_interface",
        "weight_norm_interface": "_weight_norm_interface",
        # 其他映射
        "vector_norm": "linalg_vector_norm",
        "upsample_bicubic2d_aa": "_upsample_bicubic2d_aa",
        "upsample": "_upsample_bicubic2d_aa",  # upsample 是别名
        "unique": "_unique2",
        "native_instance_norm": "instance_norm",
        "skip_rms_norm": "skip_layer_norm",
        "diagonal_backward": "diagonal",
        # 别名映射
        "trunc_divide": "div",
        "trunc_divide_": "div_",
        "rsub": "sub",
        "or_": "bitwise_or",
        # 没有对应注册算子的 label（设为 None）
        "linear": None,  # 可能是 addmm/addmv 的别名
        "matmul": None,  # 可能是 mm/bmm 的别名
        "avg_pool2d": None,  # 可能没有注册
    }
    return label_to_registered
```

**说明**:
- 这个映射表处理测试 label 和注册算子名称不一致的情况
- 有些测试使用用户友好的名称（如 `dropout`），但注册时使用底层名称（如 `native_dropout`）
- 有些 label 没有对应的注册算子（设为 `None`），可能是别名或错误

**映射示例**:
- 测试中的 `@label("dropout")` → 映射到注册的 `"native_dropout"`
- 测试中的 `@label("layer_norm")` → 映射到注册的 `"native_layer_norm"`
- 测试中的 `@label("abs")` → 没有映射，直接使用 `"abs"`（因为注册时也是 `"abs"`）

---

### 步骤 4: 匹配检查

**代码位置**: `check_missing_tests.py` 的 `main()`

```python
def main():
    # 1. 获取注册的算子
    registered_ops = get_registered_operators()
    
    # 2. 获取所有测试中的 label
    test_labels, label_to_files = get_all_test_labels()
    
    # 3. 获取 label 到注册算子的映射
    label_mapping = get_label_to_registered_mapping()
    
    # 4. 将测试 label 映射到注册的算子名称
    mapped_test_labels = set()
    unmapped_labels = set()
    
    for label in test_labels:
        if label in label_mapping:
            mapped_name = label_mapping[label]
            if mapped_name is not None:
                mapped_test_labels.add(mapped_name)
            else:
                unmapped_labels.add(label)  # 这些 label 没有对应的注册算子
        else:
            # 如果 label 本身就在注册列表中，直接使用
            if label in registered_ops:
                mapped_test_labels.add(label)
            else:
                unmapped_labels.add(label)
    
    # 5. 找出注册了但没有测试的算子
    missing_tests = registered_ops - mapped_test_labels
    
    # 6. 找出测试了但没有注册的算子（可能是别名或错误）
    extra_tests = unmapped_labels
```

**匹配逻辑**:

1. **遍历所有测试 label**
2. **检查映射表**:
   - 如果 label 在映射表中：
     - 如果映射到 `None` → 加入 `unmapped_labels`（没有对应注册算子）
     - 否则 → 将映射后的名称加入 `mapped_test_labels`
   - 如果 label 不在映射表中：
     - 如果 label 本身就在注册列表中 → 直接加入 `mapped_test_labels`
     - 否则 → 加入 `unmapped_labels`（可能是别名或错误）

3. **计算缺失的测试**:
   - `missing_tests = registered_ops - mapped_test_labels`
   - 这些是已注册但没有对应测试的算子

4. **计算多余的测试**:
   - `extra_tests = unmapped_labels`
   - 这些是测试了但没有注册的算子（可能是别名或错误）

---

## 完整示例

### 示例 1: 简单匹配（名称一致）

**注册的算子**: `"abs"` 在 `IMPL_INFO` 中

**测试文件**:
```python
@label("abs")
def test_accuracy_abs(...):
    ...
```

**匹配过程**:
1. `registered_ops` 包含 `"abs"`
2. `test_labels` 包含 `"abs"`
3. `"abs"` 不在 `label_mapping` 中
4. `"abs"` 在 `registered_ops` 中 → 加入 `mapped_test_labels`
5. ✅ 匹配成功

---

### 示例 2: 别名映射（名称不一致）

**注册的算子**: `"native_dropout"` 在 `IMPL_INFO` 中

**测试文件**:
```python
@label("dropout")
@label("native_dropout")
def test_accuracy_dropout(...):
    ...
```

**匹配过程**:
1. `registered_ops` 包含 `"native_dropout"` 和 `"dropout"`（别名）
2. `test_labels` 包含 `"dropout"` 和 `"native_dropout"`
3. `"dropout"` 在 `label_mapping` 中 → 映射到 `"native_dropout"` → 加入 `mapped_test_labels`
4. `"native_dropout"` 不在 `label_mapping` 中，但在 `registered_ops` 中 → 加入 `mapped_test_labels`
5. ✅ 匹配成功

---

### 示例 3: 缺失测试

**注册的算子**: `"topk"` 在 `IMPL_INFO` 中

**测试文件**: 没有 `@label("topk")`

**匹配过程**:
1. `registered_ops` 包含 `"topk"`
2. `test_labels` 不包含 `"topk"`
3. `"topk"` 不在 `mapped_test_labels` 中
4. `missing_tests` 包含 `"topk"`
5. ❌ 缺失测试

---

### 示例 4: 多余测试（没有注册）

**注册的算子**: 没有 `"avg_pool2d"`

**测试文件**:
```python
@label("avg_pool2d")
def test_accuracy_avg_pool2d(...):
    ...
```

**匹配过程**:
1. `registered_ops` 不包含 `"avg_pool2d"`
2. `test_labels` 包含 `"avg_pool2d"`
3. `"avg_pool2d"` 在 `label_mapping` 中 → 映射到 `None`
4. `"avg_pool2d"` 加入 `unmapped_labels`
5. ⚠️ 测试了但没有注册（可能是别名或错误）

---

## 总结

系统通过以下机制匹配 API 和测试：

1. **直接匹配**: 测试 label 和注册算子名称一致
2. **映射匹配**: 通过 `label_mapping` 将测试 label 映射到注册算子名称
3. **别名处理**: 通过 `IMPL_INFO` 中的别名映射处理

**关键点**:
- 测试文件使用 `@label("operator_name")` 标记测试
- 系统通过正则表达式提取所有 label
- 通过映射表处理名称不一致的情况
- 最终检查哪些注册的算子没有对应的测试

