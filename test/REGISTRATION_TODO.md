# 待补充注册的 10 个算子分析

## 📋 分类结果

### ✅ 已注册但名称不同（4 个）

| 测试中的名称 | IMPL_INFO 中的实际名称 | 说明 |
|-------------|----------------------|------|
| dropout | native_dropout | PyTorch 内部实现名 |
| group_norm | native_group_norm | PyTorch 内部实现名 |
| layer_norm | native_layer_norm | PyTorch 内部实现名 |
| weight_norm | _weight_norm / _weight_norm_interface | PyTorch 内部实现名 |

**处理方式**: 
- 在 test_accuracy_all.py 中添加名称映射
- 或者在 kernel_list.py 中添加别名

---

### ❌ 真正需要补充注册（6 个）

| 算子名 | 类型 | 说明 |
|--------|------|------|
| apply_rotary_pos_emb | 自定义融合算子 | Rotary Position Embedding |
| conv2d | PyTorch 标准 | 2D 卷积（可能需要确认） |
| conv_depthwise2d | 自定义 | Depthwise 2D 卷积 |
| gelu_and_mul | 自定义融合算子 | GELU + Mul 融合 |
| silu_and_mul | 自定义融合算子 | SiLU + Mul 融合 |
| skip_layer_norm | 自定义融合算子 | Skip + LayerNorm 融合 |

---

## 🔧 补充注册代码

### 方案 1: 在 kernel_list.py 中添加别名（推荐）

在 `IMPL_INFO` 字典中添加以下条目：

```python
IMPL_INFO = {
    # ... 现有条目 ...
    
    # ========== 别名映射（让测试名称与内部实现对应）==========
    "dropout": IMPL_INFO["native_dropout"],  # 别名
    "group_norm": IMPL_INFO["native_group_norm"],  # 别名
    "layer_norm": IMPL_INFO["native_layer_norm"],  # 别名
    "weight_norm": IMPL_INFO["_weight_norm_interface"],  # 别名
    
    # ========== 新增自定义融合算子 ==========
    "apply_rotary_pos_emb": [("apply_rotary_pos_emb", Autograd.disable)],
    "conv2d": [("conv2d", Autograd.default)],  # 或者映射到已有的实现
    "conv_depthwise2d": [("conv_depthwise2d", Autograd.default)],
    "gelu_and_mul": [("gelu_and_mul", Autograd.disable)],
    "silu_and_mul": [("silu_and_mul", Autograd.disable)],
    "skip_layer_norm": [("skip_layer_norm", Autograd.disable)],
}
```

### 方案 2: 修改测试文件中的 @label（不推荐）

如果这些算子已经有正确的实现，可以修改测试文件中的 @label：

```python
# 修改前
@label("dropout")
def test_accuracy_dropout(...):
    ...

# 修改后
@label("native_dropout")
def test_accuracy_dropout(...):
    ...
```

---

## 📍 需要在 kernel_list.py 中添加的位置

找到 `IMPL_INFO` 字典的定义（通常在文件开头），在合适的位置添加：

### 位置 1: 别名部分（如果存在）
```python
# Aliases for test compatibility
"dropout": IMPL_INFO["native_dropout"],
"group_norm": IMPL_INFO["native_group_norm"],
"layer_norm": IMPL_INFO["native_layer_norm"],
"weight_norm": IMPL_INFO["_weight_norm_interface"],
```

### 位置 2: 自定义融合算子部分
```python
# Custom fused operators
"apply_rotary_pos_emb": [("apply_rotary_pos_emb", Autograd.disable)],
"gelu_and_mul": [("gelu_and_mul", Autograd.disable)],
"silu_and_mul": [("silu_and_mul", Autograd.disable)],
"skip_layer_norm": [("skip_layer_norm", Autograd.disable)],
"conv_depthwise2d": [("conv_depthwise2d", Autograd.default)],
```

### 位置 3: 标准算子部分
```python
# Standard operators
"conv2d": [("conv2d", Autograd.default)],  # 如果确认是标准算子
```

---

## ✅ 推荐操作步骤

### 步骤 1: 确认这 6 个算子是否已实现
```bash
# 检查源码中是否存在这些算子的实现
cd /share/project/zpy/flagbench
grep -r "def apply_rotary_pos_emb" src/flag_gems/
grep -r "def conv2d" src/flag_gems/
grep -r "def conv_depthwise2d" src/flag_gems/
grep -r "def gelu_and_mul" src/flag_gems/
grep -r "def silu_and_mul" src/flag_gems/
grep -r "def skip_layer_norm" src/flag_gems/
```

### 步骤 2: 根据实现情况补充注册
- **如果已实现**: 在 kernel_list.py 中添加注册
- **如果未实现**: 
  - 选项 A: 注释掉对应的测试函数
  - 选项 B: 实现这些算子后再注册

### 步骤 3: 添加别名（4 个）
在 kernel_list.py 的 IMPL_INFO 末尾添加：
```python
# Test name aliases (add at the end of IMPL_INFO)
IMPL_INFO["dropout"] = IMPL_INFO["native_dropout"]
IMPL_INFO["group_norm"] = IMPL_INFO["native_group_norm"]
IMPL_INFO["layer_norm"] = IMPL_INFO["native_layer_norm"]
IMPL_INFO["weight_norm"] = IMPL_INFO["_weight_norm_interface"]
```

### 步骤 4: 验证
```bash
cd /share/project/zpy/flagbench
python test/test_accuracy_all.py --mode collect
```

---

## 🎯 预期结果

完成补充注册后：
- ✅ 可测试算子: 59 → **73** 个 (+14)
- 🔧 自定义算子: 14 → **14** 个 (不变)
- ❌ 未注册: 10 → **0** 个 (-10)

**测试覆盖率**: 71.1% → **100%** 🎉
