# 默认few-shot vs 分类few-shot 对比分析报告

## 测试配置
- **测试集**: V2_OPERATORS (50个算子)
- **模型**: gpt-5
- **温度**: 0.0
- **最大轮次**: 5 (with reflection)
- **测试时间**: 2025-12-11

## 总体结果

| 策略 | 通过数 | 通过率 |
|------|--------|--------|
| 默认few-shot (add) | 33/50 | 66.00% |
| 分类few-shot | 32/50 | 64.00% |
| **差异** | **+1** | **+2.00%** |

## 各轮次通过率对比

| 轮次 | 默认few-shot | 分类few-shot | 差异 |
|------|--------------|--------------|------|
| Round 0 | 14.00% | 12.00% | +2.00% |
| Round 1 | 50.00% | 38.00% | **+12.00%** |
| Round 2 | 60.00% | 50.00% | **+10.00%** |
| Round 3 | 62.00% | 60.00% | +2.00% |
| Round 4 | 66.00% | 64.00% | +2.00% |

**观察**: 默认few-shot在早期轮次(Round 1-2)表现明显更好，差异达10-12%。随着轮次增加，差异逐渐缩小。

## V2测试集各分类详细对比

| 分类 | 默认通过 | 分类通过 | 差异 |
|------|----------|----------|------|
| backward | 6/8 (75.00%) | 6/8 (75.00%) | 0 |
| core composite pointwise | 6/9 (66.67%) | 6/9 (66.67%) | 0 |
| core factory | 3/7 (42.86%) | 3/7 (42.86%) | 0 |
| core primitive pointwise | 4/9 (44.44%) | 4/9 (44.44%) | 0 |
| **core reduction** | **6/9 (66.67%)** | **5/9 (55.56%)** | **+1** |
| core view/reshape | 8/8 (100.00%) | 8/8 (100.00%) | 0 |

**观察**: 
- 除了core reduction分类外，其他分类的通过率完全相同
- core reduction分类中，默认few-shot表现更好 (66.67% vs 55.56%)

## 关键差异算子

### 只在默认few-shot通过的算子 (2个)
1. **huber_loss** (core reduction)
2. **reflection_pad1d_backward** (backward)

### 只在分类few-shot通过的算子 (1个)
1. **upsample_nearest2d_backward** (backward)

### 两者都通过的算子
31个算子

### 两者都失败的算子
16个算子

## 错误类型分布对比

| 错误类型 | 默认few-shot | 分类few-shot | 差异 |
|----------|--------------|--------------|------|
| AttributeError: astype | 7 | 9 | -2 |
| AttributeError: log1p | 2 | 3 | -1 |
| AttributeError: other | 42 | 34 | +8 |
| CompilationError | 5 | 12 | **-7** |
| Other | 43 | 52 | -9 |
| RuntimeError | 4 | 6 | -2 |
| TypeError | 12 | 14 | -2 |

**观察**:
- 分类few-shot的CompilationError明显更多 (-7)
- 默认few-shot的AttributeError: other更多 (+8)

## 结论

1. **总体通过率差异很小**: 仅2% (1个算子)，说明两种few-shot策略在最终效果上非常接近。

2. **默认few-shot在早期轮次表现更好**: Round 1-2的差异达10-12%，这可能是因为add算子作为few-shot示例更通用、更简单，模型更容易在早期理解并应用。

3. **分类few-shot在特定分类上未显示优势**: 虽然理论上分类few-shot应该为不同类别的算子提供更相关的示例，但实际效果并未明显优于默认few-shot。这可能是因为：
   - add算子作为基础示例已经足够通用
   - 分类few-shot的示例可能还不够精确匹配目标算子
   - 模型在理解复杂示例时可能产生更多编译错误

4. **core reduction分类的特殊性**: 默认few-shot在core reduction分类中表现更好，这可能是因为add算子的简单性更适合reduction类算子的生成模式。

5. **错误类型差异**: 分类few-shot产生更多CompilationError，这可能是因为分类示例相对复杂，导致生成的代码更容易出现编译问题。

## 建议

1. **继续使用默认few-shot**: 由于差异很小且默认few-shot在早期轮次表现更好，可以继续使用默认的add few-shot作为主要策略。

2. **优化分类few-shot**: 如果未来要继续探索分类few-shot，可以考虑：
   - 选择更简单、更通用的分类示例
   - 确保分类示例与目标算子的匹配度更高
   - 减少分类示例的复杂度，避免引入过多编译错误

3. **混合策略**: 可以考虑在特定分类（如core reduction）使用默认few-shot，在其他分类使用分类few-shot，但这需要进一步实验验证。


