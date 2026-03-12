# PYTORCH_OPERATORS 中重复的 key - 待审核列表

## 发现的重复项（共 12 个）

请检查以下重复的 key，确认哪些应该删除：

### 1. `torch.cummax`
- **第 441 行**：`'torch.cummax': torch.cummax,` ✅ (保留)
- **第 602 行**：`'torch.cummax': torch.cummax,` ❌ (删除)

### 2. `torch.diagonal`
- **第 446 行**：`'torch.diagonal': torch.diagonal,` ✅ (保留)
- **第 603 行**：`'torch.diagonal': torch.diagonal,` ❌ (删除)

### 3. `torch.dot`
- **第 453 行**：`# 'torch.dot': torch.dot,` (注释)
- **第 604 行**：`'torch.dot': torch.dot,` ✅ (保留，取消第 453 行的注释？)

### 4. `torch.nn.functional.elu_`
- **第 456 行**：`# 'torch.nn.functional.elu_': torch.nn.functional.elu_,` (注释)
- **第 605 行**：`'torch.nn.functional.elu_': torch.nn.functional.elu_,` ✅ (保留，取消第 456 行的注释？)

### 5. `torch.nn.functional.glu`
- **第 476 行**：`'torch.nn.functional.glu': torch.nn.functional.glu,` ✅ (保留)
- **第 609 行**：`'torch.nn.functional.glu': torch.nn.functional.glu,` ❌ (删除)

### 6. `torch.lerp`
- **第 494 行**：`'torch.lerp': torch.lerp,` ✅ (保留)
- **第 612 行**：`'torch.lerp': torch.lerp,` ❌ (删除)

### 7. `torch.Tensor.lerp_`
- **第 495 行**：`'torch.Tensor.lerp_': torch.Tensor.lerp_,` ✅ (保留)
- **第 613 行**：`'torch.Tensor.lerp_': torch.Tensor.lerp_,` ❌ (删除)

### 8. `torch.linspace`
- **第 496 行**：`'torch.linspace': torch.linspace,` ✅ (保留)
- **第 614 行**：`'torch.linspace': torch.linspace,` ❌ (删除)

### 9. `torch.nan_to_num`
- **第 519 行**：`'torch.nan_to_num': torch.nan_to_num,` ✅ (保留)
- **第 617 行**：`'torch.nan_to_num': torch.nan_to_num,` ❌ (删除)

### 10. `torch.polar`
- **第 530 行**：`'torch.polar': torch.polar,` ✅ (保留)
- **第 618 行**：`'torch.polar': torch.polar,` ❌ (删除)

### 11. `torch.Tensor.scatter_`
- **第 555 行**：`'torch.Tensor.scatter_': torch.Tensor.scatter_,` ✅ (保留)
- **第 619 行**：`'torch.Tensor.scatter_': torch.Tensor.scatter_,` ❌ (删除)

### 12. `torch.Tensor.to`
- **第 574 行**：`'torch.Tensor.to': torch.Tensor.to,` ✅ (保留)
- **第 625 行**：`'torch.Tensor.to': torch.Tensor.to,` ❌ (删除)

## 注意

- **`torch.sqrt` 和 `torch.Tensor.sqrt_`** 不在重复列表中，它们只出现一次（第 620-621 行）
- 所有重复项都在第 589-626 行的 "Added operators for missing tests" 部分
- 建议删除第二次出现的（第 589-626 行中的），保留第一次出现的（前面的部分）

## 需要确认的项

以下项第一次出现时是注释状态，需要确认：
- `torch.dot` (第 453 行是注释，第 604 行是定义)
- `torch.nn.functional.elu_` (第 456 行是注释，第 605 行是定义)

**建议**：
- 如果这些注释应该保留，那么第二次出现的应该删除
- 如果这些注释应该取消注释，那么需要决定保留哪一个（建议保留前面的，删除后面的）

