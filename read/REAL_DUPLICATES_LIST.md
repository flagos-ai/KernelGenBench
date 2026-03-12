# PYTORCH_OPERATORS 中真正的重复项列表

## 文件状态
- 文件总行数：633 行
- `torch.sqrt` 在第 620 行存在 ✅
- 第 611 行的注释是错误的（说 sqrt 是重复的，但前面没有 sqrt）

## 真正的重复项（共 10 个）

以下是在文件中实际出现两次的 key（排除注释）：

| Key | 第一次出现（行号） | 第二次出现（行号） | 建议操作 |
|-----|------------------|------------------|---------|
| `torch.cummax` | 441 | 602 | 删除第 602 行 |
| `torch.diagonal` | 446 | 603 | 删除第 603 行 |
| `torch.lerp` | 494 | 612 | 删除第 612 行 |
| `torch.Tensor.lerp_` | 495 | 613 | 删除第 613 行 |
| `torch.linspace` | 496 | 614 | 删除第 614 行 |
| `torch.nan_to_num` | 519 | 617 | 删除第 617 行 |
| `torch.nn.functional.glu` | 476 | 609 | 删除第 609 行 |
| `torch.polar` | 530 | 618 | 删除第 618 行 |
| `torch.Tensor.scatter_` | 555 | 619 | 删除第 619 行 |
| `torch.Tensor.to` | 574 | 625 | 删除第 625 行 |

## 需要特别处理的项

### `torch.dot` 和 `torch.nn.functional.elu_`
- `torch.dot` 第 453 行是注释，第 604 行是定义
- `torch.nn.functional.elu_` 第 456 行是注释，第 605 行是定义
- **建议**：如果这些注释应该保留，那么删除第 604 和 605 行的定义；如果应该取消注释，那么保留前面的，删除后面的

## 错误的注释

第 611 行说 "Removed duplicate: sqrt, sqrt_ (already defined above)"，但这是**错误的**：
- 前面（第 550-551 行）只有 `rsqrt` 和 `rsqrt_`
- `sqrt` 和 `sqrt_` 只在第 620-621 行出现一次
- **建议**：删除第 611 行的错误注释

## 建议操作

1. 删除第 602, 603, 612, 613, 614, 617, 609, 618, 619, 625 行的重复定义
2. 处理 `torch.dot` 和 `torch.nn.functional.elu_`（根据你的需求决定）
3. 删除或修正第 611 行的错误注释

