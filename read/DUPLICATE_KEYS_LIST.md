# PYTORCH_OPERATORS 中重复的 key 列表

## 发现的重复项（共 12 个）

请检查以下重复的 key，确认哪些应该删除：

| Key | 第一次出现（行号） | 第二次出现（行号） | 建议 |
|-----|------------------|------------------|------|
| `torch.cummax` | 441 | 602 | 删除第二次 |
| `torch.diagonal` | 446 | 603 | 删除第二次 |
| `torch.dot` | 453 (注释) | 604 | 删除第二次（第一次是注释） |
| `torch.nn.functional.elu_` | 456 (注释) | 605 | 删除第二次（第一次是注释） |
| `torch.nn.functional.glu` | 476 | 609 | 删除第二次 |
| `torch.lerp` | 494 | 612 | 删除第二次 |
| `torch.Tensor.lerp_` | 495 | 613 | 删除第二次 |
| `torch.linspace` | 496 | 614 | 删除第二次 |
| `torch.nan_to_num` | 519 | 617 | 删除第二次 |
| `torch.polar` | 530 | 618 | 删除第二次 |
| `torch.Tensor.scatter_` | 555 | 619 | 删除第二次 |
| `torch.Tensor.to` | 574 | 625 | 删除第二次 |

## 注意

- **`torch.sqrt` 和 `torch.Tensor.sqrt_`** 不在重复列表中，它们只出现一次（第 620-621 行）
- 所有重复项都在第 589-626 行的 "Added operators for missing tests" 部分
- 建议删除第二次出现的（第 589-626 行中的），保留第一次出现的（前面的部分）

## 需要确认的项

以下项第一次出现时是注释状态，需要确认：
- `torch.dot` (第 453 行是注释 `# 'torch.dot': torch.dot,`)
- `torch.nn.functional.elu_` (第 456 行是注释 `# 'torch.nn.functional.elu_': torch.nn.functional.elu_,`)

如果这些注释应该保留，那么第二次出现的应该删除。
如果这些注释应该取消注释，那么需要决定保留哪一个。

