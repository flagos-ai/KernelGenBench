# vLLM 算子测试生成流程

## 概述

本流程用于为 vLLM 自定义算子生成 test_func（精度测试函数）和 baseline wrapper（基准调用封装），最终通过 verifier 框架验证正确性。

## 目录结构

```
src/generator/vllm/                    # 生成工具
├── vllm_ops_signatures_split.json     # 算子签名数据源（129个算子）
├── vllm_test_prompt_generator.py      # Prompt 模板构建器
├── generate_testfunc.py               # test_func 生成脚本（调用 LLM API）
├── generate_baseline_wrapper.py       # baseline wrapper 生成脚本（纯模板）
├── selected_ops_parameter_specs.md    # 已选算子的详细参数规格
├── torch_ops_vllm.json                # Torch 算子与 vLLM 的映射
├── vllm_ops_list.md                   # 全量算子列表
├── vllm_ops_list_importable.md        # 可导入算子列表
└── vllm_whole_ops.md                  # 完整算子文档

src/flagbench/accuracy/vllm15/         # vLLM 0.15 的 test_func
├── __init__.py                        # 注册所有 test_func
├── test_topk_softmax.py
├── test_scaled_fp8_quant.py
└── ...

src/flagbench/accuracy/vllm13/         # vLLM 0.13 的 test_func
├── __init__.py
├── test_topk_softmax.py               # 去掉了 e_score_correction_bias
├── test_scaled_fp8_quant.py           # 去掉了 group_shape
└── ...（其余与 vllm15 相同）

src/flagbench/dataset/baseline/vllm15/ # vLLM 0.15 的 baseline wrapper
├── topk_softmax.py
├── scaled_fp8_quant.py
└── ...

src/flagbench/dataset/baseline/vllm13/ # vLLM 0.13 的 baseline wrapper
├── topk_softmax.py                    # 去掉了 e_score_correction_bias
├── scaled_fp8_quant.py                # 去掉了 group_shape
└── ...（其余与 vllm15 相同）
```

## 生成流程

### Step 1: 生成 baseline wrapper

baseline wrapper 是对 `vllm._custom_ops.xxx()` 的薄封装，不需要 LLM，纯模板生成。

```bash
# 单个算子
python src/generator/vllm/generate_baseline_wrapper.py --op-name topk_softmax

# 多个算子
python src/generator/vllm/generate_baseline_wrapper.py --op-name topk_softmax,scaled_fp8_quant

# 全部算子
python src/generator/vllm/generate_baseline_wrapper.py --all
```

输出到 `src/flagbench/dataset/baseline/vllm/` 目录。

### Step 2: 生成 test_func（调用 LLM）

test_func 是参数化的精度+性能测试函数，由 LLM 生成。

```bash
export KSYUN_API_KEY='your-api-key'

# 单个算子
python src/generator/vllm/generate_testfunc.py --op-name topk_softmax

# 多个算子（并发）
python src/generator/vllm/generate_testfunc.py --op-name topk_softmax,scaled_fp8_quant

# 指定模型
python src/generator/vllm/generate_testfunc.py --op-name topk_softmax --model mog1
```

输出到 `src/generator/vllm/output/` 目录，需要手动 copy 到 `src/flagbench/accuracy/vllm15/`。

### Step 3: 注册

在 `src/flagbench/accuracy/vllm15/__init__.py` 中添加 import：

```python
from .test_topk_softmax import test_accuracy_topk_softmax
```

在 `src/flagbench/__init__.py` 的 `accuracy_modules` 列表中添加：

```python
"flagbench.accuracy.vllm15.test_topk_softmax",
```

### Step 4: 验证

```bash
source /share/project/zhaohuxing/anaconda3/bin/activate zpy_flagbench

# 单个算子
DISPATCH_TORCH_LIB=0 python test/test_accuracy_ut.py \
    --name "vllm15::topk_softmax" \
    --test-file "flagbench.accuracy.vllm15.test_topk_softmax"

# 多个算子
DISPATCH_TORCH_LIB=0 python test/test_accuracy_ut.py \
    --name "vllm15::topk_softmax,vllm15::scaled_fp8_quant" \
    --test-file "flagbench.accuracy.vllm15.test_topk_softmax,flagbench.accuracy.vllm15.test_scaled_fp8_quant"
```

- `DISPATCH_TORCH_LIB=0`：自比较模式，baseline 同时注册到 baseline 和 triton 命名空间
- `--name "vllm15::xxx"`：verifier 用 `::` 前面的部分定位 baseline 目录，后面的部分匹配 label
- `--test-file`：指定加载哪个测试模块

## 关键文件说明

### vllm_ops_signatures_split.json

算子签名数据源，每个算子包含：

```json
{
  "topk_softmax": {
    "callable": true,
    "signature": "(topk_weights: torch.Tensor, ...) -> None",
    "input_parameters": "(topk_weights: torch.Tensor, ...)",
    "output_parameters": "None",
    "vllm_api": "_custom_ops.topk_softmax(...)",
    "doc": "..."
  }
}
```

这是 prompt 构建和 baseline 生成的唯一数据源。

### vllm_test_prompt_generator.py

`build_vllm_testfunc_prompt(op_name)` 函数读取签名 JSON，构建发给 LLM 的 prompt，包含：
- 算子签名和 API 信息
- 测试结构模板（imports、decorator、函数名）
- 参数化要求（5-8 个 @parametrize，目标 100+ 组合）
- 实现模式（in-place vs return value）
- 性能测试模板

### generate_baseline_wrapper.py

纯模板生成，从签名 JSON 解析参数列表，生成对 `_custom_ops.xxx()` 的直接调用封装。不需要 LLM。

### generate_testfunc.py

调用金山云 API（或其他 LLM），传入 prompt，获取生成的 test_func 代码，保存到 output 目录。

## vllm13 vs vllm15 版本差异

| 算子 | vllm15 (0.15) | vllm13 (0.13) |
|------|---------------|---------------|
| topk_softmax | 6 个参数（含 e_score_correction_bias） | 5 个参数（无 e_score_correction_bias） |
| scaled_fp8_quant | 7 个参数（含 group_shape） | 6 个参数（无 group_shape） |
| 其余 8 个算子 | 相同 | 相同 |

## 已知问题和注意事项

1. **LLM 生成的 test_func 通常需要手动修复**，常见问题：
   - 参数值超出算子支持范围（如 block_size=64，但只支持 8/16/32）
   - tensor shape 语义错误（如 paged_attention_v1 的 query 应该是 num_seqs 而非 total_tokens）
   - 缺少对 padding 区域的特殊处理（如 scaled_fp8_quant 的 num_token_padding）

2. **prompt 当前的不足**：
   - 只提供了类型签名（input_parameters），缺少参数约束和 tensor shape 语义
   - 要求"尽可能多的参数组合"但没强调"在合法范围内"
   - 建议后续在签名 JSON 中增加 constraints 字段

3. **silu_and_mul_scaled_fp4_experts_quant** 需要 SM90+（H100），A100 跑不了，已被 topk_softmax 替代

4. **verifier 路径映射规则**：`--name "vllm15::xxx"` → baseline 从 `baseline/vllm15/xxx.py` 加载
