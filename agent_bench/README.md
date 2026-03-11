# Agent Benchmark

用于评估 agentic 代码生成框架（如 Claude Code、Cursor、Devin 等）在 Triton kernel 生成任务上的能力。

## 快速开始

```bash
cd agent_bench/

# 配置（首次使用）
cp config.example.yaml config.yaml
cp .env.example .env
# 编辑 config.yaml 和 .env 填入你的配置

# 一键测试完整数据集
./test_ops.sh -d v2_1

# 或测试指定算子
./test_ops.sh add,softmax
```

## 一键测试脚本

`test_ops.sh` 提供一键完成 prompt 生成、agent 运行、结果验证的完整流程：

```bash
# 测试完整数据集
./test_ops.sh -d v2_1              # 测试 v2_1 全部 110 个算子
./test_ops.sh -d v2                # 测试 v2 全部 49 个算子
./test_ops.sh -d cupy              # 测试 cupy 全部 48 个算子

# 测试指定算子
./test_ops.sh add                  # 测试单个算子
./test_ops.sh add,softmax          # 测试多个算子
./test_ops.sh add -d v2            # 指定数据集

# 其他选项
./test_ops.sh -d v2_1 --skip-gen   # 跳过 prompt 生成
./test_ops.sh -d v2_1 --skip-verify # 只生成不验证
./test_ops.sh -d v2_1 --device-count 4  # 指定 GPU 数量
./test_ops.sh -d v2_1 --timeout 600     # 指定超时时间

# 查看帮助
./test_ops.sh --help
```

## 目录结构

```
agent_bench/
├── config.example.yaml      # 配置示例
├── .env.example             # 环境变量示例
├── test_ops.sh              # 一键测试脚本
├── templates/
│   └── triton_kernel.md     # Prompt 模板
├── prompts/                 # 预生成的 prompt 文件
│   ├── v2/
│   ├── v2_1/
│   └── cupy/
├── runs/                    # 运行结果
│   └── <run_name>/
│       ├── config.yaml      # 配置快照
│       ├── kernels/         # 生成的代码
│       ├── logs/            # Agent 日志
│       ├── progress.json    # 运行进度
│       └── results.json     # 验证结果
├── generate_prompts.py      # 生成 prompt
├── run.py                   # 批量运行 agent
├── verify.py                # 批量验证
└── device_manager.py        # GPU 管理
```

## 支持的 Dataset

| Dataset | 算子数量 | 说明 |
|---------|---------|------|
| v2 | 49 | PyTorch 基础算子 |
| v2_1 | 110 | PyTorch 扩展算子 |
| cupy | 48 | cuBLAS 算子 |

## 分步执行

如果需要更精细的控制，可以分步执行：

### 1. generate_prompts.py

为指定 dataset 的每个算子生成 prompt 文件。

```bash
# 生成 v2_1 所有算子的 prompt
python generate_prompts.py --dataset v2_1

# 生成所有 dataset 的 prompt
python generate_prompts.py --all

# 只生成指定算子
python generate_prompts.py --dataset v2_1 --op add,softmax

# 强制覆盖已存在的 prompt
python generate_prompts.py --dataset v2_1 --force

# 指定 Python 解释器路径
python generate_prompts.py --dataset v2_1 --python-path /path/to/python
```

### 2. run.py

批量调度 agent 生成 Triton kernel。

```bash
# 运行 v2_1 全部算子
python run.py --dataset v2_1

# 运行指定算子
python run.py --dataset v2_1 --op add,softmax

# 断点续跑
python run.py --dataset v2_1 --resume claude_v2_1_20260310_120000

# 使用自定义配置
python run.py --dataset v2_1 --config my_config.yaml
```

### 3. verify.py

批量验证生成的 kernel。

```bash
# 验证指定运行
python verify.py --run claude_v2_1_20260310_120000

# 只验证指定算子
python verify.py --run <run_name> --op add,softmax

# 指定 GPU 数量和超时
python verify.py --run <run_name> --device-count 4 --timeout 600

# 手动指定 dataset（覆盖自动检测）
python verify.py --run <run_name> --dataset v2_1
```

## 配置说明

复制 `config.example.yaml` 为 `config.yaml` 并编辑：

```yaml
# Agent 配置
agent:
  type: claude           # agent 类型
  bin: /path/to/claude   # 可执行文件路径
  timeout: 1800          # 单算子超时 (秒)
  max_retries: 3         # 失败重试次数
  budget: 50.0           # 单算子预算 (USD)

# 设备配置
device:
  gpu_ids: null          # null=自动检测, 或指定 [0,1,2,3]
  lock_dir: /tmp/agent_bench_gpu_locks

# Dataset 到测试模块的映射
test_modules:
  v2: src/flagbench/accuracy/test_v2_ops.py
  v2_1: src/flagbench/accuracy/test_v2_1_ops_with_benchmark.py
  cupy: src/flagbench/accuracy/cublas/test_cublas_ops.py
```

## 环境变量

复制 `.env.example` 为 `.env` 并配置：

```bash
ANTHROPIC_AUTH_TOKEN=your_token
ANTHROPIC_BASE_URL=https://api.anthropic.com
ANTHROPIC_MODEL=claude-sonnet-4-6
```

## 结果格式

`results.json`:

```json
{
  "run_name": "claude_v2_1_20260310_120000",
  "dataset": "v2_1",
  "summary": {
    "total": 110,
    "passed": 85,
    "failed": 25,
    "pass_rate": "77.3%"
  },
  "operators": {
    "add": {
      "status": "passed",
      "total_tests": 24,
      "passed_tests": 24,
      "failed_tests": 0
    },
    "softmax": {
      "status": "failed",
      "error": "Tensor-likes are not close..."
    }
  }
}
```

## 扩展

### 添加新 Dataset

1. 在 `generate_prompts.py` 的 `DATASET_OPERATORS` 中添加映射
2. 在 `config.yaml` 的 `test_modules` 中添加测试模块路径
3. 运行 `python generate_prompts.py --dataset <new_dataset>`

### 自定义 Prompt 模板

编辑 `templates/triton_kernel.md`，支持的变量：
- `{{OPERATOR}}` - 算子名
- `{{FULL_NAME}}` - 完整名称 (如 aten::add)
- `{{GPU_ID}}` - GPU ID
- `{{OP_SIGNATURES}}` - 函数签名
- `{{IMPL_INFO}}` - 需要实现的接口
- `{{INPUT_ARGS}}` - 输入参数
- `{{REFERENCE_CODE}}` - 参考实现 (可选)
