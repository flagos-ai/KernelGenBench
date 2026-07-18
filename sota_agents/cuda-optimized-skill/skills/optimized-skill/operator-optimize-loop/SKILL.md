---
name: operator-optimize-loop
description: Run an operator optimization loop for CUDA, CUTLASS, or Triton implementations. The loop enforces correctness validation, benchmark, backend-aware profiling/artifact capture, strategy memory (positive/negative/rejected), and final best-version selection. Use when Claude needs to iterate on a `.cu` or Triton `.py` operator for a user-specified number of rounds, compare versions by benchmark results, keep full Nsight Compute evidence for winners, and prepare the next optimization step from the latest reports.
---

<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->


# Operator Optimization Loop

这个 skill 是 `skills/optimized-skill` 的统一主入口，用来把现有能力串成闭环：

1. correctness validation
2. benchmark
3. backend-aware profiling
4. 生成本轮优化方案
5. 生成下一版算子
6. 再次 benchmark / profiling
7. 在用户指定轮数内选出最优版本

现在支持三类后端：

- `cuda`: 原生 CUDA `.cu` 算子
- `cutlass`: 基于 CUTLASS 的 `.cu` 算子
- `triton`: Triton `.py` 算子

后续迭代方法保持一致：
- 首轮广覆盖
- 后续轮针对性修正
- 每轮都要写 `optimization_proposal.md`
- 每轮都要重新 benchmark
- 三种后端每轮都要带 targeted/full profiling
- 每轮都要自动沉淀策略记忆（正向/负向/拒绝），并用于下一轮约束

## 前置条件（Pre-requisites）

**重要：本 skill 专注于性能优化，不负责代码生成或补全。**

在运行优化循环之前，输入的 kernel 文件（`kernel.py` 或 `kernel.cu`）必须满足以下要求：

### 1. 完整的算子实现（必需）

**对于 Triton (`kernel.py`)：**
- 必须包含所有 PyTorch 算子 overload 变体的 wrapper 函数
- 函数签名必须与 ATen 接口一致（参考 `operator_spec.md` 中的"需要实现的接口"）
- 必须能通过正确性验证（accuracy tests）
- 命名规范：将 `.` 替换为 `_`（如 `add.Tensor` → `add_Tensor`）

**对于 CUDA/CUTLASS (`kernel.cu`)：**
- 必须包含完整的 CUDA kernel 实现
- 必须能编译通过（`nvcc` 无错误）
- 必须能通过正确性验证

### 2. 可运行的 Kernel 实现（必需）

**Triton：**
- 使用 `@triton.jit` 装饰器定义 kernel
- 实现核心计算逻辑
- 无语法错误

**CUDA/CUTLASS：**
- 使用 `__global__` 定义 kernel
- 实现核心计算逻辑
- 无编译错误

### 3. Benchmark 接口（必需）

**Triton 必须包含：**

```python
def setup(M=1024, N=1024, K=1024, seed=None, **kwargs):
    """生成测试数据"""
    if seed is not None:
        torch.manual_seed(seed)
    
    # 生成输入张量
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    y = torch.randn(M, N, device='cuda', dtype=torch.float32)
    
    return {
        "inputs": {"x": x, "y": y},
        "outputs": [],  # 如果 wrapper 返回结果
        # 或 "outputs": ["out"] 如果需要预分配输出张量
    }

def run_kernel(x, y):
    """运行 kernel（通过主 wrapper 函数）"""
    result = add_Tensor(x, y)  # 调用主 wrapper
    return result
```

**CUDA/CUTLASS 必须包含：**
- `setup()` 函数生成测试数据
- `run_kernel()` 函数调用 kernel

### 4. 代码结构示例（Triton）

完整的 `kernel.py` 应包含以下三部分：

```python
import torch
import triton
import triton.language as tl

# Part 1: Triton kernel
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = x + y
    
    tl.store(out_ptr + offsets, out, mask=mask)

# Part 2: PyTorch 算子接口（所有 overload 变体）
def add_Tensor(self: torch.Tensor, other: torch.Tensor, *, alpha=1) -> torch.Tensor:
    """aten::add.Tensor"""
    self, other = torch.broadcast_tensors(self, other)
    self = self.contiguous()
    other = other.contiguous()
    
    out = torch.empty_like(self)
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    if alpha != 1:
        other = other * alpha
    
    add_kernel[grid](self, other, out, n_elements, BLOCK_SIZE=1024)
    return out

def add_Scalar(self: torch.Tensor, other: float, *, alpha=1) -> torch.Tensor:
    """aten::add.Scalar"""
    return add_Tensor(self, torch.full_like(self, other), alpha)

def add_out(self: torch.Tensor, other: torch.Tensor, *, alpha=1, out: torch.Tensor) -> torch.Tensor:
    """aten::add.out"""
    result = add_Tensor(self, other, alpha)
    out.copy_(result)
    return out

# Part 3: Benchmark 接口
def setup(M=1024, N=1024, seed=None, **kwargs):
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    y = torch.randn(M, N, device='cuda', dtype=torch.float32)
    return {"inputs": {"x": x, "y": y}, "outputs": []}

def run_kernel(x, y):
    return add_Tensor(x, y)
```

### 前置条件检查

如果输入的 kernel 文件不满足以上条件：
- **缺少 PyTorch wrapper 函数** → 请先使用生成流程（如 `operator_spec.md` + LLM 生成）获得完整实现
- **缺少 setup/run_kernel** → 请先添加 benchmark 接口
- **有语法错误或无法运行** → 请先修复错误，确保能通过基本测试
- **正确性验证失败** → 请先修复算法逻辑，确保结果正确

**本 skill 假设输入的 baseline 已经是功能完整、正确可运行的实现，优化循环只负责提升性能，不负责修复功能性问题。**

## 复用的现有能力

优先复用以下内容，不要重写已有逻辑：
- benchmark: `skills/optimized-skill/kernel-benchmark/scripts/benchmark.py`
- benchmark skill 约定: `skills/optimized-skill/kernel-benchmark/SKILL.md`
- NCU 分析约定: `skills/optimized-skill/ncu-rep-analyze/SKILL.md`

优化知识库按后端选择：
- CUDA:
  - `skills/optimized-skill/reference/cuda/optim.md`
  - `skills/optimized-skill/reference/cuda/memory-optim.md`
  - `skills/optimized-skill/reference/cuda/compute-optim.md`
  - `skills/optimized-skill/reference/cuda/sync-optim.md`
- CUTLASS:
  - `skills/optimized-skill/reference/cutlass/cutlass-optim.md`
- Triton:
  - `skills/optimized-skill/reference/triton/triton-optim.md`

## 输入

必需：
- 当前算子文件路径：
  - CUDA / CUTLASS: `.cu`
  - Triton: `.py`
- `--max-iterations=<N>`

非常重要：如果用户没有显式提供 `--max-iterations`，先要求用户明确给出轮数，再执行，不要自行使用默认值。

强烈建议：
- `--ref=<reference.py>`，否则不能宣称 correctness 已验证

常用可选参数：
- `--backend=cuda|cutlass|triton`
- `--M=... --N=... --K=...` 等维度参数
- `--warmup`
- `--repeat`
- `--arch`
- `--gpu`
- `--ptr-size`
- `--seed`
- `--run-dir`

说明：
- `--backend` 不给时，脚本按文件后缀推断：`.py -> triton`，否则默认 `cuda`
- `cutlass` 与 `cuda` 共用 `.cu` 评测链路，但优化策略和 proposal 依据不同

## 运行前环境检查

在执行 benchmark / profiling 之前，必须先做 preflight 检查。`scripts/optimize_loop.py` 会默认检查并记录：

通用检查：
- 本地 GPU 型号、compute capability、driver version
- `--gpu` 指定的设备是否存在且可被当前 Python / PyTorch 访问
- `torch` 是否可导入，以及 `torch.cuda.is_available()` 是否为 `true`
- benchmark 脚本、输入文件、可选 reference 文件是否存在

CUDA / CUTLASS 额外检查：
- `nvcc` 是否存在，并记录解析后的可执行文件路径与版本信息

所有后端都检查：
- `ncu` 是否存在，并记录解析后的可执行文件路径与版本信息

Triton：
- 不需要 `nvcc`
- 但同样通过 `ncu --target-processes all ... python benchmark.py <kernel.py>` 抓取 Triton kernel

检查结果会写入当前 run 目录：
- `preflight_check.md`
- `preflight_check.json`

## 每轮迭代的强制流程

对 `v0` 到 `vN` 的每一轮都必须执行：

1. 调用 `optimize_loop.py` 对当前版本做完整评测。
2. 读取本轮输出的：
   - `benchmark_result.json`
   - `iteration_summary.md`
   - targeted/full NCU 的 summary/details 文本
3. 按照“首轮广覆盖、后续针对性修正”的规则制定本轮优化方向。
4. 写出本轮 `optimization_proposal.md`（必须包含 `## Strategy tags` 结构化标签）。
5. 基于 proposal 生成下一版算子。
6. 继续下一轮，直到达到 `max_iterations` 或提前停止。

## 首轮与后续轮的优化策略

### 第一次优化（baseline -> v1）

第一次优化要尽可能多地吸收该后端知识库中已经整理好的优化方法，但前提是这些方法与当前算子的算法形态相容，且不会明显互相冲突。

#### CUDA
优先顺序：
1. 先从 `skills/optimized-skill/reference/cuda/optim.md` 获取整体迭代思路。
2. 再从以下文档中尽可能覆盖当前 kernel 能合理采用的优化项：
   - `skills/optimized-skill/reference/cuda/memory-optim.md`
   - `skills/optimized-skill/reference/cuda/compute-optim.md`
   - `skills/optimized-skill/reference/cuda/sync-optim.md`
3. 首轮允许做覆盖面较广的组合优化，但不要为了堆技巧而引入明显不适配的改法。

#### CUTLASS
优先从 `skills/optimized-skill/reference/cutlass/cutlass-optim.md` 选取与问题形态匹配的方向，例如：
- Tensor Core 路径选择
- tile shape / threadblock shape
- multistage pipeline
- epilogue fusion
- split-K / stream-K
- swizzle / scheduler
- 架构专用特性（Ampere / Hopper / Blackwell）

要求：
- 不要把 CUTLASS 当作“普通 CUDA 杂糅优化”来做
- 要明确写出当前算子更适合哪类 CUTLASS pattern
- 优先修改 template / collective / schedule / epilogue 等结构化参数

#### Triton
优先从 `skills/optimized-skill/reference/triton/triton-optim.md` 选取高收益通用项，例如：
- BLOCK_M / BLOCK_N / BLOCK_K
- `num_warps`
- `num_stages`
- coalescing / vectorization hints
- mask 处理
- fusion
- swizzle / persistent / split-K

要求：
- 首轮先拿到一个形态正确、tile 合理的实现
- 不要首轮就无节制扩大 autotune 候选集
- 要明确记录 shape 与参数的对应关系

### 后续迭代（v1 之后）

后续迭代不要再追求广泛铺开，而要针对“上一轮暴露的最主要不足”做定向修正。

#### CUDA / CUTLASS
规则：
- 先看上一轮 full NCU，再参考 targeted NCU
- 每一轮只优先解决 1 到 2 个最明确的瓶颈
- 每一轮都要显式评估双缓冲 / 多级流水线是否适合当前 kernel
- 不适合时也要写原因，例如数据复用不足、tile 结构不匹配、寄存器或 shared memory 压力过高
- 优化方向要和具体 NCU 信号绑定，例如：
  - coalescing 差 -> 优先修访存布局 / vectorization / tiling
  - occupancy 低 -> 优先修寄存器、smem、block size
  - latency bound -> 优先修 ILP、依赖链、同步粒度，并优先考虑双缓冲 / 多级流水线隐藏访存延迟
  - compute bound -> 优先修 Tensor Core、FMA、低精度路径

#### Triton
规则：
- 先看上一轮 full NCU，再参考 targeted NCU 和 benchmark
- 每一轮只优先解决 1 到 2 个最明确问题
- 重点围绕 tile、`num_warps`、`num_stages`、coalescing、fusion、grid 策略做定向修正
- 不要在后续轮次继续无差别叠加新技巧
- 每轮改动都要能解释“它是在修上一轮的哪个短板”

## 标准执行命令

### CUDA

```bash
python skills/optimized-skill/operator-optimize-loop/scripts/optimize_loop.py <kernel.cu> \
    --backend=cuda --max-iterations=<N> [--ref=<ref.py>] [--DIM=VALUE ...] \
    --warmup=10 --repeat=20 [--nvcc-bin=<nvcc>] [--ncu-bin=<ncu>]
```

### CUTLASS

```bash
python skills/optimized-skill/operator-optimize-loop/scripts/optimize_loop.py <kernel.cu> \
    --backend=cutlass --max-iterations=<N> [--ref=<ref.py>] [--DIM=VALUE ...] \
    --warmup=10 --repeat=20 [--nvcc-bin=<nvcc>] [--ncu-bin=<ncu>]
```

### Triton

```bash
python skills/optimized-skill/operator-optimize-loop/scripts/optimize_loop.py <kernel.py> \
    --backend=triton --max-iterations=<N> [--ref=<ref.py>] [--DIM=VALUE ...] \
    --warmup=10 --repeat=20 [--ncu-bin=<ncu>]
```

### 继续已有 run 目录

```bash
python skills/optimized-skill/operator-optimize-loop/scripts/optimize_loop.py <next_version_file> \
    --backend=<backend> --run-dir=<existing_run_dir> --iteration=<i> --max-iterations=<N> \
    [--ref=<ref.py>] [--DIM=VALUE ...] --warmup=10 --repeat=20
```

### 只检查环境

```bash
python skills/optimized-skill/operator-optimize-loop/scripts/optimize_loop.py <solution_file> \
    --backend=<backend> --max-iterations=<N> --preflight-only
```

## 产物约定

每次 run 都应生成一个 run 目录，目录下至少包含：
- `run_manifest.json`
- `final_summary.md`
- `preflight_check.md`
- `preflight_check.json`
- `iter_v0/`, `iter_v1/`, ...

每轮目录至少包含：
- 当前版本文件（`.cu` 或 `.py`）
- `benchmark_result.json`
- `benchmark.stdout.txt`
- `benchmark.stderr.txt`
- `iteration_summary.md`
- `optimization_proposal.md`

三种后端每轮都要求：
- `targeted.ncu-rep`
- `full.ncu-rep`
- `targeted_summary.txt`
- `targeted_details.txt`
- `full_summary.txt`
- `full_details.txt`

## 策略记忆规则（新增）

每轮都会自动把本轮策略写入两级记忆：
- 当前 run：`run_manifest.json -> strategy_memory.current_run`
- 全局记忆：`skills/optimized-skill/operator-optimize-loop/strategy-memory/global_strategy_memory.json`

`optimization_proposal.md` 必须包含：

```md
## Strategy tags
- tag_a
- tag_b
```

约定：
- 当前 iteration 的结果，会归因到上一轮（`iter_v{i-1}`）proposal 中定义的 strategy tags
- `iter_v0` 没有上一轮 proposal，默认 `baseline`

判定规则（自动）：
- `correctness` 失败 -> `rejected`
- benchmark 失败 -> `rejected`
- targeted/full NCU 执行失败或 full NCU 缺失 -> `rejected`
- 相对上一轮 `kernel median ms` 更快 -> `positive`
- 否则（慢或持平）-> `negative`

下一轮会自动生成两类约束：
- `blocked`：负向/拒绝策略指纹（避免重复）
- `preferred`：正向策略指纹（优先融合）

## 最优版本选择规则

只有满足以下条件的版本才允许参与排名：
- benchmark 成功
- 如果给了 reference，则 correctness 必须通过
- full `.ncu-rep` 必须存在

排序规则：
1. 主排序：kernel median latency 最低
2. 次排序：kernel average latency 最低
3. 再次排序：更早达到该性能的版本优先

## Claude 在循环中的行为要求

- 首轮优化按对应后端 reference 做尽可能广覆盖但仍然合理的组合优化；从第二轮开始，每轮先读上一轮证据，再做针对性修正。
- CUDA 优化建议尽量和 `memory / compute / sync` 三类文档中的已有策略对应起来。
- CUTLASS 优化建议尽量和 example/category 中的成熟 pattern 对应起来。
- Triton 优化建议尽量和 tile / pipeline / memory / fusion / grid 策略对应起来。
- 每轮改动都要能解释它解决的是哪一个具体瓶颈，不要无差别继续堆优化技巧。
- 每轮都要检查当前策略指纹是否命中 `blocked`；若命中，必须避免重复采用。
- 对 `preferred` 中的正向策略要优先评估与本轮方案的融合。
- 不要把 targeted sections 的结论当作最终结论；最终交付必须引用 winning version 的 full NCU 报告。
- correctness 失败的版本可以保留在 run 目录中，但必须明确标记为 rejected，不能参与 best 评选。
- 如果 profiling 不可用、导入失败或 full 报告缺失，要明确写出失败原因并停止把该版本当作最终答案。

## 最终回答必须包含

- 最佳版本路径
- baseline 与最佳版本的 benchmark 对比
- 最佳版本 full NCU 报告路径
- 最佳版本 targeted/full NCU 命令
- 主瓶颈判断
- 采用的关键优化思路
- 被淘汰版本及其淘汰原因（如 correctness fail、NCU 不完整、性能不如当前 best）
- 本轮策略记忆结论（positive/negative/rejected）和对应指纹
- 是否成功避开 blocked 策略，是否融合了 preferred 策略

## 常见提前停止条件

出现以下情况可以提前停止：
- correctness 失败且当前问题明显需要先修正确性
- `ncu` 无法运行或环境不允许 profiling
- 连续多轮没有任何可解释的性能改善
- 已经达到用户要求或收益明显递减
