# Competition Anti-Cheat — Triton Kernel 竞赛反作弊沙箱

独立使用的 7 层反作弊检测工具，用于验证 Triton kernel 提交是否存在作弊行为。

## 使用方式

### 命令行（推荐）

```bash
python -m sandbox.competition run path/to/kernel.py
```

kernel 文件要求：
- 定义一个 `forward` 函数，签名 `forward(*inputs) -> output`
- 定义一个 `TEST_INPUTS` 列表，包含若干个 lambda，每个 lambda 返回一组输入

`TEST_INPUTS` 示例：

```python
TEST_INPUTS = [
    lambda: (torch.randn(128, 512, device='cuda', dtype=torch.float16),),
    lambda: (torch.randn(256, 512, device='cuda', dtype=torch.float16),),
]
```

### Python API

```python
from sandbox.competition import run_check

result = run_check(
    kernel_path="path/to/kernel.py",
    generate_inputs=lambda: (torch.randn(128, 512, device='cuda'),),
)

print(result.passed)   # True = 通过
print(result.reason)   # 未通过时的具体原因
```

## 防御能力

7 层沙箱，每层防御一类作弊手段：

| 层 | 作弊手段 | 怎么拦 |
|---|---------|--------|
| 文件系统隔离 | 把结果 pickle 到 `/tmp/`，下次直接读 | HOME 重定向到临时目录，`~/.triton`、`~/.torch` 等缓存路径全部隔离 |
| 环境变量 | 利用 triton autotune 缓存最优配置 | `TRITON_DISABLE_AUTOTUNE=1`，禁用所有编译/缓存相关环境变量 |
| Import Hook | `import mmap` 共享内存、`torch.compile` 提前编译、`triton.autotune` 调优 | 拦截 import 注入 patch，禁用 `autotune`/`compile`/`cuda.graph`/`mmap`/`posix_ipc`；禁用 `print()` 防止嗅探输入 |
| CUDA 保护 | `torch.cuda.graph()` 预录制执行图、TF32 精度缓存 | 禁用 CUDA Graph、make_graphed_callables、CUDA profiler、TF32；重置 stream |
| 随机形状 + Layout | Triton kernel cache 命中，跳过实际计算 | 每轮不同 seed + Tensor layout/stride/contiguity 随机化，杀死 cache key 复用 |
| 进程隔离 | 模块级全局变量缓存（`_CACHE = {}`）跨调用复用 | 每次 `spawn` 独立子进程，全局状态归零 |
| 统计时序校验 | 时序异常：缓存命中导致的不合理低延迟 | CV < 0.15、IQR 离群点检测、收敛分数、20% 复测比对 |

## 检测类型一览

| 作弊行为 | 例子 | 会被哪层抓住 |
|---------|------|------------|
| 磁盘缓存 | `pickle.dump(result, open("/tmp/cache.pkl", "wb"))` | Layer 1 + 2 |
| CUDA Graph 预录制 | `g = torch.cuda.CUDAGraph(); g.capture(...)` | Layer 3 + 4 |
| autotune 配置缓存 | `@triton.autotune(...)` 从 `~/.triton/cache` 读最优配置 | Layer 2 + 3 |
| 禁止模块 | `import mmap` / `import posix_ipc` 共享内存传数据 | Layer 3 |
| 全局状态 | `_CACHE = {}` 模块级字典跨次复用 | Layer 6 |
| 输入嗅探 | `print(x.shape)` 拿到 shape 后做针对性优化 | Layer 3 |

## 参数说明

```python
run_check(
    kernel_path,           # str: kernel .py 文件路径
    generate_inputs,       # Callable: 无参，返回 (arg1, arg2, ...) 或 {"kw": v}
    operator_name="forward",  # str: kernel 中的函数名
    num_tests=10,          # int: 测试轮数
    warmup_runs=1,         # int: 每轮 warmup 次数
    timing_runs=4,         # int: 每轮计时次数
    retest_ratio=0.2,      # float: 复测比例
    cv_threshold=0.15,     # float: 变异系数阈值
    iqr_threshold=0.3,     # float: IQR 异常阈值
    input_is_kwargs=False, # bool: generate_inputs 返回 dict 时设为 True
    verbose=True,          # bool: 是否打印过程
)
# 返回 CheckResult: .passed (bool), .reason (str), .details (dict)
```
