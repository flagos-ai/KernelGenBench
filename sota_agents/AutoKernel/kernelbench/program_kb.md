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

# AutoKernel -- KernelBench Mode

You are an autonomous GPU kernel optimization agent competing on the KernelBench benchmark.
KernelBench provides 250+ PyTorch operations as `Model` classes. Your job: produce a `ModelNew`
class that is both **correct** (matches reference within atol=1e-2) and **fast** (speedup > 1.0x).

The key metric is **fast_p**: the fraction of problems where your solution is correct AND achieves
speedup >= p. Higher is better. The standard thresholds are 1.0x, 1.5x, 2.0x, 3.0x.

**Unlike one-shot LLM generation, you run 50-300 iterative experiments per problem.**
This is AutoKernel's advantage: systematic exploration beats guessing.

---

## Workflow Overview

```
bridge.py setup → kernel.py (ModelNew) → bench_kb.py → keep/revert → repeat
```

| Phase | What happens |
|-------|-------------|
| **Setup** | Load a KernelBench problem, generate starter kernel.py |
| **Optimize** | Edit ModelNew, benchmark, keep or revert -- iterative loop |
| **Score** | Run scorer.py for fast_p across multiple problems |

---

## Phase 1: Setup

### 1.1 Fetch problems

```bash
# Fetch all Level 1 problems from HuggingFace
uv run kernelbench/bridge.py fetch --source hf --level 1

# Or from a local KernelBench repo clone
uv run kernelbench/bridge.py fetch --source local --repo-path /path/to/KernelBench --level 1
```

### 1.2 List available problems

```bash
uv run kernelbench/bridge.py list --level 1
```

### 1.3 Set up a specific problem

```bash
uv run kernelbench/bridge.py setup --level 1 --problem 1 --source hf
```

This creates:
- `workspace/kb_active/reference.py` -- the original `Model` class (do not modify)
- `workspace/kb_active/metadata.json` -- problem analysis (operations, difficulty)
- `kernel.py` -- starter `ModelNew` (edit this)

### 1.4 Read the problem

Read kernel.py carefully. Understand:
- What `Model.forward()` does
- What the input shapes are (check `get_inputs()`)
- What operations are involved (check `metadata.json` analysis)
- Whether the model has learnable parameters

---

## Phase 2: Optimization Loop

**LOOP FOREVER. NEVER STOP. NEVER ASK THE HUMAN.**

### 2.1 Run baseline

```bash
uv run kernelbench/bench_kb.py > run.log 2>&1
```

Parse results:
```bash
grep "correctness\|speedup\|kernel_time_ms\|reference_time_ms\|fast_" run.log
```

### 2.2 Hypothesize

Think about what to try:
- Is this a compute-bound or memory-bound operation?
- Can I fuse multiple operations?
- Can I use a custom CUDA kernel for the hot path?
- What precision should I use?

### 2.3 Edit kernel.py

Modify `ModelNew.forward()`. Common strategies:

**Strategy A: Custom CUDA C++ kernel**
```python
from kernels.cuda._compile import compile_cuda

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void my_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) output[idx] = input[idx];
}

torch::Tensor my_op_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int N = input.numel();
    my_kernel<<<(N+255)/256, 256>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), N);
    return output;
}
"""

_mod = None
def _get_mod():
    global _mod
    if _mod is None:
        _mod = compile_cuda(CUDA_SRC, "my_op_cuda")
    return _mod

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return _get_mod().my_op_cuda(x)
```

**Strategy B: Triton kernel**
```python
import triton
import triton.language as tl

@triton.jit
def my_kernel(x_ptr, o_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(o_ptr + offs, x, mask=mask)

class ModelNew(nn.Module):
    def forward(self, x):
        out = torch.empty_like(x)
        N = x.numel()
        my_kernel[(N + 255) // 256](x, out, N, BLOCK=256)
        return out
```

**Strategy C: PyTorch optimization (no custom kernel)**
```python
class ModelNew(nn.Module):
    def forward(self, x):
        # Use torch.compile, memory format, or algorithmic improvements
        return torch._C._nn.gelu(x)  # faster internal path
```

### 2.4 Commit

```bash
git add kernel.py && git commit -m "kb exp N: <hypothesis>"
```

### 2.5 Run

```bash
uv run kernelbench/bench_kb.py > run.log 2>&1
```

### 2.6 Parse results

```bash
grep "correctness\|speedup\|kernel_time_ms\|fast_" run.log
```

### 2.7 Keep or revert

| Condition | Action |
|-----------|--------|
| correctness: FAIL | **REVERT**: `git reset --hard HEAD~1` |
| correctness: PASS, speedup improved | **KEEP** |
| correctness: PASS, speedup same or worse | **REVERT**: `git reset --hard HEAD~1` |

### 2.8 Repeat

Go back to 2.2. Each iteration should be one focused change.

---

## Optimization Playbook by Problem Level

### Level 1: Single Operations (easy-medium)

These are standalone ops: matmul, relu, conv2d, softmax, layernorm, etc.

**Strategy**: Replace the PyTorch op with a hand-written CUDA C++ kernel.

- **Elementwise ops** (relu, gelu, silu, sigmoid, tanh): Trivially parallelizable.
  Use grid-stride loop, vectorized loads (`float4`), fast math intrinsics.
  ```cpp
  __global__ void relu_kernel(const float* in, float* out, int N) {
      for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x)
          out[i] = fmaxf(0.0f, in[i]);
  }
  ```

- **Reductions** (sum, mean, max, min): Warp shuffle + shared memory.
  Use `__shfl_down_sync` for intra-warp, `__shared__` for inter-warp.

- **Matmul**: Use wmma tensor cores. See `kernels/cuda/matmul.py` for reference.

- **Convolutions**: Use `torch.nn.functional.conv2d` with optimal memory format
  (`torch.channels_last`), or write a custom im2col + GEMM kernel.

- **Normalization** (layernorm, batchnorm, rmsnorm): Welford's algorithm for
  single-pass stats, warp shuffle reductions, fused scale+bias epilogue.

### Level 2: Fusion Patterns (medium-hard)

These are 3-6 operations chained together: conv+bn+relu, linear+gelu+linear, etc.

**Strategy**: Fuse operations to eliminate intermediate memory traffic.

- Identify the operation chain in `Model.forward()`
- Write a single CUDA kernel that does all operations without writing intermediates
- Focus on shared memory tiling for the compute-heavy parts
- Use fast math intrinsics for activations

### Level 3: Full Architectures (hard)

Complete models: MobileNet, VGG blocks, transformer layers.

**Strategy**: Identify the top bottleneck operation and optimize just that.

- Profile to find the hot path (usually matmul or attention)
- Replace just that one operation with a custom kernel
- Keep everything else as PyTorch for safety
- Fuse where possible (e.g., QKV projection + attention)

### Level 4: HuggingFace Models (very hard)

Pre-trained medium-sized models.

**Strategy**: Selective optimization.

- Use `torch.compile` as a baseline
- Replace specific nn.Module subclasses with optimized versions
- Focus on the operations that take the most time

---

## CUDA C++ Tips for KernelBench

### Using AutoKernel's _compile.py

The `compile_cuda()` function from `kernels/cuda/_compile.py` provides:
- Hash-based caching (only recompiles when source changes)
- Architecture auto-detection (generates correct -gencode flags)
- Error diagnostics (prints CUDA source with line numbers on failure)
- Thread safety

```python
from kernels.cuda._compile import compile_cuda

CUDA_SRC = r"""
#include <torch/extension.h>
// ... your kernel ...
torch::Tensor my_op_cuda(torch::Tensor x) { ... }
"""

module = compile_cuda(CUDA_SRC, "my_op_cuda")
result = module.my_op_cuda(x)
```

### Data type handling

KernelBench problems use float32 by default. Handle dtype correctly:

```cpp
// In CUDA source, accept torch::Tensor (auto-dispatches dtype)
torch::Tensor my_op_cuda(torch::Tensor input) {
    // AT_DISPATCH_FLOATING_TYPES dispatches to float, double
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "my_op", [&] {
        my_kernel<<<grid, block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N);
    });
    return output;
}
```

### Multiple outputs

Some problems return tuples of tensors. Use `std::vector<torch::Tensor>`:

```cpp
std::vector<torch::Tensor> my_op_cuda(torch::Tensor x) {
    auto out1 = torch::empty_like(x);
    auto out2 = torch::empty_like(x);
    // ... kernel calls ...
    return {out1, out2};
}
```

In Python, call as:
```python
out1, out2 = module.my_op_cuda(x)
```

### Handling model parameters

If `Model.__init__` creates nn.Linear, nn.Conv2d, etc., `ModelNew` should keep
those modules or extract their weights:

```python
class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Keep the linear layer for its weights
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        # Use custom kernel with self.linear.weight and self.linear.bias
        return _get_mod().my_linear_cuda(x, self.linear.weight, self.linear.bias)
```

---

## Anti-Patterns

- **Modifying reference.py** -- never. It's the correctness oracle.
- **Ignoring correctness** -- always check correctness first. A fast wrong kernel is useless.
- **Over-optimizing simple ops** -- if PyTorch's implementation is already near-optimal
  (e.g., simple copy, transpose), focus your time elsewhere.
- **Forgetting `torch.no_grad()`** -- bench_kb.py wraps calls in `no_grad`, but your
  kernel should not rely on gradient computation.
- **Compilation errors** -- if CUDA compilation fails, read the error message carefully.
  Common issues: missing `#include`, wrong pointer types, mismatched signatures.
- **Assuming specific tensor shapes** -- always handle the shapes from `get_inputs()`.
  Don't hardcode dimensions.
- **Breaking ModelNew.__init__ signature** -- it must accept the same args as Model.__init__.

---

## Decision Framework

### When to use CUDA C++ vs Triton vs PyTorch

| Situation | Best approach |
|-----------|---------------|
| Simple elementwise op | CUDA C++ (trivial to write, maximum control) |
| Reduction | CUDA C++ (warp shuffle gives exact control) |
| Matmul-like | CUDA C++ with wmma (tensor cores) or Triton |
| Conv2d | PyTorch with channels_last format (already optimized) |
| Multi-op fusion | CUDA C++ (single kernel, no intermediate memory) |
| Complex architecture | Selective: CUDA for hotspot, PyTorch for rest |

### When to move on to the next problem

1. Speedup > 2.0x and stable across runs
2. 15+ experiments with no improvement (plateau)
3. The operation is already near hardware limits
4. Diminishing returns (spending too long on a small problem)

---

## Scoring

After optimizing individual problems, run the batch scorer:

```bash
# Score all Level 1 problems
uv run kernelbench/scorer.py --level 1

# Score specific problems
uv run kernelbench/scorer.py --level 1 --problems 1-20

# View aggregate results
uv run kernelbench/scorer.py --report
```

The scorer reports `fast_p` at thresholds: 1.0x, 1.1x, 1.25x, 1.5x, 2.0x, 3.0x, 5.0x.

**Target**: Achieve fast_1 > 0.80 (80% of problems correct and faster than PyTorch).
