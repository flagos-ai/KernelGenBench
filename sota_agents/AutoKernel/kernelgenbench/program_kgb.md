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

# AutoKernel -- KernelGenBench Mode

You are an autonomous GPU kernel generation and optimization agent working on the KernelGenBench benchmark.
KernelGenBench provides 210 PyTorch operators (110 aten + 50 cublas + 50 vllm). Your job: produce a standalone
Triton kernel that is both **correct** (passes all test cases) and **fast** (speedup >= 1.0x over PyTorch native).

The key metric is **fast_p**: the fraction of problems where your solution is correct AND achieves speedup >= p.

**Unlike one-shot LLM generation, you run iterative experiments per problem.**
This is AutoKernel's advantage: systematic exploration beats guessing.

---

## Kernel Format

**CRITICAL**: KernelGenBench kernels are standalone Triton Python files. They are NOT KernelBench format.

- **NO** `class Model(nn.Module)` or `class ModelNew`
- **NO** `get_inputs()` or `get_init_inputs()`
- **NO** `torch.library` registration (the framework handles this automatically)
- The kernel file contains: Triton kernel functions + Python wrapper functions
- The benchmark tool (`bench_kgb.py`) handles evaluation and registration

Example structure:
```python
import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def my_kernel(...):
    ...

def my_op(input: torch.Tensor, ...) -> torch.Tensor:
    # Launch Triton kernel
    ...
    return output

def my_op_out(input: torch.Tensor, ..., *, out: torch.Tensor) -> torch.Tensor:
    # Implement out variant
    result = my_op(input, ...)
    out.copy_(result)
    return out
```

---

## Workflow Overview

```
bridge_kgb.py setup → kernel.py (standalone Triton) → bench_kgb.py → keep/revert → repeat
```

| Phase | What happens |
|-------|-------------|
| **Setup** | Load a KernelGenBench problem, read operator spec |
| **Generate** | Write initial kernel.py based on operator spec |
| **Optimize** | Edit kernel.py, benchmark, keep or revert -- iterative loop |

---

## Phase 1: Setup

### 1.1 Read the problem

The problem is already set up. Read these files:

```bash
cat workspace/kgb_active/operator_spec.md    # Full operator specification
cat workspace/kgb_active/metadata.json        # Problem metadata (op name, dataset)
```

Understand:
- What the operator does (semantics, edge cases)
- What input shapes and dtypes are tested
- What overloads are required (e.g., `add.Tensor`, `add.Scalar`, `add.out`)
- What tolerance is expected

### 1.2 Generate initial kernel

If `kernel.py` is a stub (contains `# TODO`), write the initial implementation:

1. Read the operator spec carefully
2. Implement all required overloads (wrapper functions for each variant)
3. Handle edge cases (empty tensors, broadcasting, different dtypes)
4. Do NOT add torch.library registration (framework handles this)

### 1.3 Verify baseline

```bash
python kernelgenbench/bench_kgb.py
```

Expected: `correctness: PASS`. If it fails, fix before optimizing.

---

## Phase 2: Optimization Loop

**LOOP UNTIL DONE. NEVER STOP. NEVER ASK THE HUMAN.**

### 2.1 Run benchmark

```bash
python kernelgenbench/bench_kgb.py > run.log 2>&1
```

Parse results:
```bash
grep "correctness\|speedup\|kernel_time_ms\|reference_time_ms\|fast_\|tests" run.log
```

### 2.2 Evaluate result

- `correctness: PASS` + `speedup > current_best` → **KEEP** (new best)
- `correctness: PASS` + `speedup <= current_best` → **REVERT** to best version
- `correctness: FAIL` → **REVERT** to best version

### 2.3 Revert if needed

```bash
git checkout HEAD~1 -- kernel.py
```

Or restore from your saved best version.

### 2.4 Plan next iteration

Before each iteration, think about:
- What bottleneck to address (memory bandwidth, compute, launch overhead)
- What Triton features to try (tiling, vectorization, shared memory, autotune)
- What worked/failed in previous iterations

---

## Optimization Strategies

### Triton-specific techniques

| Technique | When to use |
|-----------|-------------|
| `@triton.autotune` | Always — let Triton search block sizes |
| Vectorized loads | Memory-bound kernels (elementwise, reductions) |
| Shared memory (`tl.load` + reuse) | Reductions, matmul-like ops |
| Loop tiling | Large reductions, softmax |
| Fused operations | Multi-step ops (e.g., add + relu) |
| `tl.dot` with `allow_tf32=False` | Matrix ops requiring FP32 precision |
| Persistent kernels | Small workloads with high launch overhead |

### Common patterns by operator type

| Operator type | Strategy |
|--------------|----------|
| Elementwise (add, mul, relu) | Simple grid, vectorized loads, fuse with neighbors |
| Reduction (sum, mean, norm) | Two-pass or tree reduction, shared memory |
| Softmax/LogSoftmax | Online softmax (numerically stable), fused |
| MatMul/GEMM | Tiled with `tl.dot`, split-K for thin matrices |
| Attention | Flash-attention pattern, tiled Q/K/V |
| Normalization (layernorm, rmsnorm) | Fused mean+var, vectorized |

### What NOT to do

1. **Never wrap in KernelBench format** — no `class Model`, no `get_inputs()`
2. **Never modify bench_kgb.py** or any file in `kernelgenbench/`
3. **Never hack the benchmark** — no timing manipulation, no monkey-patching
4. **Never skip correctness** — every iteration must pass all tests
5. **Never use CUDA C++** — KernelGenBench requires Triton kernels

---

## Output Format

`bench_kgb.py` prints greppable lines:

```
correctness: PASS
speedup: 1.234x
kernel_time_ms: 0.4523
reference_time_ms: 0.5580
tests: 10/10
fast_1.0: PASS
fast_1.5: FAIL
fast_2.0: FAIL
```

---

## When to stop

1. `speedup >= 2.0x` and stable across runs
2. 10+ iterations with no improvement (plateau)
3. All viable Triton optimization approaches exhausted
4. `correctness: PASS` achieved but further speedup requires non-Triton approaches
